/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Segment-level result cache for the token-producer DataProducer.
//
// Agent workloads resend the whole conversation on every turn, so the exact
// render of turn N is the exact prefix of turn N+1. This file turns that
// repetition into a cache: the render backend is wrapped in a decorator that
// keeps the token arrays of previously rendered conversations and, on a prefix
// hit, renders only the new tail instead of the full conversation.
//
// A prefix hit is reusable only when the model's chat template renders each
// message independently of its neighbours ("concatenative"). That is a property
// of (model, template), not of a request, so it is established once by a
// startup probe. Until the probe passes, and whenever it fails, the decorator
// degrades to an exact whole-body cache and then to a plain pass-through, so it
// can only ever return renderer output -- never a synthesised estimate.
package tokenizer

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/cespare/xxhash/v2"
	lru "github.com/hashicorp/golang-lru/v2"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

const (
	// segmentCacheDefaultSize is the number of rendered conversations kept per
	// plugin instance. One entry is one prefix token array, so the default
	// holds a few thousand turns of an agent session.
	segmentCacheDefaultSize = 4096

	// segmentCacheMaxMessages caps the conversations the segment layer will
	// key. Prefix keys are hashed once per message boundary, so the key cost
	// grows quadratically in the message count; past this bound the request
	// still gets the exact whole-body cache and nothing else.
	segmentCacheMaxMessages = 64

	// segmentCacheMinPrefixTokens keeps short prefixes out of the segment
	// layer: below this the extra tail render costs more than it saves.
	segmentCacheMinPrefixTokens = 128

	// segmentProbeAttempts and segmentProbeRetryInterval bound the startup
	// probe, mirroring the render backend's warmup retry policy. The probe
	// keeps retrying because a render endpoint may still be rolling out.
	segmentProbeAttempts      = 12
	segmentProbeRetryInterval = 10 * time.Second

	// segmentReporterInterval is how often the cache logs its counters.
	segmentReporterInterval = 5 * time.Minute

	// segmentProbeAnchorText is the synthetic user message prepended to a tail
	// render. A tail can start on an assistant or tool message, which some chat
	// templates reject as a standalone conversation; the anchor restores a
	// leading user turn. Its own tokens are subtracted from the result.
	segmentProbeAnchorText = "llm-d segment cache anchor"

	roleUser = "user"
)

// segmentCacheConfig configures the segment-level result cache. It is defined
// under the token-producer plugin's parameters:
//
//	{"modelName": "...", "segmentCache": {"enabled": true, "size": 4096}}
//
// The layer is off unless enabled is explicitly true: it trades memory and a
// startup render probe for latency, and both are deployment decisions.
type segmentCacheConfig struct {
	// Enabled turns the segment layer on. The exact whole-body layer is
	// independent of this flag and always on when the block is present.
	Enabled *bool `json:"enabled,omitempty"`
	// Size is the LRU capacity in rendered conversations. Defaults to 4096.
	Size int `json:"size,omitempty"`
	// MinPrefixTokens is the shortest reusable prefix. Defaults to 128.
	MinPrefixTokens *int `json:"minPrefixTokens,omitempty"`
	// Probe runs the startup chat-template probe. Defaults to true; only
	// useful for tests that want to exercise the degraded paths.
	Probe *bool `json:"probe,omitempty"`
}

// segmentCacheStats is a point-in-time copy of the cache counters.
type segmentCacheStats struct {
	Entries       int
	ProbeOK       bool
	ExactHits     uint64
	PrefixHits    uint64
	Misses        uint64
	TailErrors    uint64
	ProbeFailures uint64
}

// segmentCache stores the token IDs of previously rendered conversations,
// keyed by a digest over (envelope, messages[:k]).
//
// The LRU guards itself. The probe result is published once, under mu, and read
// on the request path; the counters are atomic because the request path is
// concurrent and every increment is on a different field.
type segmentCache struct {
	entries *lru.Cache[uint64, []uint32]

	// pluginName identifies the owning plugin instance in logs.
	pluginName string
	// minPrefixTokens is the shortest prefix the segment layer will reuse.
	minPrefixTokens int
	// probeAttempts and probeInterval bound the startup probe. They are fields
	// rather than constants so tests can run the probe without waiting out the
	// production retry budget; production always keeps the defaults below.
	probeAttempts int
	probeInterval time.Duration

	mu sync.RWMutex
	// probeOK reports whether the chat template passed the concatenativity
	// probe. Prefix reuse is enabled only while it is true.
	probeOK bool
	// anchorTokens is the render of the probe anchor as a standalone
	// conversation. A tail render must start with exactly these tokens.
	anchorTokens []uint32

	exactHits     atomic.Uint64
	prefixHits    atomic.Uint64
	misses        atomic.Uint64
	tailErrors    atomic.Uint64
	probeFailures atomic.Uint64
}

// newSegmentCache builds the cache from plugin configuration. It returns nil
// when the block is absent or explicitly disabled, which the caller reads as
// "do not wrap the backend".
func newSegmentCache(cfg *segmentCacheConfig, pluginName string) (*segmentCache, error) {
	if cfg == nil {
		return nil, nil
	}
	if cfg.Enabled != nil && !*cfg.Enabled {
		return nil, nil
	}
	size := cfg.Size
	if size <= 0 {
		size = segmentCacheDefaultSize
	}
	entries, err := lru.New[uint64, []uint32](size)
	if err != nil {
		return nil, fmt.Errorf("invalid segment cache size %d: %w", size, err)
	}
	minPrefix := segmentCacheMinPrefixTokens
	if cfg.MinPrefixTokens != nil && *cfg.MinPrefixTokens >= 0 {
		minPrefix = *cfg.MinPrefixTokens
	}
	return &segmentCache{
		entries:         entries,
		pluginName:      pluginName,
		minPrefixTokens: minPrefix,
		probeAttempts:   segmentProbeAttempts,
		probeInterval:   segmentProbeRetryInterval,
	}, nil
}

// probeEnabled reports whether the startup probe should run.
func (cfg *segmentCacheConfig) probeEnabled() bool {
	return cfg == nil || cfg.Probe == nil || *cfg.Probe
}

// snapshot copies the counters for logging and tests.
func (c *segmentCache) snapshot() segmentCacheStats {
	c.mu.RLock()
	probeOK := c.probeOK
	c.mu.RUnlock()
	return segmentCacheStats{
		Entries:       c.entries.Len(),
		ProbeOK:       probeOK,
		ExactHits:     c.exactHits.Load(),
		PrefixHits:    c.prefixHits.Load(),
		Misses:        c.misses.Load(),
		TailErrors:    c.tailErrors.Load(),
		ProbeFailures: c.probeFailures.Load(),
	}
}

// anchor returns the probe anchor token array and whether prefix reuse is on.
func (c *segmentCache) anchor() ([]uint32, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if !c.probeOK || len(c.anchorTokens) == 0 {
		return nil, false
	}
	return c.anchorTokens, true
}

// setProbe publishes a successful probe result.
func (c *segmentCache) setProbe(anchorTokens []uint32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.anchorTokens = anchorTokens
	c.probeOK = true
}

// startReporter logs the counters periodically so operators can see whether the
// layer pays for itself. The loop exits with ctx.
func (c *segmentCache) startReporter(ctx context.Context) {
	logger := log.FromContext(ctx)
	ticker := time.NewTicker(segmentReporterInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s := c.snapshot()
			logger.V(logutil.DEBUG).Info("token-producer segment cache",
				"plugin", c.pluginName,
				"entries", s.Entries,
				"probeOK", s.ProbeOK,
				"exactHits", s.ExactHits,
				"prefixHits", s.PrefixHits,
				"misses", s.Misses,
				"tailErrors", s.TailErrors,
				"probeFailures", s.ProbeFailures,
			)
		}
	}
}

// segmentPlan is the cache-relevant decomposition of a chat-completions body.
//
// envelope holds the canonical bytes of every body field except "messages";
// messages holds the canonical bytes of each message, in order; keys[k] is the
// digest of envelope followed by messages[:k], so keys has one more element
// than messages and its last element is the whole conversation.
type segmentPlan struct {
	model    string
	envelope map[string]any
	raw      []any
	messages [][]byte
	keys     []uint64
	// reusable reports whether the prefix layer may touch this request. It is
	// false for multimodal bodies, for conversations with a render flag that
	// moves the generation prompt, and for bodies that override the template.
	reusable bool
}

// planSegments decomposes a chat-completions body for the cache. It returns
// (nil, false) for every protocol the segment layer does not model, which is
// the signal to pass the body straight through.
//
// Multimodal and template-overriding bodies are still planned so the exact
// whole-body layer can serve them; only the prefix layer is gated off.
func planSegments(body *fwkrh.InferenceRequestBody) (*segmentPlan, bool) {
	if body == nil || body.ChatCompletions == nil {
		return nil, false
	}
	envelope, messagesValue, ok := payloadEnvelope(body)
	if !ok {
		return nil, false
	}
	raw, ok := messagesValue.([]any)
	if !ok || len(raw) == 0 || len(raw) > segmentCacheMaxMessages {
		return nil, false
	}

	messages := make([][]byte, len(raw))
	reusable := true
	for i, msg := range raw {
		encoded, err := json.Marshal(msg)
		if err != nil {
			return nil, false
		}
		messages[i] = encoded
		if !reusable {
			continue
		}
		if containsMultimodal(msg) || !structuredRoles(msg) {
			reusable = false
		}
	}
	if flagged(envelope, "add_generation_prompt") ||
		flagged(envelope, "continue_final_message") ||
		flagged(envelope, "return_assistant_tokens_mask") ||
		envelope["chat_template"] != nil ||
		envelope["documents"] != nil {
		reusable = false
	}

	model, _ := envelope["model"].(string)
	return &segmentPlan{
		model:    model,
		envelope: envelope,
		raw:      raw,
		messages: messages,
		keys:     prefixKeys(envelope, messages),
		reusable: reusable,
	}, true
}

// payloadEnvelope returns the request body as a payload map without the
// "messages" entry, together with the messages value the segment layer keys and
// rebuilds separately. Messages is removed from the envelope so that the
// envelope digest is the same for every turn of a session.
//
// WirePayload keeps the parser's raw bytes until something mutates the body, so
// both representations are decoded here rather than at the call site.
func payloadEnvelope(body *fwkrh.InferenceRequestBody) (map[string]any, any, bool) {
	payload := body.WirePayload()
	m, ok := payload.AsMap()
	if !ok {
		raw, isRaw := payload.(fwkrh.RawPayload)
		if !isRaw || len(raw) == 0 {
			return nil, nil, false
		}
		decoded := map[string]any{}
		if err := json.Unmarshal(raw, &decoded); err != nil {
			return nil, nil, false
		}
		m = fwkrh.PayloadMap(decoded)
	}
	envelope := make(map[string]any, len(m))
	var messages any
	for k, v := range m {
		if k == "messages" {
			messages = v
			continue
		}
		envelope[k] = v
	}
	return envelope, messages, true
}

// prefixKeys digests every message boundary. The envelope is re-encoded with
// encoding/json, which sorts map keys, so the digest is stable across requests
// that carry the same body in a different field order.
func prefixKeys(envelope map[string]any, messages [][]byte) []uint64 {
	keys := make([]uint64, len(messages)+1)
	buf := make([]byte, 0, 4096)
	if encoded, err := json.Marshal(envelope); err == nil {
		buf = append(buf, encoded...)
	}
	keys[0] = xxhash.Sum64(buf)
	for i, msg := range messages {
		buf = append(buf, '\n')
		buf = append(buf, msg...)
		keys[i+1] = xxhash.Sum64(buf)
	}
	return keys
}

// flagged reports whether a top-level body field is present and truthy. The
// parser leaves body fields as decoded JSON, so booleans arrive as bool.
func flagged(envelope map[string]any, key string) bool {
	v, ok := envelope[key]
	if !ok {
		return false
	}
	b, ok := v.(bool)
	return ok && b
}

// structuredRoles reports whether a decoded message carries a string role,
// which every chat template needs in order to render it in isolation.
func structuredRoles(msg any) bool {
	m, ok := msg.(map[string]any)
	if !ok {
		return false
	}
	role, ok := m["role"].(string)
	return ok && role != ""
}

// multimodalKeys are the content-block keys that make a message's token
// positions depend on an image/audio processor, which the segment layer cannot
// splice by concatenation.
var multimodalKeys = []string{"image_url", "audio_url", "input_audio", "video_url", "image", "audio", "video"}

// containsMultimodal reports whether a decoded message holds multimodal
// content anywhere in its content blocks.
func containsMultimodal(msg any) bool {
	m, ok := msg.(map[string]any)
	if !ok {
		return false
	}
	return anyMultimodal(m["content"])
}

func anyMultimodal(v any) bool {
	switch t := v.(type) {
	case map[string]any:
		for _, key := range multimodalKeys {
			if present, ok := t[key]; ok && present != nil {
				return true
			}
		}
		for _, child := range t {
			if anyMultimodal(child) {
				return true
			}
		}
	case []any:
		for _, child := range t {
			if anyMultimodal(child) {
				return true
			}
		}
	}
	return false
}

// cachingProducer wraps a render backend with the segment and exact caches.
// It implements the backend interfaces the plugin and director rely on, so the
// wrapper is invisible to them.
type cachingProducer struct {
	inner      tokenInputProducer
	cache      *segmentCache
	pluginName string
}

var (
	_ tokenInputProducer = &cachingProducer{}
	_ warmer             = &cachingProducer{}
	_ timeoutAware       = &cachingProducer{}
)

// warmup forwards the render backend's warmup so wrapping does not skip it.
func (c *cachingProducer) warmup(ctx context.Context) {
	if w, ok := c.inner.(warmer); ok {
		w.warmup(ctx)
	}
}

// produceTimeout forwards the wrapped backend's render timeout.
func (c *cachingProducer) produceTimeout() time.Duration {
	if ta, ok := c.inner.(timeoutAware); ok {
		return ta.produceTimeout()
	}
	return 0
}

// produce resolves a request from the cache when it can and from the wrapped
// backend otherwise.
//
// The lookup order is: exact whole-body hit, prefix hit plus tail render, full
// render. Every path returns renderer output for the request's own bytes; a
// failed tail render falls back to the full render rather than propagating, so
// the cache can add latency but not errors.
func (c *cachingProducer) produce(ctx context.Context, body *fwkrh.InferenceRequestBody) (*fwkrh.TokenizedRequest, error) {
	plan, ok := planSegments(body)
	if !ok {
		return c.inner.produce(ctx, body)
	}
	full := plan.keys[len(plan.keys)-1]

	if tok, hit := c.cache.entries.Get(full); hit && len(tok) > 0 {
		c.cache.exactHits.Add(1)
		return tokenizedFrom(tok), nil
	}

	if plan.reusable {
		if tr, ok := c.produceFromPrefix(ctx, plan, full); ok {
			return tr, nil
		}
	}

	c.cache.misses.Add(1)
	tr, err := c.inner.produce(ctx, body)
	if err != nil {
		return nil, err
	}
	if tok, ok := singleTextPrompt(tr); ok {
		c.cache.entries.Add(full, tok)
	}
	return tr, nil
}

// produceFromPrefix serves a request by concatenating a cached prefix with a
// freshly rendered tail. It reports false, without an error, for every case the
// caller should resolve with a full render.
func (c *cachingProducer) produceFromPrefix(ctx context.Context, plan *segmentPlan, full uint64) (*fwkrh.TokenizedRequest, bool) {
	anchorTokens, ready := c.cache.anchor()
	if !ready {
		return nil, false
	}

	// The previous turns of a session are the prefixes of the current one, so
	// the longest match is usually the whole conversation minus one message.
	for k := len(plan.messages) - 1; k >= 1; k-- {
		prefix, hit := c.cache.entries.Get(plan.keys[k])
		if !hit || len(prefix) < c.cache.minPrefixTokens {
			continue
		}
		tail, err := c.renderTail(ctx, plan, k, anchorTokens)
		if err != nil {
			c.cache.tailErrors.Add(1)
			log.FromContext(ctx).V(logutil.DEBUG).Info(
				"token-producer segment cache: tail render failed, falling back to a full render",
				"plugin", c.pluginName, "prefixMessages", k, "err", err)
			return nil, false
		}
		merged := make([]uint32, 0, len(prefix)+len(tail))
		merged = append(merged, prefix...)
		merged = append(merged, tail...)
		c.cache.entries.Add(full, merged)
		c.cache.prefixHits.Add(1)
		return tokenizedFrom(merged), true
	}
	return nil, false
}

// renderTail renders messages[k:] as an anchor-led conversation and returns the
// tokens the anchor did not produce.
//
// The anchor check is not a formality: it is what makes the splice sound. If
// the renderer returns anything but the known anchor tokens up front, the
// template is not concatenative for this body and the caller must not splice.
func (c *cachingProducer) renderTail(ctx context.Context, plan *segmentPlan, k int, anchorTokens []uint32) ([]uint32, error) {
	tail := make([]any, 0, 1+len(plan.raw)-k)
	tail = append(tail, map[string]any{"role": roleUser, "content": segmentProbeAnchorText})
	tail = append(tail, plan.raw[k:]...)

	payload := make(fwkrh.PayloadMap, len(plan.envelope)+1)
	for key, value := range plan.envelope {
		payload[key] = value
	}
	payload["messages"] = tail

	body := &fwkrh.InferenceRequestBody{
		ChatCompletions: &fwkrh.ChatCompletionsRequest{},
		Payload:         payload,
		Mutated:         true,
	}
	tr, err := c.inner.produce(ctx, body)
	if err != nil {
		return nil, err
	}
	tokens, ok := singleTextPrompt(tr)
	if !ok {
		return nil, errors.New("tail render returned no single text prompt")
	}
	if len(tokens) <= len(anchorTokens) || !equalTokens(tokens[:len(anchorTokens)], anchorTokens) {
		return nil, errors.New("tail render did not reproduce the anchor prefix")
	}
	return tokens[len(anchorTokens):], nil
}

// tokenizedFrom wraps a cached token array in a fresh TokenizedRequest. The
// slice is copied because consumers may hold or mutate the request's prompts
// while the cache keeps serving the same entry.
func tokenizedFrom(tokens []uint32) *fwkrh.TokenizedRequest {
	cloned := make([]uint32, len(tokens))
	copy(cloned, tokens)
	return &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{{TokenIDs: cloned}}}
}

// singleTextPrompt returns the token IDs of a text-only single-prompt result.
// Multimodal results are excluded: their features carry token offsets, so a
// cached array cannot be reused without re-deriving them.
func singleTextPrompt(tr *fwkrh.TokenizedRequest) ([]uint32, bool) {
	if tr == nil || len(tr.Prompts) != 1 {
		return nil, false
	}
	prompt := tr.Prompts[0]
	if len(prompt.MultiModalFeatures) > 0 || len(prompt.TokenIDs) == 0 {
		return nil, false
	}
	return prompt.TokenIDs, true
}

// equalTokens compares two token arrays element-wise.
func equalTokens(a, b []uint32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// runProbe establishes whether the configured renderer's chat template is
// concatenative, and publishes the anchor tokens the tail render depends on.
//
// It retries like the render backend warmup because the endpoint may still be
// rolling out. Until it succeeds the decorator serves exact hits and plain
// renders only, which is the same contract as a cache that is warming up.
func (c *segmentCache) runProbe(ctx context.Context, inner tokenInputProducer, model string) {
	logger := log.FromContext(ctx)
	for attempt := 0; attempt < c.probeAttempts; attempt++ {
		anchor, err := probeConcatenative(ctx, inner, model)
		if err == nil {
			c.setProbe(anchor)
			logger.V(logutil.DEFAULT).Info(
				"token-producer segment cache probe passed; prefix reuse enabled",
				"plugin", c.pluginName, "anchorTokens", len(anchor))
			return
		}
		c.probeFailures.Add(1)
		if c.probeInterval <= 0 {
			continue
		}
		select {
		case <-time.After(c.probeInterval):
		case <-ctx.Done():
			return
		}
	}
	logger.V(logutil.DEFAULT).Info(
		"token-producer segment cache probe did not pass; prefix reuse stays disabled",
		"plugin", c.pluginName,
		"attempts", c.probeAttempts)
}

// probeConcatenative checks the identity the prefix layer relies on:
//
//	tokens([a, b, c, d]) == tokens([a, b]) ++ (tokens([anchor, c, d]) - tokens([anchor]))
//
// Both the message roles and the text are synthetic, so a passing probe proves
// the template adds a fixed amount of glue per message rather than conditioning
// a message on its neighbours. The anchor subtraction is checked directly,
// because a template that special-cases the first message would otherwise turn
// the anchor into part of the tail.
func probeConcatenative(ctx context.Context, inner tokenInputProducer, model string) ([]uint32, error) {
	first := probeMessage(roleUser, "segment probe alpha")
	second := probeMessage("assistant", "segment probe beta")
	third := probeMessage(roleUser, "segment probe gamma")
	fourth := probeMessage("assistant", "segment probe delta")

	anchor, err := renderProbe(ctx, inner, model, probeMessage(roleUser, segmentProbeAnchorText))
	if err != nil {
		return nil, err
	}
	if len(anchor) == 0 {
		return nil, errors.New("probe anchor rendered no tokens")
	}
	head, err := renderProbe(ctx, inner, model, first, second)
	if err != nil {
		return nil, err
	}
	whole, err := renderProbe(ctx, inner, model, first, second, third, fourth)
	if err != nil {
		return nil, err
	}
	tail, err := renderProbe(ctx, inner, model, probeMessage(roleUser, segmentProbeAnchorText), third, fourth)
	if err != nil {
		return nil, err
	}

	if len(whole) <= len(head) || !equalTokens(whole[:len(head)], head) {
		return nil, errors.New("template is not prefix-stable at a message boundary")
	}
	if len(tail) <= len(anchor) || !equalTokens(tail[:len(anchor)], anchor) {
		return nil, errors.New("template did not reproduce the probe anchor")
	}
	if !equalTokens(whole[len(head):], tail[len(anchor):]) {
		return nil, errors.New("template renders messages differently in context")
	}
	return anchor, nil
}

// probeMessage builds a synthetic chat message for the probe.
func probeMessage(role, text string) any {
	return map[string]any{"role": role, "content": text}
}

// renderProbe renders a synthetic conversation through the wrapped backend.
func renderProbe(ctx context.Context, inner tokenInputProducer, model string, messages ...any) ([]uint32, error) {
	body := &fwkrh.InferenceRequestBody{
		ChatCompletions: &fwkrh.ChatCompletionsRequest{},
		Payload:         fwkrh.PayloadMap{"model": model, "messages": messages},
		Mutated:         true,
	}
	tr, err := inner.produce(ctx, body)
	if err != nil {
		return nil, err
	}
	tokens, ok := singleTextPrompt(tr)
	if !ok {
		return nil, errors.New("probe render returned no single text prompt")
	}
	return tokens, nil
}

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

package tokenizer

import (
	"context"
	"errors"
	"testing"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// fakeRenderer stands in for the render backend. Its "template" is either
// context-free per message (the case prefix reuse is built for) or sensitive to
// a message's index in the conversation (the case the probe must reject).
type fakeRenderer struct {
	calls            int
	contextSensitive bool
}

func (f *fakeRenderer) produce(_ context.Context, body *fwkrh.InferenceRequestBody) (*fwkrh.TokenizedRequest, error) {
	f.calls++
	payload, ok := body.WirePayload().AsMap()
	if !ok {
		return nil, errors.New("fake renderer got a non-map payload")
	}
	messages, ok := payload["messages"].([]any)
	if !ok {
		return nil, errors.New("fake renderer got no messages")
	}
	var tokens []uint32
	for i, raw := range messages {
		msg, ok := raw.(map[string]any)
		if !ok {
			return nil, errors.New("fake renderer got a non-object message")
		}
		if f.contextSensitive {
			tokens = append(tokens, uint32(i+1))
		}
		role, _ := msg["role"].(string)
		for _, r := range role {
			tokens = append(tokens, uint32(r))
		}
		text, _ := msg["content"].(string)
		for _, r := range text {
			tokens = append(tokens, uint32(r))
		}
		tokens = append(tokens, 0)
	}
	return &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{{TokenIDs: tokens}}}, nil
}

// chatBody builds a chat-completions body from role/content pairs.
func chatBody(model string, pairs ...string) *fwkrh.InferenceRequestBody {
	messages := make([]any, 0, len(pairs)/2)
	for i := 0; i+1 < len(pairs); i += 2 {
		messages = append(messages, map[string]any{"role": pairs[i], "content": pairs[i+1]})
	}
	return &fwkrh.InferenceRequestBody{
		ChatCompletions: &fwkrh.ChatCompletionsRequest{},
		Payload:         fwkrh.PayloadMap{"model": model, "messages": messages},
		Mutated:         true,
	}
}

// startCache wraps a fake renderer and runs the probe synchronously so the test
// does not depend on goroutine scheduling.
func startCache(t *testing.T, inner tokenInputProducer, contextSensitive bool) *cachingProducer {
	t.Helper()
	renderer, ok := inner.(*fakeRenderer)
	if !ok {
		t.Fatalf("inner is %T, want *fakeRenderer", inner)
	}
	renderer.contextSensitive = contextSensitive
	enabled, probe, minTokens := true, true, 0
	cache, err := newSegmentCache(
		&segmentCacheConfig{Enabled: &enabled, Probe: &probe, MinPrefixTokens: &minTokens},
		"test")
	if err != nil {
		t.Fatalf("newSegmentCache: %v", err)
	}
	// One attempt, no retry delay: the probe either passes on the synthetic
	// template or it does not, and the tests should not wait out the
	// production retry budget.
	cache.probeAttempts = 1
	cache.probeInterval = 0
	cache.runProbe(context.Background(), inner, "test-model")
	// runProbe renders its own probes; only the calls made by produce() are of
	// interest to the assertions below.
	renderer.calls = 0
	return &cachingProducer{inner: inner, cache: cache, pluginName: "test"}
}

func TestProduceFromPrefixReusesRenderedTail(t *testing.T) {
	renderer := &fakeRenderer{}
	producer := startCache(t, renderer, false)
	ctx := context.Background()

	first, err := producer.produce(ctx, chatBody("m", "user", "hello", "assistant", "hi"))
	if err != nil {
		t.Fatalf("first produce: %v", err)
	}
	if renderer.calls != 1 {
		t.Fatalf("after the first turn calls = %d, want 1", renderer.calls)
	}

	// The second turn resends the conversation with one more message. Only the
	// tail may be rendered.
	second, err := producer.produce(ctx, chatBody("m", "user", "hello", "assistant", "hi", "user", "more"))
	if err != nil {
		t.Fatalf("second produce: %v", err)
	}
	if renderer.calls != 2 {
		t.Fatalf("after the second turn calls = %d, want 2 (prefix reuse)", renderer.calls)
	}

	// The spliced result must equal what a full render of the second turn
	// produces, token for token.
	full := &fakeRenderer{}
	producer2 := startCache(t, full, false)
	want, err := producer2.produce(ctx, chatBody("m", "user", "hello", "assistant", "hi", "user", "more"))
	if err != nil {
		t.Fatalf("reference produce: %v", err)
	}
	if !equalTokens(second.Prompts[0].TokenIDs, want.Prompts[0].TokenIDs) {
		t.Fatalf("spliced tokens %v != full render %v", second.Prompts[0].TokenIDs, want.Prompts[0].TokenIDs)
	}
	if len(first.Prompts[0].TokenIDs) == 0 {
		t.Fatal("first turn produced no tokens")
	}
	if got := producer.cache.prefixHits.Load(); got != 1 {
		t.Fatalf("prefixHits = %d, want 1", got)
	}
}

func TestProduceExactHitSkipsRender(t *testing.T) {
	renderer := &fakeRenderer{}
	producer := startCache(t, renderer, false)
	ctx := context.Background()

	a, err := producer.produce(ctx, chatBody("m", "user", "hello", "assistant", "hi"))
	if err != nil {
		t.Fatalf("first produce: %v", err)
	}
	b, err := producer.produce(ctx, chatBody("m", "user", "hello", "assistant", "hi"))
	if err != nil {
		t.Fatalf("second produce: %v", err)
	}
	if renderer.calls != 1 {
		t.Fatalf("calls = %d, want 1", renderer.calls)
	}
	if !equalTokens(a.Prompts[0].TokenIDs, b.Prompts[0].TokenIDs) {
		t.Fatal("cached tokens differ from the rendered ones")
	}
	// A cache hit must hand out an independent slice: consumers own their copy.
	b.Prompts[0].TokenIDs[0] = 0xffff
	c, err := producer.produce(ctx, chatBody("m", "user", "hello", "assistant", "hi"))
	if err != nil {
		t.Fatalf("third produce: %v", err)
	}
	if c.Prompts[0].TokenIDs[0] == 0xffff {
		t.Fatal("a cache hit aliased the caller's slice")
	}
}

func TestProbeRejectsContextSensitiveTemplate(t *testing.T) {
	renderer := &fakeRenderer{}
	producer := startCache(t, renderer, true)
	ctx := context.Background()

	if _, ready := producer.cache.anchor(); ready {
		t.Fatal("probe accepted a template that renders messages in context")
	}
	if _, err := producer.produce(ctx, chatBody("m", "user", "hello", "assistant", "hi")); err != nil {
		t.Fatalf("first produce: %v", err)
	}
	if _, err := producer.produce(ctx, chatBody("m", "user", "hello", "assistant", "hi", "user", "more")); err != nil {
		t.Fatalf("second produce: %v", err)
	}
	if renderer.calls != 2 {
		t.Fatalf("calls = %d, want 2 (no prefix reuse without a passing probe)", renderer.calls)
	}
	if got := producer.cache.probeFailures.Load(); got == 0 {
		t.Fatal("probe failure was not recorded")
	}
}

func TestPlanSegmentsGatesPrefixReuse(t *testing.T) {
	long := "padding padding padding padding padding"
	base := chatBody("m", "user", long)
	plan, ok := planSegments(base)
	if !ok || !plan.reusable {
		t.Fatalf("plain chat body: ok=%v reusable=%v, want true/true", ok, plan.reusable)
	}

	withGenerationPrompt := chatBody("m", "user", long)
	withGenerationPrompt.Payload.(fwkrh.PayloadMap)["add_generation_prompt"] = true
	if plan, _ := planSegments(withGenerationPrompt); plan.reusable {
		t.Fatal("add_generation_prompt body was marked reusable")
	}

	withImage := chatBody("m", "user", long)
	withImage.Payload.(fwkrh.PayloadMap)["messages"] = []any{
		map[string]any{"role": "user", "content": []any{
			map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64,AA"}},
		}},
	}
	if plan, _ := planSegments(withImage); plan.reusable {
		t.Fatal("multimodal body was marked reusable")
	}

	completions := &fwkrh.InferenceRequestBody{Completions: &fwkrh.CompletionsRequest{}}
	if _, ok := planSegments(completions); ok {
		t.Fatal("completions body was planned for the segment cache")
	}
}

func TestSegmentCacheDisabledByDefault(t *testing.T) {
	cache, err := newSegmentCache(nil, "test")
	if err != nil {
		t.Fatalf("newSegmentCache(nil): %v", err)
	}
	if cache != nil {
		t.Fatal("a nil config enabled the segment cache")
	}
	disabled := false
	cache, err = newSegmentCache(&segmentCacheConfig{Enabled: &disabled}, "test")
	if err != nil {
		t.Fatalf("newSegmentCache(disabled): %v", err)
	}
	if cache != nil {
		t.Fatal("an explicitly disabled config enabled the segment cache")
	}
}

func TestCacheSaltIsolatesEntries(t *testing.T) {
	renderer := &fakeRenderer{}
	producer := startCache(t, renderer, false)
	ctx := context.Background()

	a := chatBody("m", "user", "hello")
	a.Payload.(fwkrh.PayloadMap)["cache_salt"] = "tenant-a"
	if _, err := producer.produce(ctx, a); err != nil {
		t.Fatalf("first produce: %v", err)
	}
	b := chatBody("m", "user", "hello")
	b.Payload.(fwkrh.PayloadMap)["cache_salt"] = "tenant-b"
	if _, err := producer.produce(ctx, b); err != nil {
		t.Fatalf("second produce: %v", err)
	}
	if renderer.calls != 2 {
		t.Fatalf("calls = %d, want 2 (cache salt must not share entries)", renderer.calls)
	}
}

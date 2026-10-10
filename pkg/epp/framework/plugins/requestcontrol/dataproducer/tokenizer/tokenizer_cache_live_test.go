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
	"fmt"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
)

// TestTokenizerCacheParityLive compares complete OFF/ON render IDs through Produce.
func TestTokenizerCacheParityLive(t *testing.T) {
	on := renderEndpointForTest(t)
	off := on
	off.url = strings.TrimRight(os.Getenv("VLLM_RENDER_TEST_BASELINE_URL"), "/")
	if off.url == "" {
		t.Skip("set VLLM_RENDER_TEST_BASELINE_URL to a matching cache-disabled renderer")
	}
	for _, tc := range []struct{ name, fields string }{
		{"unicode", `"messages":[{"role":"user","content":"Explain Python: 你好 👋"}]`},
		{"salt", `"messages":[{"role":"user","content":"Explain Python."}],"cache_salt":"tenant-a"`},
		{"growing", `"messages":[{"role":"user","content":"Explain Python."},{"role":"assistant","content":"Python is a language."},{"role":"user","content":"Continue with examples."}]`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := []byte(fmt.Sprintf(`{"model":%q,"max_tokens":1,%s}`, on.model, tc.fields))
			expected := off.render(t, chatRenderPath, raw)
			for _, endpoint := range []liveRenderEndpoint{off, on} {
				target, err := url.Parse(endpoint.url)
				require.NoError(t, err)
				p := endpoint.plugin(t, httputil.NewSingleHostReverseProxy(target))
				for range 3 {
					parsed, err := openai.NewOpenAIParser().ParseRequest(t.Context(), raw, map[string]string{":path": "/v1/chat/completions"})
					require.NoError(t, err)
					req := &scheduling.InferenceRequest{Body: parsed.Body, Headers: map[string]string{"authorization": endpoint.auth}}
					require.NoError(t, p.Produce(t.Context(), req, nil))
					require.Equal(t, expected, req.Body.TokenizedRequest.Prompts)
				}
			}
		})
	}
}

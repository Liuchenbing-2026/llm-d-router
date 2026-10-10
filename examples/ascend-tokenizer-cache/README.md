# Ascend tokenizer-cache backend integration

| Delivery | Status |
| --- | --- |
| Backend startup and offline verification helpers | Implemented and locally checked |
| Native tokenizer data producer | Existing vLLM /render path reused; OFF/ON live acceptance added |
| End-to-end integration and hardware acceptance | Pending |

## Component integration

Keep the existing token-producer plugin and configure its native vLLM renderer:

~~~yaml
- type: token-producer
  parameters:
    modelName: local-model
    vllm:
      url: http://127.0.0.1:8000
~~~

This is a plugin fragment, not a complete EndpointPickerConfig. Preserve the
precise-prefix producer, cache hashing/group settings and scheduling profile
from the deployment. The configured service must expose the native render
endpoints used by the plugin and contain the compatible Ascend frontend patch.

The /tokenize comparison below verifies a subset of frontend IDs. It does not
validate the token-producer's /v1/chat/completions/render response, multimodal
offsets, forwarded token IDs, or P/D handoff. Those need separate service
integration tests. The Go plugin performs no local segment-cache lookup.
Speculative residency and history hints must not be reported as event-backed KV.

## Backend contract

| Item | Value |
| --- | --- |
| vLLM-Ascend source | 4ea839b813fecbd1aba3affa63e3c896e75b123b |
| Paired vLLM reference | ced6857afa0ea7b2e3f0846a62e1394e90f15607 |
| Cache option | additional_config.tokenizer_cache_gb; default 0 |
| Local checks | Python 3.9.6, stdlib; no image or inference service used |
| Hardware, throughput, quality | Not run |
| Task branch | xxx_tokenizer_cache |
| Component base SHA | b2fe9257abe72aa7650265f32a965e14fc235fa6 |
| Delivery SHA | Commit containing this example; inspect with git log -1 -- . |

Install the paired sources in an Ascend environment with its compatible
CANN, torch-npu and model dependencies. source_contract.json records fingerprints
of the relevant installed patch/config/renderer files. The launcher verifies
these files before execution. This detects this specific interface mismatch;
it does not certify the complete environment or all model templates.

Source: [Ascend frontend cache](https://github.com/vllm-project/vllm-ascend/pull/16298).
The patch remains an unmerged dependency. Engine CLI arguments belong to the
backend startup command; do not insert them into an OpenAI request body.

## Commands

Dry-run command construction without installing the backend:

~~~bash
python3 launch_backend.py --model Qwen/Qwen3-0.6B --cache-gb 4 --dry-run \
  -- --served-model-name local-model --max-model-len 8192
python3 -m unittest -v test_example.py
~~~

After preparing the pinned backend, start OFF and ON servers in separate
terminals, with separately allocated resources:

~~~bash
python3 launch_backend.py --model Qwen/Qwen3-0.6B --cache-gb 0 --port 8000 \
  -- --served-model-name local-model --max-model-len 8192
python3 launch_backend.py --model Qwen/Qwen3-0.6B --cache-gb 4 --port 8001 \
  -- --served-model-name local-model --max-model-len 8192
python3 verify_tokens.py --baseline http://127.0.0.1:8000 \
  --cached http://127.0.0.1:8001 --model local-model --rounds 3 \
  --output token-conformance.json
~~~

The service commands above are provided for hardware execution and were not
run locally. The comparison requires complete token-ID equality across OFF,
ON and repeated synthetic inputs. Report a failure on different IDs,
nondeterminism or malformed responses. Its per-call timing is diagnostic;
sequential /tokenize calls are not an inference load benchmark.

For workload evaluation, keep the same model, template, API-process count,
hardware, arrival trace and KV-routing policy. Measure tokenizer CPU time,
process RSS, TTFT, completion latency and success-task cost in the existing
benchmark harness. Include tools, truncation and template changes in model
conformance before deployment. No performance result is supplied here.

## Boundaries

The cache stores text-segment-to-token-ID results in host memory. Its budget
is an estimate per tokenizer instance rather than an RSS limit. API-process
count and DP-rank count are separate. A cache hit does not establish physical
KV residency, reduce logical request token count, or enable cross-model KV
reuse. Cache-OFF preserves the budget switch; wrappers still require the
paired renderer interfaces.

Public offline checks cover launcher argument preservation, invalid budgets,
installed-file mismatch and complete-ID comparison failure. These checks do
not substitute for native Router builds, real backend conformance, cluster
tests, generation/eviction recovery or P/D acceptance.

## Existing native EPP producer and new live acceptance

The existing `backendType: vllm` token producer calls the model's native
`/render`, preserves the original wire request and shares exact PromptTokens
within that request. Use this path for compatible Ascend renderer backends;
no second Go encoding-result cache is introduced. `estimate` produces an
estimate and is not an equivalent exact-token backend.

Actual package test command (Go 1.27.2, macOS arm64):

```bash
go test ./pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer -count=1 -json
```

The final run has 449 passing test entries **including subtests**, zero
failures, and 10 skipped live/model tests. This is a native Go producer /
mock transport result, not a running Kubernetes Gateway or NPU acceptance.
The new `TestTokenizerCacheParityLive` invokes the real producer against
OFF/ON renderers, checks full IDs and multimodal features, and skips explicitly
when the required endpoints are absent. To execute it on prepared services:

```bash
export VLLM_RENDER_TEST_BASELINE_URL=http://127.0.0.1:8000
export VLLM_RENDER_TEST_URL=http://127.0.0.1:8001
export VLLM_RENDER_TEST_MODEL=local-model
export VLLM_RENDER_TEST_ALIAS=local-model
export VLLM_RENDER_TEST_SERVE=1
go test ./pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer \
  -run 'TestTokenizerCacheParityLive|TestRenderServingLive' -count=1 -v
```

Set credentials using the existing live-test environment contract if needed.
Deployment must point render metadata at the same model/template as the target
engine. Cache-OFF/ON equivalence does not establish physical KV residency.

## Native render and serving conformance (2026-10-10)

`verify_serving.py` supplements `/tokenize` checks with the actual chat-render
protocol and, with `--serve`, generation request ingress. It compares complete
OFF/ON token IDs, repetition, growing conversations, Unicode, tenant salts and
tools. Serving must return the same `prompt_token_ids` and prompt usage.

```bash
python3 -m unittest -v test_example.py test_serving.py
python3 verify_serving.py --baseline http://127.0.0.1:8000 \
  --cached http://127.0.0.1:8001 --model local-model --rounds 3 --serve \
  --output render-serving-conformance.json
```

`BACKEND_API_KEY`, if required, is read from the environment. Never commit it.
The six harness tests use local HTTP fixture servers, not an inference model.
The serving command requires prepared OFF/ON backends and is **not executed**
on NPU in this delivery. Output timing is diagnostic only; the harness does not
measure generation quality, inference throughput, or full Agent success.
Generation text hashes are recorded without treating nondeterministic text as
a tokenizer-equivalence failure. Real workload evaluation must use held-out
Agent tasks and include retries, failures, complete latency and total cost.

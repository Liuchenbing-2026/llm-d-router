# Native vLLM tokenizer-cache backend integration

The backend uses vLLM's `--tokenizer-cache-gb` option. The paired Ascend
`additional_config.tokenizer_cache_gb` is fixed at zero; it does not enable this
native cache. Pin vLLM `b37f01391ec3b0c028ab6947e4ea3b6f7846504d` for the Ascend
0.30 compatibility environment. Its segment-cache module is byte-identical to
the implementation at `45a0e62f241dffeefdb701c4c1c60b874c1be373`. Full serving
of that newer upstream-main base with the paired Ascend plugin fails at an
incompatible private KV-cache import, so these versions are not interchangeable.

The source-file contract is checked before launch. Cache activation and hits
require runtime verification; matching files alone does not establish a gain.
The native CLI budget is a backend startup option, never an OpenAI request field.

| Delivery | Status |
| --- | --- |
| Backend startup and offline verification helpers | Implemented and locally checked |
| Native tokenizer data producer | Existing vLLM /render path reused; OFF/ON live acceptance added |
| CPU render / native producer acceptance | Passed in the declared Qwen3 text scope; see 2026-10-11 deployment section |
| Full runtime and hardware acceptance | Pending |

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
| Paired vLLM reference | b37f01391ec3b0c028ab6947e4ea3b6f7846504d |
| Cache option | --tokenizer-cache-gb; default 0; legacy additional-config budget 0 |
| Local checks | Python 3.12.13; 11 offline helper tests passed |
| Hardware, throughput, quality | Per-launcher bootstrap untested; compatibility-backend matrix tracked separately |
| Task branch | native_tokenizer_cache_acceptance_20261013 |
| Component base SHA | b2fe9257abe72aa7650265f32a965e14fc235fa6 |
| Delivery SHA | Commit containing this example; inspect with git log -1 -- . |

Install the paired sources in an Ascend environment with its compatible
CANN, torch-npu and model dependencies. source_contract.json records fingerprints
of the installed native config, renderer, and segment-cache files. The launcher verifies
these files before execution. This detects this specific interface mismatch;
it does not certify the complete environment or all model templates.

Source: [Native HF renderer cache](https://github.com/vllm-project/vllm/pull/61335).
The native implementation remains an unmerged dependency. Engine CLI arguments belong to the
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

## Verification plan — 2026-10-11

| Evidence boundary | Current scope |
| --- | --- |
| This change | Repository-specific acceptance plan; no new production-routing implementation |
| Existing software checks | Historical native/fixture results above; not a real model or NPU qualification |
| Live acceptance | Single-NPU compatibility matrix in progress; full Agent / DP / P-D acceptance pending |
| Release | Correctness → native ingress → fault recovery → isolated performance → full-task evaluation |

先验收 **token ID 一致性**，再验收 **事件索引真实性**，最后验收 **任务收益**。三者不能互相替代。
Host 分词缓存、Router whole-prompt L0、引擎物理 KV residency 是三个独立对象；逻辑 token 数不因编码缓存命中而下降。

### Installation prerequisites and immutable experiment record

| Field | Required record |
| --- | --- |
| Repository | Base SHA from this README; task branch `xxx_tokenizer_cache`; tested HEAD from `git rev-parse HEAD` |
| Paired engine | Ascend `4ea839b813fecbd1aba3affa63e3c896e75b123b`; vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`; check whether each dependency has merged at execution time |
| Container | Full registry/tag and actually pulled digest; additional dependencies, OS/architecture, Python/Rust/Go versions; no invented image or digest |
| Model | Weight revision/digest, tokenizer files, chat-template digest, LoRA revision, dtype/quantization and context length |
| Hardware | Allocated physical cards, device model, driver/runtime versions, interconnect, TP/PP/DP topology, API-process count |
| Runtime | Complete launcher command and additional-config, routing config digest, connector/version, event endpoints, seeds and TTL/capacity |
| Workload | Dataset revision/hash, task/repository/time split, prompt/output lengths, arrivals/concurrency, warm-up, timeout and retry budget |

Run the installed-file contract check in `launch_backend.py` before starting workers. Keep credentials in environment variables. Public artifacts contain synthetic inputs and aggregate evidence; keep restricted raw inputs and results local. Model serving, load tests and NPU occupancy need explicitly assigned resources; none are started by this document.

### Shared backend conformance commands

Run from this example directory. All real-service commands below are **plans, not completed tests**. OFF and ON need separately allocated comparable resources; a sequential crossover on the same cards must restart/reset between conditions and alternate order. Keep KV policy, API-process count, weights and template identical.

~~~bash
python3 -m unittest -v test_example.py test_serving.py
python3 launch_backend.py --model Qwen/Qwen3-0.6B --cache-gb 0 --port 8000 \
  -- --served-model-name local-model --max-model-len 8192
python3 launch_backend.py --model Qwen/Qwen3-0.6B --cache-gb 4 --port 8001 \
  -- --served-model-name local-model --max-model-len 8192
python3 verify_tokens.py --baseline http://127.0.0.1:8000 \
  --cached http://127.0.0.1:8001 --model local-model --rounds 3 \
  --output token-conformance.json
python3 verify_serving.py --baseline http://127.0.0.1:8000 \
  --cached http://127.0.0.1:8001 --model local-model --rounds 3 --serve \
  --output render-serving-conformance.json
~~~

`Qwen/Qwen3-0.6B` is an inexpensive text smoke-test example, not DeepSeek/DP16 certification. Use the actual production weights/template for final conformance. `verify_serving.py` calls backend render/generate, **not the entire upstream router/Agent**. Capture actual upstream ingress separately under the cases below. Check every complete token ID, not only count; reject malformed IDs, changed `usage.prompt_tokens`, and render/serving mismatch. Extend the live corpus to empty/truncated inputs, tools and unsupported modalities without silently skipping failures. Output text hashes are diagnostic; generation quality requires an independent task judge.

### Repository-specific cases

| ID | Test object | Input / failure / execution | Independent oracle and pass condition | Current state |
| --- | --- | --- | --- | --- |
| EPP-01 | 原生Token Producer | 配置backendType=vllm；原生Produce通过/render；OFF/ON长上下文、tools、Unicode、salt | 完整IDs和多模态feature metadata一致；不能仅token count相同 | 449项含子测试历史通过、10项live/model skip；实服务未测 |
| EPP-02 | Producer到Scorer接线 | 同request IDs追踪token producer、prefix/group hash、cache signal、filter与scorer | 参与prefix算分的是验证后的tokens和正确group；配套参数相同；缺信号可解释 | 待完整EPP链 |
| EPP-03 | 端点与DP粒度 | 先确认endpoint是否暴露rank；2实例×DP2，再4实例×DP16；预热一个指定rank | 输出的endpoint/rank必须能寻址；未暴露rank时只宣称endpoint粒度，不能默认rank精确 | DP粒度依部署和引擎契约；待验证 |
| EPP-04 | 故障与选择保护 | 事件滞后、满缓存、容量过载、pod下线、模型身份改变、session冲突 | 硬资格和容量保护先于亲和；无stale exact信号；session不强制绑死过载端点 | 待执行 |
| EPP-05 | Sidecar/Coordinator与P/D | request→prefill→handoff→decode；注入坏元数据、迟到结果、断流和取消 | stage、generation与request绑定；P/D所有reservation释放；D失败不复用旧handoff | 待真实联调 |
| EPP-06 | 插件价值消融 | 相同pool：负载、session、prefix、联合策略；前缀命中只作中间指标 | 成功率、TTFT P99、任务完成和成功任务成本同时达预登记门槛；不中途换权重 | 待执行 |

### Failure reproduction and result recording

Each case has five fields in the actual run record: **phenomenon**, **steps** (fixed versions/input/flags), **expected versus observed**, **confirmed cause versus hypothesis**, **fix and rerun result**. Preserve the first failing trace and every unresolved/skipped condition. Assign the request/attempt ID before ingress; attach route ID, model/template identity, worker incarnation, endpoint/rank, event sequence and engine response to the same trace. A recovered request does not establish a repaired root cause.

### Performance and quality gates (proposed; not achieved)

| Gate | Required comparison / measurement |
| --- | --- |
| G0 — correctness | Zero full-token-ID mismatches, unauthorized cross-identity credit, unsafe handoff and resource leaks in the declared deterministic corpus; explicitly count unknown/skipped cases |
| G1 — native ingress | Actual router/Agent ingress reaches the registered model/endpoint/rank with unchanged semantics; standalone backend probes cannot pass this gate |
| G2 — recovery | Scripted gaps, clear/restart, cancellation, saturation and malformed events revoke trust correctly; no exact claim before validated reconciliation |
| G3 — isolated benefit | Cache OFF/ON × routing policy factorial; fixed hardware/trace/quality; cold/warm/churn separately; measure CPU encode, RSS, actual KV reuse, TTFT P99, completion P95, SLO goodput |
| G4 — task value | Fixed strong, fixed main and simple rules as controls; held-out tasks; total spend including failed attempts, retries, classifier and fusion divided by successful tasks; quality and latency constraints decided before running |

Use at least five paired seeds/arrival traces for the initial comparison and report paired bootstrap 95% confidence intervals. This repeat count and all performance thresholds are **project proposals**, not an upstream guarantee; compute the necessary sample size from variance and the business non-inferiority margin. Zero observed failures does not prove impossibility outside the tested corpus. Do not select the best run or tune on the held-out set. A busy server with no completions must report zero goodput and undefined successful-task cost rather than omit the run.

Before a load test, inspect the **pinned** engine benchmark entrypoint with `vllm bench serve --help`; record the exact supported flags. If that revision lacks this CLI, use its checked-in serving benchmark entrypoint and record its SHA. Do not copy speculative `kv_aware` flags into a runnable command: at Router base `91b65ee4`, that policy is not wired. A trace runner, task judge and fault injector are still required beyond the current conformance helpers.

The paired vLLM revision above was inspected at `vllm/entrypoints/cli/benchmark/serve.py`, `vllm/benchmarks/serve.py` and `vllm/benchmarks/datasets/datasets.py`; the following flag names exist there. This is a **planned synthetic cold-load example**, not a measured workload, warmed-prefix replay or Agent evaluation. Run only after G0/G1 pass and resources are assigned. The rate/length/concurrency values are illustrative and must be replaced by the preregistered workload; `127.0.0.1:8080` must be the actual reachable upstream ingress. Use the installed tokenizer matching production weights, not the model alias as a tokenizer path.

~~~bash
vllm bench serve --backend openai-chat \
  --base-url http://127.0.0.1:8080 --endpoint /v1/chat/completions \
  --model local-model --tokenizer Qwen/Qwen3-0.6B \
  --dataset-name random --random-input-len 4096 --random-output-len 128 \
  --num-prompts 1000 --request-rate 4 --max-concurrency 32 --seed 1 \
  --percentile-metrics ttft,e2el --metric-percentiles 50,95,99 \
  --save-result --result-dir benchmark-results
~~~

Repeat across the fixed policy/cache conditions with separate result directories, alternating run order and identical generated inputs/arrival seeds. `e2el` is end-to-end request latency, not full Agent-task time. Random prompts do not establish prefix-reuse accuracy; warm/churn/PD and model-selection quality require the dedicated traces and oracles in the table. This CLI does not measure classifier/Agent loop costs unless those are actually on the tested path.

Full RFC implementation and shared validation design:
https://github.com/Liuchenbing-2026/Model_test/blob/router/agent-routing-evaluation/项目交付/任务列表/任务13%20llm-d%20Router调研与可复现指南/详细任务/路由方案/README.md#validation-plan-20261011

### Native verification entrypoints

Run from repository root. The live test must **run**, not silently skip:

~~~bash
VLLM_RENDER_TEST_BASELINE_URL=http://127.0.0.1:8000 \
VLLM_RENDER_TEST_URL=http://127.0.0.1:8001 \
VLLM_RENDER_TEST_MODEL=local-model \
go test -json ./pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer \
  -run '^TestTokenizerCacheParityLive$' -count=1 > tokenizer-live.json
go test ./pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer -count=1
go vet ./pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer
~~~

Parse the first JSON output for `Action=pass` of `TestTokenizerCacheParityLive` and reject `Action=skip` or zero cases; process exit zero alone is insufficient. `VLLM_RENDER_TEST_AUTHORIZATION` carries the prepared backend authorization header when needed; never put secrets in commands/archives. This package check does not exercise full EPP, Sidecar, Coordinator, Gateway or P/D. Repository-wide `make presubmit` remains the broader gate for future code changes.

## CPU render deployment and actual EPP acceptance (2026-10-11)

`launch_cpu_renderer.py` uses the pinned Ascend segment-cache implementation
with vLLM's standalone CPU render service. Each renderer owns its tokenizer
cache. The adapter checks the source contract, preserves the native HTTP routes,
and writes cache statistics on graceful shutdown. It does not load model weights.

The CPU factory supplies the cache budget without initializing AscendConfig.
This is an explicit adapter against the fixed patch's private functions. An
engine upgrade requires renewed source-contract and behavior verification.
The native Ascend plugin and NPU generation require separate acceptance.

| Actual check | Result and scope |
| --- | --- |
| Native render OFF/ON | 30 comparisons across five cases; complete IDs equal |
| Independent HF oracle | Eight text-chat cases, both modes, 16 comparisons; complete IDs equal |
| Native Go producer | 59 passing entries including parents; zero failures/skips; 54 native protocol variants and three parity cases |
| Cache activation | OFF: zero instances/lookups; ON: one instance, 1043 hits / 1082 lookups in this synthetic run |
| Existing vLLM launch CLI | 11 tests passed; 15 existing runtime/deprecation warnings recorded |
| NPU, generation quality, inference throughput | Not evaluated |

The producer run invokes `Produce` with parsed requests and forwards its native
render HTTP calls through a test proxy to actual services. It covers chat,
Anthropic-compatible messages, tools, template controls, aliases, batches and
truncation. This does not start a Kubernetes Gateway, EPP scheduler or KV scorer.
The first run rejected six automatic-tool variants because startup omitted the
tool parser. Its failure log is retained; the configured rerun passes.

### Fixed environment

| Item | Actual value |
| --- | --- |
| Image tag | quay.nju.edu.cn/ascend/vllm-omni:v0.30.0 |
| Registry digest | quay.io/ascend/vllm-omni@sha256:274d65369c912c14cc241f53c9c900d95e844b91652ca604cc71f07a9e1b3aaf |
| Local image ID | sha256:96c8901953dbe758199d97795279762131b58faadb02fa593289ea01e3e68dbd |
| vLLM base | ced6857afa0ea7b2e3f0846a62e1394e90f15607; native image binaries reused |
| vLLM render change | [router/tokenizer-cache-render-20261011](https://github.com/Liuchenbing-2026/vllm/tree/router/tokenizer-cache-render-20261011); entry.py preserves additional_config |
| Ascend cache source | 4ea839b813fecbd1aba3affa63e3c896e75b123b; loaded directly from its fixed Python file |
| Python / tokenizer stack | 3.12.13; torch 2.10.0+cpu; transformers 5.14.1; tokenizers 0.22.2 |
| Model assets | Local Qwen3-8B tokenizer, Qwen2Tokenizer class; weights not loaded |
| CPU isolation | arm64 Linux; 4 CPU / 12 GiB container limit; no device mappings or privileged access |

The CPU test container removes inherited `torch-npu`, `vllm-ascend`, `triton`
and `triton-ascend` packages. The original image and the native Ascend build
container remain unchanged. Use `TORCH_DEVICE_BACKEND_AUTOLOAD=0`,
`VLLM_TARGET_DEVICE=cpu`, and `VLLM_PLUGINS=''` for this renderer.
The uv 0.11.6 Linux arm64 archive is fixed by SHA256
`d5be4bf7015ea000378cb3c3aba53ba81a8673458ace9c7fa25a0be005b74802`.
Its system-site-packages virtual environment resolves Python 3.12.13.

### Actual service and independent comparison commands

From this example directory, use an environment containing the paired vLLM
render change and the CPU dependencies above. The tested model is mounted at
`/models/qwen3-8b`; the fixed Ascend source is mounted at the path below.
Create the output directory first. Launch each mode in its own terminal:

```bash
export VLLM_PLUGINS='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_TARGET_DEVICE=cpu
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false DO_NOT_TRACK=1
mkdir -p /work/results
/work/.venv-render/bin/python launch_cpu_renderer.py \
  --cache-source /work/source/engine-tokenizer-cache-20261011/vllm_ascend/patch/platform/patch_tokenizer_cache.py \
  --cache-stats-output /work/results/cpu-render-tools-off-stats.json \
  launch render /models/qwen3-8b --host 127.0.0.1 --port 18480 \
  --served-model-name router-qwen3-8b router-qwen3-8b-alias \
  --dtype float32 --max-model-len 4096 \
  --additional-config '{"tokenizer_cache_gb":0}' \
  --disable-uvicorn-access-log --enable-auto-tool-choice --tool-call-parser hermes
/work/.venv-render/bin/python launch_cpu_renderer.py \
  --cache-source /work/source/engine-tokenizer-cache-20261011/vllm_ascend/patch/platform/patch_tokenizer_cache.py \
  --cache-stats-output /work/results/cpu-render-tools-on-stats.json \
  launch render /models/qwen3-8b --host 127.0.0.1 --port 18481 \
  --served-model-name router-qwen3-8b router-qwen3-8b-alias \
  --dtype float32 --max-model-len 4096 \
  --additional-config '{"tokenizer_cache_gb":1}' \
  --disable-uvicorn-access-log --enable-auto-tool-choice --tool-call-parser hermes
/work/.venv-render/bin/python verify_serving.py \
  --baseline http://127.0.0.1:18480 --cached http://127.0.0.1:18481 \
  --model router-qwen3-8b --rounds 3 --output /work/results/cpu-native-render-tools.json
/work/.venv-render/bin/python verify_render_oracle.py \
  --baseline http://127.0.0.1:18480 --cached http://127.0.0.1:18481 \
  --model router-qwen3-8b --tokenizer /models/qwen3-8b \
  --output /work/results/cpu-native-oracle-fixed.json
```

`verify_render_oracle.py` executes in a separate process without importing the
cache adapter. It uses the tokenizer's native chat template and retains the
complete expected and observed IDs. It accepts the Transformers 5.14.1
BatchEncoding return type by extracting `input_ids`. This oracle covers the
specified Qwen3 text-chat policy; it does not certify every model template.

Run the producer from the repository root with Go 1.27.2. The actual macOS
producer run reached these services through a task-scoped SSH loopback tunnel,
using local ports 28480/28481:

```bash
VLLM_RENDER_TEST_BASELINE_URL=http://127.0.0.1:28480 \
VLLM_RENDER_TEST_URL=http://127.0.0.1:28481 \
VLLM_RENDER_TEST_MODEL=router-qwen3-8b \
VLLM_RENDER_TEST_ALIAS=router-qwen3-8b-alias \
go test ./pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer \
  -run 'TestTokenizerCacheParityLive|TestNativeRenderLive' -count=1 -json
```

Stop only the test-owned service PIDs with SIGTERM after verification. Require
ON `cache_hits > 0`, ON `renderer_cache_instances == 1`, and OFF zero instances
and hits in the emitted statistics. The deployed services and tunnel were
stopped after collecting evidence. Timings and hit ratios here describe a
synthetic correctness sample; no load or generation benchmark was executed.

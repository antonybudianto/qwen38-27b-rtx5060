# The patch series, one line each

What every file in `patches/` (and `kvarn/`) is, where it came from, and what retires it. The Dockerfile applies
them in the order of `patches/series` onto the installed vLLM wheel; `verify.sh` checks each one is in place. Kinds:

- **backport**: a merged or open upstream change carried early. Retires when the pin carries it.
- **fix**: a defect in upstream or in this stack, fixable upstream. Retires when upstream takes it.
- **feature**: something upstream does not have. Stays until upstreamed as a feature.
- **local**: this hardware or environment (WSL2, sm80, a tuned build, env knobs). Stays.
- **own**: a fix to a feature this repo introduced. Rides with that feature.

Cut against: the pin the current hunks were generated on. Every file is exported from its commit on the fork
branch (`cpuchip/vllm` `qwen38/0.28` = v0.28.0 + one commit per row, in series order, subject `[qwen38] <topic>`),
so the series applies to the 0.28.0 tree with exact context; the Dockerfile, `patches/check_vllm_series.sh`,
`kvarn/install.sh` and `verify.sh` apply and check with `--fuzz 0`, and a hunk whose context has moved fails the
build by name instead of landing by guess. Regenerate a file with `bash scripts/export-patch.sh <fork checkout>
<commit> patches/<topic>.patch`; do not edit the files by hand. A patch that reads an env knob registers it in
`envs.py` in its own hunk (so the knob is in the torch.compile cache key), and reads it through `vllm.envs`.

| patch | kind | what | upstream | cut against | retires when |
|---|---|---|---|---|---|
| dflash2-backport | backport, RETIRED | DFlash2 speculator on 0.27.1 | vllm #52816 (in 0.28.0) | 0.27.1 | done; kept for history, skipped by the Dockerfile |
| dflash2-lookup-drafting | feature | lookup-augmented drafting for DFlash2 (n-gram search over the context); registers its `VLLM_DFLASH2_LOOKUP*`, `VLLM_DFLASH2_GRAPH_BOTH`, `VLLM_DFLASH2_DRAFT_TOPK_TOPP` knobs | none | 0.28.0 | upstreamed |
| dflash2-ngram-chains | feature | quantized candidate chains for the drafter; `propose` override; registers `VLLM_DFLASH2_CHAIN*` | none | 0.28.0 | upstreamed |
| dflash2-prewarm | fix | compile every DFlash2 rung at boot instead of at first request | none yet | 0.28.0 | upstream PR |
| dflash2-z-adaptive-emitted | fix | adaptive z counts emitted tokens, not sampling slots | none yet | 0.28.0 | upstream PR |
| dspark-draft-quant-config | fix | bf16 DSpark drafter beside a quantized target (callable `hf_overrides`) | none yet | 0.28.0 | upstream PR |
| engine-completion-log | fix | one ungated INFO line per request that leaves the scheduler finished (#94, #110) | none yet | 0.28.0 | upstream PR |
| engine-stall-sentinel | fix | warn once per episode when the core completes no step for `VLLM_ENGINE_STALL_SENTINEL_S` while requests are live (#107, #110); registers the knob | none yet | 0.28.0 | upstream PR |
| hybrid-kv-groups-v2-cudagraph | fix | KV group sizing when the smallest bucket is the drafter's sliding-window layers; explicit CUDA-graph memory reserve for the V2 runner; registers `VLLM_V2_CUDAGRAPH_MEM_MIB` | none yet | 0.28.0 | upstream PR; the reserve hunk retires when vLLM profiles V2 graphs (0.29.0 does) |
| hybrid-sw-block-promote | fix | promote a draft SW layer's block to a divisor of the primary block instead of padding its page | none yet (upstream pads) | 0.28.0 | upstream PR |
| int4-kv-per-token-head | feature | int4 per-token-head KV cache with the DFlash2 drafter | none | 0.28.0 | upstreamed |
| int4-mq3d-envs | own | registers `VLLM_INT4_MQ_3D` and `VLLM_INT4_MQ_3D_DEBUG`, read through `vllm.envs` (#93) | none | 0.28.0 | rides with mq3d |
| mamba-align-checkpoint-order | fix | keep reachable Mamba state snapshots alive until request end (fork #52); registers `VLLM_MAMBA_ALIGN_KEEP_CHECKPOINTS` | vllm #45238 (not merged) | 0.28.0 | check against upstream #52789 (internal prefill checkpoints, in 0.29) at each pin |
| mamba-align-retire-null-gaps | backport | align mode retires Mamba state blocks across null gaps instead of stopping at the first one (fork #101) | vllm #55450 (merged 2026-09-11) | 0.28.0 | the pin that carries #55450 |
| mamba-chunked-prefill-align | fix | state loss and NaN during chunked prefill on Mamba/GDN | none yet | 0.28.0 | upstream PR |
| marlin-int8-layer-select | local | env vars to pick which layers run W4A8 with the Marlin kernel; registers and reads `VLLM_MARLIN_INT8_INCLUDE_RE` / `_EXCLUDE_RE` | none | 0.28.0 | stays |
| marlin-int8-negative-scales | fix | Marlin W4A8 reads group scales as unsigned; AutoRound exports negative ones | none yet | 0.28.0 | upstream PR |
| marlin-repack-staged-sm80 | local | one grow-only staging buffer for the sm80 Marlin repack (fork #27); registers `VLLM_MARLIN_REPACK_STAGED` | none | 0.28.0 | stays |
| marlin-tune-table | local | wiring for a locally built tunable Marlin extension, off by default (`VLLM_MARLIN_TUNE`, registered by speed-knobs-envs) | none | 0.28.0 source | stays |
| offload-dflash-eagle-groups | fix | OffloadingConnector under dflash flagged every KV group as draft attention (fork #33) | none yet | 0.28.0 | upstream PR |
| offload-mtp-serve | backport | OffloadingConnector serves stored hits under MTP/EAGLE instead of vetoing the request; load boundary from the computed offset; finished-request store watermark clamped (#100) | vllm #52771, #52807, #54288 (merged) | 0.28.0 | the pin that carries all three |
| offload-wsl2-devptr | local | CPU offload tier device pointers on WSL2 | none | 0.28.0 | stays |
| prefill-attn-int8 | feature | int8-QK Triton prefill attention for head_dim 256; registers `VLLM_PREFILL_ATTN` | none | 0.28.0 | upstreamed |
| qwen3_5-embed-quant | fix | pass `quant_config` to the token embedding (main model and MTP module) | none yet | 0.28.0 | upstream PR |
| qwen3_5-mtp-draft-vocab | feature | vocab-truncated draft head for MTP | none | 0.28.0 | upstreamed |
| sampler-small-topk-fast-softmax | feature | sort-free top-k/top-p for small k, multi-block row softmax; registers `VLLM_DRAFT_TOPK_TOPP` and `VLLM_DRAFT_TEMP_SCALE` (both read once at import) | none | 0.28.0 | upstreamed or superseded |
| spec-decode-attn | feature | split-KV verify attention on FLASH_ATTN with query-row tiling; registers `VLLM_SPEC_DECODE_ATTN`, `VLLM_SPEC_DECODE_ATTN_QMAX`, `VLLM_SPEC_ATTN_BLOCK_M` (#114) | none | 0.28.0 | upstreamed |
| spec-decode-int4-kv-mq3d | feature | multi-query 3D int4 verify path | none | 0.28.0 | rides with int4-kv-per-token-head |
| spec-decode-int8-kv | feature | split-KV verify attention over an int8 per-token-head cache | none | 0.28.0 | rides with spec-decode-attn |
| spec-decode-scratch-token-units | own | mq3d scratch sized in tokens, not sequences (fork #46, #57) | none | 0.28.0 | rides with mq3d |
| spec-decode-scratch-within-budget | own | mq3d scratch allocated inside the memory budget (fork #57) | none | 0.28.0 | rides with mq3d |
| spec-sampler-prewarm | fix | compile the rejection sampler's Triton kernels at boot (fork #48) | none yet | 0.28.0 | upstream PR |
| speed-knobs-envs | local | registers `VLLM_MARLIN_TUNE` and `VLLM_MARLIN_TUNE_DIR` for marlin-tune-table (the other knobs it used to register now live in the patches that read them) | none | 0.28.0 | stays while marlin-tune-table does |
| sse-keep-alive | backport | SSE keep-alive comments on idle streaming responses so a proxy's idle timer does not drop a long prefill (#85, #115) | vllm #51034 | 0.28.0 | the pin that carries #51034 |
| triton-spec-attn-fp8-kv | feature | split-KV verify attention on the per-tensor fp8 KV cache (TRITON_ATTN, sm89+); registers `VLLM_SPEC_ATTN_DEBUG` (#90) | none | 0.28.0 | upstreamed |
| vision-tower-cpu-offload | local | Qwen3 vision tower bulk weights in host RAM; registers `VLLM_VISION_CPU_OFFLOAD_GB` | none | 0.28.0 | stays |
| vllm-pr50021-gdn-spec-bounds | backport | bounds checks in GDN/KDA spec-decode state lookups | vllm #50021 (open) | 0.28.0 | the pin that carries #50021 |
| vllm-pr54282-draft-gumbel-salt | backport | the draft's Gumbel noise stream decoupled from the target's (probabilistic draft sampling) | vllm #54282 (in 0.29.0) | 0.28.0 | 0.29.0 |
| xgrammar-spec-terminated | fix | structured output + speculative decoding: tokens accepted past the grammar's end are ignored, not fatal (#31) | in 0.29.0 | 0.28.0 | 0.29.0 |
| kvarn/kvarn-0.28.0 | feature | KVarN cache dtypes, quant mode, backend registration, page size (applied by `kvarn/install.sh` after the series, onto the modules it copies from `kvarn/files`) | none (KVarN is Huawei CSL's, Apache-2.0) | 0.28.0 | upstreamed |
| kvarn/kvarn-v2-runner-0.28.0 | own | KVarN with the V2 runner and DFlash2 (SW groups, Mamba block index, selector guards) | none | 0.28.0 | rides with KVarN |

The `KVARN_*` knobs the copied modules read are not registered on this line (they are copied files, not patches,
and vLLM's unknown-variable warning fires for `VLLM_` names only); the 0.29 port registers them.

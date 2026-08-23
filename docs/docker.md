# Docker, and WSL2

The container image (same stack, frozen) and an independent WSL2 reproduction with its memory caveats.

[← back to the main README](../README.md)

The container image is the same stack, frozen: Python 3.12 venv, vLLM 0.27.1 pinned
(torch 2.13 / cu130 / Triton 3.7.1), every patch in `patches/` applied and
`verify.sh --install` run at build time, KVarN preinstalled. Host prerequisites:
an NVIDIA driver that speaks CUDA 13 (≥ 580), Docker with the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
configured as a runtime. The 250 W power limit is a host setting
(`sudo nvidia-smi -pl 250`), the container cannot set it.

```bash
git clone https://github.com/syv-ai/qwen38-27b-rtx3090 && cd qwen38-27b-rtx3090
echo "VLLM_API_KEY=$(openssl rand -hex 24)" > .env   # all knobs live in .env (gitignored)
docker compose --profile single up -d               # or --profile batch
docker compose logs -f single
```

The first `up` builds the image (9.5 GB), then the `prepare` service downloads
the model into `./models` and runs the same requantization scripts as above
(CPU only, idempotent, ~20 GB + a few minutes; `FAST_VARIANT=0` in `.env`
skips the ~1 GB fast-variant download), then the server starts. The first
start also does the torch.compile / CUDA-graph / FlashInfer-JIT work (2–3
minutes); that lands in the `qwen-cache` volume, so later starts take ~1
minute. `docker compose ps` shows the healthcheck (`/health`, 15-minute start
period). Measured in the container on the 3090: single-user 112.6 / 115.7 tok/s
(e2e / decode, default sampling), batch 950 tok/s on the 128/512 × 64 row, the
same KV pools as the venv install — no container tax; the only first-start
difference is gotcha 16 below.

- Modes are compose profiles: `single` runs `single-user/start_qwen.sh`, `batch`
  runs `batch/start_qwen.sh`. One GPU, so one at a time
  (`docker compose --profile single down` before `--profile batch up -d`).
- Every start-script knob works from `.env`, which is passed straight into the
  container: `CTX=long`, `KV=kvarn`, `SPEC=dflash2`, `PREFIX_CACHE=1`, `MAX_LEN=`,
  `MAX_SEQS=`, `SPEC_ATTN=0`, `EXTRA_ARGS=...` (`prepare` also fetches the DFlash2 drafter;
  `DFLASH2=0` skips it). `PORT` (default 18020) and `MODELS_DIR` (default `./models`,
  so a venv install and the container can share one download) are read by
  compose itself.
- `docker compose run --rm single verify` runs `verify.sh` inside the container
  (GPU, patches, model). The entrypoint runs `verify.sh --no-server` before every
  start and refuses to serve on a FAIL (`VERIFY=0` skips that).
- `docker compose --profile bench run --rm bench` runs `bench/run_benchmarks.sh`
  in a client-only container against the server that is already up, printing the
  same ROW lines as the venv install:

  ```bash
  docker compose --profile batch up -d
  docker compose --profile bench run --rm bench                    # the batch tables
  docker compose --profile bench run --rm bench single --prefill   # mode + flags
  ```

  Arguments after the service name go straight to the script, since the service
  overrides the entrypoint with the script itself; `BENCH_ARGS` in `.env` sets the
  default (`batch`). `BENCH_TARGET` picks the service to drive (default `single`;
  use `host.docker.internal` for a server running on the host rather than in
  compose), and `PORT` is the one the server already reads from `.env`. `./bench`
  is bind-mounted, so raw vllm logs land in `./bench/results` on the host and
  editing the script needs no image rebuild. `BENCH_WAIT` (default 1800s) is how
  long the client polls `/health` before giving up, which covers a first server
  start's compile. Run it twice and keep the second numbers, exactly as on the
  venv install.
- Files that `prepare` writes to `./models` are root-owned: the container runs
  as root, like vLLM's own image.
- The image carries an nvcc (CUDA "base" + `cuda-nvcc`, not the 8 GB "devel"
  image) because FlashInfer JIT-compiles its fp8-KV attention kernel on first
  use (batch mode, `CTX=long`) and Triton needs a C compiler for its launchers.
- On WSL2 the batch default may fail vLLM's free-memory gate; put
  `GPU_UTIL=0.93` in `.env` (see the WSL2 notes below, an independent
  containerized reproduction that predates this compose file).

### WSL2 notes

An independent WSL2 reproduction at `e81fa39` used kernel
`6.6.87.2-microsoft-standard-WSL2`, Ubuntu 24.04, NVIDIA driver 591.86,
Docker Engine 29.2.0 / Compose 5.0.2, and one RTX 3090 exposed to the
container. All six launch configurations passed authenticated API/chat and
GPU-isolation checks with zero failed benchmark requests. The full failure
signatures and earlier five-profile matrix are in [issue #1](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/1).

| profile | measured cache | representative output throughput |
|---|---:|---:|
| `single-long` | 159,326 tokens, fixed as described below | 95.39 tok/s greedy, C1 |
| `single-fast` | 93,791 tokens | 114.17 tok/s greedy, C1 |
| `single-huge` | 320,000 cold / 327,272 warm | 79.84 / 81.66 tok/s, C1 sampled |
| `batch` | 201,832 tokens | 1,041.99 / 1,038.25 tok/s, C64 |
| `batch`, `KV=int4pth` | 437,414 tokens | 1,043.84 / 1,044.06 tok/s, C64 |
| `batch`, `KV=kvarn` | 334,183 cold / 350,192 warm | 843.72 / 852.42 tok/s, C64 |

Three WSL-specific memory behaviors are worth accounting for:

1. **The ordinary batch default may fail vLLM's startup free-memory gate.**
   On an otherwise clean card, WSL reported 22.75/24.0 GiB free, less than
   the 23.33 GiB requested by `GPU_UTIL=0.972`. Launching with
   `GPU_UTIL=0.93 bash batch/start_qwen.sh` retained a 201,832-token FP8
   pool, preserving the 150k context contract and expected C64 throughput.
   Keep 0.972 as the tuned native-Linux default; 0.93 is a WSL fallback.
2. **Cold and cached starts can profile different activation peaks.** A warm
   start may turn the difference into extra KV pages and leave less transient
   headroom than the cold start. For a deterministic service, compile once
   from a cold cache, record vLLM's conservative
   `Replace gpu_memory_utilization config with --kv-cache-memory=...`
   recommendation, verify that the resulting token pool exceeds
   `MAX_LEN`, and pass that machine/profile-specific byte value through
   `EXTRA_ARGS` on later starts. Stress concurrent prefill or
   `prompt_logprobs` before promoting it. Do not copy a byte value from a
   different card or profile.
3. **`expandable_segments` can crash Marlin repack on some driver/dxgkrnl
   combinations.** Both start scripts default to
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; its CUDA VMM calls
   crashed the engine with `RuntimeError: CUDA driver error: device not
   ready` inside `gptq_marlin_repack` on Windows driver 610.74 (WSL 2.1.5
   and 2.7.12 alike — the `e81fa39` reproduction on driver 591.86 did not
   hit this). The scripts respect a pre-set value, so put
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` in `.env` (Docker)
   or the environment (venv) if you see that signature.

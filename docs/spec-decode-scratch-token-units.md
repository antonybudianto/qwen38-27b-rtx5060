# The scratch-buffer sizing fix for #46's int4 speculative attention: measurements and verification

Companion to `patches/spec-decode-scratch-token-units.patch` and the pull request that
carries it. The PR description says what changed and why; this file carries the conditions
behind every number and the detail of how it was checked.

Terms used below. vLLM's Triton attention has a serial decode kernel that walks the whole KV
cache in one pass (2D) and a split-KV one that divides the walk into segments and reduces
them (3D); the 3D kernel keeps partial results in three scratch buffers indexed by query
token, so "token rows" means rows of those buffers. DFlash2 is the draft model this repo
ships; at draft depth `d` it proposes `d` tokens per step and the verify batch is `d + 1`
tokens per sequence (multi-query). A "breach" is an accepted batch that exceeded the scratch
capacity.

## Conditions

Qwen3.8-27B-W4A16-AutoRound with its DFlash2 drafter (`Qwen3.8-27B-DFlash2-W4A16`), int4
per-token-head KV cache (`single-user/alternative.sh`), draft depth 7, `max_num_seqs=1`,
`max_num_batched_tokens=2048`, CUDA graphs on, `--prefix-match-unit 848` (not required on current
main, where the engine derives the same unit; passed for parity with earlier runs), temperature 0,
thinking off, context limit 262,144. RTX 3090, native Ubuntu 26.04, Python 3.14, vLLM 0.27.1,
torch 2.13.0+cu130, CUDA 13.0, driver 610.43.02. Nothing else ran on the box during the
throughput runs (each boot waited for load average below 1.3). The independent numerical
check ran on an RTX 4090 under WSL2.

## The defect, in numbers

`seq_threshold_3D` is `128 // num_heads_kv`; this model has 4 KV heads, so 32. With CUDA
graphs on, that 32 is replaced by the entry of the graph capture list closest to it by
absolute difference, which at `max_num_seqs=1` is `[1, 2, 4, 8]`: the buffers get 8 rows. The
dispatch log for one 6,747-token request, with the multi-query kernel enabled and debug
output on:

| batch size (query tokens) | CUDA graphs on | fully eager | cause |
|---|---|---|---|
| 1, 2, 4, 8 | 3D kernel | 3D kernel | |
| 9 | **2D kernel** | 3D kernel | 9 tokens, 8 rows |
| 24 and larger | 2D kernel | 2D kernel | over the 16-token per-sequence limit (intended) |

Only 9-token batches occurred in our runs; 10 to 16 were not observed live. The same request,
same config; only the graph mode differs. Holding the mode fixed and moving only the row
count flips the 9-token batch in both directions (eager with 8 rows falls back; graph mode
with 16 rows serves it), so the row count is the deciding predicate, and the graph mode is
what shrinks the row count at this configuration. At larger `max_num_seqs` the sequence-unit
sizing is short even without CUDA graphs, since `max_num_seqs x 16` tokens exceeds
`seq_threshold_3D` rows.

## Cost

One token row is `heads x 16 segments x 256 (padded head dim) x fp32` plus two small
max/expsum buffers: 396,288 bytes for the target model (24 query heads) and 528,384 bytes for
the drafter (32 query heads). There are 4 live allocations: one attention backend instance
per attention shape, two shapes, two instances each. Not one per layer (the model has 64
layers, 16 of them full attention).

| `max_num_seqs` | token rows | total across 4 allocations | source |
|---|---|---|---|
| 1 (this repo's default) | 16 | 28.22 MiB (14.11 MiB more than the 8-row allocation) | boot log |
| 8 | 128 | 225.76 MiB | boot log |
| 32 (the `seq_threshold_3D` ceiling) | 512 | about 903 MiB | arithmetic from the per-row cost |

The boot log prints the derived capacity with its inputs, so the cost is auditable per
deployment; there is no silent cap.

## Throughput: a null result, and why it is expected

Metric: decode-only tokens per second, `usage.completion_tokens` divided by the time from the
first content token to the last (one SSE chunk is one speculative step, not one token, so
chunk counts are not used). 384 tokens generated after a NASA history prompt. Boots alternate
patched and unpatched so drift cannot align with the arm. The timed runs below have the
dispatch instrumentation off; which kernel served the 9-token batches was proven in separate
boots with it on (the table above), because the patch adds a debug line and an instrument
that runs during the measurement is part of the measurement. Arm identity in the timed runs
comes from the patch's boot-record line, which prints only on the patched arm. Nothing else
ran on the box; each boot waited for load average below 1.3.

| prompt | boots per arm | patched | unpatched | difference |
|---|---|---|---|---|
| 6,747 tokens, instrumentation off | 2 | 110.88 tok/s | 110.67 tok/s | +0.18% |
| 6,747 tokens, instrumentation on | 3 | 110.77 tok/s | 110.63 tok/s | +0.13% |
| 68,013 tokens, instrumentation on | 2 | 44.6 tok/s | 44.6 tok/s | within 0.2% (the reporting resolution) |

Spreads within an arm: 0.36% and 0.09% (off), 0.54% (on, three boots); the 68k arm has too few
boots for a spread estimate of its own. The reason for the null is arithmetic. At draft depth
7 the ordinary verify batch is 8 tokens, 8 is a capture size that already ran the 3D kernel,
and the band this fixes occurred in 1 of 152 decode steps in the 68k run (a 9-token batch,
hitting all 16 int4 attention layers of that step). Those batches occur because the drafter's
n-gram chain extension, which lengthens a draft when recent tokens match an earlier n-gram,
occasionally proposes more than the base 7 tokens. So #46's gain on 8-token batches is
unaffected, and only the 9 to 16 band was falling back.

**RTX 4090, and a method finding.** The first attempt on that card produced decode rates
between 48 and 144 tok/s across boots with no code change. The cause was one artifact in
the shared torch.compile cache: an ahead-of-time compile artifact written by an earlier boot
from stock code, in an unusually short compile, ran at full speed in the process that produced
it and about 2.6x slower in every process that later loaded it, on both cards, with identical
output. The cache does not distinguish boots, so every later boot, stock or patched, inherited
it (same card, empty cache: 148 tok/s; shared cache thirteen minutes earlier: 80). Dropping
that one artifact restored the original rates to the decimal. The affected artifact is the one a
boot writes when it loads one compiled module from the cache and recompiles another; a boot
that compiles everything, or loads everything, does not produce or hit it. The patched boot's
own artifact was harmless. To reproduce the numbers here, use an empty compile cache per boot
or `VLLM_DISABLE_COMPILE_CACHE=1`. Redone with a fresh, empty cache for every
boot, the card quiet and the other card idle, stock/patched/stock/patched: fresh compiles of
the same code give different generated text and different draft acceptance at temperature 0
(steps to generate 512 tokens at 6,747 tokens of prompt: 133 and 153 for the two stock boots,
143 and 152 for the two patched), so two stock boots differ in tok/s as much as stock differs
from patched, and tok/s is not an arm metric on that card. Milliseconds per decode step is,
with a spread of at most 0.9% over all four boots.

| prompt, RTX 4090, fresh cache per boot, 512 tokens generated | boots per arm | stock | difference per step, patched vs stock, each pair |
|---|---|---|---|
| 6,747 tokens | 2 | 26.0 ms per step | 0.08% and 0.07% |
| 68,013 tokens | 2 | 38.1 ms per step | 0.44% and 0.20% |

So there is no throughput effect at draft depth 7 on either card. The 3090 result above is in
tokens per second because every one of its boots shared one compile lineage (the cache was
not cleared between arms), which is fine for the difference between arms and means the
absolute tok/s figures belong to that lineage rather than to the hardware.

A deeper draft would put that band in the common case, and it is not reachable with this
drafter on this card. The checkpoint is trained at depth 7; at depth 11 its own log reads
"drafting 7 tokens per step (the block the checkpoint was trained for); the remaining 4 of 11
verify positions are filled from context", so the extra positions are padding, not draft. The
engine also failed to boot at depth 11 in all eight attempts: at 262,144 context, out of
device memory during graph capture; at 98,304 context, patched and unpatched, with no
prefix-match unit and with `--prefix-match-unit 888` (the depth-11 block size is 1,776, which
848 does not divide), on the KV-cache coordinator's block-divisibility assert
(`kv_cache_coordinator.py`, `scheduler_block_size % hash_block_size == 0`) or out of device
memory. That regime, and multi-sequence serving, are untested for throughput here.

## Verification

**Property test**, `bench/mq3d_capacity_property.py` (run with `python`): every accepted batch
must fit the derived allocation. It takes the full grid of `max_num_seqs` in {1, 2, 4, 7, 8,
9, 16, 31, 32, 33, 64, 256}, `seq_threshold_3D` in {1, 2, 4, 8, 16, 32, 64, 128},
`max_num_batched_tokens` in {8, 16, 64, 128, 2048, 8192, 131072} and per-sequence query
length in {1, 2, 8, 16, 17, 32}, and for each configuration the corner vectors plus ragged
random query-length vectors from a fixed seed (20260901): 238,276 accepted batches over 4,032
configurations, 0 violations. `--mutate seq-rows` (sequence-unit sizing, the eager-mode
pre-fix allocation) fails 129,498 of them; `--mutate snapped-rows` (sequence-unit sizing after
the capture-list substitution, what #46 ran under CUDA graphs) fails 140,008.

**Independent numerical check**, `bench/mq3d_layer2_oracle.py`, with its verdict records in
`bench/mq3d_layer2_verdicts.jsonl`. Written on the 4090 box without reference to this patch's
code, it imports the production allocation and dispatch (the real
`TritonAttentionMetadataBuilder` and `_launch_packed_attn`, entered through
`unified_attention()` as the backend enters it), runs every case through the multi-query
kernel disabled (`VLLM_INT4_MQ_3D=0`, the 2D leg) and enabled (the 3D leg) with NaN-prefilled
scratch, and compares both legs against a separate fp32 attention over its own
dequantisation of the cached bytes. `_launch_packed_attn` is call-counted per case, so a
green that never reached the production path cannot occur. Declared configuration:
`max_num_seqs=2`, `max_num_batched_tokens=8192`, eager (no CUDA graphs; the capture-list
substitution is a separate change), which gives capacity 32 so that the 17-token
`[16, 1]` case is admissible. Real geometry from the served model (24 query heads, 4 KV
heads, head dim 256). Run with `python bench/mq3d_layer2_oracle.py` on a machine with the
patched stack and a CUDA device; it locates the installed vLLM and the patch file from the
checkout (`VLLM_ROOT`, `ORACLE_PATCH` and `ORACLE_OUT` override), rewrites the verdict file
beside itself, and exits non-zero if any case fails. The original wrote to container paths;
those three lines are the only change from the 4090 author's version besides four wording
edits that removed machine names.

| case | query lengths | 3D-leg reason | 2D vs 3D, max abs diff | vs reference, max abs (2D / 3D) |
|---|---|---|---|---|
| row0-decode-[1] | `[1]` | `NONE(3d eligible)` | 0.0000 | 0.0312 / 0.0312 |
| row1-boundary-[16] | `[16]` | `NONE(3d eligible)` | 0.0078 | 0.0308 / 0.0309 |
| row2-policy-[17] | `[17]` | `max_query_len_policy>16` | 0.0000 | 0.0372 / 0.0372 |
| row3-even-[8,8] | `[8, 8]` | `NONE(3d eligible)` | 0.0088 | 0.0306 / 0.0340 |
| row4-units-[16,1] | `[16, 1]` | `NONE(3d eligible)` | 0.0078 | 0.0314 / 0.0314 |
| row5-ragged-[1,8,3] | `[1, 8, 3]` | `NONE(3d eligible)` | 0.0078 | 0.0336 / 0.0293 |
| row6-zerorow-[4,0,3] | `[4, 0, 3]` | `NONE(3d eligible)` | 0.0078 | 0.0344 / 0.0322 |
| row7-hist-[9] | `[9]` | `NONE(3d eligible)` | 0.0059 | 0.0286 / 0.0285 |
| row9-wild-[848] | `[848]` | `max_query_len_policy>16` | 0.0000 | 0.0320 / 0.0320 |
| row8-breach-[16,16,16] | `[16, 16, 16]` | `q_token_capacity_failed` |  |  /  |

All 10 cases: both legs finite, the 3D leg writes the scratch and the 2D leg leaves it
untouched, the two legs agree to within 0.0088 (fp16 reduction order), and both are within
0.0372 of the reference on outputs of magnitude 17 to 26 (relative 1.9e-3). The 2D leg's
`mq_but_flag_off` reason is the forced-off flag, as intended; the case under test is the 3D
leg. The 17-token single sequence is rejected by the per-sequence limit and never reaches the
capacity question; the `[16, 1]` batch, the same 17 tokens, is accepted. The padded
zero-length case `[4, 0, 3]` leaks nothing. The breach cases assert `mq3d_breach_state()`
(count incremented, sticky flag, first specimen kept across a second breach). Reverting the
production file to sequence-unit sizing makes every multi-query case breach and leaves
exactly the one-token and 17-token cases green, which is the original defect's signature;
restoring the file returns 10 of 10.

**Rerun on the 3090** from this branch, unchanged apart from the path fixes below, after the
patch's header text was rewritten: 10 of 10 cases, 20 of 20 launch calls counted, and every
case's reason bit and 2D-vs-3D difference identical to the 4090 run. Its verdict file is
`bench/mq3d_layer2_verdicts-3090.jsonl`, and it records the shipped patch's sha. So the
independent check holds on both architectures, and the run instruction above is the one that
produced it.

Provenance recorded in the 4090 verdict file: the three surface files by sha256
(`triton_attn.py` aa3fa23692a2…, `int4_per_token_head.py`
cf2bf54178d2…, `triton_unified_attention.py`
24ec41175a58…) and the patch file at c2b5a002e743…, which is
this patch's content before its header text was rewritten; the diff hunks are identical to
the shipped file. Scope: the reference reads the cached bytes the production write path
produced, so it bounds the attention and dispatch, not the cache write.

**Live at `max_num_seqs=8`** (function check only; pre-fix not run at this setting, no
throughput measured): capacity 128 rows, and a batch of 64 tokens across 8 sequences is
served by the 3D kernel.

**Mutations**, on the 6,747-token request above. Allocation re-expressed in sequence units:
the construction assert fires and the engine refuses to boot. Declared capacity shrunk to 8:
the 9-token batch breaches, the counter reads 1 after the first and 16 at the end of the run,
one diagnostic line is written, and the first specimen is unchanged by later breaches.

**Patch application.** The full patch set, this one included, applies without failure or
fuzz to the stock vLLM 0.27.1 wheel in the Dockerfile's order. That is an apply check, not a
build; the same set booted and served on the 3090 for every measurement above.

## Known nit

The boot log states the capacity formula as a sentence. The inputs and the result are printed
and can be recomputed, which is how the second mutation was caught, but the sentence could go
stale relative to the code. Left as a follow-up rather than folded in because the independent
check verified these exact bytes.

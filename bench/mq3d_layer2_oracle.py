"""Layer-2 operator oracle for the int4 3D scratch unit-split patch (Patch A).

Registered spec (the registered spec and the 3090 box's scope notes, recorded before the patch was received):
  - IMPORTS the production dispatch/capacity implementation. Layer 1 tests the
    formula; Layer 2 tests the PATH: the real TritonAttentionMetadataBuilder
    allocation code and the real _launch_packed_attn dispatch, entered through
    unified_attention() exactly as the backend enters it.
  - Matrix: [16] boundary . [17] policy rejection . [8,8] . [16,1] .
    ragged [1,8,3] . padded zero-length row . total-over-capacity mutation.
    [17] vs [16,1] both total 17: if [16,1] rejects on "17" the units are
    still conflated.
  - Forced 2D (VLLM_INT4_MQ_3D=0) and forced 3D (=1) through the patched
    launcher for every row; NaN-prefilled scratch before every call; 2D must
    leave scratch untouched, 3D must write it; outputs finite both legs and
    allclose across legs.
  - Exact reason bits under VLLM_INT4_MQ_3D_DEBUG=1.
  - Breach row asserts mq3d_breach_state() (count, sticky degraded, first
    specimen) -- state, not log lines.
  - Reached-production proof: _launch_packed_attn is call-counted per row; a
    green that never touched the path is structurally impossible.
  - Declared config recorded beside every verdict: MAX_SEQS=2 =>
    capacity 32 so [16,1]=17 is admissible (agreed with the 3090 box).

Geometry is REAL (read this session from the served model's config.json,
Qwen3.8-27B-W4A16-AutoRound-fast): 24 q heads, 4 kv heads, head_dim 256.
Config VALUES are declared; the capacity/allocation CODE is production.
"""

import contextlib
import hashlib
import io
import json
import math
import os
import sys
import types

import torch

DEV = torch.device("cuda:0")
torch.manual_seed(0)

# Resolve the installed vllm package; VLLM_ROOT overrides. (The original hardcoded a
# container venv path.)
VLLM_ROOT = os.environ.get("VLLM_ROOT") or os.path.dirname(__import__("vllm").__file__)
# Verdicts are written beside this script by default; ORACLE_OUT overrides. (The original
# ran in a container and wrote to /work, which does not exist on a native checkout.)
OUT_PATH = os.environ.get(
    "ORACLE_OUT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "mq3d_layer2_verdicts.jsonl"),
)

# ---------------------------------------------------------------- provenance
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

PROVENANCE = {
    "surface": {
        p: sha256(os.path.join(VLLM_ROOT, p))
        for p in (
            "v1/attention/backends/triton_attn.py",
            "v1/attention/ops/int4_per_token_head.py",
            "v1/attention/ops/triton_unified_attention.py",
        )
    },
    "patch_sha256": sha256(os.environ.get("ORACLE_PATCH", os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "patches",
        "spec-decode-scratch-token-units.patch"))),
}

# ---------------------------------------------------------------- prod imports
import vllm.v1.attention.ops.int4_per_token_head as ipth
from vllm.v1.attention.ops.int4_per_token_head import (
    mq3d_breach_state,
    reshape_and_cache_int4,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode
import vllm.v1.attention.backends.triton_attn as ta

# ----------------------------------------------------- declared configuration
# REAL geometry (config.json, read this session); DECLARED scheduler limits.
NUM_HEADS_Q = 24
NUM_HEADS_KV = 4
HEAD_DIM = 256
BLOCK_SIZE = 16
MAX_NUM_SEQS = 2          # DECLARED: makes capacity 32 so [16,1]=17 admissible
MAX_NUM_BATCHED_TOKENS = 8192
KV_HISTORY = 48           # decode-verify shape: history + current queries

DECLARED = {
    "max_num_seqs": MAX_NUM_SEQS,
    "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
    "num_heads_q": NUM_HEADS_Q,
    "num_heads_kv": NUM_HEADS_KV,
    "head_dim": HEAD_DIM,
    "block_size": BLOCK_SIZE,
    "cudagraph": "NONE (eager; capture-snap is a separate change per the patch)",
}

# ------------------------------------------- production builder, stub config
# The capacity derivation, buffer allocation, boot log, and construction
# assert are executed from the PRODUCTION class __init__. Only the config
# OBJECT is a stub carrying the declared values above.
def build_production_builder():
    mc = types.SimpleNamespace(
        get_num_attention_heads=lambda pc: NUM_HEADS_Q,
        get_num_kv_heads=lambda pc: NUM_HEADS_KV,
        get_head_size=lambda: HEAD_DIM,
        rswa_window=None,
    )
    vc = types.SimpleNamespace(
        model_config=mc,
        parallel_config=types.SimpleNamespace(),
        scheduler_config=types.SimpleNamespace(
            max_num_seqs=MAX_NUM_SEQS,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        ),
        compilation_config=types.SimpleNamespace(
            cudagraph_mode=ta.CUDAGraphMode.NONE,
            cudagraph_capture_sizes=[],
            static_forward_context={},
        ),
        cache_config=types.SimpleNamespace(block_size=BLOCK_SIZE),
        speculative_config=None,
    )
    spec = types.SimpleNamespace(block_size=BLOCK_SIZE)
    return ta.TritonAttentionMetadataBuilder(spec, [], vc, DEV)


builder = build_production_builder()
CAPACITY = builder.scratch_token_capacity_3d
SEQ_THRESHOLD = builder.seq_threshold_3D
QLEN_CAP = builder.max_query_len_3d
SEGM = builder.num_par_softmax_segments
BUFS = (
    builder.softmax_segm_output,
    builder.softmax_segm_max,
    builder.softmax_segm_expsum,
)
print(
    f"[oracle] production builder: capacity={CAPACITY} "
    f"seq_threshold_3D={SEQ_THRESHOLD} qlen_cap={QLEN_CAP} segments={SEGM} "
    f"buf_rows={[b.shape[0] for b in BUFS]}"
)
if os.environ.get("ORACLE_SKIP_FORMULA_CHECK") != "1":
    assert CAPACITY == min(
        MAX_NUM_BATCHED_TOKENS, min(MAX_NUM_SEQS, SEQ_THRESHOLD) * QLEN_CAP
    ), "builder capacity disagrees with its own declared formula"

# ------------------------------------------------------------- KV cache build
NUM_BLOCKS = 96
PACKED_HS = HEAD_DIM // 2

kv_raw = torch.zeros(
    NUM_BLOCKS, NUM_HEADS_KV, BLOCK_SIZE, 2 * PACKED_HS,
    dtype=torch.uint8, device=DEV,
)
key_cache, value_cache = kv_raw.transpose(1, 2).split(PACKED_HS, dim=-1)
k_scale_cache = torch.zeros(
    NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS_KV, dtype=torch.float32, device=DEV
)
v_scale_cache = torch.zeros_like(k_scale_cache)


def populate_seq(seq_idx, kv_len, block_table_row):
    """Write kv_len tokens of fresh K/V for one sequence via the production
    write path (reshape_and_cache_int4), into this sequence's blocks.
    Returns the RHT'd source so the reference's dequant math can be
    CALIBRATED against what was actually stored (format-reading check)."""
    k = torch.randn(kv_len, NUM_HEADS_KV, HEAD_DIM, dtype=torch.float16, device=DEV)
    v = torch.randn(kv_len, NUM_HEADS_KV, HEAD_DIM, dtype=torch.float16, device=DEV)
    slots = torch.tensor(
        [
            int(block_table_row[t // BLOCK_SIZE]) * BLOCK_SIZE + t % BLOCK_SIZE
            for t in range(kv_len)
        ],
        dtype=torch.int64, device=DEV,
    )
    reshape_and_cache_int4(
        k, v, key_cache, value_cache, slots,
        k_scale_cache=k_scale_cache, v_scale_cache=v_scale_cache,
    )
    from vllm.v1.attention.ops.int4_per_token_head import single_rht
    return (single_rht(k.float()).to(k.dtype), single_rht(v.float()).to(v.dtype))


# --------------------------------------------------- reached-production proof
CALLS = {"n": 0}
CAPTURE = {"last": None}
_orig_launch = ipth._launch_packed_attn


def _counting_launch(**kw):
    CALLS["n"] += 1
    r = _orig_launch(**kw)
    torch.cuda.synchronize()
    # Capture at the LAUNCHER layer: q here is post-RHT and out is pre-inverse-
    # RHT, so the reference compare involves no RHT at all -- the property
    # under test is the attention kernel, and the transform cancels by layer.
    CAPTURE["last"] = {
        "q": kw["q"].detach().clone(),
        "out": kw["out"].detach().clone(),
        "cu": kw["cu_seqlens_q"].detach().clone(),
        "seqused": kw["seqused_k"].detach().clone(),
        "bt": kw["block_table"].detach().clone(),
        "scale": float(kw["softmax_scale"]),
    }
    return r


ipth._launch_packed_attn = _counting_launch


# ------------------------------------------------- independent reference leg
# Sol's hold (#1640): inter-arm agreement cannot prove both arms right. This
# reference shares NOTHING with the kernel but the cache bytes and their
# documented format: nibble unpack (byte b -> elements 2b low, 2b+1 high),
# zero-point recovery from the scale float's low 4 mantissa bits
# ((bits & -16) | zp -- read from the pack kernel this session), dequant
# (nibble - zp) * scale, then plain fp32 torch attention with an explicit
# causal mask. Computed at the launcher layer where RHT cancels.
def _dequant_tokens(cache, scale_cache, bt_row, kv_len):
    t = torch.arange(kv_len, device=DEV)
    blk = bt_row[(t // BLOCK_SIZE)].long()
    slot = (t % BLOCK_SIZE).long()
    packed = cache[blk, slot].contiguous()          # [kv_len, kvh, PACKED_HS] u8
    low = (packed & 0xF).to(torch.float32)
    high = (packed >> 4).to(torch.float32)
    nib = torch.stack((low, high), dim=-1).reshape(kv_len, NUM_HEADS_KV, HEAD_DIM)
    sbits = scale_cache[blk, slot].contiguous().view(torch.int32)
    zp = (sbits & 0xF).to(torch.float32)
    scale = (sbits & -16).view(torch.float32)
    return (nib - zp.unsqueeze(-1)) * scale.unsqueeze(-1)


def reference_attn(kcap, q_lens):
    """fp32 reference from the cache bytes; zero-length rows have NO output
    rows by construction (cu adjacency) -- the declared/ignored value."""
    ref = torch.zeros_like(kcap["out"], dtype=torch.float32)
    gqa = NUM_HEADS_Q // NUM_HEADS_KV
    for i, ql in enumerate(q_lens):
        if ql == 0:
            continue
        lo, hi = int(kcap["cu"][i]), int(kcap["cu"][i + 1])
        kv_len = int(kcap["seqused"][i])
        kd = _dequant_tokens(key_cache, k_scale_cache, kcap["bt"][i], kv_len)
        vd = _dequant_tokens(value_cache, v_scale_cache, kcap["bt"][i], kv_len)
        qf = kcap["q"][lo:hi].to(torch.float32)      # [ql, hq, hd]
        kf = kd.repeat_interleave(gqa, dim=1)        # [kv, hq, hd]
        vf = vd.repeat_interleave(gqa, dim=1)
        scores = torch.einsum("qhd,khd->hqk", qf, kf) * kcap["scale"]
        qpos = kv_len - ql + torch.arange(ql, device=DEV)
        mask = torch.arange(kv_len, device=DEV)[None, :] > qpos[:, None]
        scores.masked_fill_(mask[None, :, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        ref[lo:hi] = torch.einsum("hqk,khd->qhd", probs, vf)
    return ref

# ------------------------------------------------------------------ one call
def run_case(q_lens, force_3d, seed):
    """Build the row's batch, call unified_attention through the production
    entry, and return the observation record. Both legs of a row share a seed
    so the 2D-vs-3D compare sees IDENTICAL inputs (first run's harness bug:
    per-leg randn made even 2D-vs-2D 'differ' by 1.5)."""
    torch.manual_seed(seed)
    num_seqs = len(q_lens)
    tokens = sum(q_lens)
    max_q = max(q_lens) if q_lens else 0

    blocks_per_seq = math.ceil((KV_HISTORY + max(q_lens or [0])) / BLOCK_SIZE) + 1
    block_table = torch.zeros(num_seqs, blocks_per_seq, dtype=torch.int32, device=DEV)
    next_block = 1  # block 0 stays a zero-filled dummy
    seqused = []
    rht_src = []
    for s, ql in enumerate(q_lens):
        kv_len = KV_HISTORY + ql
        nb = math.ceil(kv_len / BLOCK_SIZE)
        row = list(range(next_block, next_block + nb))
        next_block += nb
        block_table[s, :nb] = torch.tensor(row, dtype=torch.int32, device=DEV)
        rht_src.append(populate_seq(s, kv_len, row))
        seqused.append(kv_len)
    assert next_block <= NUM_BLOCKS

    q = torch.randn(tokens, NUM_HEADS_Q, HEAD_DIM, dtype=torch.float16, device=DEV)
    out = torch.zeros_like(q)
    cu = torch.zeros(num_seqs + 1, dtype=torch.int32, device=DEV)
    cu[1:] = torch.cumsum(torch.tensor(q_lens, device=DEV), 0)
    seqused_k = torch.tensor(seqused, dtype=torch.int32, device=DEV)

    # NaN-prefill the production scratch: 2D must leave it untouched, 3D must
    # overwrite what it reads back, and an unwritten-row read leaks NaN into
    # the output where the finite check catches it.
    for b in BUFS:
        b.fill_(float("nan"))

    os.environ["VLLM_INT4_MQ_3D"] = "1" if force_3d else "0"
    os.environ["VLLM_INT4_MQ_3D_DEBUG"] = "1"

    breach_before = mq3d_breach_state()
    calls_before = CALLS["n"]
    cap = io.StringIO()
    with contextlib.redirect_stdout(cap):
        unified_attention(
            q=q,
            k=key_cache,
            v=value_cache,
            out=out,
            cu_seqlens_q=cu,
            max_seqlen_q=max_q,
            seqused_k=seqused_k,
            max_seqlen_k=max(seqused) if seqused else 0,
            softmax_scale=HEAD_DIM ** -0.5,
            causal=True,
            window_size=(-1, -1),
            block_table=block_table,
            softcap=0.0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            seq_threshold_3D=SEQ_THRESHOLD,
            max_query_len_3d=QLEN_CAP,
            scratch_token_capacity_3d=CAPACITY,
            num_par_softmax_segments=SEGM,
            softmax_segm_output=BUFS[0],
            softmax_segm_max=BUFS[1],
            softmax_segm_expsum=BUFS[2],
            kv_quant_mode=KVQuantMode.INT4_PER_TOKEN_HEAD,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
        )
    torch.cuda.synchronize()

    # Reference leg: my dequant + fp32 attention vs the captured kernel out,
    # plus dequant calibration against the RHT'd source actually stored.
    kcap = CAPTURE["last"]
    ref = reference_attn(kcap, q_lens)
    ref_delta = (
        float((kcap["out"].to(torch.float32) - ref).abs().max().item())
        if sum(q_lens) else 0.0
    )
    calib = 0.0
    for i, ql in enumerate(q_lens):
        kv_len = KV_HISTORY + ql
        kd = _dequant_tokens(key_cache, k_scale_cache, block_table[i], kv_len)
        calib = max(calib, float(
            (kd - rht_src[i][0].to(torch.float32)).abs().max().item()))

    reason_lines = [
        ln for ln in cap.getvalue().splitlines() if "reasons=" in ln
    ]
    reasons = (
        reason_lines[-1].split("reasons=")[-1].strip() if reason_lines else "NO-DEBUG-LINE"
    )
    scratch_touched = not bool(torch.isnan(BUFS[1][:max(tokens, 1)]).all().item())
    return {
        "calls_delta": CALLS["n"] - calls_before,
        "reasons": reasons,
        "scratch_touched": scratch_touched,
        "out_finite": bool(torch.isfinite(out).all().item()),
        "breach_delta": mq3d_breach_state()["count"] - breach_before["count"],
        "ref_max_abs": ref_delta,
        "out_max_abs": float(kcap["out"].abs().max().item()),
        "dequant_calib_max_abs": calib,
        "out": out,
    }


# ---------------------------------------------------------------- the matrix
MATRIX = [
    ("row0-decode-[1]", [1], "informational: q=1 decode, units coincide"),
    ("row1-boundary-[16]", [16], "positive boundary: qlen==cap, 3D expected"),
    ("row2-policy-[17]", [17], "policy rejection: qlen>cap, capacity never asked"),
    ("row3-even-[8,8]", [8, 8], "3D expected"),
    ("row4-units-[16,1]", [16, 1], "17 TOKENS admissible: the units discriminator"),
    ("row5-ragged-[1,8,3]", [1, 8, 3],
     "ragged reducer (operator-level construction: num_seqs exceeds declared MAX_SEQS=2)"),
    ("row6-zerorow-[4,0,3]", [4, 0, 3],
     "padded zero-length row: reducer hazard (operator-level: num_seqs>MAX_SEQS=2)"),
    ("row7-hist-[9]", [9], "informational: the original bug's shape"),
    ("row9-wild-[848]", [848],
     "policy in the wild: the 3090 box's organic q=848 shape, 2D both legs"),
]
# row8 (breach mutation) runs last: [16,16,16] = 48 tokens > capacity 32,
# policy-eligible, so it must breach, fall back 2D, and mark DEGRADED.

EXPECT_3D = {  # under force_3d=1
    "row0-decode-[1]": True,
    "row1-boundary-[16]": True,
    "row2-policy-[17]": False,
    "row3-even-[8,8]": True,
    "row4-units-[16,1]": True,
    "row5-ragged-[1,8,3]": True,
    "row6-zerorow-[4,0,3]": True,
    "row7-hist-[9]": True,
    "row9-wild-[848]": False,
}

verdicts = []
FAIL = 0

for name, q_lens, note in MATRIX:
    seed = 100 + MATRIX.index((name, q_lens, note))
    r2d = run_case(q_lens, force_3d=False, seed=seed)
    r3d = run_case(q_lens, force_3d=True, seed=seed)
    mq = (max(q_lens) if q_lens else 0) > 1

    checks = {
        "reached_production_2d": r2d["calls_delta"] == 1,
        "reached_production_3d": r3d["calls_delta"] == 1,
        # q=1 rows go 3D regardless of the flag (stock behavior: the flag only
        # gates multi-query), so untouched-scratch applies to mq rows only.
        "2d_scratch_untouched": (not mq) or (not r2d["scratch_touched"]),
        "3d_path_taken": r3d["scratch_touched"] == EXPECT_3D[name],
        "2d_out_finite": r2d["out_finite"],
        "3d_out_finite": r3d["out_finite"],
        "no_breach": r2d["breach_delta"] == 0 and r3d["breach_delta"] == 0,
        "2d_reason_flag_off": (not mq) or ("mq_but_flag_off" in r2d["reasons"]),
    }
    if name == "row2-policy-[17]":
        checks["policy_reason_exact"] = "max_query_len_policy>16" in r3d["reasons"]
        checks["not_capacity_reason"] = "q_token_capacity_failed" not in r3d["reasons"]
    if name == "row4-units-[16,1]":
        checks["units_not_conflated"] = r3d["reasons"].startswith("NONE")
    if EXPECT_3D[name] and mq:
        checks["3d_reason_none"] = r3d["reasons"].startswith("NONE")

    d = (r2d["out"] - r3d["out"]).abs()
    maxdiff = float(d.max().item()) if d.numel() else 0.0
    checks["2d_vs_3d_close"] = maxdiff < 5e-2
    # advisory harness-sanity bound only; numerical ACCEPTANCE is Gate 3's call.
    checks["ref_sane_advisory"] = max(r2d["ref_max_abs"], r3d["ref_max_abs"]) < 0.1

    ok = all(checks.values())
    FAIL += 0 if ok else 1
    verdicts.append({
        "row": name, "q_lens": q_lens, "note": note,
        "configured_capacity": CAPACITY, "seq_threshold_3D": SEQ_THRESHOLD,
        "declared": DECLARED,
        "reasons_2d": r2d["reasons"], "reasons_3d": r3d["reasons"],
        "max_abs_diff_2d_vs_3d": maxdiff,
        "ref_max_abs_2d": r2d["ref_max_abs"], "ref_max_abs_3d": r3d["ref_max_abs"],
        "out_max_abs": r3d["out_max_abs"],
        "dequant_calib_max_abs": max(
            r2d["dequant_calib_max_abs"], r3d["dequant_calib_max_abs"]),
        "reference": ("independent fp32 torch attention over own-math dequant "
                      "of the cache bytes (nibble + mantissa-zp), "
                      "launcher-layer compare (RHT cancels)"),
        "checks": checks, "ok": ok,
    })
    print(f"[oracle] {name}: {'OK' if ok else 'FAIL'} "
          f"maxdiff={maxdiff:.4g} ref2d={r2d['ref_max_abs']:.4g} "
          f"ref3d={r3d['ref_max_abs']:.4g} "
          f"calib={r2d['dequant_calib_max_abs']:.4g} "
          f"r3d.reasons={r3d['reasons']}")

# ------------------------------------------------------- row8: breach mutation
pre = mq3d_breach_state()
r_breach = run_case([16, 16, 16], force_3d=True, seed=200)
post1 = mq3d_breach_state()
# Second breach uses a DIFFERENT shape (64 tokens): if "first" were being
# overwritten instead of kept, tokens would read 64 -- same-shape re-breach
# could never falsify the sticky-first claim (anti-vacuous, ledger #10).
r_breach2 = run_case([16, 16, 16, 16], force_3d=True, seed=201)
post2 = mq3d_breach_state()

bchecks = {
    "reached_production": r_breach["calls_delta"] == 1,
    "fell_back_2d": not r_breach["scratch_touched"],
    "out_finite": r_breach["out_finite"],
    "count_incremented": post1["count"] == pre["count"] + 1,
    "degraded_sticky": post1["degraded"] is True and post2["degraded"] is True,
    "count_increments_again": post2["count"] == post1["count"] + 1,
    "first_specimen_kept": (
        post1["first"] is not None
        and post1["first"]["tokens"] == 48
        and post1["first"]["declared"] == CAPACITY
        and post2["first"]["tokens"] == 48  # NOT 64: first survived breach 2
    ),
    "reason_exact": "q_token_capacity_failed" in r_breach["reasons"],
    "policy_reason_absent": "max_query_len_policy" not in r_breach["reasons"],
}
bok = all(bchecks.values())
FAIL += 0 if bok else 1
verdicts.append({
    "row": "row8-breach-[16,16,16]", "q_lens": [16, 16, 16],
    "note": "deliberate scheduler-contract violation under MAX_SEQS=2, exercising the runtime invariant wall (48 tokens > capacity 32, policy-eligible)",
    "configured_capacity": CAPACITY, "seq_threshold_3D": SEQ_THRESHOLD,
    "declared": DECLARED,
    "reasons_3d": r_breach["reasons"],
    "breach_state_after": {k: v for k, v in post2.items()},
    "checks": bchecks, "ok": bok,
})
print(f"[oracle] row8-breach: {'OK' if bok else 'FAIL'} state={post2}")

# --------------------------------------------------------------------- emit
with open(OUT_PATH, "w") as f:
    f.write(json.dumps({"provenance": PROVENANCE, "declared": DECLARED,
                        "capacity": CAPACITY, "seq_threshold_3D": SEQ_THRESHOLD}) + "\n")
    for v in verdicts:
        v.pop("out", None)
        f.write(json.dumps(v) + "\n")

total = len(verdicts)
print(f"[oracle] === {total - FAIL}/{total} rows OK; "
      f"launch calls total={CALLS['n']}; verdicts -> {OUT_PATH} ===")
sys.exit(1 if FAIL else 0)

#!/usr/bin/env python3
"""Property test for the int4 3D scratch capacity derivation (spec-decode-scratch-token-units.patch).

THE PROPERTY: for every batch the 3D policy declares eligible, the total query tokens
in that batch fit in the derived scratch capacity.

    capacity = min(max_num_batched_tokens,
                   min(max_num_seqs, seq_threshold_3D) * max_query_len_3d)

This is the invariant the shipped patch relies on. If it can be violated by any config
the engine accepts, the runtime guard becomes reachable in production -- which is the
defect the patch exists to remove, not a safety net it is allowed to lean on.

Pure arithmetic: no GPU, no engine, no server. Runs in a second, so it can cover config
space that a live boot never will -- including max_num_seqs above the sequence threshold
and max_num_batched_tokens as the active limiter, neither of which the single-user
launcher can produce (it hardcodes max_num_seqs=1).
"""
import argparse
import itertools
import random
import sys


def derived_capacity(max_num_batched_tokens, max_num_seqs, seq_threshold_3d, max_query_len_3d):
    return min(max_num_batched_tokens,
               min(max_num_seqs, seq_threshold_3d) * max_query_len_3d)


def policy_eligible(q_lens, max_num_seqs, seq_threshold_3d, max_query_len_3d,
                    max_num_batched_tokens):
    """The 3D policy's own eligibility rules, stated once."""
    n = len(q_lens)
    return (
        n >= 1
        and n <= seq_threshold_3d               # the num_seqs threshold check
        and n <= max_num_seqs                   # the scheduler cannot exceed its own cap
        and max(q_lens) <= max_query_len_3d     # the query-length policy cap
        and sum(q_lens) <= max_num_batched_tokens  # the scheduler's token budget
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--random-vectors', type=int, default=20000,
                    help='ragged query-length vectors sampled per config tuple')
    ap.add_argument('--seed', type=int, default=20260901)
    ap.add_argument('--mutate', choices=['seq-rows', 'snapped-rows'],
                    help="NEGATIVE CONTROL. Replace the derivation with a known-broken one "
                         "and require the property to FAIL. A property test that has never "
                         "been shown to go red is a decoration: 'seq-rows' is the pre-patch "
                         "bug (capacity = the SEQUENCE threshold, in the wrong unit); "
                         "'snapped-rows' additionally snaps that threshold to the nearest "
                         "capture size, which is what shipped.")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    # seq thresholds that 128 // num_heads_kv can actually produce, plus the snapped
    # capture sizes the capture path selects.
    SEQ_THRESHOLDS = [1, 2, 4, 8, 16, 32, 64, 128]
    # deliberately spans below / equal / above every seq threshold
    MAX_NUM_SEQS = [1, 2, 4, 7, 8, 9, 16, 31, 32, 33, 64, 256]
    # small values make max_num_batched_tokens the ACTIVE limiter
    MAX_BATCHED = [8, 16, 64, 128, 2048, 8192, 131072]
    MAX_QLEN = [1, 2, 8, 16, 17, 32]

    checked = violations = eligible_seen = 0
    limiter = {"batched": 0, "product": 0}
    failures = []

    for mnbt, mns, thr, mql in itertools.product(MAX_BATCHED, MAX_NUM_SEQS,
                                                 SEQ_THRESHOLDS, MAX_QLEN):
        if a.mutate == 'seq-rows':
            cap = thr                       # the original defect: rows sized in SEQUENCES
        elif a.mutate == 'snapped-rows':
            cap = min([1, 2, 4, 8, 16, 24, 32], key=lambda x: abs(x - thr))
        else:
            cap = derived_capacity(mnbt, mns, thr, mql)
        limiter["batched" if mnbt <= min(mns, thr) * mql else "product"] += 1

        # 1. exhaustive corners: every-sequence-at-max, single sequence at max, ones
        corners = []
        n_max = min(mns, thr)
        if n_max >= 1:
            corners += [[mql] * n_max, [mql], [1] * n_max,
                        [mql] * max(1, n_max - 1) + [1]]
            if n_max >= 2:
                corners.append([mql] + [1] * (n_max - 1))
        # 2. ragged random vectors
        for _ in range(max(0, a.random_vectors // (len(MAX_BATCHED) * len(MAX_NUM_SEQS)))):
            n = rng.randint(1, max(1, min(mns, thr) + 2))     # +2 probes ABOVE the cap
            corners.append([rng.randint(1, mql + 2) for _ in range(n)])

        for q_lens in corners:
            checked += 1
            if not policy_eligible(q_lens, mns, thr, mql, mnbt):
                continue
            eligible_seen += 1
            if sum(q_lens) > cap:
                violations += 1
                if len(failures) < 5:
                    failures.append((mnbt, mns, thr, mql, cap, q_lens, sum(q_lens)))

    print(f"  config tuples : {len(MAX_BATCHED)*len(MAX_NUM_SEQS)*len(SEQ_THRESHOLDS)*len(MAX_QLEN):,}")
    print(f"  vectors tested: {checked:,}   policy-eligible: {eligible_seen:,}")
    print(f"  active limiter: max_num_batched_tokens in {limiter['batched']:,} tuples, "
          f"the seq x qlen product in {limiter['product']:,}")
    if eligible_seen == 0:
        print("\n  REFUSING TO PASS: zero eligible vectors were exercised.")
        return 2
    if violations and a.mutate:
        print(f"\n  NEGATIVE CONTROL OK: mutation '{a.mutate}' violates the property in "
              f"{violations:,} case(s), e.g.")
        for mnbt, mns, thr, mql, cap, q, tot in failures[:3]:
            print(f"    max_num_seqs={mns} thr={thr} max_qlen={mql} -> capacity={cap}, "
                  f"but eligible {q} sums to {tot}")
        return 0
    if violations:
        print(f"\n  PROPERTY VIOLATED in {violations:,} case(s):")
        for mnbt, mns, thr, mql, cap, q, tot in failures:
            print(f"    max_num_batched_tokens={mnbt} max_num_seqs={mns} thr={thr} "
                  f"max_qlen={mql} -> capacity={cap}, but eligible {q} sums to {tot}")
        return 1
    if a.mutate:
        print(f"\n  NEGATIVE CONTROL FAILED: mutation '{a.mutate}' did NOT violate the "
              f"property. The test cannot detect the bug it was written for.")
        return 3
    print(f"\n  PROPERTY HOLDS: every one of {eligible_seen:,} policy-eligible batches "
          f"fits the derived capacity.")
    return 0


if __name__ == '__main__':
    sys.exit(main())

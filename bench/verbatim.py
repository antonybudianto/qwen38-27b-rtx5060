"""How much of an answer is a copy of its source, and when that counts as broken.

Shared by bench/residue_sweep.py and bench/bugb_sweep.py, which both ask the server to
reproduce a document verbatim and then have to decide whether it did.

Three different damage shapes have now been seen at the SAME broken prompt length
(issue #25, `SPEC=mtp CTX=huge` under FULL capture, residue 4): an empty answer with
`finish_reason=stop`, a one-character answer, and 400 tokens of fluent Danish that
opens with a malformed `<think>` and invents a translation task the prompt never asked
for. The location is deterministic; the manifestation is not. Any detector keyed on one
signature -- "it repeats", "it came back empty" -- files at least one of the three as
healthy, and that is how the mtp failure survived several full sweeps.

What survives all three is coverage: how much of the answer is actually a copy of the
source. Two rules use it here, and the second is the one that matters:

  length      an answer too short to be the copy is broken however well it scores.
              Not redundant: the live residue-4 break returns ONE character, that
              character occurs in the document, and its coverage is therefore 1.00.
  absolute    a copy task that returns almost nothing from the document is broken
              whatever it returned instead.
  neighbours  compare against what the OTHER prompt lengths on the same server produced.
              A model that is simply bad at this prompt scores low everywhere and is a
              measurement problem; a model that scores low at one length out of 128 is
              this bug.

Coverage rather than a longest-prefix match, because a prefix match cannot do this job:
one wrong character at offset 38 pins it at 38 however perfect the next 750 are, which
files an ordinary greedy divergence as a collapse. Same for a model that answers the
copy task with a one-line preamble — under a prefix rule that is 0 characters verbatim
and BROKEN; under coverage it costs one window.

  venv/bin/python bench/verbatim.py     # self-test against all three recorded shapes
"""

WINDOW = 40      # long enough that a window matches the source by content, not by luck
STRIDE = 20      # overlapping, so a divergence costs two windows rather than the tail


def coverage(ans, doc):
    """Fraction of the answer's 40-character windows that appear verbatim in the source."""
    if not ans:
        return 0.0
    stop = max(1, len(ans) - WINDOW + 1)
    wins = [ans[i:i + WINDOW] for i in range(0, stop, STRIDE)]
    return sum(w in doc for w in wins) / len(wins)


def prefix_match(ans, doc):
    """Longest prefix of the answer that appears in the source. Reported, not judged on."""
    n = 0
    while n < len(ans) and ans[:n + 1] in doc:
        n += 1
    return n


def repeats(ans):
    """Most times any 40-character window of the answer occurs in it."""
    return max((ans.count(ans[i:i + WINDOW]) for i in range(0, max(1, len(ans) - WINDOW), WINDOW)),
               default=0)


def invented_repeats(ans, doc):
    """Repetition the MODEL added, i.e. beyond what the source itself repeats.

    Coverage alone lets a long enough loop through: repeat a 500-character passage three
    times and only the two seam windows fail to match. Counting occurrences in the answer
    alone would instead punish a document that legitimately repeats a line.
    """
    return max((ans.count(ans[i:i + WINDOW]) - doc.count(ans[i:i + WINDOW])
                for i in range(0, max(1, len(ans) - WINDOW), WINDOW)), default=0)


def classify(ans, doc, ref=None, min_len=40, floor=0.5, rel=0.6, max_loop=3):
    """-> (flag, coverage, reason). flag is "ok" or "BROKEN".

    ref is the coverage the neighbouring prompt lengths reached on this server; pass None
    on the first few samples, before there is a neighbourhood to compare against.
    """
    cov = coverage(ans, doc)
    if not ans:
        return "BROKEN", cov, "empty"
    if len(ans) < min_len:
        return "BROKEN", cov, f"{len(ans)} chars"
    if cov < floor:
        return "BROKEN", cov, f"cov {cov:.2f} < {floor}"
    if ref is not None and ref > 0 and cov < rel * ref:
        return "BROKEN", cov, f"cov {cov:.2f} < {rel:g}x neighbours {ref:.2f}"
    loop = invented_repeats(ans, doc)
    if loop > max_loop:
        return "BROKEN", cov, f"loops {loop}x"
    return "ok", cov, ""


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2


def _selftest():
    """The outcomes this has to tell apart: the three shapes from the issue-25 thread,
    and the healthy answers a rule written for them would wrongly condemn."""
    doc = ("Kapitel 1. Den lange rejse begyndte en tirsdag i november, og ingen af dem "
           "vidste hvor den ville ende. Vejret var koldt, og toget var forsinket med "
           "fyrre minutter. Anna satte sig ved vinduet og kiggede ud paa marker, der "
           "laa graa og tomme under en lav himmel. Hun havde pakket for lidt toej med.")
    good = doc[:300]
    cases = [
        # answer,                              expect,   what it is
        (good, "ok", "the correct verbatim copy"),
        ("", "BROKEN", "empty answer, finish_reason=stop"),
        ("K", "BROKEN", "one character, then nothing"),
        ("**<think>\nThe user wants me to translate the first 60 lines of the document "
         "into Danish.Let\n\nLet me look at the first 40 lines and see what they say "
         "before I begin the translation work in earnest, since the user did not "
         "specify a target language for the output text.", "BROKEN",
         "400 tokens of confident derailment"),
        (good[:120] + "X" + good[121:300], "ok", "one divergent character mid-copy"),
        (("Vejret var koldt, og toget var forsinket med fyrre minutter. " * 6), "BROKEN",
         "degenerate repetition of one source line, the dflash2 shape"),
        (doc[100:180] * 4 + doc[:200], "BROKEN", "a long loop, which coverage alone misses"),
        (doc[:200] + doc[40:80] + doc[200:300], "ok",
         "a source passage that repeats a line is still a copy"),
        ("Her er de første 60 linjer:\n\n" + good, "ok", "a copy behind a one-line preamble"),
    ]
    ref = coverage(good, doc)
    bad = 0
    for ans, expect, what in cases:
        flag, cov, why = classify(ans, doc, ref=ref)
        ok = flag == expect
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {flag:<7} cov={cov:4.2f} "
              f"prefix={prefix_match(ans, doc):>4} rep={repeats(ans):>2}  {what}"
              + (f"  [{why}]" if why else ""))
    # A prefix match would call case 5 broken at 120 of 300 characters; coverage does not.
    assert prefix_match(cases[4][0], doc) < 130, "the divergent-character case should trip a prefix rule"
    print("all shapes classified correctly" if not bad else f"{bad} FAILURES")
    return bad


if __name__ == "__main__":
    import sys
    sys.exit(1 if _selftest() else 0)

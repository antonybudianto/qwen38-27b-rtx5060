"""Bug B: sweep prompt length and report exact prompt tokens, acceptance, and whether
the verbatim answer is correct.

  venv/bin/python bench/bugb_sweep.py 19500 19800 20000 20200 ...

Against SPEC=dflash2 CTX=huge PREFIX_CACHE=1 CUDAGRAPH_MODE=FULL_AND_PIECEWISE, this
finds one broken length in every 128: on this box every prompt length ==124 (mod 128)
collapses to 1.97 tok/step and degenerate repetition, while every other residue is
clean -- see docs/gotchas.md gotcha 37. The `mod128` column is the point of the script.
The default PIECEWISE has no such length and this sweep is flat against it.

Sweep in steps of 1 token near a broken length, not in steps of 100: at a coarse grid
this reads as a threshold, which is how it was first (mis)diagnosed.
"""
import json, os, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)


def _key(path):  # a key is optional; keyless servers ignore the header
    try:
        return open(path).read().strip()
    except OSError:
        return ""


KEY = os.environ.get("VLLM_API_KEY") or _key(os.path.join(REPO, "api_key.txt"))
BASE = "http://127.0.0.1:" + os.environ.get("PORT", "18020")
DOC = open(os.path.expanduser(os.environ.get("CORPUS", "~/bench/labd_corpus.txt"))).read()

from transformers import AutoTokenizer
TOK = AutoTokenizer.from_pretrained(os.path.join(REPO, "models", "Qwen3.8-27B-W4A16-AutoRound-fast"))


def metrics():
    req = urllib.request.Request(BASE + "/metrics", headers={"Authorization": "Bearer " + KEY})
    d = {}
    for line in urllib.request.urlopen(req).read().decode().splitlines():
        for k in ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_accepted_tokens_total"):
            if line.startswith(k + " ") or line.startswith(k + "{"):
                d[k] = float(line.split()[-1])
    return d.get("vllm:spec_decode_num_drafts_total", 0.0), d.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)


print(f"{'ctx':>7} {'prompt tok':>11} {'mod128':>7} {'tok/step':>9} {'verbatim':>12} {'repeats':>8}")
for ctx in [int(a) for a in sys.argv[1:]]:
    doc = DOC[: int(ctx * 3.6)]
    content = ("Dokument:\n\n" + doc +
               "\n\nGengiv ordret de første 60 linjer af dokumentet. Ingen kommentarer, kun teksten.")
    # The ENGINE sees the chat-templated prompt, not this string: the Qwen3 wrapper
    # with enable_thinking=false is +12 tokens. Measuring the raw content was how the
    # residue got misread as "R = 117 + k" and then as a mysterious constant 12 -- the
    # real rule is just (templated length) == L (mod 128). Never report the raw count.
    ptok = len(TOK.encode(TOK.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False), add_special_tokens=False))
    body = json.dumps({"model": "qwen3.8-27b",
                       "messages": [{"role": "user", "content": content}],
                       "max_tokens": 400, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    d0 = metrics()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}),
        timeout=1200).read().decode())
    d1 = metrics()
    # `or ""`: a collapse-to-stop returns content=null, and indexing that raised a
    # TypeError that ended the sweep silently -- which is how the mtp failure stayed
    # invisible through several full runs (issue #25, mjungnickel18).
    ans = r["choices"][0]["message"].get("content") or ""
    dn = d1[0] - d0[0]
    tps = (d1[1] - d0[1]) / dn + 1 if dn else float("nan")
    # longest prefix of the answer that appears verbatim in the source document
    n = 0
    while n < len(ans) and ans[:n + 1] in doc:
        n += 1
    rep = max((ans.count(ans[i:i + 40]) for i in range(0, max(1, len(ans) - 40), 40)), default=0)
    # Judge on the OUTPUT, not on an absolute tok/step: the ceiling is the verify
    # block, so a healthy k=3 run sits at 3.96 and an absolute threshold calls it
    # broken. A working verbatim task reproduces the whole answer; a broken one
    # returns a few characters, a degenerate repeat, or nothing at all.
    # n is a LONGEST-PREFIX match, so one wrong character at offset 38 pins it at 38
    # however perfect the next 750 are -- `repeats` is what separates a real collapse
    # from a single divergent token. But do NOT require repetition to call it BROKEN:
    # a collapse-to-stop returns one or two characters and repeats nothing, and filing
    # that as DIVERGED is what hid the mtp failure at residue 4.
    if len(ans) > 40 and n >= len(ans) - 2:
        flag = "ok"
    elif len(ans) < 40 or rep > 3:
        flag = "BROKEN"          # truncated to nothing, or degenerate repetition
    else:
        flag = "DIVERGED"        # full-length answer that stops matching the source
    print(f"{ctx:>7} {ptok:>11} {ptok % 128:>7} {tps:>9.2f} "
          f"{str(n) + '/' + str(len(ans)):>12} {rep:>8}  {flag}")

"""Every residue mod 128, one prompt each, on a warm server.

Sampling a handful of residues is not a test: with one broken length in 128, five
samples miss it 82% of the time, and this repo shipped a wrong claim on exactly that
arithmetic (issue #25). Sweep all of them or say you sampled.

Stepping the pad by 1 covers all 128 residues exactly once, because one pad token is
one prompt token here -- no coverage gaps. Each prompt is sent TWICE and the second
(a full self-hit) is the measurement, which is the condition the bug needs.

  venv/bin/python residue_all.py <label> [start] [count]
"""
import hashlib, json, os, re, sys, urllib.request

LABEL = sys.argv[1]
START = int(sys.argv[2]) if len(sys.argv) > 2 else 0
COUNT = int(sys.argv[3]) if len(sys.argv) > 3 else 128
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)


def _key(path):  # a key is optional; keyless servers ignore the header
    try:
        return open(path).read().strip()
    except OSError:
        return ""


KEY = os.environ.get("VLLM_API_KEY") or _key(os.path.join(REPO, "api_key.txt"))
BASE = "http://127.0.0.1:18020"
DOC = open(os.path.expanduser("~/bench/labd_corpus.txt")).read()[:72000]
from transformers import AutoTokenizer
TOK = AutoTokenizer.from_pretrained(os.path.join(REPO, "models", "Qwen3.8-27B-W4A16-AutoRound-fast"))


def metrics():
    txt = urllib.request.urlopen(urllib.request.Request(
        BASE + "/metrics", headers={"Authorization": "Bearer " + KEY}), timeout=30).read().decode()
    g = lambda n: sum(float(x) for x in re.findall(
        rf"^{re.escape(n)}\{{[^}}]*}} ([0-9.e+]+)$", txt, re.M)) or 0.0
    return g("vllm:spec_decode_num_drafts_total"), g("vllm:spec_decode_num_accepted_tokens_total")


def once(content):
    body = json.dumps({"model": "qwen3.8-27b", "messages": [{"role": "user", "content": content}],
                       "max_tokens": 300, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    d0 = metrics()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}),
        timeout=1800).read().decode())
    d1 = metrics()
    ans = r["choices"][0]["message"].get("content") or ""
    dn = d1[0] - d0[0]
    tps = (d1[1] - d0[1]) / dn + 1 if dn else float("nan")
    n = 0
    while n < len(ans) and ans[:n + 1] in DOC:
        n += 1
    rep = max((ans.count(ans[i:i + 40]) for i in range(0, max(1, len(ans) - 40), 40)), default=0)
    return tps, n, len(ans), rep, hashlib.sha1(ans.encode()).hexdigest()[:12]


bad = []
for pad in range(START, START + COUNT):
    content = ("Dokument:\n\n" + DOC +
               "\n\nGengiv ordret de første 60 linjer af dokumentet. Ingen kommentarer, kun teksten."
               + " x" * pad)
    real = len(TOK.encode(TOK.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False), add_special_tokens=False))
    once(content)                                  # arm the prefix cache
    tps, n, ln, rep, h = once(content)             # measure the self-hit
    ok = ln > 40 and n >= ln - 2
    if not ok:
        bad.append((real % 128, real, tps, n, ln, rep, h))
        print(f"[{LABEL}] BROKEN residue={real % 128:3d} len={real} tok/step={tps:.2f} "
              f"verbatim={n}/{ln} rep={rep} sha={h}", flush=True)
    elif pad % 16 == 0:
        print(f"[{LABEL}] .. residue={real % 128:3d} ok ({pad - START + 1}/{COUNT})", flush=True)
print(f"[{LABEL}] DONE: {len(bad)} broken of {COUNT} residues"
      + ("" if not bad else "  -> " + ", ".join(str(b[0]) for b in bad)), flush=True)

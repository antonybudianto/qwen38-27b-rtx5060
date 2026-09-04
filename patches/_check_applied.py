"""Is this patch's content present in the installed vLLM tree?

verify.sh checks patches with a reverse dry-run, which is exact but cannot see a patch whose
hunks were disturbed by a second patch touching the same file (the DFlash2 pair does that).
This is the fallback: take the substantial lines a patch adds and look for them in the tree.

Per file, not tree-wide. The tree-wide version passed a patch whose hunks had been rejected
outright, because the lines it adds to one file also occur in a sibling file the patch does
not touch (PR #43: a patch that failed every hunk still cleared an 80% tree-wide threshold).
A patch's added lines must be in the files that patch claims to change.

  python patches/_check_applied.py <patch> <site-packages/vllm>   # exit 0 = present
"""
import os
import sys

patch, root = sys.argv[1], sys.argv[2]

# {path relative to the vllm package: [substantial added lines]}
per_file: "dict[str, list[str]]" = {}
current = None
for line in open(patch, errors="replace").read().splitlines():
    if line.startswith("+++ "):
        p = line[4:].split("\t")[0].strip()
        if p == "/dev/null":
            current = None
            continue
        # patches here are -p1 against the vllm package: strip a leading a/ or b/
        parts = p.split("/")
        if parts and parts[0] in ("a", "b"):
            parts = parts[1:]
        current = "/".join(parts)
        per_file.setdefault(current, [])
        continue
    if current and line.startswith("+") and not line.startswith("+++"):
        text = line[1:].strip()
        if len(text) > 24 and not text.startswith(("#", '"', "'")):
            per_file[current].append(text)

per_file = {f: lines[:400] for f, lines in per_file.items() if lines}
if not per_file:
    sys.exit(1)

for rel, added in per_file.items():
    path = os.path.join(root, rel)
    try:
        blob = open(path, errors="replace").read()
    except OSError:
        sys.exit(1)
    hits = sum(1 for a in added if a in blob)
    # ceil(0.8 * n), so a file contributing one or two substantial lines still has to
    # have them -- the old max(3, ...) floor would have failed such a file outright.
    if hits < -(-8 * len(added) // 10):
        sys.exit(1)
sys.exit(0)

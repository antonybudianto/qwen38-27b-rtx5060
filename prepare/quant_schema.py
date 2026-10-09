"""Refuse a checkpoint these scripts cannot extend, before they write anything (#241).

prepare/quant_lm_head.py, quant_embed.py, quant_mtp.py and quant_heads_stream.py do not
build a quantization config. They read the one already in config.json, clone
`config_groups.group_0` and append a group of their own. A checkpoint whose config.json is
not already in compressed-tensors pack-quantized form has neither group_0 nor an `ignore`
list, so that read raises KeyError -- and because the shard is written before config.json
and the index (that order is deliberate, see atomic_publish.py), the error arrives after
the shard has already been replaced.

A native AutoRound export is the usual cause: its `quantization_config` carries
bits/group_size/sym/data_type and `quant_method: "auto-round"`, which is not a shape these
scripts can extend -- it has to be converted, not modified. The directory name is not a
guide either way: the base model they were written for is a compressed-tensors re-export
that keeps "-AutoRound" in its name.

load_config() runs before the first shard is opened, so an unsupported checkpoint gets one
line and an untouched directory instead of a half-rewritten one.
"""

import json
import sys

SEE = ('prepare/README.md, "Which checkpoints these scripts modify" -- they extend an '
       "existing\n  compressed-tensors config, they do not convert one.")


def _reject(path, reason, detail, qc):
    sys.exit(f"{path}: {reason}\n  {detail}\n"
             f"  quantization_config has: {', '.join(sorted(qc)) or '(nothing)'}\n"
             f"  {SEE}\n  Nothing was written.")


def load_config(d):
    """Read <d>/config.json and return (config, quantization_config), after checking that
    it is the compressed-tensors schema the in-place scripts extend. Exits non-zero, before
    any write, when it is not."""
    path = d + "config.json"
    try:
        with open(path) as f:
            c = json.load(f)
    except FileNotFoundError:
        sys.exit(f"{path} does not exist: {d} is not a model directory")
    except json.JSONDecodeError as e:
        sys.exit(f"{path} is not valid JSON ({e}); an interrupted run may have left it half-written")

    qc = c.get("quantization_config")
    if not isinstance(qc, dict):
        sys.exit(f"{path} has no quantization_config, so there is no group_0 to extend.\n"
                 f"  {SEE}\n  Nothing was written.")

    method = qc.get("quant_method")
    if method is not None and method != "compressed-tensors":
        _reject(path, f"quant_method is {method!r}, not 'compressed-tensors'",
                "this looks like a native quantizer export (an AutoRound one carries bits/\n"
                "  group_size/sym/data_type). It has to be converted to compressed-tensors\n"
                "  pack-quantized first; these scripts cannot do that. A checkpoint already\n"
                "  prepared for HyperQwen needs none of prepare/ at all.", qc)

    if not isinstance(qc.get("ignore"), list):
        _reject(path, "quantization_config has no 'ignore' list",
                "the scripts rewrite it (they drop lm_head and the mtp.* linears from it).", qc)

    groups = qc.get("config_groups")
    if not isinstance(groups, dict) or "group_0" not in groups:
        _reject(path, "quantization_config has no config_groups.group_0",
                "every group these scripts add is a clone of group_0.", qc)

    print(f"config.json: compressed-tensors, extending config_groups.group_0 "
          f"({len(groups)} existing group(s), {len(qc['ignore'])} ignored modules)")
    return c, qc

#!/bin/bash
# Container entrypoint. First argument selects what to run:
#   single   single-user/start_qwen.sh  (MTP speculative decoding, low latency)
#   batch    batch/start_qwen.sh        (throughput)
#   prepare  docker/prepare.sh          (download + requantize the model into /app/models)
#   bench    bench/run_benchmarks.sh    (client only: benchmarks $HOST:$PORT, no GPU work)
#   verify   verify.sh [args]
#   <anything else> is exec'd as a command (e.g. bash)
# Before serving, verify.sh --no-server runs and aborts on FAIL (model not
# requantized, patches missing, ...); VERIFY=0 skips that.
set -e
cd /app
cmd=${1:-single}; shift || true
case "$cmd" in
  single|batch)
    if [ "${VERIFY:-1}" != "0" ]; then
      bash verify.sh --no-server || { echo "entrypoint: verify.sh FAILED — fix the above or set VERIFY=0"; exit 1; }
    fi
    if [ "$cmd" = single ]; then exec bash single-user/start_qwen.sh "$@"; else exec bash batch/start_qwen.sh "$@"; fi ;;
  prepare) exec bash docker/prepare.sh "$@" ;;
  verify)  exec bash verify.sh "$@" ;;
  bench)   exec bash bench/run_benchmarks.sh "$@" ;;
  *)       exec "$cmd" "$@" ;;
esac

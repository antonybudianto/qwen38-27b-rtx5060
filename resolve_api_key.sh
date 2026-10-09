#!/bin/bash
# resolve_api_key.sh - one key-precedence resolver for the scripts that serve the
# API and the scripts that call it (issue #113: bench/warmup.sh re-derived the
# launcher's chain, and a mismatch means every request 401s).
#
# Sourced after REPO is set. It defines functions and sets nothing by itself, so
# each caller opts into the side it needs:
#
#   resolve_vllm_key    server side. vLLM binds `--api-key` from VLLM_API_KEY, so
#                       this is the key the server will require. api_key.txt is
#                       the documented file fallback and an exported
#                       VLLM_API_KEY wins. No placeholder: with no key anywhere,
#                       the server binds none, and a placeholder here would make
#                       it demand a key nobody configured.
#
#   resolve_bind_host   server side, after resolve_vllm_key. Sets BIND_HOST, the
#                       --host the launchers pass. An explicit HOST wins. Else a
#                       server with a key listens on every interface (0.0.0.0,
#                       as before), and one without a key listens on 127.0.0.1
#                       only, because with no key nothing on the network is
#                       protected (#204). Inside a container the default stays
#                       0.0.0.0, since a published port cannot reach a loopback
#                       bind; there the port mapping is what limits exposure.
#
#   resolve_client_key  client side. A client presents OPENAI_API_KEY, so it must
#                       equal the key the server bound. An explicit
#                       OPENAI_API_KEY wins (the client may point at a server on
#                       another host, where the local file is not the right key),
#                       else the server key, else api_key.txt, else a harmless
#                       placeholder - a value to send to a server that bound no
#                       key and ignores it. This reads the server side but does
#                       not export it, so a client process gains no variable the
#                       launcher did not set.

resolve_vllm_key() {
  if [ -z "${VLLM_API_KEY:-}" ] && [ -f "$REPO/api_key.txt" ]; then
    export VLLM_API_KEY="$(cat "$REPO/api_key.txt")"
  fi
}

resolve_client_key() {
  if [ -z "${OPENAI_API_KEY:-}" ]; then
    if [ -n "${VLLM_API_KEY:-}" ]; then
      export OPENAI_API_KEY="$VLLM_API_KEY"
    elif [ -f "$REPO/api_key.txt" ]; then
      export OPENAI_API_KEY="$(cat "$REPO/api_key.txt")"
    else
      export OPENAI_API_KEY="EMPTY"
    fi
  fi
}

resolve_bind_host() {
  if [ -n "${HOST:-}" ]; then
    BIND_HOST=$HOST
    if [ -z "${VLLM_API_KEY:-}" ]; then
      case "$HOST" in 127.*|localhost|::1) ;; *)
        echo "WARNING: no API key and HOST=$HOST: anything that can reach this port can use the server. Set VLLM_API_KEY or api_key.txt (openssl rand -hex 24)." >&2 ;;
      esac
    fi
  elif [ -n "${VLLM_API_KEY:-}" ]; then
    BIND_HOST=0.0.0.0
  elif [ -f /.dockerenv ]; then
    BIND_HOST=0.0.0.0
    echo "WARNING: no API key: this container listens on 0.0.0.0, so whatever the published port reaches is open. Set VLLM_API_KEY (make keygen) or publish the port on 127.0.0.1 only." >&2
  else
    BIND_HOST=127.0.0.1
    echo "no API key: binding 127.0.0.1 only. To serve other machines, set VLLM_API_KEY (openssl rand -hex 24 > api_key.txt) or HOST=0.0.0.0." >&2
  fi
}

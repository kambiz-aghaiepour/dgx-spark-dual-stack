#!/bin/sh
# Test shim that emulates the second spark without ssh.
#
#   fake-node.sh <host> <helper> <args...>
#
# Runs the helper with HOME=$FAKE_NODE_HOME, so the "worker" gets its own cache,
# config and archive path while the orchestrator still talks to it through the
# same subprocess boundary it uses for the real worker.
host="$1"
shift
exec env HOME="${FAKE_NODE_HOME:?FAKE_NODE_HOME must be set}" \
    SPARK_VLLM_NODE="${FAKE_NODE_NAME:-worker}" python3 "$@"

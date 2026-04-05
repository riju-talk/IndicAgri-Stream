#!/usr/bin/env bash
set -euo pipefail

FLINK_HOME="${FLINK_HOME:-/opt/flink}"
FLINK_BIN="${FLINK_HOME}/bin/flink"
DEFAULT_JOB="${DEFAULT_JOB:-${FLINK_HOME}/jobs/main.py}"

export PATH="/opt/flink/pyflink-venv/bin:${PATH}"

print_help() {
  cat <<'EOF'
Usage:
  /docker-entrypoint.sh help
  /docker-entrypoint.sh jobmanager
  /docker-entrypoint.sh taskmanager
  /docker-entrypoint.sh run [job_file.py] [additional flink args]

Examples:
  /docker-entrypoint.sh run /opt/flink/jobs/pipeline.py --python /opt/flink/jobs/pipeline.py
EOF
}

MODE="${1:-help}"

case "${MODE}" in
  help)
    print_help
    ;;
  jobmanager)
    exec "${FLINK_HOME}/bin/jobmanager.sh" start-foreground
    ;;
  taskmanager)
    exec "${FLINK_HOME}/bin/taskmanager.sh" start-foreground
    ;;
  run)
    JOB_FILE="${2:-${DEFAULT_JOB}}"
    if [[ ! -f "${JOB_FILE}" ]]; then
      echo "PyFlink job file not found: ${JOB_FILE}"
      exit 1
    fi
    shift 2 || true
    exec "${FLINK_BIN}" run -py "${JOB_FILE}" "$@"
    ;;
  *)
    exec "$@"
    ;;
esac

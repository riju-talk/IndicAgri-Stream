#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"

run_lint() {
  local lint_targets=()

  for dir in api ml dags; do
    if [[ -d "${dir}" ]]; then
      lint_targets+=("${dir}")
    fi
  done

  if ((${#lint_targets[@]} == 0)); then
    echo "No lint targets found (api/, ml/, dags/). Skipping flake8/black."
    return
  fi

  echo "Running flake8 on: ${lint_targets[*]}"
  flake8 "${lint_targets[@]}" --max-line-length=120

  echo "Running black check on: ${lint_targets[*]}"
  black --check "${lint_targets[@]}"
}

run_tests() {
  if [[ ! -d tests ]]; then
    echo "No tests directory found. Skipping pytest."
    return
  fi

  local cov_args=()
  [[ -d api ]] && cov_args+=(--cov=api)
  [[ -d ml ]] && cov_args+=(--cov=ml)

  echo "Running pytest..."
  pytest tests/ -v "${cov_args[@]}" --cov-report=xml
}

run_protolint() {
  if [[ ! -d schemas ]]; then
    echo "No schemas directory found. Skipping protolint."
    return
  fi

  if ! command -v protolint >/dev/null 2>&1; then
    echo "protolint is not installed. Skipping proto lint."
    return
  fi

  echo "Running protolint..."
  protolint lint schemas/
}

case "${MODE}" in
  lint)
    run_lint
    run_protolint
    ;;
  test)
    run_tests
    ;;
  all)
    run_lint
    run_tests
    run_protolint
    ;;
  *)
    echo "Unknown mode '${MODE}'. Use: lint, test, or all."
    exit 1
    ;;
esac

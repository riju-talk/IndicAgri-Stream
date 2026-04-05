#!/usr/bin/env bash
set -euo pipefail

BASE_BRANCH="${1:-main}"

if [[ ! -d schemas ]]; then
  echo "No schemas directory found. Skipping schema validation."
  exit 0
fi

if ! command -v buf >/dev/null 2>&1; then
  echo "buf is required for schema validation."
  exit 1
fi

echo "Running buf lint..."
buf lint schemas/

if ! git rev-parse --verify "${BASE_BRANCH}" >/dev/null 2>&1; then
  git fetch origin "${BASE_BRANCH}:${BASE_BRANCH}" --depth=1 >/dev/null 2>&1 || true
fi

if git rev-parse --verify "${BASE_BRANCH}" >/dev/null 2>&1; then
  echo "Running buf breaking check against ${BASE_BRANCH}..."
  buf breaking schemas/ --against ".git#branch=${BASE_BRANCH}"
else
  echo "Base branch '${BASE_BRANCH}' not available locally. Skipping buf breaking check."
fi

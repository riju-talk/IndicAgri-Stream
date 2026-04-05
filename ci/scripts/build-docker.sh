#!/usr/bin/env bash
set -euo pipefail

SERVICE="${1:-}"
IMAGE_TAG="${2:-latest}"
REF_TAG="${3:-}"
PUSH_IMAGE="${PUSH_IMAGE:-false}"
REGISTRY="${REGISTRY:-ghcr.io/local/indicagri-stream}"

if [[ -z "${SERVICE}" ]]; then
  echo "Usage: $0 <api|flink|airflow> [image-tag] [ref-tag]"
  exit 1
fi

DOCKERFILE="ci/docker/Dockerfile.${SERVICE}"

if [[ ! -f "${DOCKERFILE}" ]]; then
  echo "Dockerfile not found: ${DOCKERFILE}"
  exit 1
fi

case "${SERVICE}" in
  api)
    if [[ ! -d api ]]; then
      echo "api/ directory not found. Skipping API image build."
      exit 0
    fi
    ;;
  flink)
    if [[ ! -d flink ]]; then
      echo "flink/ directory not found. Skipping Flink image build."
      exit 0
    fi
    if [[ ! -f scripts/flink-entrypoint.sh ]]; then
      echo "scripts/flink-entrypoint.sh not found. Skipping Flink image build."
      exit 0
    fi
    ;;
  airflow)
    if [[ ! -d dags ]]; then
      echo "dags/ directory not found. Skipping Airflow image build."
      exit 0
    fi
    ;;
  *)
    echo "Unknown service '${SERVICE}'. Expected one of: api, flink, airflow."
    exit 1
    ;;
esac

IMAGE_NAME="${REGISTRY}/${SERVICE}:${IMAGE_TAG}"
echo "Building ${IMAGE_NAME} from ${DOCKERFILE}..."
docker build -f "${DOCKERFILE}" -t "${IMAGE_NAME}" .

if [[ "${PUSH_IMAGE}" == "true" ]]; then
  echo "Pushing ${IMAGE_NAME}..."
  docker push "${IMAGE_NAME}"

  if [[ -n "${REF_TAG}" ]]; then
    SAFE_REF_TAG="${REF_TAG//\//-}"
    REF_IMAGE_NAME="${REGISTRY}/${SERVICE}:${SAFE_REF_TAG}"
    docker tag "${IMAGE_NAME}" "${REF_IMAGE_NAME}"
    docker push "${REF_IMAGE_NAME}"
  fi
else
  echo "PUSH_IMAGE=false; build complete, push skipped."
fi

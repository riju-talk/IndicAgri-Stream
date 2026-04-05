SHELL := /usr/bin/env bash
.RECIPEPREFIX := >

.PHONY: lint test schema build build-api build-flink build-airflow iac ge ci

lint:
>bash ci/scripts/run-tests.sh lint

test:
>bash ci/scripts/run-tests.sh test

schema:
>bash ci/scripts/validate-schemas.sh main

build-api:
>PUSH_IMAGE=false REGISTRY=local/indicagri-stream bash ci/scripts/build-docker.sh api latest local

build-flink:
>PUSH_IMAGE=false REGISTRY=local/indicagri-stream bash ci/scripts/build-docker.sh flink latest local

build-airflow:
>PUSH_IMAGE=false REGISTRY=local/indicagri-stream bash ci/scripts/build-docker.sh airflow latest local

build: build-api build-flink build-airflow

iac:
>if [ -z "$$(find infra/terraform -name '*.tf' -print -quit)" ]; then echo "No Terraform files found in infra/terraform."; else cd infra/terraform && terraform init && terraform plan; fi

ge:
>ge checkpoint run agri_quality_suite --details

ci: lint test schema

# Pull Request Checklist

## Summary

- What changed:
- Why this change is needed:
- Rollout impact:

## Architecture Validation

- [ ] Kafka topic partitioning strategy documented (`plot_id` or compound key)
- [ ] Schema version bump matches registry compatibility rule
- [ ] Idempotent sink logic updated if data model changed
- [ ] DLQ routing added for new event types
- [ ] Terraform state tagged with `env=staging/prod`
- [ ] Grafana dashboard IDs updated in README if new metrics added
- [ ] Backpressure/failover runbook updated if pipeline topology changed

## Testing Evidence

- [ ] `make lint`
- [ ] `make test`
- [ ] `make schema`
- [ ] `make iac` (if infra changed)
- [ ] `make ge` (if data pipeline changed)

## Deployment and Rollback

- [ ] Runbook link added or updated:
- [ ] Rollback steps validated for affected services
- [ ] Required environment approvals confirmed (staging/prod)

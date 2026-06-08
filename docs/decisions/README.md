# Architecture Decisions

Record durable repository architecture and governance decisions here.

Current refactor decisions:

- Prefer additive reusable modules over broad moves that could break evidence
  references.
- Keep one-off evidence-producing runners in `scripts/research/`.
- Centralize strict split, claim, artifact, metrics, baseline, forecast-panel,
  selector, feature-builder, path, and serialization behavior under `src/`.
- Preserve both `config/` and `configs/` until a dedicated migration can update
  every import and runtime reference.

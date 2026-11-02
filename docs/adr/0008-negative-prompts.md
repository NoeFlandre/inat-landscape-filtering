# ADR 0008: Negative prompts

Status: accepted (2026-11-02)

## Context

In the pilot, many photos above the thresholds show leaves or single plants. Diego Marcos proposed negative prompts: prompts that describe what we do not want, subtracted from the landscape prompts.

## Decision

- Add `score_contrast`: the cosine between the image and the unit direction "mean landscape prompt − mean negative prompt".
- Negative prompts: a single plant, leaf, flower or animal; a natural specimen; animal tracks, droppings, bones or a skull. The animal prompts prepare the next round with mammals and birds.
- Keep `score_landscape`. `publish --score` selects the score that defines the configs.

## Consequences

- `score_contrast` is not a probability. It is a cosine in [-1, 1], so its thresholds are different. A visual check must set them.
- The change needs only the `score` step. It does not need new embeddings.

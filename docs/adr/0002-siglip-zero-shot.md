# ADR 0002: Zero-shot scores with SigLIP2

Status: accepted (2026-11-02)

## Context

We want to find landscape photos without labelled data first.
CLIP was trained with a softmax over a batch. Its raw similarity has no absolute meaning. A softmax score depends on the other prompts in the list.
SigLIP was trained with a sigmoid on each image–text pair. `sigmoid(scale × cosine + bias)` gives a score for one prompt alone.

## Decision

- Use `google/siglip2-base-patch16-224` (Apache-2.0).
- Main score: the sigmoid score of the mean text embedding of three landscape prompts.
- Also store the score of each landscape prompt and a close-up score.
- Store the image embeddings. Calculate the scores from the stored embeddings.

## Consequences

- A change of prompts needs only the `score` step. It does not need new embeddings.
- The scores are not calibrated. The thresholds come from the score distribution and from a visual check.
- A linear probe on the stored embeddings can replace the zero-shot score later. It needs labels.

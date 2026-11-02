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

## Pilot result (10 000 photos)

- The scores are small. The median is less than 0.0001. The 99th percentile is 0.0012. The maximum is 0.0135.
- The model output is correct. The Hugging Face example ("2 cats") gives 0.17 with the same code. SigLIP sigmoid scores are low for short generic prompts.
- The order of the scores is useful. A visual check of samples gave:
  - `>= 0.002` (63 photos): almost all are scene views.
  - `0.001` to `0.002` (63 photos): most are scene views. Some are vegetation close-ups.
  - `0.0005` to `0.001` (89 photos): about half are scene views.
  - `< 0.0005`: most are close views of plants or ground.
- The published thresholds are 0.0005, 0.001 and 0.002.

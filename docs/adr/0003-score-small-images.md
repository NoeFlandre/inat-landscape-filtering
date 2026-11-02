# ADR 0003: Score the 240 px photos

Status: accepted (2026-11-02)

## Context

The model input is 224 × 224 px. The bucket has each photo at 75, 240, 500, 1024 and 2048 px (longest side).
A 240 px file is about 20–40 KB. A 1024 px file is about 200–560 KB.
At scale, the download and the JPEG decode cost more than the model.

## Decision

- Score the `small` (240 px) photo.
- Download the `medium` (500 px) photo only for photos that the dataset keeps with the image.

## Consequences

- The download is about 10 times smaller than with `large`.
- For a 4:3 photo, the short side is 180 px. The model input is a small upscale of it. We accept this. We did not measure the effect.

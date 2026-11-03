# ADR 0010: Images in all threshold configs, uploaded in parts

Status: accepted (2026-11-03)

## Context

The threshold configs are nested: each photo of a strict config is also in the lower configs. Each config has its own files, so an image is stored once per config that keeps it. The machine has about 12 GB of free disk. A full build of the three configs with images needs more than that.

## Decision

- Each threshold config keeps the 500 px image (the user chose this option).
- During the run, `publish --no-images` refreshes the metadata, the figures and the card every hour. It does not download images.
- At the end, `publish` (with images) deletes the threshold configs on the hub. Then it reads the lowest config in parts of 2,000 photos. For each part, it downloads the images once, writes one file for each config, uploads the files in one commit and deletes them.
- A photo without a 500 px image is not in the threshold configs.
- There is no local image cache.

## Consequences

- The disk use of a build is one part (about 300 MB), not the full set.
- The hub stores each image up to three times. The final size is about 20 GB.
- During the final build, the threshold configs on the hub are not complete.

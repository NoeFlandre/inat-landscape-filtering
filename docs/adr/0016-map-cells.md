# ADR 0016: The maps show H3 cells of resolution 4 on the Equal Earth projection

Status: accepted (2026-11-05)

## Context

The maps used hexagons of 2° (and of 0.25° on a fourth map) on a longitude and latitude grid. That grid makes the land near the poles too large. A hexagon of degrees has no relation to the H3 cells of the data, and its area changes with the latitude.

## Decision

- All the maps use the Equal Earth projection. It keeps the areas. The code has the formula (`domain/projection.py`): there is no new dependency.
- The data stays at H3 resolution 5 (about 250 km²).
- Each map shows the H3 cells of resolution 4 (about 1,800 km²). A cell of resolution 4 has the mean of the values of its cells of resolution 5 with a value: each place counts once, also if it has many photos.
- Light grey is a land cell of resolution 4 with no value. White is sea.
- A cell over the 180° meridian is drawn on the two sides of the map. The frame of the map cuts it.
- The maps are 400 dpi: a cell is a few pixels wide. Thus the fourth map (the colour map in detail) is not necessary, and it is removed.

## Consequences

- The maps are about 5000 px wide.
- The area of a colour on a map is the area of the land it shows.

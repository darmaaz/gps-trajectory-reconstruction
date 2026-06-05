"""Rendering primitives for the demonstration notebook and scripts.

The package exposes a small set of matplotlib helpers tuned for the
trajectory-reconstruction story: a clean white-background axes, a
light-gray road network backdrop, and overlays for candidate paths,
ground-truth observations, and position-at-time predictions.

Design choices:

- **No tile basemap.** All maps are pure data on white, with the road
  network rendered as a faint LineCollection. The visual budget goes to
  the data, not the basemap.
- **Distinct categorical colors for candidate paths.** Posterior weight
  is shown via linewidth and a labelled legend rather than alpha, so the
  paths remain visually identifiable even at low weights.
- **Equal aspect, no axis ticks.** Maps look like maps, not scatter plots.
"""

from .primitives import (
    clean_map_axes,
    draw_candidate_paths,
    draw_network_backdrop,
    draw_observations,
    draw_path_edges,
    draw_position_predictions,
)

__all__ = [
    "clean_map_axes",
    "draw_candidate_paths",
    "draw_network_backdrop",
    "draw_observations",
    "draw_path_edges",
    "draw_position_predictions",
]

"""PCM_aCAM instrument layer: tree-to-CAM conversion, PCM analog CAM model
(threshold mapping, programming noise, drift, read noise, drift tracking),
matching and voting; ``cell.py`` adds the 6T2M bound equation with transistor
mismatch and global drift-compensation nodes (used from E0.2 on).

Experiment scripts live in ``experiments/``; this package carries no
experiment numbers. ``pcm_model.py`` and ``io_utils.py`` are the same as in
PCM_IRDrift (AIHWKit PCM model transcription and run-stamp helpers).
"""

__version__ = "0.1.0"

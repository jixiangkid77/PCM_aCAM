# Time to an rms budget versus the budget (post hoc, from e54_curves.csv of E5_20261024_0210)

Lower bounds; rms over all bounds without the bound failures (|e| > 15 LSB), as in E5.

## 6T2M cell, L6T, with the PCM noise model

| design | budget (LSB) | rms at t0 | input remap | divider supply | ideal scaling | supply / remap |
|---|---|---|---|---|---|---|
| P1 | 1.00 | 0.769 | 3.84 min | 8.84 min | 10 min | 2.30 |
| P1 | 1.25 | 0.769 | 13.3 h | 1.45 d | 2.85 d | 2.61 |
| P1 | 1.50 | 0.769 | 23.4 d | 89.6 d | 240 d | 3.82 |
| P1 | 1.75 | 0.769 | 1.63 y | > 10 y | > 10 y | - |
| P1 | 2.00 | 0.769 | > 10 y | > 10 y | > 10 y | - |
| P1 | 2.50 | 0.769 | > 10 y | > 10 y | > 10 y | - |
| P3 | 1.00 | 0.661 | 2.53 h | 1.34 h | 2.46 h | 0.53 |
| P3 | 1.25 | 0.661 | 17.2 d | 5.73 d | 17 d | 0.33 |
| P3 | 1.50 | 0.661 | 3 y | 268 d | 3.01 y | 0.25 |
| P3 | 1.75 | 0.661 | > 10 y | > 10 y | > 10 y | - |
| P3 | 2.00 | 0.661 | > 10 y | > 10 y | > 10 y | - |
| P3 | 2.50 | 0.661 | > 10 y | > 10 y | > 10 y | - |

## GS cell, LBUF, with the PCM noise model

| design | budget (LSB) | rms at t0 | input remap | divider supply | ideal scaling | supply / remap |
|---|---|---|---|---|---|---|
| P1 | 1.00 | 0.738 | 10.1 min | 19.7 min | 20.7 min | 1.95 |
| P1 | 1.25 | 0.738 | 19.1 h | 3.5 d | 4.28 d | 4.38 |
| P1 | 1.50 | 0.738 | 30.5 d | 243 d | 313 d | 7.96 |
| P1 | 1.75 | 0.738 | 2.01 y | > 10 y | > 10 y | - |
| P1 | 2.00 | 0.738 | > 10 y | > 10 y | > 10 y | - |
| P1 | 2.50 | 0.738 | > 10 y | > 10 y | > 10 y | - |
| P3 | 1.00 | 0.655 | 2.81 h | 2.37 h | 2.84 h | 0.84 |
| P3 | 1.25 | 0.655 | 18.2 d | 14 d | 18.6 d | 0.77 |
| P3 | 1.50 | 0.655 | 3.14 y | 2.28 y | 3.29 y | 0.73 |
| P3 | 1.75 | 0.655 | > 10 y | > 10 y | > 10 y | - |
| P3 | 2.00 | 0.655 | > 10 y | > 10 y | > 10 y | - |
| P3 | 2.50 | 0.655 | > 10 y | > 10 y | > 10 y | - |

## 6T2M cell, L6T, without PCM noise (uniform drift)

| design | budget (LSB) | rms at t0 | input remap | divider supply | ideal scaling | supply / remap |
|---|---|---|---|---|---|---|
| P1 | 1.00 | 0.376 | > 10 y | > 10 y | > 10 y | - |
| P1 | 1.25 | 0.376 | > 10 y | > 10 y | > 10 y | - |
| P1 | 1.50 | 0.376 | > 10 y | > 10 y | > 10 y | - |
| P1 | 1.75 | 0.376 | > 10 y | > 10 y | > 10 y | - |
| P1 | 2.00 | 0.376 | > 10 y | > 10 y | > 10 y | - |
| P1 | 2.50 | 0.376 | > 10 y | > 10 y | > 10 y | - |
| P3 | 1.00 | 0.086 | > 10 y | > 10 y | > 10 y | - |
| P3 | 1.25 | 0.086 | > 10 y | > 10 y | > 10 y | - |
| P3 | 1.50 | 0.086 | > 10 y | > 10 y | > 10 y | - |
| P3 | 1.75 | 0.086 | > 10 y | > 10 y | > 10 y | - |
| P3 | 2.00 | 0.086 | > 10 y | > 10 y | > 10 y | - |
| P3 | 2.50 | 0.086 | > 10 y | > 10 y | > 10 y | - |

## GS cell, LBUF, without PCM noise (uniform drift)

| design | budget (LSB) | rms at t0 | input remap | divider supply | ideal scaling | supply / remap |
|---|---|---|---|---|---|---|
| P1 | 1.00 | 0.308 | > 10 y | > 10 y | > 10 y | - |
| P1 | 1.25 | 0.308 | > 10 y | > 10 y | > 10 y | - |
| P1 | 1.50 | 0.308 | > 10 y | > 10 y | > 10 y | - |
| P1 | 1.75 | 0.308 | > 10 y | > 10 y | > 10 y | - |
| P1 | 2.00 | 0.308 | > 10 y | > 10 y | > 10 y | - |
| P1 | 2.50 | 0.308 | > 10 y | > 10 y | > 10 y | - |
| P3 | 1.00 | 0.014 | > 10 y | > 10 y | > 10 y | - |
| P3 | 1.25 | 0.014 | > 10 y | > 10 y | > 10 y | - |
| P3 | 1.50 | 0.014 | > 10 y | > 10 y | > 10 y | - |
| P3 | 1.75 | 0.014 | > 10 y | > 10 y | > 10 y | - |
| P3 | 2.00 | 0.014 | > 10 y | > 10 y | > 10 y | - |
| P3 | 2.50 | 0.014 | > 10 y | > 10 y | > 10 y | - |

# Reprogramming interval of the array (earlier of the lower-bound and upper-bound branch type)

## 6T2M cell, with the PCM noise model

| design | budget (LSB) | input remap | divider supply | ideal scaling | better node, factor |
|---|---|---|---|---|---|
| P1 | 1.00 | 3.27 min (upper) | 8.55 min (upper) | 8.74 min (upper) | supply, 2.62 |
| P1 | 1.25 | 11.3 h (upper) | 1.45 d (lower) | 2.77 d (upper) | supply, 3.09 |
| P1 | 1.50 | 18.9 d (upper) | 89.6 d (lower) | 228 d (upper) | supply, 4.75 |
| P1 | 1.75 | 1.28 y (upper) | > 10 y (lower) | > 10 y (lower) | - |
| P1 | 2.00 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P1 | 2.50 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P3 | 1.00 | 2.38 h (upper) | 1.34 h (lower) | 2.46 h (lower) | remap, 1.78 |
| P3 | 1.25 | 13.8 d (upper) | 5.73 d (lower) | 17 d (lower) | remap, 2.41 |
| P3 | 1.50 | 2.21 y (upper) | 268 d (lower) | 3.01 y (lower) | remap, 3.00 |
| P3 | 1.75 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P3 | 2.00 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P3 | 2.50 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |

## GS cell, with the PCM noise model

| design | budget (LSB) | input remap | divider supply | ideal scaling | better node, factor |
|---|---|---|---|---|---|
| P1 | 1.00 | 3.76 min (upper) | 9.55 min (upper) | 9.98 min (upper) | supply, 2.54 |
| P1 | 1.25 | 11.4 h (upper) | 2.24 d (upper) | 2.72 d (upper) | supply, 4.71 |
| P1 | 1.50 | 18 d (upper) | 164 d (upper) | 215 d (upper) | supply, 9.09 |
| P1 | 1.75 | 1.14 y (upper) | > 10 y (lower) | > 10 y (lower) | - |
| P1 | 2.00 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P1 | 2.50 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P3 | 1.00 | 2.13 h (upper) | 1.8 h (upper) | 2.16 h (upper) | remap, 1.18 |
| P3 | 1.25 | 13.5 d (upper) | 10.5 d (upper) | 14 d (upper) | remap, 1.29 |
| P3 | 1.50 | 2.25 y (upper) | 1.63 y (upper) | 2.38 y (upper) | remap, 1.38 |
| P3 | 1.75 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P3 | 2.00 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P3 | 2.50 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |

## 6T2M cell, without PCM noise (uniform drift)

| design | budget (LSB) | input remap | divider supply | ideal scaling | better node, factor |
|---|---|---|---|---|---|
| P1 | 1.00 | 8.1 y (upper) | > 10 y (lower) | > 10 y (lower) | - |
| P1 | 1.25 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P1 | 1.50 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P1 | 1.75 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P1 | 2.00 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P1 | 2.50 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P3 | 1.00 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P3 | 1.25 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P3 | 1.50 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P3 | 1.75 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P3 | 2.00 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P3 | 2.50 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |

## GS cell, without PCM noise (uniform drift)

| design | budget (LSB) | input remap | divider supply | ideal scaling | better node, factor |
|---|---|---|---|---|---|
| P1 | 1.00 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P1 | 1.25 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P1 | 1.50 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P1 | 1.75 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P1 | 2.00 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P1 | 2.50 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P3 | 1.00 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P3 | 1.25 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P3 | 1.50 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P3 | 1.75 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P3 | 2.00 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |
| P3 | 2.50 | > 10 y (lower) | > 10 y (lower) | > 10 y (lower) | - |


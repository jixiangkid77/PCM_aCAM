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


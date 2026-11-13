# Bootstrap 95 % intervals of the Monte Carlo rms errors (one year) and pull-down offset spread (post hoc)

## E1

| condition | node | rms (LSB) | 95 % interval | relative | prediction | in interval |
|---|---|---|---|---|---|---|
| P1 | dac_remap | 0.726 | [0.674, 0.775] | -7.1 / +6.9 % | - | - |
| P1 | supply | 0.523 | [0.481, 0.567] | -8.0 / +8.3 % | - | - |
| P2 | dac_remap | 0.324 | [0.299, 0.349] | -7.8 / +7.9 % | - | - |
| P2 | supply | 0.538 | [0.498, 0.577] | -7.5 / +7.2 % | - | - |
| P3 | dac_remap | 0.120 | [0.111, 0.129] | -7.5 / +7.9 % | - | - |
| P3 | supply | 0.541 | [0.500, 0.581] | -7.6 / +7.4 % | - | - |
| P4 | dac_remap | 0.150 | [0.137, 0.163] | -8.9 / +9.1 % | - | - |
| P4 | supply | 0.534 | [0.494, 0.573] | -7.5 / +7.3 % | - | - |
- dac_remap: relative half-width 7.0-9.0 %
- supply: relative half-width 7.4-8.2 %

## E2cal

| condition | node | rms (LSB) | 95 % interval | relative | prediction | in interval |
|---|---|---|---|---|---|---|
| P1 | dac_remap | 0.813 | [0.751, 0.873] | -7.6 / +7.4 % | 0.889 | False |
| P1 | supply | 0.597 | [0.537, 0.656] | -10.0 / +10.0 % | 0.615 | True |
| P2 | dac_remap | 0.329 | [0.303, 0.354] | -7.9 / +7.9 % | 0.339 | True |
| P2 | supply | 0.588 | [0.534, 0.644] | -9.3 / +9.5 % | 0.615 | True |
| P3 | dac_remap | 0.121 | [0.112, 0.130] | -7.5 / +7.5 % | 0.120 | True |
| P3 | supply | 0.584 | [0.533, 0.638] | -8.8 / +9.1 % | 0.615 | True |
| S1 | dac_remap | 0.757 | [0.694, 0.819] | -8.2 / +8.3 % | 0.805 | True |
| S1 | supply | 0.291 | [0.265, 0.316] | -8.9 / +8.7 % | 0.300 | True |
| S2 | dac_remap | 0.212 | [0.195, 0.229] | -7.7 / +8.0 % | 0.219 | True |
| S2 | supply | 0.289 | [0.266, 0.313] | -8.1 / +8.2 % | 0.300 | True |
| S4 | dac_remap | 0.115 | [0.107, 0.124] | -7.0 / +7.1 % | 0.116 | True |
| S4 | supply | 0.295 | [0.271, 0.318] | -8.1 / +7.9 % | 0.306 | True |
- dac_remap: relative half-width 7.1-8.3 %
- supply: relative half-width 8.0-10.0 %

## E2

| condition | node | rms (LSB) | 95 % interval | relative | prediction | in interval |
|---|---|---|---|---|---|---|
| C1 | dac_remap | 0.762 | [0.689, 0.835] | -9.6 / +9.5 % | 0.712 | True |
| C1 | supply | 0.578 | [0.525, 0.633] | -9.2 / +9.4 % | 0.617 | True |
| C2 | dac_remap | 0.972 | [0.903, 1.039] | -7.1 / +6.9 % | 1.006 | True |
| C2 | supply | 0.582 | [0.525, 0.639] | -9.8 / +9.9 % | 0.608 | True |
| C3 | dac_remap | 0.094 | [0.087, 0.102] | -7.4 / +7.7 % | 0.096 | True |
| C3 | supply | 0.582 | [0.532, 0.632] | -8.6 / +8.6 % | 0.617 | True |
| C4 | dac_remap | 0.174 | [0.160, 0.189] | -8.4 / +8.3 % | 0.173 | True |
| C4 | supply | 0.574 | [0.525, 0.623] | -8.5 / +8.6 % | 0.608 | True |
| G1 | dac_remap | 0.371 | [0.340, 0.402] | -8.4 / +8.2 % | 0.359 | True |
| G1 | supply | 0.582 | [0.530, 0.635] | -9.0 / +9.0 % | 0.615 | True |
| G2 | dac_remap | 0.128 | [0.117, 0.139] | -8.8 / +8.9 % | 0.125 | True |
| G2 | supply | 0.582 | [0.530, 0.634] | -9.0 / +9.0 % | 0.615 | True |
| G3 | dac_remap | 0.110 | [0.101, 0.119] | -8.2 / +8.1 % | 0.108 | True |
| G3 | supply | 0.194 | [0.178, 0.210] | -8.1 / +8.1 % | 0.203 | True |
| G4 | dac_remap | 0.038 | [0.035, 0.042] | -8.4 / +8.6 % | 0.038 | True |
| G4 | supply | 0.582 | [0.531, 0.634] | -8.8 / +8.9 % | 0.615 | True |
| G5 | dac_remap | 0.620 | [0.569, 0.673] | -8.3 / +8.4 % | 0.603 | True |
| G5 | supply | 0.799 | [0.718, 0.884] | -10.2 / +10.5 % | 0.844 | True |
| G6 | dac_remap | 0.073 | [0.067, 0.080] | -8.7 / +8.5 % | 0.074 | True |
| G6 | supply | 0.389 | [0.356, 0.421] | -8.4 / +8.2 % | 0.410 | True |
| T1 | dac_remap | 0.796 | [0.711, 0.883] | -10.8 / +10.9 % | 0.698 | False |
| T1 | supply | 0.582 | [0.527, 0.636] | -9.4 / +9.3 % | 0.622 | True |
| T2 | dac_remap | 0.853 | [0.789, 0.914] | -7.4 / +7.3 % | 0.869 | True |
| T2 | supply | 0.580 | [0.525, 0.636] | -9.5 / +9.7 % | 0.607 | True |
| T3 | dac_remap | 0.105 | [0.096, 0.112] | -7.7 / +7.5 % | 0.110 | True |
| T3 | supply | 0.589 | [0.536, 0.641] | -8.9 / +8.9 % | 0.622 | True |
| T4 | dac_remap | 0.177 | [0.162, 0.192] | -8.3 / +8.2 % | 0.178 | True |
| T4 | supply | 0.572 | [0.522, 0.620] | -8.7 / +8.4 % | 0.607 | True |
- dac_remap: relative half-width 7.0-10.8 %
- supply: relative half-width 8.1-10.4 %

## E3a_linear

| condition | node | rms (LSB) | 95 % interval | relative | prediction | in interval |
|---|---|---|---|---|---|---|
| G1 | dac_remap | 0.366 | [0.335, 0.398] | -8.5 / +8.6 % | - | - |
| G1 | supply | 0.719 | [0.637, 0.803] | -11.4 / +11.7 % | - | - |
| P1 | dac_remap | 0.887 | [0.812, 0.964] | -8.4 / +8.7 % | - | - |
| P1 | supply | 0.722 | [0.634, 0.809] | -12.2 / +12.0 % | - | - |
| P3 | dac_remap | 0.116 | [0.107, 0.125] | -7.9 / +7.5 % | - | - |
| P3 | supply | 0.709 | [0.632, 0.786] | -10.8 / +10.9 % | - | - |
| S1 | dac_remap | 0.854 | [0.770, 0.939] | -9.8 / +10.0 % | - | - |
| S1 | supply | 0.334 | [0.303, 0.366] | -9.5 / +9.4 % | - | - |
- dac_remap: relative half-width 7.7-9.9 %
- supply: relative half-width 9.5-12.1 %

## E3b

| condition | node | rms (LSB) | 95 % interval | relative | prediction | in interval |
|---|---|---|---|---|---|---|
| U1 | dac_remap | 0.849 | [0.785, 0.912] | -7.6 / +7.3 % | 0.899 | True |
| U1 | supply | 0.287 | [0.262, 0.311] | -8.7 / +8.6 % | 0.293 | True |
| U2 | dac_remap | 0.342 | [0.316, 0.367] | -7.4 / +7.5 % | 0.358 | True |
| U2 | supply | 0.288 | [0.264, 0.311] | -8.2 / +8.1 % | 0.293 | True |
| U3 | dac_remap | 0.104 | [0.095, 0.112] | -8.1 / +7.7 % | 0.103 | True |
| U3 | supply | 0.288 | [0.265, 0.311] | -8.1 / +8.1 % | 0.293 | True |
| U4 | dac_remap | 0.896 | [0.832, 0.960] | -7.2 / +7.1 % | 0.965 | False |
| U4 | supply | 0.502 | [0.460, 0.543] | -8.4 / +8.2 % | 0.478 | True |
| U5 | dac_remap | 0.098 | [0.091, 0.106] | -7.7 / +7.6 % | 0.097 | True |
| U5 | supply | 0.506 | [0.463, 0.547] | -8.4 / +8.2 % | 0.478 | True |
- dac_remap: relative half-width 7.2-7.9 %
- supply: relative half-width 8.1-8.6 %

## Pull-down offsets (standard deviation of the switching-point offset in each draw)

| experiment | condition | branch | sigma (mV) |
|---|---|---|---|
| E1 | P1 | lower | 22.15 |
| E1 | P2 | lower | 22.15 |
| E1 | P3 | lower | 22.15 |
| E1 | P4 | lower | 22.15 |
| E2 | C1 | lower | 21.91 |
| E2 | C2 | lower | 21.65 |
| E2 | C3 | lower | 21.91 |
| E2 | C4 | lower | 21.65 |
| E2 | G1 | lower | 21.86 |
| E2 | G2 | lower | 21.86 |
| E2 | G3 | lower | 11.14 |
| E2 | G4 | lower | 21.86 |
| E2 | G5 | lower | 21.86 |
| E2 | G6 | lower | 11.14 |
| E2 | T1 | lower | 22.12 |
| E2 | T2 | lower | 21.60 |
| E2 | T3 | lower | 22.12 |
| E2 | T4 | lower | 21.60 |
| E2cal | P1 | lower | 22.15 |
| E2cal | P2 | lower | 22.15 |
| E2cal | P3 | lower | 22.15 |
| E2cal | S1 | lower | 22.15 |
| E2cal | S2 | lower | 22.15 |
| E2cal | S4 | lower | 11.30 |
| E3a | G1 | lower | 24.88 |
| E3a | P1 | lower | 24.88 |
| E3a | P3 | lower | 24.88 |
| E3a | S1 | lower | 24.88 |
| E3b | U1 | upper (trip point) | 10.57 |
| E3b | U2 | upper (trip point) | 10.57 |
| E3b | U3 | upper (trip point) | 10.57 |
| E3b | U4 | upper (trip point) | 17.67 |
| E3b | U5 | upper (trip point) | 17.67 |

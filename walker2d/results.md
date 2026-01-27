# Tables for results - ongoing - Testing on best config found (C) - target torso mass +1.6kg shift
## models are evaluated over 100 test episodes
## Table for config C performances - baseline
| seed | source -> source | source -> target | target -> target |
| --- | --- | --- | --- |
| 42 | 2896.55 +/- 17.35 | 832.70 +/- 771.08 | 2845.33 +/- 14.56 |
| 10 | 2693.26 +/- 13.13 | 1571.88 +/- 207.79 | 2720.77 +/- 7.74 |
| 0 (random) | 2667.78 +/- 27.45 | 2446.55 +/- 236.85 | 2709.77 +/- 14.92 |
### Mean results across the 3 runs
| source -> source | source -> target | target -> target |
| --- | --- | --- |
| 2752.53 +/- 19.31 | 1617.04 +/- 405.24 | 2758.62 +/- 12.41 |
## Table for config C - with UDR 0.3
| seed | source -> source | source -> target |
| --- | --- | --- |
| 42 | 2665.27 +/- 311.07 | 2568.22 +/- 348.99 |
| 10 | 2420.95 +/- 9.48 | 1461.25 +/- 128.11 |
| 0 (random) | 2613.97 +/- 9.88 | 2695.73 +/- 16.80 |
### Mean results across the 3 runs
| source -> source | source -> target |
| --- | --- |
| 2566.73 +/- 110.14 | 2241.73 +/- 164.63 |
## Table for config C - with UDR 0.28
| seed | source -> source | source -> target |
| --- | --- | --- |
| 42 | 2706.81 +/- 15.83 | 1276.49 +/- 281.29 |
| 10 | 2199.77 +/- 13.25 | 1541.17 +/- 326.62 |
| 0 (random) | 2741.50 +/- 5.87 | 2795.16 +/- 272.50 |
### Mean results across the 3 runs
| source -> source | source -> target |
| --- | --- |
| 2549.36 +/- 11.65 | 1870.94 +/- 293.47 |

## Table for config C - with UDR 0.20
| seed | source -> source | source -> target |
| --- | --- | --- |
| 42 | 2237.59 +/- 258.11 | 763.60 +/- 61.09 |
| 10 | 2591.53 +/- 35.45 | 2407.76 +/- 248.34 |
| 0 (random) | 2556.06 +/- 22.53 | 2603.25 +/- 30.69 |

### Mean results across the 3 runs
| source -> source | source -> target |
| --- | --- |
| 2461.73 +/- 105.36 | 1924.87 +/- 125.37 |

### Table for config C - with symmetric UDR 0.30
| seed | source -> source | source -> target |
| --- | --- | --- |
| 42 | 2396.75 +/- 165.43 | 1975.87 +/- 288.04 |
| 10 | 2458.03 +/- 11.11 | 2451.91 +/- 325.88 |
| 0 (random) | 2698.06 +/- 19.23 | 2127.37 +/- 148.15 |
### Mean results across the 3 runs
| source -> source | source -> target |
| --- | --- |
| 2517.61 +/- 65.26 | 2185.05 +/- 254.02 |
### Table for config C - with symmetric UDR 0.50
| seed | source -> source | source -> target |
| --- | --- | --- |
| 42 | 2119.00 +/- 17.26 | 2119.84 +/- 88.82 |
| 10 | 826.50 +/- 11.01 | 771.37 +/- 10.54 |
| 0 (random) | 2083.09 +/- 272.05 | 1238.27 +/- 409.92 |
### Mean results across the 3 runs
| source -> source | source -> target |
| --- | --- |
| 1676.20 +/- 100.11 | 967.86 +/- 509.28 |

--------------------------------------------------------------
### network architecthure results
### UDR 30
### Table for udr30 network small
| seed | source -> source | source -> target |
| --- | --- | --- |
| 42 | 2587.12 +/- 22.80 |  2654.20+/-195.12 |
| 10 |2615.45 +/- 26.84  | 2578.15+/- 164.55 |
| 0 (random) | 2609.67 +/- 22.15 |  2598.70+/- 191.82 |
### Mean results across the 3 runs
| source -> source | source -> target |
| --- | --- |
| 2604.08 +/- 23.93 | 2610.35 +/- 183.83 |

### Table for udr30 network medium
| seed | source -> source | source -> target |
| --- | --- | --- |
| 42 | 2687.93 +/- 9.96  | 2079.44 +/- 363.02 |
| 10 | 2698.89 +/- 11.0  |  2322.06+/-327.86 |
| 0 (random) |  2709.85+/-10.92 |  2564.68+/-36.98|
### Mean results across the 3 runs
| source -> source | source -> target |
| --- | --- |
| 2698.89 +/- 10.96 | 2322.06 +/- 242.62 |

### Table for udr30 network deep
| seed | source -> source | source -> target |
| --- | --- | --- |
| 42 | 795.42+/- 108.45  |  752.30+/- 142.60  |
| 10 |824.15+/- 119.32   |  789.64+/- 165.14  |
| 0 (random) | 813.43+/- 114.80   |  771.60+/- 169.17 |
### Mean results across the 3 runs
| source -> source | source -> target |
| --- | --- |
| 811.00 +/- 114.19 | 771.18 +/- 158.97 |

### Baseline
### Table for baseline network small
| seed | source -> source | source -> target |
| --- | --- | --- |
| 42 | 2265.40+/-14.12 |  1542.80+/-176.40 |
| 10 |2265.40+/-17.55  |  1620.45+/-192.18 |
| 0 (random) | 2265.40+/-15.97 |  1594.12+/-174.05 |
### Mean results across the 3 runs
| source -> source | source -> target |
| --- | --- |
| 2290.37 +/- 15.88 | 1585.79 +/- 182.21 |

### Table for baseline network medium
| seed | source -> source | source -> target |
| --- | --- | --- |
| 42 | 1041.25 +/-214.60  |  510.30+/- 94.10 |
| 10 | 1070.12 +/-235.18  |  518.28+/- 105.75 |
| 0 (random) |  1059.15 +/-231.05  |  502.45+/-98.20|
### Mean results across the 3 runs
| source -> source | source -> target |
| --- | --- |
| 1056.84 +/- 226.81 | 511.01 +/- 99.35 |

### Table for udr30 network deep
| seed | source -> source | source -> target |
| --- | --- | --- |
| 42 | 1042.15+/- 151.25  |  612.48+/- 163.12  |
| 10 |1088.60+/- 158.40  |  641.11+/- 154.98   |
| 0 (random) | 1065.45+/- 161.05  |  625.10+/- 156.95  |
### Mean results across the 3 runs
| source -> source | source -> target |
| --- | --- |
| 1065.40 +/- 156.90 | 626.23 +/- 158.35 |
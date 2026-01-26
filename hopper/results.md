# Tables for results - searching best hyper config - source torso mass -1.0kg shift
## trainings are done over 1M timesteps, tests are done over 50 episodes
### Table for config setup - 1
| hyper | value |
| --- | --- |
| lr | 3e-4 |
| n_steps | 4096 |
| batch_size | 64 |
| n_epochs | 5 |
| ent_coef | 0.0 |
| clip_range | 0.2 |
### Results
| seed | source -> source | source -> target | target -> target |
| --- | --- | --- | --- |
| 42 | 1617.39 +/- 9.57 | 1397.18 +/- 66.38 | ... |
| 10 | 1657.99 +/- 117.63 | 885.95 +/- 356.87 | ... |
| 0 | 1685.91 +/- 3.92 | 1031.63 +/- 37.31 | ... |
| Mean | 1653.76 +/- 43.71 | 1104.92 +/- 153.52 | ... |

### Table for config setup - 2
| hyper | value |
| --- | --- |
| lr | 3e-4 |
| n_steps | 4096 |
| batch_size | 64 |
| n_epochs | 10 |
| ent_coef | 0.0 |
| clip_range | 0.2 |
### Results - not promising
| seed | source -> source | source -> target | target -> target |
| --- | --- | --- | --- |
| 42 | 1601.15 +/- 1.71 | 687.87 +/- 4.86 | ... |
| 10 |  |  | ... |
| 0 |  | ... |
| Mean | 

### Table for config setup - 3
| hyper | value |
| --- | --- |
| lr | 3e-4 |
| n_steps | 4096 |
| batch_size | 64 |
| n_epochs | 10 |
| ent_coef | 0.001 |
| clip_range | 0.2 |
### Results
| seed | source -> source | source -> target | target -> target |
| --- | --- | --- | --- |
| 42 | 1698.70 +/- 1.56 | 748.32 +/- 35.89 | ... |
| 10 | 1710.54 +/- 6.17 | 909.63 +/- 14.71 | ... |
| 0 | 1637.97 +/- 3.29 | 733.97 +/- 20.36 | ... |
| Mean | 1682.40 +/- 3.67 | 797.31 +/- 23.65 | ... |

### Table for config setup - 4
| hyper | value |
| --- | --- |
| lr | 3e-4 |
| gamma | 0.99 |
| n_steps | 8192 |
| batch_size | 64 |
| n_epochs | 10 |
| ent_coef | 0.0 |
| clip_range | 0.2 |

### Results
| seed | source -> source | source -> target | target -> target |
| --- | --- | --- | --- |
| 42 | 1715.95 +/- 4.80 | 1053.64 +/- 14.77 | ... |
| 10 | 1715.90 +/- 5.88 | 877.11 +/- 23.89 | ... |
| 0 | 1629.30 +/- 59.51 | 781.67 +/- 28.46 | ... |
| Mean | 1686.95 +/- 23.40 | 904.14 +/- 22.37 | ... |

### Table for config setup - 5
| hyper | value |
| --- | --- |
| lr | 3e-4 |
| n_steps | 4096 |
| batch_size | 32 |
| n_epochs | 5 |
| ent_coef | 0.0 |
| clip_range | 0.2 |
### Results
| seed | source -> source | source -> target | target -> target |
| --- | --- | --- | --- |
| 42 | 1718.85 +/- 7.99 | 1045.89 +/- 26.11 | ... |
| 10 | 1722.95 +/- 51.91 | 888.22 +/- 10.13 | ... |
| 0 | 1707.49 +/- 26.92 | 905.24 +/- 27.70 | ... |
| Mean | 1714.30 +/- 27.63 | 943.57 +/- 30.82 | ... |

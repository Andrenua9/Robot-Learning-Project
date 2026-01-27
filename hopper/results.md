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
| 0 | 1721.49 +/- 48.06 | 721.63 +/- 7.69 | ... |
| Mean | 1714.30 +/- 27.63 | 943.57 +/- 30.82 | ... |
### Results (1 env loyal)
| seed | source -> source | source -> target | target -> target |
| --- | --- | --- | --- |
| 42 | 1718.20 +/- 7.85 | 1047.69 +/- 31.66 | ... |
| 0 | 1721.49 +/- 48.06 | 721.63 +/- 7.69 | ... |
| 100 | 1636.20 +/- 110.58 | 1001.61 +/- 44.88 | ... |
| Mean | 1691.96 +/- 55.50 | 923.64 +/- 28.08 | ... |

### Table for config 5 UDR 0.15
| seed | source -> source | source -> target |
| --- | --- | --- |
| 42 | 1707.69 +/- 2.52 | 1715.59 +/- 15.05 |
| 10 | 1412.98 +/- 81.85 | 1107.17 +/- 76.87 |
| 0 | 1558.93 +/- 21.12 | 1647.58 +/- 41.61 |
| 100 | 1199.43 +/- 13.26 | 698.50 +/- 3.79 |
| Mean | 1469.76 +/- 29.69 | 1292.21 +/- 34.33 |

### Table for config 5 UDR 0.20
| seed | source -> source | source -> target |
| --- | --- | --- |
| 42 | 1568.57 +/- 159.38 | 754.09 +/- 14.44 |
| 10 | 1724.82 +/- 36.11 | 979.87 +/- 7.40 |
| 0 | 1729.39 +/- 38.02 | 980.46 +/- 6.52 |
|100| 1724.45 +/- 36.39 | 980.43 +/- 6.78 |
| Mean | 1686.81 +/- 67.48 | 923.72 +/- 8.78 |

### Table for config 5 UDR 0.25
| seed | source -> source | source -> target |
| --- | --- | --- |
| 42 | 1627.11 +/- 103.73 | 790.37 +/- 15.95 |
| 10 | 1390.66 +/- 90.86 | 1000.88 +/- 27.78 |
| 0 | 1166.99 +/- 55.43 | 1566.68 +/- 164.54 |
| 100 | 1641.50 +/- 3.92 | 973.77 +/- 10.01 |
| Mean | 1456.57 +/- 63.48 | 1082.93 +/- 54.57 |

### Table for config 5 UDR 0.30
| seed | source -> source | source -> target |
| --- | --- | --- |
| 42 | 952.70 +/- 8.98 | 968.41 +/- 15.24 |
| 10 | 1604.44 +/- 60.28 | 1198.27 +/- 43.76 |
| 0 | 1698.52 +/- 9.07 | 964.31 +/- 8.75 |
| 100 | 1717.50 +/- 28.50 | 905.44 +/- 10.35 |
| Mean | 1493.29 +/- 26.71 | 1009.11 +/- 19.52 |

### Table for config 5 UDR 0.40
| seed | source -> source | source -> target |
| --- | --- | --- |
| 42 | 1661.65 +/- 7.35 | 1155.72 +/- 54.16 |
| 10 | 1661.42 +/- 5.76 | 986.50 +/- 4.00 |
| 0 | 1651.23 +/- 3.58 | 830.78 +/- 6.15 |
| 100 | 1752.01 +/- 64.21 | 1442.68 +/- 46.16 |
| Mean | 1681.58 +/- 20.22 | 1103.92 +/- 27.62 |

### ADR RESULTS
| seed |  source --> source | source -> target |
| --- | --- | --- |
| 20 | 1611.69+/- 1.07 | 1665.54 +/- 1.84 |
| 42 | 1612.42+/- 2.34 | 1604.60+/- 9.99 |

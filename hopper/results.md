# Tables for results - searching best hyper config - source torso mass -1.0kg shift
## trainings are done over 1M timesteps, tests are done over 50 episodes
## n_envs=8
### Table for config setup - 1
| hyper | value |
| --- | --- |
| lr | 3e-4 |
| n_steps | 512 |
| batch_size | 64 |
| n_epochs | 5 |
| ent_coef | 0.0 |
| clip_range | 0.2 |
### Results - in corso
| seed | source -> source | source -> target | target -> target |
| --- | --- | --- | --- |
| 42 | 1617.39 +/- 9.57 | 1397.18 +/- 66.38 | ... |
| 10 | ... | ... | ... |
| 0 | ... | ... | ... |


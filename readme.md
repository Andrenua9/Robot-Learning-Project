# Starting code for final course project extension of Robot Learning - 01HFNOV

Official assignment at [Google Doc](https://docs.google.com/document/d/1XWE2NB-keFvF-EDT_5muoY8gdtZYwJIeo48IMsnr9l0/edit?usp=sharing).

# Project Overview

This project investigates the effectiveness of Domain Randomization (DR) techniques in reducing the reality gap for robotic locomotion tasks in simulated environments.

The study begins with the Hopper environment, comparing:

Uniform Domain Randomization (UDR)

Automatic Domain Randomization (ADR)

In the ADR framework, the randomization range is dynamically adapted based on the agent’s performance, effectively acting as an automatic curriculum learning mechanism that improves training stability.

The analysis is then extended to the more complex Walker2d environment, which introduces additional challenges due to its bipedal structure and higher degrees of freedom.

# Experimental Analysis

On Walker2d, we performed:

- Evaluation of different neural network architectures (Small [64,64], Medium, Deep)

- Robustness testing under systematic torso mass shifts

Comparison between:

- Standard UDR

- Symmetry-Constrained UDR

# Key Findings

- Domain Randomization significantly improves transfer robustness.

- Model capacity plays a critical role in generalization.

Surprisingly, the Small architecture ([64, 64]) outperformed larger networks, eliminating the performance drop in the target environment.

Lower-capacity models appear to introduce a beneficial inductive bias, reducing overfitting to simulation-specific noise.

Symmetry-Constrained UDR did not outperform standard UDR.
Training with asymmetric randomization acted as an effective regularizer and achieved slightly better performance.

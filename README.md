# Learning While Scheduling in Multi-Server Systems With Unknown Statistics: MaxWeight with Discounted UCB

The code is an implementation of the paper
[**Learning While Scheduling in Multi-Server Systems With Unknown Statistics: MaxWeight with Discounted UCB**].

We proposed MaxWeight with discounted UCB algorithm, and we compare the proposed algorithm with 4 baselines,
frame-based MaxWeight [1], DAM.UCB [2], MaxWeight with empirical mean (EM), and MaxWeight with discounted EM.
We consider four settings, stationary, nonstationary aperiodic, nonstationary periodic, and nonstationary periodic with
a larger period.
Please go to the paper for details.

## Getting started

- Install python 3.7

- Install the requirements in "requirements.txt"

## Files in the folder

- "max_weight.py": implementation of MaxWeight with discounted UCB, MaxWeight with EM, and MaxWeight with discounted EM
- "frame_based_max_weight.py": implementation of the frame-based MaxWeight algorithm.
- "dam_ucb_centralized.py": implementation of the DAM.UCB algorithm in the centralized setting.
- "plotfig_10_users_stationary_3_curves.py": plotting the total queue lengths for the MaxWeight with discounted UCB, frame-based MaxWeight, and DAM.UCB algorithms in the stationary setting.
- "plotfig_10_users_all_curves.py": plotting the total queue lengths for all the five algorithms in all four settings.
- "plotfig_10_users_diff_T.py": plotting the total queue length for MaxWeight with discounted UCB algorithm with difference $g(\gamma)$ in all the four settings.
- "plotfig_10_users_diff_frame_size.py": plotting the total queue length for frame-based MaxWeight algorithm with difference frame sizes in all the four settings.

## Simulation commands

To generate the figures in the simulation part or the appendix part of the paper, you need to first run the commands for simulation and then run the corresponding "plotfig_*.py" file for plotting figures.
The output figures will be saved to PDF files in the corresponding folders.
All the commands are presented in the following for different settings.

### Stationary Setting

First run the following commands for the five methods:

| Method                       | Command                                  |
| :--------------------------: | ---------------------------------------- |
|  MaxWeight with discounted UCB  | ```python max_weight.py --setting stationary --discount yes --ucb yes --T 6071209191677812 --simulation_horizon 100000``` |
|  Frame-based MaxWeight  | ```python frame_based_max_weight.py --setting stationary --frame_size 120 --simulation_horizon 100000``` |
|  DAM.UCB  | ```python dam_ucb_centralized.py --setting stationary --epoch_size 161 --simulation_horizon 100000``` |
|  MaxWeight with EM  | ```python max_weight.py --setting stationary --discount no --ucb no --simulation_horizon 100000``` |
|  MaxWeight with discounted EM  | ```python max_weight.py --setting stationary --discount yes --ucb no --T 6071209191677812 --simulation_horizon 100000``` |

Then, run the following command for plotting the comparison figures for the stationary setting.
```bash
python plotfig_10_users_all_curves.py --setting stationary
python plotfig_10_users_stationary_3_curves.py
```


### Nonstationary Aperiodic Setting

First run the following commands for the five methods:

| Method                       | Command                                  |
| :--------------------------: | ---------------------------------------- |
|  MaxWeight with discounted UCB  | ```python max_weight.py --setting aperiodic --discount yes --ucb yes --T 8192 --simulation_horizon 30000``` |
|  Frame-based MaxWeight  | ```python frame_based_max_weight.py --setting aperiodic --frame_size 30 --simulation_horizon 30000``` |
|  DAM.UCB  | ```python dam_ucb_centralized.py --setting aperiodic --epoch_size 113 --simulation_horizon 30000``` |
|  MaxWeight with EM  | ```python max_weight.py --setting aperiodic --discount no --ucb no --simulation_horizon 30000``` |
|  MaxWeight with discounted EM  | ```python max_weight.py --setting aperiodic --discount yes --ucb no --T 8192 --simulation_horizon 30000``` |

Then, run the following command for plotting the comparison figures for the nonstationary aperiodic setting.
```bash
python plotfig_10_users_all_curves.py --setting aperiodic
```

### Nonstationary Periodic Setting

First run the following commands for the five methods:

| Method                       | Command                                  |
| :--------------------------: | ---------------------------------------- |
|  MaxWeight with discounted UCB  | ```python max_weight.py --setting periodic --discount yes --ucb yes --T 8192 --simulation_horizon 4000``` |
|  Frame-based MaxWeight  | ```python frame_based_max_weight.py --setting periodic --frame_size 25 --simulation_horizon 4000``` |
|  DAM.UCB  | ```python dam_ucb_centralized.py --setting periodic --epoch_size 161 --simulation_horizon 4000``` |
|  MaxWeight with EM  | ```python max_weight.py --setting periodic --discount no --ucb no --simulation_horizon 4000``` |
|  MaxWeight with discounted EM  | ```python max_weight.py --setting periodic --discount yes --ucb no --T 8192 --simulation_horizon 4000``` |

Then, run the following command for plotting the comparison figures for the nonstationary aperiodic setting.
```bash
python plotfig_10_users_all_curves.py --setting periodic
```

### Nonstationary Periodic With a Larger Period Setting

First run the following commands for the five methods:

| Method                       | Command                                  |
| :--------------------------: | ---------------------------------------- |
|  MaxWeight with discounted UCB  | ```python max_weight.py --setting periodic_larger_period --discount yes --ucb yes --T 8192 --simulation_horizon 40000``` |
|  Frame-based MaxWeight  | ```python frame_based_max_weight.py --setting periodic_larger_period --frame_size 20 --simulation_horizon 40000``` |
|  DAM.UCB  | ```python dam_ucb_centralized.py --setting periodic_larger_period --epoch_size 161 --simulation_horizon 40000``` |
|  MaxWeight with EM  | ```python max_weight.py --setting periodic_larger_period --discount no --ucb no --simulation_horizon 40000``` |
|  MaxWeight with discounted EM  | ```python max_weight.py --setting periodic_larger_period --discount yes --ucb no --T 8192 --simulation_horizon 40000``` |

Then, run the following command for plotting the comparison figures for the nonstationary aperiodic setting.
```bash
python plotfig_10_users_all_curves.py --setting periodic_larger_period
```

### MaxWeight with discounted UCB algorithm with different $g(\gamma)$

First run the following command for simulating MaxWeight with discounted UCB algorithm with different $g(\gamma)$:
```bash
python max_weight.py --setting * --discount yes --ucb yes --T ** --simulation_horizon 100000
```
where you need to replace "\*" with one of the four different settings, "stationary", "aperiodic", "periodic", or "periodic_larger_period",
and replace "\**" with different $g(\gamma)$ that you want to simulate. 
Before plotting figures, make sure you have run the command with the following different $g(\gamma)$:

| Setting                       | $g(\gamma)$                                  |
| :--------------------------: | ---------------------------------------- |
|  stationary  | 4096, 8192, 33554432, 137438953472 |
|  aperiodic  | 2048, 4096, 8192, 16384 |
|  periodic  | 2048, 4096, 8192, 16384 |
|  periodic_larger_period  | 2048, 4096, 8192, 16384 |


After that, run the following command for plotting figures:
```bash
python plotfig_10_users_diff_T.py --setting *
```
where you need to replace "\*" with one of the four different settings.
 

### Frame-based MaxWeight algorithm with different frame sizes

First run the following command for simulating frame-based MaxWeight algorithm with different frame sizes:
```bash
python frame_based_max_weight.py --setting * --frame_size ** --simulation_horizon 100000
```
where you need to replace "\*" with one of the four different settings, "stationary", "aperiodic", "periodic", or "periodic_larger_period",
and replace "\**" with different frame sizes that you want to simulate. 
Before plotting figures, make sure you have run the command with the following different frame sizes:

| Setting                       | Frame sizes                                  |
| :--------------------------: | ---------------------------------------- |
|  stationary  | 100, 110, 120, 130, 140, 150 |
|  aperiodic  | 15, 20, 25, 30, 35, 40, 45, 50 |
|  periodic  | 15, 20, 25, 30, 35, 40, 45, 50 |
|  periodic_larger_period  | 15, 20, 25, 30, 35, 40, 45, 50 |


After that, run the following command for plotting figures:
```bash
python plotfig_10_users_diff_frame_size.py --setting *
```
where you need to replace "\*" with one of the four different settings.
 

## Results

All the results can be found in "SIMULATION RESULTS" (Section 6) of the main text of the paper
and also in "ADDITIONAL DETAILS OF THE SIMULATIONS" (Section 12) of the supplementary materials. 

## Reference

[1] Stahlbuhk, T., Shrader, B., and Modiano, E. (2019). Learning algorithms for scheduling in wireless networks with unknown channel statistics. Ad Hoc Networks, 85:131–144.

[2] Freund, D., Lykouris, T., and Weng, W. (2022). Efficient decentralized multi-agent learning in asymmetric queuing systems. In Proc. Conf. Learning Theory (COLT), volume 178, pages 4080–4084.

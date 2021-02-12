# RL and Deep-RL implementations

This is a **modular** implementation, meaning: you can plug-and-play **almost any environment** (in the corresponding file, within the same base folder) **with any algorithm**.

Table of contents:

* [Tabular RL Algorithms](https://github.com/EliorBenYosef/reinforcement-learning#tabular-rl-algorithms)
* [Deep RL Algorithms](https://github.com/EliorBenYosef/reinforcement-learning#deep-rl-algorithms) 
* [Results](https://github.com/EliorBenYosef/reinforcement-learning#results) 
* [How to use](https://github.com/EliorBenYosef/reinforcement-learning#how-to-use) 

## Tabular RL Algorithms

**Implemented [Algorithms](../master/reinforcement_learning/tabular_RL/algorithms):**

. | Prediction | Control
--- | --- | ---
**[Monte Carlo](../master/reinforcement_learning/tabular_RL/algorithms/monte_carlo.py) (MC)** | MC policy evaluation | MC non-exploring-starts control, off-policy MC control
**[Temporal Difference 0](../master/reinforcement_learning/tabular_RL/algorithms/td_zero.py) (TD0)** | TD0 policy evaluation | SARSA, Expected SARSA, Q Learning, Double Q Learning

**Implemented [Environments](../master/reinforcement_learning/tabular_RL/envs_dss.py):**

Discrete State Space | Discretized State Space
--- | ---
**Toy Text** - FrozenLake, Taxi, Blackjack | **Classic Control** - MountainCar, CartPole, Acrobot

## Deep RL Algorithms

Most of the cases, you can select the desired library type (`lib_type`) implementation: 
`LIBRARY_TF`, `LIBRARY_TORCH`, `LIBRARY_KERAS`. 

**Implemented Control [Algorithms](../master/reinforcement_learning/deep_RL/algorithms):**

* **[Deep Q Learning](../master/reinforcement_learning/deep_RL/algorithms/deep_q_learning.py) (DQL)** 
* **[Policy Gradient](../master/reinforcement_learning/deep_RL/algorithms/policy_gradient.py) (PG)**
  * set `ep_batch_num = 1` for the **Monte-Carlo PG (REINFORCE)** algorithm
* **[Actor-Critic](../master/reinforcement_learning/deep_RL/algorithms/actor_critic.py) (AC)**
* **[Deep Deterministic Policy Gradient](../master/reinforcement_learning/deep_RL/algorithms/deep_deterministic_policy_gradient.py) (DDPG)**

**Implemented [Environments](../master/reinforcement_learning/deep_RL/envs.py):**

*(environments with **Continuous State Space**)*

. | Discrete Action Space | Continuous Action Space
--- | --- | ---
**Observation Vector Input Type** | CartPole, LunarLander | Pendulum, MountainCarContinuous, LunarLanderContinuous, BipedalWalker
**Stacked Frames Input Type** | Breakout, SpaceInvaders |

### Algorithms restrictions

Note that some some algorithms have restrictions.

* Innate restrictions:

Discrete Action Space | Continuous Action Space
--- | ---
Deep Q Learning | Deep Deterministic Policy Gradient

* Some current restrictions are due to the fact that there's more work to be done (code-wise), 
meaning: writing for **every** -
  * **library implementation** (tensorflow, torch, keras).
  * **input (state) type** (observation vector, stacked frames).
  * **action space type** (discrete, continuous).

### [cmdline_play.py](../master/reinforcement_learning/deep_RL/utils/cmdline_play.py)

Enables **playing from the command-line**.

Running this file performs the algorithm on a single environment, 
through the command-line (using the `argparse` module to parse command-line options). 
The major benefit from this is that it enables concatenating multiple independent runs via `&&`
(so you can run multiple tests in one go).

### [grid_search.py](../master/reinforcement_learning/deep_RL/utils/grid_search.py)

Enables **performing grid search**.

Running this file performs a comparative grid search for a single environment, and plots the results.
This is mostly done for hyper-parameters tuning. Note that currently I added 16 colors 
(more than that will raise an error, so add more colors if you need more than 16 combinations)

Currently, grid search is tuned to DQL, but it's applicable to every algorithm with only minor changes
(the relevant imports are there at the top of the file, just commented out).

## Results

Algorithms Performance Examples.
Training & Test results come in the forms of graphs and statistics (for some of the environments)
of both: running average of episode scores, and accumulated scores.

### Tabular RL Algorithms

#### AI agent before and after training

**Mountain Car**

<p align="left">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/results/tabular_RL/mountain-car_untrained.gif" width="400">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/results/tabular_RL/mountain-car_trained.gif" width="400">
</p>

**Cart Pole**

<p align="left">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/results/tabular_RL/cart-pole_untrained.gif" width="400">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/results/tabular_RL/cart-pole_trained.gif" width="400">
</p>

**Acrobot**

<p align="left">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/results/tabular_RL/acrobot_untrained.gif" width="400">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/results/tabular_RL/acrobot_trained.gif" width="400">
</p>

#### `environment_test()` (in [test_tabular_rl.py](../master/tests/test_tabular_rl.py)) result graphs

<p align="left">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/results/tabular_RL/algorithms_comparison_legend.png" width="200">
</p>

<p align="left">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/results/tabular_RL/cart-pole-v0-score-training.png" width="390">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/results/tabular_RL/cart-pole-v0-accumulated-score-training.png" width="410">
</p>

<p align="left">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/results/tabular_RL/cart-pole-v0-scores-test.png" width="390">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/results/tabular_RL/cart-pole-v0-accumulated-scores-test.png" width="410">
</p>

### Deep RL Algorithms

#### [Grid Search](../master/reinforcement_learning/deep_RL/utils/grid_search.py)

Performance of DQL Grid Search on first & second FC layers' sizes (number of nodes \ neurons):

<p align="left">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/results/deep_RL/cart-pole-v1_dql.png" width="500">
</p>

## How to use

The test files contain examples of how to use:

* [Tabular RL Algorithms](../master/tests/test_tabular_rl.py)
* Deep RL Algorithms:
  * [DQL](../master/tests/test_deep_rl/test_dql.py)
  * [PG](../master/tests/test_deep_rl/test_pg.py)
  * [AC](../master/tests/test_deep_rl/test_ac.py)
  * [DDPG](../master/tests/test_deep_rl/test_ddpg.py)

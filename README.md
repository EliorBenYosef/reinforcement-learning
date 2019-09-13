# RL and Deep-RL implementations

This is a **modular** implementation, meaning: you can plug-and-play **almost any environment** (in the corresponding file, within the same base folder) **with any algorithm**.

Table of contents:

1. [Tabular Methods](https://github.com/EliorBenYosef/reinforcement-learning#1-tabular-methods)
   * [How to use](https://github.com/EliorBenYosef/reinforcement-learning#how-to-use)
   * [Implemented Algorithms](https://github.com/EliorBenYosef/reinforcement-learning#implemented-algorithms-rl_tabularpy)
   * [Implemented Environments](https://github.com/EliorBenYosef/reinforcement-learning#implemented-environments-envs_dsspy)
   * [Algorithms Performance Examples](https://github.com/EliorBenYosef/reinforcement-learning#algorithms-performance-examples)

2. [Deep Reinforcement Learning](https://github.com/EliorBenYosef/reinforcement-learning#2-deep-reinforcement-learning) 
   * [How to use](https://github.com/EliorBenYosef/reinforcement-learning#how-to-use-1)
   * [Implemented Algorithms](https://github.com/EliorBenYosef/reinforcement-learning#implemented-algorithms)
   * [Implemented Environments](https://github.com/EliorBenYosef/reinforcement-learning#implemented-environments-envspy)
   * [Algorithms Performance Examples](https://github.com/EliorBenYosef/reinforcement-learning#algorithms-performance-examples-1)

## 1. Tabular Methods

### How to use

Simply run the testing file ([rl_tabular_testing.py](../master/tabular_methods/rl_tabular_testing.py)).

There are 3 main operations (leave the one you need, comment out the rest):
* `policy_evaluation_algorithms_test()` - performs either Monte Carlo or TD-0 policy evaluation.
* `learning_algorithms_test()` - performs each algorithm separately for multiple environments.
* `environments_test()` - performs a comparative algorithms test for each environment.
Training & Test results come in the forms of graphs and statistics (for some of the environments)
of both: running average of episode scores, and accumulated scores.

### Implemented Algorithms ([rl_tabular.py](../master/tabular_methods/rl_tabular.py))

* **Monte Carlo**
  * MC policy evaluation
  * MC non-exploring-starts control
  * off-policy MC control

* **TD-0** policy evaluation

* (under the `GeneralModel` class)
  * **SARSA**
  * **Expected SARSA**
  * **Q-learning**
  * **Double Q-learning**

### Implemented Environments ([envs_dss.py](../master/tabular_methods/envs_dss.py))

*(environments with **Discrete\Discretized State Space**)*

* Toy Text
  * FrozenLake
  * Taxi
  * Blackjack
  
* Classic Control
  * MountainCar
  * CartPole
  * Acrobot

### Algorithms Performance Examples

#### AI agent before and after training

**Mountain Car**

<p float="left">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/tabular_methods/performance/mountain-car_untrained.gif" width="400">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/tabular_methods/performance/mountain-car_trained.gif" width="400">
</p>

**Cart Pole**

<p float="left">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/tabular_methods/performance/cart-pole_untrained.gif" width="400">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/tabular_methods/performance/cart-pole_trained.gif" width="400">
</p>

**Acrobot**

<p float="left">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/tabular_methods/performance/acrobot_untrained.gif" width="400">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/tabular_methods/performance/acrobot_trained.gif" width="400">
</p>

#### `environments_test()` result graphs

<p float="left">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/tabular_methods/performance/algorithms_comparison_legend.png" width="200">
</p>

<p float="left">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/tabular_methods/performance/cart-pole-v0-score-training.png" width="390">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/tabular_methods/performance/cart-pole-v0-accumulated-score-training.png" width="410">
</p>

<p float="left">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/tabular_methods/performance/cart-pole-v0-scores-test.png" width="390">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/tabular_methods/performance/cart-pole-v0-accumulated-scores-test.png" width="410">
</p>

## 2. Deep Reinforcement Learning

### How to use

Simply run the desired algorithm file to perform it on the desired environment:

* [deep_q_learning.py](../master/deep_reinforcement_learning/algorithms/deep_q_learning.py) - 
`play()` example environments: CartPole (0), Breakout (1), SpaceInvaders (2).
* [policy_gradient.py](../master/deep_reinforcement_learning/algorithms/policy_gradient.py) - 
`play()` example environments: CartPole (0), Breakout (1), SpaceInvaders (2).
* [actor_critic.py](../master/deep_reinforcement_learning/algorithms/actor_critic.py) - 
`play()` example environments: CartPole (0), Pendulum (1), MountainCarContinuous (2).
* [deep_deterministic_policy_gradient.py](../master/deep_reinforcement_learning/algorithms/deep_deterministic_policy_gradient.py) - 
`play()` example environments: Pendulum (0), MountainCarContinuous (1).

Most of the cases, you can select the desired library type (`lib_type`) implementation: `LIBRARY_TF`, `LIBRARY_TORCH`, `LIBRARY_KERAS`. 

In each algorithm file, there are 3 main operations (leave the one you need, comment out the rest):
* `play()` - performs the algorithm on a single environment.
* `command_line_play()` - performs the algorithm on a single environment, 
through the command-line (using the `argparse` module to parse command-line options). 
This enables concatenating multiple independent runs via `&&`.
* `perform_grid_search()` - performs a comparative grid search for a single environment, and plots the results.
Training & Test results come in the forms of graphs and statistics (for some of the environments)
of both: running average of episode scores, and accumulated scores.

### Implemented Algorithms

* **[Deep Q-learning](../master/deep_reinforcement_learning/algorithms/deep_q_learning.py) (DQL)**
* **[Policy Gradient](../master/deep_reinforcement_learning/algorithms/policy_gradient.py) (PG)**
* **[Actor-Critic](../master/deep_reinforcement_learning/algorithms/actor_critic.py) (AC)**
* **[Deep Deterministic Policy Gradient](../master/deep_reinforcement_learning/algorithms/deep_deterministic_policy_gradient.py) (DDPG)**

Note that some some algorithms have restrictions.

* Innate restrictions:

Discrete Action Space | Continuous Action Space
--- | ---
Deep Q-learning, Policy Gradient | Deep Deterministic Policy Gradient

* or because I haven't completed the code, meaning: writing for **every library implementation** (tensorflow, torch, keras) or for **every input type** (observation vector, stacked frames).

### Implemented Environments ([envs.py](../master/deep_reinforcement_learning/envs.py))
*(environments with **Continuous State Space**)*

. | Discrete Action Space | Continuous Action Space
--- | --- | ---
**Observation Vector Input Type** | CartPole, LunarLander | Pendulum, MountainCarContinuous, LunarLanderContinuous, BipedalWalker
**Stacked Frames Input Type** | Breakout, SpaceInvaders |

### Algorithms Performance Examples

#### DQL

performance of `perform_grid_search()` on first & second FC layers' sizes (number of nodes \ neurons):

<p float="left">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/deep_reinforcement_learning/performance/cart-pole-v1_dql.png" width="500">
</p>

## Dependencies
* Python 3.7.1
* OpenAI Gym 0.12.1
* Tensorflow 1.13.1
* Tensorflow-Probability 0.7
* PyTorch 1.1.1
* Keras 1.0.8
* Numpy
* Matplotlib


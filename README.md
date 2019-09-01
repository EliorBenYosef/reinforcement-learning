# RL and Deep-RL implementations

This repository has two parts:
1. Tabular Methods
2. Deep Reinforcement Learning

This is a **modular** implementation, meaning: you can plug-and-play **almost any environment** (in the corresponding file, within the same base folder) **with any algorithm**.

## 1. Tabular Methods

### How to use

Simply run the testing file ([rl_tabular_testing.py](../blob/master/tabular_methods/rl_tabular_testing.py)).

There are 3 main operations (comment out what you don't need):
* `policy_evaluation_algorithms_test()` - performs either Monte Carlo or TD-0 policy evaluation.
* `learning_algorithms_test()` - performs each algorithm separately for multiple environments.
* `environments_test()` - performs a comparative algorithms test for each environment.
Training & Test results come in the forms of graphs and statistics (for some of the environments)
of both: running average of episode scores, and accumulated scores.

<p float="left">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/samples/algorithms_comparison_legend.png" width="200">
</p>

<p float="left">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/samples/cart-pole-v0-score-training.png" width="400">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/samples/cart-pole-v0-accumulated-score-training.png" width="400">
</p>

<p float="left">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/samples/cart-pole-v0-scores-test.png" width="400">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/samples/cart-pole-v0-accumulated-scores-test.png" width="400">
</p>

### Implemented Algorithms ([rl_tabular.py](../blob/master/tabular_methods/rl_tabular.py))

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

### Implemented Environments ([envs_dss.py](../blob/master/tabular_methods/envs_dss.py))

*(environments with **Discrete\Discretized State Space**)*

* Toy Text
  * FrozenLake
  * Taxi
  * Blackjack
  
* Classic Control
  * MountainCar
  * CartPole
  * Acrobot

## 2. Deep Reinforcement Learning

### How to use

Simply run the testing file ([rl_deep_testing.py](../blob/master/deep_reinforcement_learning/rl_deep_testing.py)).

There are 4 main operation (comment out what you don't need):
* `DQL.play()` - performs the **Deep Q-learning** algorithm on one of the environments: CartPole (0), Breakout (1), SpaceInvaders (2).
* `PG.play()` - performs the **Policy Gradient** algorithm on one of the environments: CartPole (0), Breakout (1), SpaceInvaders (2).
* `AC.play()` - performs the **Actor-Critic** algorithm on one of the environments: CartPole (0), Pendulum (1), MountainCarContinuous (2).
* `DDPG.play()` - performs the **Deep Deterministic Policy Gradient** algorithm on one of the environments: Pendulum (0), MountainCarContinuous (1).

Most of the cases, you can select the desired library type (`lib_type`) implementation: `LIBRARY_TF`, `LIBRARY_TORCH`, `LIBRARY_KERAS`. 

### Implemented Algorithms ([rl_deep.py](../blob/master/deep_reinforcement_learning/rl_deep.py))

* **Deep Q-learning (DQL)**
* **Policy Gradient (PG)**
* **Actor-Critic (AC)**
* **Deep Deterministic Policy Gradient (DDPG)**

Note that some some algorithms have restrictions.

* Innate restrictions:

Discrete Action Space | Continuous Action Space
--- | ---
Deep Q-learning, Policy Gradient | Deep Deterministic Policy Gradient

* or because I haven't completed the code, meaning: writing for **every library implementation** (tensorflow, torch, keras) or for **every input type** (observation vector, stacked frames).

### Implemented Environments ([envs.py](../blob/master/deep_reinforcement_learning/envs.py))
*(environments with **Continuous State Space**)*

. | Discrete Action Space | Continuous Action Space
--- | --- | ---
**Observation Vector Input Type** | CartPole, LunarLander | Pendulum, MountainCarContinuous, LunarLanderContinuous, BipedalWalker
**Stacked Frames Input Type** | Breakout, SpaceInvaders |

## AI agent before and after training

### Mountain Car

<p float="left">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/samples/mountain-car_untrained.gif" width="400">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/samples/mountain-car_trained.gif" width="400">
</p>

### Cart Pole

<p float="left">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/samples/cart-pole_untrained.gif" width="400">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/samples/cart-pole_trained.gif" width="400">
</p>

### Acrobot

<p float="left">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/samples/acrobot_untrained.gif" width="400">
  <img src="https://github.com/EliorBenYosef/reinforcement-learning/blob/master/samples/acrobot_trained.gif" width="400">
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


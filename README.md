# RL and Deep-RL implementations

This repository has two parts:
1. Tabular Methods
2. Deep Reinforcement Learning

This is a **modular** implementation, meaning: you can plug-and-play **almost any environment** (in the corresponding file, within the same base folder) **with any algorithm**.

## 1. Tabular Methods

### Implemented Algorithms

* **Monte Carlo**
  * MC policy evaluation
  * MC non-exploring-starts control
  * off-policy MC control

* **TD-0**

* (under the `GeneralModel` class)
  * **SARSA**
  * **Expected SARSA**
  * **Q-learning**
  * **Double Q-learning**

### Implemented Environments
*(environments with **Discrete\Discretized State Space**)*
* FrozenLake
* Taxi
* Blackjack
* CartPole
* Acrobot
* MountainCar

## 2. Deep Reinforcement Learning

### Implemented Algorithms

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

### Implemented Environments
*(environments with **Continuous State Space**)*

. | Discrete Action Space | Continuous Action Space
--- | --- | ---
**Observation Vector Input Type** | CartPole, LunarLander | Pendulum, MountainCarContinuous, LunarLanderContinuous, BipedalWalker
**Stacked Frames Input Type** | Breakout, SpaceInvaders |

## Dependencies
* Python 3.7.1
* OpenAI Gym 0.12.1
* Tensorflow 1.13.1
* Tensorflow-Probability 0.7
* PyTorch 1.1.1
* Keras 1.0.8
* Numpy
* Matplotlib


# Adaptation-to-criticality-through-organizational-invariance

Code reproducing the models used in this article https://arxiv.org/abs/1712.05284

## Abstract

Many biological and cognitive systems do not operate deep within one or other regime of activity. Instead, they are poised at critical points located at transitions of their parameter space. The pervasiveness of criticality suggests that there may be general principles inducing this behaviour, yet there is no well-founded theory for understanding how criticality is found at a wide range of levels and contexts. In this paper we present a general adaptive mechanism that maintains an internal organizational structure in order to drive a system towards critical points while it interacts with different environments. We implement the mechanism in artificial embodied agents controlled by a neural network maintaining a correlation structure randomly sampled from an Ising model at critical temperature. Agents are evaluated in two classical reinforcement learning scenarios: the Mountain Car and the Acrobot double pendulum. In both cases the neural controller reaches a point of criticality, which coincides with a transition point between two regimes of the agent's behaviour. These results suggest that adaptation to criticality could be used as a general adaptive mechanism in some circumstances, providing an alternative explanation for the pervasive presence of criticality in biological and cognitive systems. 

## Description of the code

The code in the [Network](Network/) folder generates the distribution of correlations used in the article, and trains an isolated network of different sizes to be poised near a critical point.

The code in the [Mountain-Car](Mountain-Car/) and [Acrobot](Acrobot/) includes scripts for training agents in teh Mountain-Car and Acrobot environments. As well, videos of the behaviour of trained agents with $N_h=64$ hidden neurons and $N=70$ total neurons for the [Mountain-Car](Video-MountainCar.avi) and [Acrobot](Acrobot.avi) environments.

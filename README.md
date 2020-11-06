# Adaptation-to-criticality-through-organizational-invariance

Code reproducing the models used in this article
Aguilera, M & Bedia, MG (2018). [Adaptation to criticality through organizational invariance in embodied agents].(https://www.nature.com/articles/s41598-018-25925-4). _Scientific Reports_ volume 8, Article number: 7723 (2018). doi:10.1038/s41598-018-25925-4

## Abstract

Many biological and cognitive systems do not operate deep within one or other regime of activity. Instead, they are poised at critical points located at transitions of their parameter space. The pervasiveness of criticality suggests that there may be general principles inducing this behaviour, yet there is no well-founded theory for understanding how criticality is found at a wide range of levels and contexts. In this paper we present a general adaptive mechanism that maintains an internal organizational structure in order to drive a system towards critical points while it interacts with different environments. We implement the mechanism in artificial embodied agents controlled by a neural network maintaining a correlation structure randomly sampled from an Ising model at critical temperature. Agents are evaluated in two classical reinforcement learning scenarios: the Mountain Car and the Acrobot double pendulum. In both cases the neural controller reaches a point of criticality, which coincides with a transition point between two regimes of the agent's behaviour. These results suggest that adaptation to criticality could be used as a general adaptive mechanism in some circumstances, providing an alternative explanation for the pervasive presence of criticality in biological and cognitive systems. 

## Description of the code

The code in the [Network](Network/) folder generates the distribution of correlations used in the article, and trains an isolated network of different sizes to be poised near a critical point.

The code in the [Mountain-Car](Mountain-Car/) and [Acrobot](Acrobot/) includes scripts for training agents in teh Mountain-Car and Acrobot environments, as well as for visualizing their behaviour and reproduce results in the paper related with the entrpy and heat capacity of the system. As well, videos of the behaviour of trained agents with N_h=64 hidden neurons and N=70 total neurons for the [Mountain-Car](https://github.com/MiguelAguilera/Adaptation-to-criticality-through-organizational-invariance/raw/master/Video-MountainCar.avi) and [Acrobot](https://github.com/MiguelAguilera/Adaptation-to-criticality-through-organizational-invariance/raw/master/Video-Acrobot.avi) environments.

* 'train.py' generates agents trained according to the adaptation to criticality algorithm
* 'simulate.py' picks one agent generated from 'train.py' and simulates and visualizes its behaviour
* 'compute-entropy.py' picks one agent generated from 'train.py' and simulates its behaviour to compute values of entropy
* 'visualize-entropy.py' visualizes the results from 'compute-entropy.py' 

# About

This Python module defines Numpy and Torch methods for solving a Bayesian Nash equilibrium using Neural Replicator Dynamics. Many iterations of the NeuRD update rule are applied to a uniform strategy to produce another strategy with low exploitability.

# Definitions

A game involves two players. A player is a set of types $T$ and a set of actions labeled $\{1, ..., n_t\}$ for each $t \in T$ and a probability distribution function over the types $\Omega \in \Delta_T$.

A policy for a player is a collection of type-specific strategies $\pi_t \in \Delta_{n_t}$ for $t \in T$.

For each pair of types $s, t$ we have a zero-sum payoff matrix $M_{s, t} \in \mathbb{R}^{n_s \times n_t}$ of player 1's payoffs where player 1 is the row player.
Then the payoff for policies $\pi_i \in \prod_{t \in T_i} \Delta_{n_t}$, $i = 1, 2$ is
$\sum_{s \in T_1} \sum_{t \in T_2} \Omega_1(s) \Omega_2(t) P(M_{s,t}, \pi_1, \pi_2)$ for player 1 where $P$ is the usual row player payoff:
$P(M, \pi_1, \pi_2) = \pi_1^T \times M \times \pi_2$.

A Bayesian Nash equilibrium is then a pair of policies over the game where both players are indifferent to changing strategies. The exploitability of a pair of strategies is the sum of the maximal gain for both players if they could unilaterally change strategies. Therefore a strategy pair is Nash if the exploitability is zero.

# Interface

Player data is distinguished by the subscript $i = 1, 2$

A `Player` is specified by two lists with the same length, which corresponds to the number of "types" $T_i$ for that player.

The first list `n : List[int]` the number of actions for that type

The second list `o : List[float]` is the "priors" over those types, $\Omega_i$.

A game is then specified by two players `p1 : Player`, `p2 :Player` completed by a `Dict[(int, int), np.array]` of matrix games. The keys are for thie dictionary are pairs indices of types for players 1 and 2, resp. The values are Numpy arrays with shape `(p1.n[i], p2.n[i])`.

This data defines a `Solver` class which stores the data in padded tensors so that a single NeuRD update is simply batched matrix multiplication.

This class has a method which takes `n : int, lr : float, lr_decay : float`. It starts with a uniform strategy for both players and performs `n` iterations of the algorithm and returns the final and average strategies for both players. Each iterations the learning rate decay is applied `lr *= lr_decay`. The function returns the average strategyies for player 1 and 2, and also the latest iteration's strategies.

It also has a method that takes a strategy (say the output of the previous method) and computes the exploitability.

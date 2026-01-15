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

A player is specified by a `List[int]` and `List[float]`, both the same length. The length is the number of types $T = \{0, ..., n - 1\}$. The values of the first list are the number of actions for that type and the second is $\Omega_i$ the pdf over those types.

A game is then specified by two players and a list[matrix] of some 2D array of floats. The list has length $|T_1| \times |T_2|$ and is in row major order.

This data defines a `Solver` class which stores the data in padded tensors so that a single NeuRD update is simply batched matrix multiplication.

This class has a method which takes a single `n : int`. It starts with a uniform strategy for both players and performs `n` iterations of the algorithm and returns the final and average strategies for both players. The pair of average strategies should be Nash.

It also has a method that takes a strategy (say the output of the previous method) and computes the exploitability.

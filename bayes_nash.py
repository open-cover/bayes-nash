import numpy as np
from typing import Dict, List
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--main",
    type=str,
    required=True,
    help="The function to run as main()",
)
parser.add_argument(
    "--iterations",
    type=int,
    required=True,
    help="Number of iterations to use when solving",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.01,
    help="Learning rate for NeuRD update.",
)
parser.add_argument(
    "--games",
    type=int,
    default=1000,
    help="Number of Bayesian games to try solving",
)

args = parser.parse_args()


def softmax(x, axis=-1):
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


class Player:

    def __init__(self, actions, omega):
        assert len(actions) == len(omega), "Mismatched actions and omega lengths"
        assert all(k >= 1 for k in actions), "Some types have number of actions < 1"
        assert sum(omega) == 1, "Omega pdf does not sum to 1"
        self.n = len(actions)
        self.K = max(actions)
        self.actions = np.array(actions)
        self.omega = np.array(omega)

    def logits(self):
        logits = np.zeros((self.n, self.K))
        for i in range(self.n):
            logits[i, self.actions[i] :] = -np.inf
        return logits


class Solver:

    def __init__(self, p1: Player, p2: Player, payoffs: Dict[[int, int], np.array]):
        self.n1 = p1.n
        self.n2 = p2.n
        self.pairs = p1.n * p2.n
        self.omega = np.outer(p1.omega, p2.omega)[..., None]
        self.batched_payoffs = np.zeros((self.n1, self.n2, p1.K, p2.K))
        for i in range(p1.n):
            for j in range(p2.n):
                self.batched_payoffs[i, j, 0 : p1.actions[i], 0 : p2.actions[j]] = (
                    payoffs[i, j]
                )

        # TODO remove
        self.p1 = p1
        self.p2 = p2

    def go(self, iterations: int, lr: float, p: bool = False):
        p1_logits, p2_logits = self.p1.logits(), self.p2.logits()

        for _ in range(iterations):
            p1_policies = softmax(p1_logits)
            p2_policies = softmax(p2_logits)
            p1_returns = np.einsum("ijmn,jn->ijm", self.batched_payoffs, p2_policies)
            p2_returns = -np.einsum("im,ijmn->ijn", p1_policies, self.batched_payoffs)

            # payoff = np.einsum('ijn,jn->ij', p2_returns, p2_policies)[..., None] # mind the negative!
            p1_payoffs = np.einsum("im,ijm->ij", p1_policies, p1_returns)[..., None]

            p1_advantages = p1_returns - p1_payoffs
            p2_advantages = p2_returns + p1_payoffs

            p1_gradient = np.sum(p1_advantages * self.omega, axis=1)
            p2_gradient = np.sum(p2_advantages * self.omega, axis=0)
            # assert p1_gradient.shape == (self.p1.n, self.p1.K)
            # assert p2_gradient.shape == (self.p2.n, self.p2.K)

            p1_logits += p1_gradient
            p2_logits += p2_gradient

        return p1_logits, p2_logits

    def expl(self, p1_logits: np.array, p2_logits: np.array) -> float:
        p1_policies = softmax(p1_logits)
        p2_policies = softmax(p2_logits)
        p1_returns = np.einsum("ijmn,jn->ijm", self.batched_payoffs, p2_policies)
        p2_returns = -np.einsum("im,ijmn->ijn", p1_policies, self.batched_payoffs)
        p1_options = np.sum(self.omega * p1_returns, axis=1)
        p2_options = np.sum(self.omega * p2_returns, axis=0)
        p1_best = np.max(p1_options, axis=1).sum()
        p2_best = np.max(p2_options, axis=1).sum()
        return p1_best + p2_best


def simple():

    iterations = args.iterations

    p1 = Player([2, 3], [0.5, 0.5])
    p2 = Player([2, 2], [0.5, 0.5])

    def draw(a, b):
        return np.zeros((a, b))

    def win(a, b):
        x = -np.ones((a, b))
        x[0, :] = 1
        return x

    matrices = {}
    matrices[(0, 0)] = draw(2, 2)
    matrices[(1, 0)] = draw(3, 2)
    matrices[(0, 1)] = win(2, 2)
    matrices[(1, 1)] = win(3, 2)

    solver = Solver(p1, p2, matrices)
    p1_logits, p2_logits = solver.go(iterations=iterations, lr=args.lr)
    e = solver.expl(p1_logits, p2_logits)
    print(f"expl: {e}")


def test():
    import random

    games = args.games
    iterations = args.iterations

    n1 = random.randint(1, 5)
    n2 = random.randint(1, 5)
    k1 = [random.randint(1, 4) for _ in range(n1)]
    k2 = [random.randint(1, 4) for _ in range(n2)]
    o1 = [1.0 / n1 for _ in range(n1)]
    o2 = [1.0 / n2 for _ in range(n2)]

    total_expl = 0
    max_expl = 0

    for _ in range(games):
        p1 = Player(k1, o1)
        p2 = Player(k2, o2)

        matrices = {}
        for i in range(n1):
            for j in range(n2):
                matrices[(i, j)] = np.random.rand(k1[i], k2[j])

        solver = Solver(p1, p2, matrices)
        p1_logits, p2_logits = solver.go(iterations=iterations, lr=args.lr)

        e = solver.expl(p1_logits, p2_logits)
        max_expl = max(e, max_expl)
        total_expl += e

    print(f"Average exploitability: {total_expl / games}")
    print(f"Max exploitability: {max_expl}")


if __name__ == "__main__":
    if args.main == "simple":
        simple()
    elif args.main == "test":
        test()

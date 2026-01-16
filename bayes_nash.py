import numpy as np
from typing import Dict, List
import argparse
import random

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
    "--lr-decay",
    type=float,
    default=1.0,
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
        eps = 0.001
        assert len(actions) == len(omega), "Mismatched actions and omega lengths"
        assert all(k >= 1 for k in actions), "Some types have number of actions < 1"
        assert (
            abs(sum(omega) - 1) < eps
        ), f"Omega pdf does not sum to [1 - eps, 1 + eps], eps = {eps}"
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

    def go(self, iterations: int, lr: float, lr_decay: float, p: bool = False):
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

            p1_logits += lr * p1_gradient
            p2_logits += lr * p2_gradient
            lr *= lr_decay

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


def generate_random_game_solver(seed, max_types=5, max_actions=4):
    rng = random.Random(seed)
    n1 = rng.randint(1, max_types)
    n2 = rng.randint(1, max_types)
    k1 = [rng.randint(1, max_actions) for _ in range(n1)]
    k2 = [rng.randint(1, max_actions) for _ in range(n2)]
    raw_o1 = [rng.random() for _ in range(n1)]
    o1 = [x / sum(raw_o1) for x in raw_o1]
    raw_o2 = [rng.random() for _ in range(n2)]
    o2 = [x / sum(raw_o2) for x in raw_o2]

    p1 = Player(k1, o1)
    p2 = Player(k2, o2)
    np_rng = np.random.default_rng(seed)
    matrices = {}
    for i in range(n1):
        for j in range(n2):
            matrices[(i, j)] = np_rng.random((k1[i], k2[j]))
    solver = Solver(p1, p2, matrices)
    return solver


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
    p1_logits, p2_logits = solver.go(
        iterations=iterations, lr=args.lr, lr_decay=args.lr_decay
    )
    e = solver.expl(p1_logits, p2_logits)
    print(f"expl: {e}")


def test():

    games = args.games
    iterations = args.iterations

    total_expl = 0
    max_expl = 0
    max_expl_seed = None

    for _ in range(games):
        seed = random.randint(0, 2**32 - 1)
        solver = generate_random_game_solver(seed)
        p1_logits, p2_logits = solver.go(
            iterations=iterations, lr=args.lr, lr_decay=args.lr_decay
        )
        e = solver.expl(p1_logits, p2_logits)
        if e > max_expl:
            max_expl = e
            max_expl_seed = seed
        total_expl += e

    print(f"Average exploitability: {total_expl / games}")
    print(f"Max exploitability: {max_expl} with seed {max_expl_seed}")


HARD_SEEDS = [400845770, 2894948770, 2847287987, 826824531]

if __name__ == "__main__":
    if args.main == "simple":
        simple()
    elif args.main == "test":
        test()

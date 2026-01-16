import numpy as np

from typing import Dict, List


def softmax(x, axis=-1):
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


class Player:

    def __init__(self, actions, omega):
        assert len(actions) == len(omega), "Mismatched actions and omega lengths"
        # assert actions >= 1
        # assert sum omega = 1
        self.n = len(actions)
        self.K = max(actions)
        print(f"Player - n:  {self.n}, k: {self.K}")
        self.actions = np.array(actions)
        self.omega = np.array(omega)

    def logits(self):
        logits = np.zeros((self.n, self.K))
        for i in range(self.n):
            logits[i, self.actions[i] :] = -np.inf
        return logits


class Solver:

    def __init__(self, p1: Player, p2: Player, payoffs: Dict[[int, int], np.matrix]):
        self.n1 = p1.n
        self.n2 = p2.n
        self.pairs = p1.n * p2.n
        self.omega = np.outer(p1.omega, p2.omega)[..., None]
        print(p1.omega.shape, p2.omega.shape)
        print("omega", self.omega.shape)
        print("omega sum", np.sum(self.omega))

        self.batched_payoffs = np.zeros((self.n1, self.n2, p1.K, p2.K))
        for i in range(p1.n):
            for j in range(p2.n):
                self.batched_payoffs[i, j, 0 : p1.actions[i], 0 : p2.actions[j]] = (
                    payoffs[i, j]
                )

        # TODO remove
        self.p1 = p1
        self.p2 = p2

    def get_expanded_policies(self, p1_logits, p2_logits):

        x = softmax(p1_logits)
        y = softmax(p2_logits)
        x = np.repeat(x, self.n2, axis=0)
        y = np.tile(y, (self.n1, 1))
        x = x[:, None, :]
        y = y[:, :, None]
        return x, y

    def go(self, iterations: int):
        p1_logits, p2_logits = self.p1.logits(), self.p2.logits()

        for _ in range(iterations):

            x = softmax(p1_logits)
            y = softmax(p2_logits)
            # x = x.reshape()
            print("logits:" , x.shape, y.shape)
            p1_returns = np.einsum('ijmn,jn->ijm', self.batched_payoffs, y)
            p2_returns = np.einsum('im,ijmn->ijn', x, self.batched_payoffs)
            print("returns: ", p1_returns.shape, p2_returns.shape)
            payoff = np.einsum('ijn,jn->ij', p2_returns, y)[..., None]
            # payoff_same = np.einsum('im,ijm->ij', x, p1_returns)
            print("payoff", payoff.shape)
            # assert(payoff == payoff_same)
            # print(payoff == payoff_same)

            p1_advantages = p1_returns - payoff
            p2_advantages = p2_returns + payoff

            print("advantages:", p1_advantages.shape, p2_advantages.shape)

            p1_gradient = np.sum(p1_advantages * self.omega, axis=1)
            p2_gradient = np.sum(p2_advantages * self.omega, axis=0)
            print("gradients:", p1_gradient.shape, p2_gradient.shape)
            assert p1_gradient.shape == (self.p1.n, self.p1.K)
            assert p2_gradient.shape == (self.p2.n, self.p2.K)



def test():
    import random
    tries = 1000

    n1 = random.randint(1, 5)
    n2 = random.randint(1, 5)

    k1 = [random.randint(1, 4) for _ in range(n1)]
    k2 = [random.randint(1, 4) for _ in range(n2)]
    o1 = [1.0 / n1 for _ in range(n1)]
    o2 = [1.0 / n2 for _ in range(n2)]


    for _ in range(tries):
        print(f"try: {_}")
        p1 = Player(k1, o1)
        p2 = Player(k2, o2)

        matrices = {}
        for i in range(n1):
            for j in range(n2):
                matrices[(i, j)] = np.random.rand(k1[i], k2[j])

        solver = Solver(p1, p2, matrices)
        solver.go(1)

if __name__ == "__main__":
    test()

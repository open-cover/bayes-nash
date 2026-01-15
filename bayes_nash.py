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
        self.n = len(actions)
        self.K = max(actions)
        self.actions = np.array(actions)
        self.omega = np.array(omega)


class Solver:

    def __init__(self, p1 : Player, p2: Player, payoffs: Dict[[int, int], np.matrix]):

        self.n1 = p1.n
        self.n2 = p2.n
        self.pairs = p1.n * p2.n

        # should be new data, not view
        self.omega = np.outer(p1.omega, p2.omega).reshape(self.pairs, 1)

        # new data
        p1_logits, p2_logits = self.new_logits(p1, p2)

        p1_policy_flat, p2_policy_flat = self.get_expanded_policies(p1_logits, p2_logits)

        self.batched_payoffs = np.zeros((self.pairs, p1.K, p2.K))
        for i in range(p1.n):
            for j in range(p2.n):
                index = i * p2.n + j
                self.batched_payoffs[index, 0 : p1.actions[i], 0 : p2.actions[j]] = payoffs[i, j]

        p1_rewards = np.matmul(self.batched_payoffs, p2_policy_flat)
        p2_rewards = np.matmul(p1_policy_flat, self.batched_payoffs)

        assert p1_rewards.shape == (self.pairs, p1.K, 1)
        assert p2_rewards.shape == (self.pairs, 1, p2.K)

    def get_expanded_policies(self, p1_logits, p2_logits):

        x = softmax(p1_logits)
        y = softmax(p2_logits)
        x = np.repeat(x, self.n2, axis=0)
        y = np.tile(y, (self.n1, 1))
        x = x[:, None, :]
        y = y[:, :, None]
        return x, y


    def new_logits(self, p1 : Player, p2 : Player):
        p1_logits = np.zeros((p1.n, p1.K))
        p2_logits = np.zeros((p2.n, p2.K))
        for i in range(p1.n):
            p1_logits[i, p1.actions[i] : ] = -np.inf
        for i in range(p2.n):
            p2_logits[i, p2.actions[i] : ] = -np.inf
        return p1_logits, p2_logits
        


if __name__ == "__main__":
    
    p1 = Player([2, 1], [.5, .5])
    p2 = Player([3, 3], [.9, .1])

    matrices = {}
    for i in range(p1.n):
        for j in range(p2.n):
            matrix = np.random.rand(p1.actions[i], p2.actions[j])
            matrices[(i, j)] = matrix

    solver = Solver(p1, p2, matrices)
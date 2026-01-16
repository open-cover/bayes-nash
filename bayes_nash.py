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
        self.omega = np.outer(p1.omega, p2.omega).reshape(self.pairs, 1)

        self.batched_payoffs = np.zeros((self.pairs, p1.K, p2.K))
        for i in range(p1.n):
            for j in range(p2.n):
                index = i * p2.n + j
                self.batched_payoffs[index, 0 : p1.actions[i], 0 : p2.actions[j]] = payoffs[i, j]

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


    def go(self, iterations : int):
        p1_logits, p2_logits = self.new_logits(p1, p2)


        for _ in range(iterations):

            p1_policy_flat, p2_policy_flat = self.get_expanded_policies(p1_logits, p2_logits)


            p1_returns = np.squeeze(np.matmul(self.batched_payoffs, p2_policy_flat), axis=-1)
            p2_returns = -np.squeeze(np.swapaxes(np.matmul(p1_policy_flat, self.batched_payoffs), 1, 2), axis=-1)
            assert p1_returns.shape == (self.pairs, p1.K)
            assert p2_returns.shape == (self.pairs, p2.K)

            rewards = np.matmul(p1_policy_flat, p1_returns[:, :, None]).squeeze(-1)
            assert rewards.shape == (self.pairs, 1)
            
            p1_advantages = p1_returns - rewards
            p2_advantages = p2_returns + rewards
            assert p1_advantages.shape == (self.pairs, p1.K)
            assert p2_advantages.shape == (self.pairs, p2.K)

            p1_gradient = np.sum(p1_advantages * self.omega, axis=0)
            p2_gradient = np.sum(p2_advantages * self.omega, axis=0)
            assert p1_gradient.shape == (p1.K,)
            assert p2_gradient.shape == (p2.K,)

            print(p1_gradient)
            print(p2_gradient)


if __name__ == "__main__":
    
    p1 = Player([2, 2], [.5, .5])
    p2 = Player([2, 3], [0.5, 0.5])

    matrices = {}

    draw_2 = np.zeros((2, 2))
    draw_3 = np.zeros((2, 3))

    win_2 = np.array([[1, 1], [-1, -1]])
    win_3 = np.array([[1, 1, 1], [-1, -1, -1]])

    matrices[(0, 0)] = draw_2
    matrices[(0, 1)] = win_3
    matrices[(1, 0)] = draw_2
    matrices[(1, 1)] = win_3

    solver = Solver(p1, p2, matrices)
    solver.go(1)
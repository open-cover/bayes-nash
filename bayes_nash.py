import numpy as np

class Player:

    def __init__(self, actions, omega):
        assert len(actions) == len(omega), "Mismatched actions and omega lengths"
        self.n = len(actions)
        self.actions = np.array(actions)
        self.omega = np.array(omega)


class Solve:

    def __init__(self, p1 : Player, p2: Player, payoffs):

        assert len(payoffs) ==(p1.n * p2.n), "Mismatched number of payoff matrices"

        for i in range(p1.n):
            for j in range(p2.n):
                pass


if __name__ == "__main__":
    pass

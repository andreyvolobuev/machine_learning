import numpy as np

class Infoset:
    def __init__(self, actions):
        self.actions = actions
        self.strategy = np.zeros(len(self.actions))
        self.regrets = np.zeros(len(self.actions))

    def normalize(self, strategy):
        if sum(strategy) > 0:
            strategy /= sum(strategy)
        else:
            strategy = np.array([1 / len(self.actions)] * len(self.actions))
        return strategy

    def get_strategy(self, reach_probabilities):
        strategy = np.maximum(0, self.regrets)
        strategy = self.normalize(strategy)

        self.strategy += reach_probabilities * strategy
        return strategy

    def get_average_strategy(self):
        return self.normalize(self.strategy)


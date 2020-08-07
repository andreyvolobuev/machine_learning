import numpy as np
from leduk import Game
from utils import Infoset


class CFRTrainer:
    def __init__(self):
        self.imap = {}

    def get_infoset(self, card, history):
        available_actions = Game.get_available_actions(history)
        key = str(card) + history
        self.imap[key] = self.imap.get(key, Infoset(available_actions))
        return self.imap[key]

    def train(self, iters):
        value = 0
        for _ in range(iters):
            card1, card2 = Game.deal_cards()
            reach_prob1, reach_prob2 = 1, 1
            history = ''
            value += self.cfr(card1, card2, history, reach_prob1, reach_prob2)
        return value

    def cfr(self, card1, card2, history, reach_prob1, reach_prob2):
        if Game.is_terminal(history):
            return Game.get_payoff(history, card1, card2)

        if Game.is_chance(history):
            value = 0
            for card in Game.deck:
                if card != card1 and card != card2:
                    value += self.cfr(card2, card1,\
                                      history + f'_{card}_',\
                                      reach_prob2, reach_prob1)
            return value

        infoset = self.get_infoset(card1, history)
        strategy = infoset.get_strategy(reach_prob1)
        counterfactual_values = np.zeros(len(infoset.actions))

        for i, action in enumerate(infoset.actions):
            action_probability = strategy[i]
            new_reach_prob = reach_prob1 * action_probability
            counterfactual_values[i] += -self.cfr(card2, card1,\
                                                  history + action,\
                                                  reach_prob2, new_reach_prob)

        node_value = counterfactual_values.dot(strategy)

        for i, action in enumerate(infoset.actions):
            infoset.regrets[i] += reach_prob2 * (counterfactual_values[i] - node_value)

        return node_value


if __name__ == '__main__':
    trainer = CFRTrainer()
    trainer.train(10000)
    np.set_printoptions(precision=2, floatmode='fixed', suppress=True)
    for key, infoset in trainer.imap.items():
        key = key[:2] + ' ' + key[2:].replace('_', ' ')
        print(f'{key} \t {infoset.get_average_strategy()}')


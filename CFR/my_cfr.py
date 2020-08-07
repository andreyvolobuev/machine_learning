import numpy as np

class Infoset:
    def __init__(self):
        self.actions = ['B', 'C']
        self.regrets = np.zeros(len(self.actions))
        self.strategy = np.zeros(len(self.actions))

    def normalize(self, strategy):
        """
        Normalize the strategy to match positive regrets.
        If there are no positive regrets, then use uniform random strategy.
        """
        if sum(strategy) > 0:
            strategy /= sum(strategy)
        else:
            strategy = np.array([1.0 / len(self.actions)] * len(self.actions))
        return strategy

    def get_strategy(self, reach_probability):
        """
        Return regret matching strategy.
        """
        strategy = np.maximum(0, self.regrets)
        strategy = self.normalize(strategy)

        self.strategy += reach_probability * strategy
        return strategy

    def get_average_strategy(self):
        return self.normalize(self.strategy.copy())


class Game:
    values = {'K': 3, 'Q': 2, 'J': 1}

    @staticmethod
    def is_terminal(history):
        result =  history[-2:] in ['BB', 'CC', 'BC']
        return result

    @staticmethod
    def get_payoff(history, cards):
        if history[-2:] in ['BC']:
            return +1
        else:
            payoff = 2 if 'B' in history else 1
            active_value, op_value = Game.values[cards[0]],\
                                     Game.values[cards[1]]

            if active_value > op_value:
                return payoff
            else:
                return -payoff


class CFR:
    def __init__(self):
        self.imap = {}

    def get_infoset(self, card, history):
        key = str(card) + history
        self.imap[key] = self.imap.get(key, Infoset())
        return self.imap[key]

    def train(self, iters):
        value = 0
        deck = ['J', 'Q', 'K']
        for _ in range(iters):
            print('... new iter ...')
            cards = np.random.choice(deck, 2, replace=False)
            card1, card2 = cards
            history = ''
            reach_prob1, reach_prob2 = 1, 1
            value += self.cfr(card1, card2, history, reach_prob1, reach_prob2)
        return value

    def cfr(self, card1, card2, history, reach_prob1, reach_prob2):
        print(f'run cfr. {card1} {card2}, {history}, {reach_prob1}, {reach_prob2}')
        if Game.is_terminal(history):
            return Game.get_payoff(history, [card1, card2])

        infoset = self.get_infoset(card1, history)
        strategy = infoset.get_strategy(reach_prob1)
        counterfactual_values = np.zeros(len(infoset.actions))

        for i, action in enumerate(infoset.actions):
            action_prob = strategy[i]
            new_reach_prob = reach_prob1 * action_prob
            counterfactual_values[i] = -self.cfr(card2, card1,
                                                 history + action,
                                                 reach_prob2, new_reach_prob)

        node_value = counterfactual_values.dot(strategy)

        # print(f'node val: {node_value}, cfr: {counterfactual_values}')
        for i, action in enumerate(infoset.actions):
            infoset.regrets[i] += reach_prob2 * (counterfactual_values[i] - node_value)

        return node_value


if __name__ == '__main__':
    game = Game()
    trainer = CFR()

    np.set_printoptions(precision=2, floatmode='fixed', suppress=True)

    value = trainer.train(10)
    print(value)

    for name, infoset in sorted(trainer.imap.items(), key=lambda s: len(s[0])):
        print(f"{name:3}:    {infoset.get_average_strategy()}")


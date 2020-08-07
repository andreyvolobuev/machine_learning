import numpy as np

class Game:
    deck = ['As', 'Ah', 'Ks', 'Kh', 'Qs', 'Qh', 'Js', 'Jh']
    values = {'A': 4,
              'K': 3,
              'Q': 2,
              'J': 1}

    @staticmethod
    def deal_cards():
        return np.random.choice(Game.deck, 2, replace=False)

    @staticmethod
    def is_terminal(history):
        if 'F' in history or '_' in history and history[-2:] in ['PP', 'BC', 'RC']:
            return True

    @staticmethod
    def get_payoff(history, card1, card2):
        payoff = 2
        for stage in history:
            if stage == 'B' or stage == 'C':
                payoff += 1
            elif stage == 'R':
                payoff += 2

        if 'F' in history:
            return payoff

        chance = history.index('_')+1
        card = history[chance:chance+2]
        if card1[0] == card[0]:
            return payoff
        elif card2[0] == card[0]:
            return -payoff
        elif Game.values[card1[0]] > Game.values[card2[0]]:
            return payoff
        elif Game.values[card1[0]] < Game.values[card2[0]]:
            return -payoff
        else:
            return payoff / 2

    @staticmethod
    def is_chance(history):
        if '_' not in history and history[-2:] in ['PP', 'RC', 'BC']:
            return True

    @staticmethod
    def get_available_actions(history):
        actions = []

        if len(history) > 0:
            last_action = history[-1]
            if last_action == 'P' or last_action == '_':
                actions = ['P', 'B']
            elif last_action == 'B':
                actions = ['F', 'C', 'R']
            elif last_action == 'R':
                actions = ['F', 'C']
        else:
            actions = ['P', 'B']

        return actions


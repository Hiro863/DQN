from agent import Agent
from parameters import *
import random


class Humanoid(Agent):
    def __init__(self, name, loc, nutrientA, nutrientB, state):
        Agent.__init__(self, name, loc, nutrientA, nutrientB, state)

        # agent type
        self.type = HUMANOID

        # possessions
        self.possessions = []       # things carried around

    # actions

    def act(self, visible, dqn, observe=True):

        if observe:
            action = ACTIONS[random.randint(0, 7)]
            while action == 'run':
                action = ACTIONS[random.randint(0, 7)]
        else:
            action = ACTIONS[dqn.decide(visible)]
            while action == 'run':
                action = ACTIONS[dqn.decide(visible)]

        # initialise words
        words = None

        if action == 'attack':
            parameter = self.attack()
        elif action == 'move':
            parameter = self.move()
        elif action == 'eat':
            parameter = self.eat()
        elif action == 'pick_up':
            parameter = self.pick_up()
        elif action == 'drink':
            parameter = self.drink()
        elif action == 'speak':
            parameter, words = self.speak()
        else:
            parameter = self.remain()

        # do action in turn
        return action, parameter, words

    def attack(self):
        # attack
        # TODO: make this more sophisticated

        # direction of the food
        direction = DIRECTIONS[random.randint(0, 3)]

        # food location
        row = MOVE_VEC[direction][0] + self.row
        col = MOVE_VEC[direction][1] + self.col

        return row, col

    def pick_up(self):
        # picks an object up
        # TODO: make this more sophisticated

        # direction of the food
        direction = DIRECTIONS[random.randint(0, 3)]

        # food location
        row = MOVE_VEC[direction][0] + self.row
        col = MOVE_VEC[direction][1] + self.col
        return row, col

    def speak(self):
        # speaks to a fellow humanoid
        # TODO: make this more sophisticated

        # direction of the food
        direction = DIRECTIONS[random.randint(0, 3)]

        # food location
        row = MOVE_VEC[direction][0] + self.row
        col = MOVE_VEC[direction][1] + self.col

        words = None
        return (row, col), words

    #TODO eat food in possession


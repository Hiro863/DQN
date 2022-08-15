from agent import Agent
from parameters import *
import random


class Predator(Agent):
    def __init__(self, name, loc, nutrientA, nutrientB, state):
        Agent.__init__(self, name, loc, nutrientA, nutrientB, state)

        # agent type
        self.type = PREDATOR

    # actions

    def act(self, visible, dqn, observe=True):

        if observe:
            action = ACTIONS[random.randint(0, 7)]
            while action in ('pick_up', 'speak'):
                action = ACTIONS[random.randint(0, 7)]
        else:
            action = ACTIONS[dqn.decide(visible)]
            while action in ('pick_up', 'speak'):
                action = ACTIONS[dqn.decide(visible)]

        if action == 'attack':
            parameter = self.attack()
        elif action == 'run':
            parameter = self.run()
        elif action == 'move':
            parameter = self.move()
        elif action == 'eat':
            parameter = self.eat()
        elif action == 'drink':
            parameter = self.drink()
        else:
            parameter = self.remain()

        # do action in turn
        return action, parameter, None

    def run(self):
        # moves double speed
        # move in one of four directions
        # TODO: make this more sophisticated

        row = self.row
        col = self.col

        for _ in range(2):
            direction = DIRECTIONS[random.randint(0, 3)]

            # destination
            row += MOVE_VEC[direction][0]
            col += MOVE_VEC[direction][1]

        return row, col

    def attack(self):
        # attack
        # TODO: make this more sophisticated

        # direction of the food
        direction = DIRECTIONS[random.randint(0, 3)]

        # food location
        row = MOVE_VEC[direction][0] + self.row
        col = MOVE_VEC[direction][1] + self.col

        return row, col



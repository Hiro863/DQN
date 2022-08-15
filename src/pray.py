from agent import Agent
from parameters import *
import random


class Pray(Agent):
    def __init__(self, name, loc, nutrientA, nutrientB, state):
        Agent.__init__(self, name, loc, nutrientA, nutrientB, state)

        # agent type
        self.type = PRAY

    # actions
    def act(self, visible, dqn, observe=True):

        if observe:
            action_index = random.randint(0, 12)
            action, direction = DECISIONS[action_index]
            while action in ('attack', 'pick_up', 'speak', 'run'):
                action_index = random.randint(0, 12)
                action, direction = DECISIONS[action_index]
            debug = 'Random'
        else:
            action_index, debug = dqn.decide(visible)
            action, direction = DECISIONS[action_index]
            while action in ('attack', 'pick_up', 'speak', 'run'):
                action_index, debug = dqn.decide(visible)
                action, direction = DECISIONS[action_index]

        if action == 'move':
            parameter = self.move(direction)
        elif action == 'eat':
            parameter = self.eat(direction)
        elif action == 'drink':
            parameter = self.drink(direction)
        else:
            parameter = self.remain()

        # print moves for debugging purposes
        if direction is None:
            direction = 'none'
        #print(debug + ': ' + action + ', Direction: ' + direction)

        # do action in turn
        return action_index, action, parameter, None

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


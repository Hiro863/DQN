from src.parameters import *


class Agent:
    def __init__(self, name, loc, nutrientA, nutrientB, state):

        # name
        self.name = name        # integer

        # bodily state
        self.alive = True       # alive
        self.stamina = 1.0      # stamina, needed to act
        self.hunger = 1.0       # hunger, 0 is death
        self.thirst = 1.0       # thirst, 0 is death
        self.nutrientA = 1.0    # nutrient A, 0 is death
        self.nutrientB = 1.0    # nutrient B, 0 is death

        # location
        self.row = loc[0]
        self.col = loc[1]

        # nutrients, if eaten
        self.body_nutrientA = nutrientA
        self.body_nutrientB = nutrientB

        # visible area
        self.visible = self.get_visible(state)

    def get_visible(self, state):
        # get visible area VISIBLE_RADIUS x VISIBLE_RADIUS

        # default is -2
        visible = np.full((2 * VISIBLE_RADIUS + 1, 2 * VISIBLE_RADIUS + 1), OUTSIDE)

        # for all points in the area
        for row in range(2 * VISIBLE_RADIUS + 1):
            for col in range(2 * VISIBLE_RADIUS + 1):
                # convert coordinates
                map_row, map_col = self.map_convert((row, col))

                # copy content
                if (0 <= map_row < state.shape[0]) and (0 <= map_col < state.shape[1]):
                    visible[row, col] = state[map_row, map_col]

        # self is 1
        visible[VISIBLE_RADIUS, VISIBLE_RADIUS] = SELF

        return visible

    # actions

    def move(self, parameter):
        # move in one of four directions

        row = MOVE_VEC[parameter][0] + self.row
        col = MOVE_VEC[parameter][1] + self.col


        return row, col

    def eat(self, parameter):
        # eat something in vicinity
        # TODO: make this more sophisticated

        row = MOVE_VEC[parameter][0] + self.row
        col = MOVE_VEC[parameter][1] + self.col

        return row, col

    def drink(self, parameter):
        # drink water
        # TODO: make this more sophisticated

        row = MOVE_VEC[parameter][0] + self.row
        col = MOVE_VEC[parameter][1] + self.col

        return row, col

    def remain(self):
        # do nothing
        return self.row, self.col

    def map_convert(self, loc):
        # converts visible coordinate to map coordinate
        agent_row, agent_col = loc
        map_coord = agent_row + self.row - VISIBLE_RADIUS, agent_col + self.col - VISIBLE_RADIUS

        return map_coord

    def state_update(self):
        # body state update
        self.stamina -= STAMINA_RATE
        self.hunger -= HUNGER_RATE
        self.thirst -= THIRST_RATE
        self.nutrientA -= NUTRIENT_A_RATE
        self.nutrientB -= NUTRIENT_B_RATE





#todo remember places
#TODO debug
class Food:
    def __init__(self, loc, nutrientA, nutrientB, poisonous=False):

        # location
        self.row = loc[0]
        self.col = loc[1]

        # nutrients
        self.nutrientA = nutrientA
        self.nutrientB = nutrientB

        # poisonous
        self.poisonous = poisonous

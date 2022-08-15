from src.food import Food


class DeadBody(Food):
    def __init__(self, loc, nutrientA, nutrientB, poisonous=False):
        Food.__init__(self, loc, nutrientA, nutrientB, poisonous=False)


import numpy as np
import random

# path names============================================================================================================
# directory names
pickle_dir = 'pickle_files'
graphs_dir = '../graphs'
weights_dir = '../weights'
pray_dir = 'pray'
predator_dir = 'predator'
humanoid_dir = 'humanoid'

# file names
training_file = 'training.p'
score_file = 'score.p'
weights_file = 'weights(1).h5'

# playgame parameters===================================================================================================
NUMBER_OF_GAMES = 10000
TRAIN = False
RECORD = False

# game parameters=======================================================================================================
MAX_TURN = 1000
PLAY_RATE = 1

NUMBER_OF_FOODS = 40
NUMBER_OF_PRAYS = 10
NUMBER_OF_PREDATORS = 0
NUMBER_OF_HUMANOIDS = 0

# DQN parameters========================================================================================================
# model
INPUT_SIZE = 16
CHANNEL_NUMBER = 8
L_RATE = 0.0001
BATCH_SIZE = 32

# dqn
MEMORY_CAPACITY = 1000000
OBSERVE = 1000000
UPDATE_TARGET_FREQUENCY = 1000
GAMMA = 0.99

# epsilon
MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.0000005

# save
SAVE_WEIGHTS = 100000

# backend parameters====================================================================================================
# channels
WATER_CHANNEL = 0
FOOD1_CHANNEL = 1
FOOD2_CHANNEL = 2
FOOD3_CHANNEL = 3
PRAY_CHANNEL = 4
PREDATOR_CHANNEL = 5
HUMANOID_CHANNEL = 6
DEAD_BODY_CHANNEL = 7

# map
FOOD_AREA1 = -5
FOOD_AREA2 = -4
FOOD_AREA3 = -3
OUTSIDE = -2
SELF = -1
EMPTY = 0
WATER = 1
FOOD1 = 2
FOOD2 = 3
FOOD3 = 4
PRAY = 5
PREDATOR = 6
HUMANOID = 7
DEAD_BODY = 8


# game
GRID_SIZE = 10
COLORS = {EMPTY: np.array([0, 0, 0]),               # black
          WATER: np.array([100, 0, 0]),             # dark blue
          FOOD1: np.array([0, 255, 0]),              # green
          FOOD2: np.array([0, 100, 0]),             # dark green
          FOOD3: np.array([0, 255, 255]),           # yellow
          PRAY: np.array([211, 0, 148]),            # dark violet
          PREDATOR: np.array([255, 0, 0]),          # blue
          HUMANOID: np.array([0, 0, 255]),          # red
          DEAD_BODY: np.array([0, 76, 153])}        # brown

# food
POISONOUS1 = 0.0        # probability of getting a poisonous food in area 1
POISONOUS2 = 0.0        # probability of getting a poisonous food in area 2
POISONOUS3 = 0.0        # probability of getting a poisonous food in area 3

FOOD_RATE1 = 1          # Number of turns before food appears in area 1
FOOD_RATE2 = 2          # Number of turns before food appears in area 2
FOOD_RATE3 = 3          # Number of turns before food appears in area 3

NUTRIENT_A_MAX1 = 0.9
NUTRIENT_A_MIN1 = 0.8
NUTRIENT_B_MAX1 = 0.7
NUTRIENT_B_MIN1 = 0.3

NUTRIENT_A_MAX2 = 0.7
NUTRIENT_A_MIN2 = 0.1
NUTRIENT_B_MAX2 = 0.3
NUTRIENT_B_MIN2 = 0.1

NUTRIENT_A_MAX3 = 0.2
NUTRIENT_A_MIN3 = 0.1
NUTRIENT_B_MAX3 = 0.7
NUTRIENT_B_MIN3 = 0.3

NUTRIENT_A_PRAY_MAX = 0.9
NUTRIENT_A_PRAY_MIN = 0.7
NUTRIENT_B_PRAY_MAX = 0.8
NUTRIENT_B_PRAY_MIN = 0.7

NUTRIENT_A_PREDATOR_MAX = 0.9
NUTRIENT_A_PREDATOR_MIN = 0.7
NUTRIENT_B_PREDATOR_MAX = 0.9
NUTRIENT_B_PREDATOR_MIN = 0.7


ADJACENT = ((-1, 0), (0, 1), (1, 0), (0, -1))

VISIBLE_SIZE = 16

# agent related
MAX_POSSESSION = 3
VISIBLE_RADIUS = 7

DIRECTIONS = {0: 'n',  # north
              1: 'e',  # east
              2: 's',  # south
              3: 'w'}  # west

MOVE_VEC = {'n': (-1, 0),
            'e': (0, 1),
            's': (1, 0),
            'w': (0, -1)}

ACTIONS = {0: 'attack',
           1: 'run',
           2: 'move',
           3: 'eat',
           4: 'pick_up',
           5: 'drink',
           6: 'speak',
           7: 'remain'}

ACTION_INDEX_PRAY = {'move': 0,
                     'eat': 1,
                     'drink': 2,
                     'remain': 3}

ACTION_INDEX_PREDATOR = {'attack': 0,
                         'run': 1,
                         'move': 2,
                         'eat': 3,
                         'drink': 4,
                         'remain': 5}

ACTION_INDEX_HUMANOID = {'attack': 0,
                         'move': 1,
                         'eat': 2,
                         'pick_up': 3,
                         'drink': 4,
                         'speak': 5,
                         'remain': 6}

MAX_STR_LENGTH = 10

DEAD_BODY_DECOMPOSE = 100

STAMINA_RATE = 0#0.005
HUNGER_RATE = 0#0.005
THIRST_RATE = 0#0.005
NUTRIENT_A_RATE = 0#0.005
NUTRIENT_B_RATE = 0#0.005

AGENT_ACTION_NUMBER = 4
PRAY_ACTION_NUMBER = AGENT_ACTION_NUMBER
PREDATOR_ACTION_NUMBER = AGENT_ACTION_NUMBER + 2
HUMANOID_ACTION_NUMBER = AGENT_ACTION_NUMBER + 3

PRAY_DECISION_NUMBER = 13

DECISIONS = {0: ('move', 'n'),
             1: ('move', 'e'),
             2: ('move', 's'),
             3: ('move', 'w'),
             4: ('eat', 'n'),
             5: ('eat', 'e'),
             6: ('eat', 's'),
             7: ('eat', 'w'),
             8: ('drink', 'n'),
             9: ('drink', 'e'),
             10: ('drink', 's'),
             11: ('drink', 'w'),
             12: ('remain', None)}





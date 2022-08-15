from src.parameters import *
from src.food import Food
from src.pray import Pray
from src.predator import Predator
from src.humanoid import Humanoid
from dead_body import DeadBody
import random
import collections


class Environment:
    def __init__(self, world_map, num_foods, num_prays, num_predators, num_humanoids, turn, train):

        # initial map of the world
        self.world_map = world_map
        self.rows = world_map.shape[0]
        self.cols = world_map.shape[1]

        # initilaise the state
        self.state = world_map.copy()
        self.state[np.where(self.state < 0)] = 0

        # list of creatures
        self.alive = []             # list of names of creatures that are alive

        # correspondences between location and name
        self.loc_to_name = {}
        self.name_to_loc = {}
        self.name_to_agent = {}
        self.name_to_killed = {}            # agent killed in current turn

        # dead bodies, (location, turn)
        self.dead_bodies = []

        # food correspondences, location: food
        self.loc_to_food = {}

        # place food
        for _ in range(num_foods):
            self.place_food(FOOD_AREA1)
            self.place_food(FOOD_AREA2)
            self.place_food(FOOD_AREA3)

        # list of names of all creatures
        self.num_creatures = 0

        # place prays
        for _ in range(num_prays):
            self.place_prays()

        # place predators
        for _ in range(num_predators):
            self.place_predators()

        # place humanoids
        for _ in range(num_humanoids):
            self.place_humanoids()

        # turn number
        self.turn = turn

        # train state
        self.train = train

    def refresh(self, num_foods, num_prays, num_predators, num_humanoids):

        # initialise the state
        self.state = self.world_map.copy()
        self.state[np.where(self.state < 0)] = 0

        # list of creatures
        self.alive = []  # list of names of creatures that are alive

        # correspondences between location and name
        self.loc_to_name = {}
        self.name_to_loc = {}
        self.name_to_agent = {}
        self.name_to_killed = {}  # agent killed in current turn

        # dead bodies, (location, turn)
        self.dead_bodies = []

        # food correspondences, location: food
        self.loc_to_food = {}

        # place food
        for _ in range(num_foods):
            self.place_food(FOOD_AREA1)
            self.place_food(FOOD_AREA2)
            self.place_food(FOOD_AREA3)

        # list of names of all creatures
        self.num_creatures = 0

        # place prays
        for _ in range(num_prays):
            self.place_prays()

        # place predators
        for _ in range(num_predators):
            self.place_predators()

        # place humanoids
        for _ in range(num_humanoids):
            self.place_humanoids()

        # turn number
        self.turn = 0

    def random_empty_loc(self):
        # return random empty location in state
        row = random.randint(0, self.rows - 1)
        col = random.randint(0, self.cols - 1)

        # repeat until find the empty pos
        while self.state[row, col] != EMPTY:
            row = random.randint(0, self.rows - 1)
            col = random.randint(0, self.cols - 1)

        return row, col

    def place_food(self, area_name):
        # random empty location
        row, col = self.random_empty_loc()
        while self.world_map[row, col] != area_name:
            row, col = self.random_empty_loc()

        # nutrients
        if area_name == FOOD_AREA1:
            nutrientA = random.uniform(NUTRIENT_A_MIN1, NUTRIENT_A_MAX1)
            nutrientB = random.uniform(NUTRIENT_B_MIN1, NUTRIENT_B_MAX1)
        elif area_name == FOOD_AREA2:
            nutrientA = random.uniform(NUTRIENT_A_MIN2, NUTRIENT_A_MAX2)
            nutrientB = random.uniform(NUTRIENT_B_MIN2, NUTRIENT_B_MAX2)
        elif area_name == FOOD_AREA3:
            nutrientA = random.uniform(NUTRIENT_A_MIN3, NUTRIENT_A_MAX3)
            nutrientB = random.uniform(NUTRIENT_B_MIN3, NUTRIENT_B_MAX3)

        # update state
        if area_name == FOOD_AREA1:
            self.state[row, col] = FOOD1
        if area_name == FOOD_AREA2:
            self.state[row, col] = FOOD2
        if area_name == FOOD_AREA3:
            self.state[row, col] = FOOD3

        food = Food((row, col), nutrientA, nutrientB)

        # add to dictionary
        self.loc_to_food[(row, col)] = food

    def place_prays(self):
        # TODO improve nutrient setting
        # name
        name = self.num_creatures
        self.num_creatures += 1

        # random empty location
        row, col = self.random_empty_loc()

        # nutrients
        nutrientA = random.uniform(NUTRIENT_A_PRAY_MIN, NUTRIENT_A_PRAY_MAX)
        nutrientB = random.uniform(NUTRIENT_B_PRAY_MIN, NUTRIENT_B_PRAY_MAX)

        # update state
        self.state[row, col] = PRAY

        # list as alive
        self.alive.append(name)

        # append to correspondences dictionary
        self.name_to_loc[name] = (row, col)
        self.loc_to_name[(row, col)] = name

        # create an instance
        pray = Pray(name, (row, col), nutrientA, nutrientB, self.state)

        # append to dictionary
        self.name_to_agent[name] = pray

    def place_predators(self):
        # TODO improve nutrient setting
        # name
        name = self.num_creatures
        self.num_creatures += 1

        # random empty location
        row, col = self.random_empty_loc()

        # nutrients
        nutrientA = random.uniform(NUTRIENT_A_PREDATOR_MIN, NUTRIENT_A_PREDATOR_MAX)
        nutrientB = random.uniform(NUTRIENT_B_PREDATOR_MIN, NUTRIENT_B_PREDATOR_MIN)

        # update state
        self.state[row, col] = PREDATOR

        # list as alive
        self.alive.append(name)

        # append to correspondences dictionary
        self.name_to_loc[name] = (row, col)
        self.loc_to_name[(row, col)] = name

        # create an instance
        predator = Predator(name, (row, col), nutrientA, nutrientB, self.state)
        self.name_to_agent[name] = predator

    def place_humanoids(self):
        # TODO improve nutrient setting
        # name
        name = self.num_creatures
        self.num_creatures += 1

        # random empty location
        row, col = self.random_empty_loc()

        # nutrients
        nutrientA = random.random()
        nutrientB = random.random()

        # update state
        self.state[row, col] = HUMANOID

        # list as alive
        self.alive.append(name)

        # append to correspondences dictionary
        self.name_to_loc[name] = (row, col)
        self.loc_to_name[(row, col)] = name

        # create an instance
        humanoid = Humanoid(name, (row, col), nutrientA, nutrientB, self.state)
        self.name_to_agent[name] = humanoid

    def eat_food(self, loc, agent_name):
        # get food
        food = self.loc_to_food[loc]

        # update state
        row, col = loc
        if self.state[row, col] == DEAD_BODY:
            for dead_body in self.dead_bodies:
                if dead_body[0] == loc:
                    self.dead_bodies.remove(dead_body)

        self.state[row, col] = EMPTY

        # update dictionary
        del self.loc_to_food[loc]

        # update agent
        agent = self.name_to_agent[agent_name]
        agent.hunger = 1.0
        agent.nutrientA = min(1.0, agent.nutrientA + food.nutrientA)
        agent.nutrientB = min(1.0, agent.nutrientB + food.nutrientB)
        if food.poisonous:
            self.set_dead(agent_name)

    def pick_item(self, loc, agent_name):
        row, col = loc
        agent = self.name_to_agent[agent_name]

        # pick
        item = self.loc_to_food[loc]
        agent.possessions.append(item)

        # update state
        self.state[row, col] = EMPTY

    def set_dead(self, agent_name):
        # sets dead
        agent = self.name_to_agent[agent_name]
        agent.alive = False

        # copy properties
        dead_body = DeadBody((agent.row, agent.col), agent.body_nutrientA, agent.body_nutrientB, poisonous=False)

        # append to food list
        self.loc_to_food[(agent.row, agent.col)] = dead_body

        # get location
        loc = self.name_to_loc[agent_name]

        # add to dictionary
        self.dead_bodies.append((loc, self.turn))

        # delete from dictionaries
        del self.loc_to_name[loc]
        del self.name_to_loc[agent_name]
        del self.name_to_agent[agent_name]

        # delete from the list
        for i, name in enumerate(self.alive):
            if name == agent_name:
                del self.alive[i]

        # add to killed
        self.name_to_killed[agent_name] = agent

        # update state
        self.state[agent.row, agent.col] = DEAD_BODY

    def move_agent(self, agent_name, new_loc):

        # get values
        new_row, new_col = new_loc
        agent = self.name_to_agent[agent_name]

        # old location
        old_row = agent.row
        old_col = agent.col
        # update
        agent.row = new_row
        agent.col = new_col

        # update dictionaries
        #TODO loc_to_name KeyError
        #TODO: seems to be it’s trying to move to occupied location

        del self.loc_to_name[(old_row, old_col)]
        self.loc_to_name[(new_row, new_col)] = agent_name
        self.name_to_loc[agent_name] = (new_row, new_col)

    def battle(self, agent1, agent2):
        # TODO: more sophisticated version of this
        # both dead
        self.set_dead(agent1)
        self.set_deat(agent2)

    def out_of_bound(self, new_loc):
        new_row, new_col = new_loc

        if new_row < 0 or new_row >= self.rows or new_col < 0 or new_col >= self.cols:
            return True
        else:
            return False

    def check_adjacent(self, loc1, loc2):
        # check if loc2 is adjacent to loc1

        row1, col1 = loc1

        adjacent = []
        for d_row, d_col in ADJACENT:
            adjacent.append((d_row + row1, d_col + col1))

        if loc2 in adjacent:
            return True
        else:
            return False

    def map_modify(self):
        # modify the map slightly
        pass

    def apply_attacks(self, attacks, action_parameters):
        # kills victims specified in parameters
        # if the attack is mutual, battle happens

        # get list of attack targets
        attack_list = []
        for attacker_name in attacks:
            attack_list.append(action_parameters[attacker_name])

        # get names of victims from locations
        victims = []
        for row, col in attack_list:
            if not self.out_of_bound((row, col)):
                if self.state[row, col] in (PRAY, PREDATOR, HUMANOID):
                    victims.append(self.loc_to_name[(row, col)])

        #TODO check for same attacks (KeyError)

        # check for mutual attacks
        for attacker in attack_list:
            if attacker in victims:
                victim = self.loc_to_name[action_parameters[attacker]]
                self.battle(attacker, victim)

        # set the rest as dead
        for victim in victims:
            self.set_dead(victim)

    def apply_moves(self, moves, action_parameters):
        # apply both moves and runs

        # update moves list
        dead = []
        for mover_name in moves:
            if mover_name not in self.alive:
                dead.append(mover_name)
        for i, mover_name in enumerate(moves):
            if mover_name in dead:
                del moves[i]

        # get list of destinations
        destinations = []
        for mover_name in moves:
            destinations.append(action_parameters[mover_name])

        # check if valid
        valid = []
        for destination in destinations:
            if not self.out_of_bound(destination):
                row, col = destination
                if self.state[row, col] == EMPTY:
                    valid.append((row, col))

        # check if more than one ants are heading to the same location
        same_loc = [loc for loc, count in collections.Counter(destinations).items() if count > 1]

        # apply movements

        # erase first
        for i, mover_name in enumerate(moves):
            if destinations[i] in valid:
                if destinations[i] not in same_loc:
                    # erase
                    mover = self.name_to_agent[mover_name]
                    row, col = mover.row, mover.col
                    self.state[row, col] = EMPTY

        # move the agents
        for i, mover_name in enumerate(moves):
            if destinations[i] in valid:
                if destinations[i] not in same_loc:
                    # move
                    new_row, new_col = destinations[i]
                    mover = self.name_to_agent[mover_name]
                    self.state[new_row, new_col] = mover.type

                    # tell the agent
                    self.move_agent(mover_name, (new_row, new_col))

    def fight_for_food(self, agent_names):
        # TODO: make this more sophisticated
        winner_index = random.randint(0, len(agent_names) - 1)

        return agent_names[winner_index]

    def apply_eats(self, eats, action_parameters):
        # check if edible stuff is in vicinity
        # if true, eat the food

        #TODO: remove this later
        successful = []

        # update moves list
        dead = []
        for eater_name in eats:
            if eater_name not in self.alive:
                dead.append(eater_name)
        for i, mover_name in enumerate(eats):
            if mover_name in dead:
                del eats[i]

        # list of eat attempts
        eat_attempts = []

        for eater_name in eats:
            eater = self.name_to_agent[eater_name]

            # sort by agent type

            # pray only eats food
            if eater.type == PRAY:
                food_row, food_col = action_parameters[eater_name]
                if not self.out_of_bound((food_row, food_col)):
                    if self.state[food_row, food_col] in (FOOD1, FOOD2, FOOD3):
                        eat_attempts.append((food_row, food_col))

            # predator only dead bodies
            if eater.type == PREDATOR:
                agent_row, agent_col = action_parameters[eater_name]
                if not self.out_of_bound((agent_row, agent_col)):
                    if self.state[agent_row, agent_col] == DEAD_BODY:
                        eat_attempts.append((agent_row, agent_col))

            # humanoid eats food and dead bodies
            if eater.type == HUMANOID:
                edible_row, edible_col = action_parameters[eater_name]
                if not self.out_of_bound((edible_row, edible_col)):
                    if self.state[edible_row, edible_col] in (FOOD1, FOOD2, FOOD3, DEAD_BODY):
                        eat_attempts.append((edible_row, edible_col))

        # check if the same food is being eaten
        same_food = [loc for loc, count in collections.Counter(eat_attempts).items() if count > 1]

        # reverse the dictionary to get a multidict,
        rev_multidict = {}
        for name, food_loc in action_parameters.items():
            rev_multidict.setdefault(food_loc, set()).add(name)

        # apply eating
        for eater_name in eats:
            food_loc = action_parameters[eater_name]

            if food_loc in eat_attempts:
                # both edible and sole eater

                if food_loc not in same_food:
                    # successful
                    self.eat_food(food_loc, eater_name)

                    #TODO remove later
                    successful.append(eater_name)

                # edible but there is a rival
                elif food_loc in same_food:
                    pass
                    # list of rivals
                    '''
                    conflicts = [names for food_loc, names in rev_multidict.items() if len(names) > 1]
                    for conflict in conflicts:
                        winner_name = self.fight_for_food(conflict) #TODO something wrong here

                        #debug
                        print(conflicts)

                        # update agent state
                        self.eat_food(food_loc, winner_name)

                        #TODO remove later
                        successful.append(winner_name)
                    '''

        return successful

    def apply_pick_ups(self, pick_ups, action_parameters):
        # pick up food

        # update moves list
        dead = []
        for picker_name in pick_ups:
            if picker_name not in self.alive:
                dead.append(picker_name)
        for i, mover_name in enumerate(pick_ups):
            if mover_name in dead:
                del pick_ups[i]

        # pick up attempts
        pick_up_attempts = []

        # only food and dead bodies can be picked up
        for pick_up_name in pick_ups:
            item_row, item_col = action_parameters[pick_up_name]
            if not self.out_of_bound((item_row, item_col)):
                if self.state[item_row, item_col] in (FOOD1, FOOD2, FOOD3, DEAD_BODY):
                    pick_up_attempts.append(action_parameters[pick_up_name])

        # check if the same food is being eaten
        same_item = [loc for loc, count in collections.Counter(pick_up_attempts).items() if count > 1]

        # reverse the dictionary to get a multidict,
        rev_multidict = {}
        for name, item_loc in action_parameters.items():
            rev_multidict.setdefault(item_loc, set()).add(name)

        # apply pick up
        for picker_name in pick_ups:
            item_loc = action_parameters[picker_name]
            picker = self.name_to_agent[picker_name]

            if item_loc in pick_up_attempts:
                # if no one else picks and has enough space
                if item_loc not in same_item and len(picker.possessions) < MAX_POSSESSION:
                    # successful
                    self.pick_item(item_loc, picker_name)

                # there is a rival
                elif item_loc in same_item and len(picker.possessions) < MAX_POSSESSION:
                    # fight for item
                    names = [names for food_loc, names in rev_multidict.items() if len(names) > 1]
                    winner_name = self.fight_for_food(names)

                    # update agent state
                    self.pick_item(item_loc, winner_name)

    def apply_drinks(self, drinks, action_parameters):

        # update drinks list
        dead = []
        for drinker_name in drinks:
            if drinker_name not in self.alive:
                dead.append(drinker_name)
        for i, mover_name in enumerate(drinks):
            if mover_name in dead:
                del drinks[i]

        # if specified location is water then drink
        for drinker_name in drinks:
            row, col = action_parameters[drinker_name]
            if not self.out_of_bound((row, col)):
                if self.state[row, col] == WATER:
                    drinker = self.name_to_agent[drinker_name]
                    drinker.thirst = 1.0

    def apply_speaks(self, speaks, action_parameters, words):
        #TODO
        pass

    def get_action_index(self, action, parameter):
        print(action)
        print(parameter)

        if action == 'move':
            if parameter == 'n':
                action_index = 0
            if parameter == 'e':
                action_index = 1
            if parameter == 's':
                action_index = 2
            if parameter == 'w':
                action_index = 3
        if action == 'eat':
            if parameter == 'n':
                action_index = 4
            if parameter == 'e':
                action_index = 5
            if parameter == 's':
                action_index = 6
            if parameter == 'w':
                action_index = 7
        if action == 'drink':
            if parameter == 'n':
                action_index = 8
            if parameter == 'e':
                action_index = 9
            if parameter == 's':
                action_index = 10
            if parameter == 'w':
                action_index = 11
        if action == 'remain':
            action_index = 12

        return action_index


    def update_state(self, pray_dqn, predator_dqn, humanoid_dqn):
        # updates state

        # total rewards
        R = 0

        # eat successful #TODO, to be reconsidered later
        eat_successful = []

        # modify map
        self.map_modify()

        # place food
        if self.turn % FOOD_RATE1 == 0:
            self.place_food(FOOD_AREA1)
        if self.turn % FOOD_RATE2 == 0:
            self.place_food(FOOD_AREA2)
        if self.turn % FOOD_RATE3 == 0:
            self.place_food(FOOD_AREA3)

        # get action attempts
        action_attempts = {}
        action_parameters = {}
        action_words = {}

        # acted agents
        acted = []

        # dictionary for state, action, reward, state'
        name_to_s = {}
        name_to_a = {}
        name_to_r = {}

        for name in self.alive:
            agent = self.name_to_agent[name]
            if agent.type == PRAY:
                # get visible area
                visible = agent.get_visible(self.state)

                # save s to dictionary
                name_to_s[name] = pray_dqn.convert(visible)

                # act
                if pray_dqn.steps < OBSERVE and self.train:
                    action_index, action, parameter, words = agent.act(visible, pray_dqn)
                else:
                    action_index, action, parameter, words = agent.act(visible, pray_dqn, observe=False)

                # save a to dictionary
                name_to_a[name] = action_index

            if agent.type == PREDATOR:
                # get visible area
                visible = agent.get_visible(self.state)

                # save s to dictionary
                name_to_s[name] = predator_dqn.convert(visible)

                # act
                if predator_dqn.steps < OBSERVE and self.train:
                    action, parameter, words = agent.act(visible, predator_dqn)
                else:
                    action, parameter, words = agent.act(visible, predator_dqn, observe=False)

                # save a to dictionary
                name_to_a[name] = self.get_action_index(action, parameter)

            if agent.type == HUMANOID:
                # get visible area
                visible = agent.get_visible(self.state)

                # save s to dictionary
                name_to_s[name] = humanoid_dqn.convert(visible)

                # act
                if humanoid_dqn.steps < OBSERVE and self.train:
                    action, parameter, words = agent.act(visible, humanoid_dqn)
                else:
                    action, parameter, words = agent.act(visible, humanoid_dqn, observe=False)

                # save a to dictionary
                name_to_a[name] = self.get_action_index(action, parameter)

            action_attempts[name] = action
            action_parameters[name] = parameter
            action_words[name] = words

        attacks = []
        moves = []
        eats = []
        pick_ups = []
        drinks = []
        speaks = []

        # sort attempts
        for agent in self.alive:
            # remember who acted
            acted.append(agent)

            if action_attempts[agent] == 'attack':
                attacks.append(agent)
            elif action_attempts[agent] in ('move', 'run'):
                moves.append(agent)
            elif action_attempts[agent] == 'eat':
                eats.append(agent)
            elif action_attempts[agent] == 'pick_up':
                pick_ups.append(agent)
            elif action_attempts[agent] == 'drink':
                drinks.append(agent)
            elif action_attempts[agent] == 'speak':
                speaks.append(agent)

        # apply attacks
        self.apply_attacks(attacks, action_parameters)

        # apply moves
        self.apply_moves(moves, action_parameters)

        # apply eats
        #TODO remove later
        eat_successful = self.apply_eats(eats, action_parameters)

        # apply pick ups
        self.apply_pick_ups(pick_ups, action_parameters)

        # apply drinks
        self.apply_drinks(drinks, action_parameters)

        # apply speaks
        self.apply_speaks(speaks, action_parameters, action_words)

        # update state
        for agent_name in self.alive:
            agent = self.name_to_agent[agent_name]
            agent.state_update()

        # check for health
        for agent_name in self.alive:
            agent = self.name_to_agent[agent_name]
            if min(agent.hunger, agent.thirst, agent.nutrientA, agent.nutrientB) <= 0:
                self.set_dead(agent_name)

        # delete dead bodies
        decomposed = []
        for dead_body in self.dead_bodies:
            if self.turn - dead_body[1] > DEAD_BODY_DECOMPOSE:
                decomposed.append(dead_body[0])

        for row, col in decomposed:
            if not self.out_of_bound((row, col)):
                self.state[row, col] = EMPTY

                # remove from the list
                for dead_body in self.dead_bodies:
                    if dead_body[0] == (row, col):
                        self.dead_bodies.remove(dead_body)

        # append reward and state'
        for name in acted:
            if name in self.alive:
                # s
                s = name_to_s[name]

                # a
                a = name_to_a[name]

                # if alive reward 0
                #TODO remove later
                if name in eat_successful:
                    r = 1
                    R += r
                    s_ = None
                else:
                    name_to_r[name] = 0
                    r = name_to_r[name]
                    R += r

                    # cutoff after 15
                    if self.turn % 15 == 0:
                        r = -1
                        R += r
                        s_ = None
                    else:
                        # state'
                        agent = self.name_to_agent[name]
                        visible = agent.get_visible(self.state)
                        s_ = pray_dqn.convert(visible)

                # agent type
                agent = self.name_to_agent[name]
                if agent.type == PRAY:
                    pray_dqn.observe((s, a, r, s_))
                if agent.type == PREDATOR:
                    predator_dqn.observe((s, a, r, s_))
                if agent.type == HUMANOID:
                    humanoid_dqn.observe((s, a, r, s_))

            else:
                # agent type
                agent = self.name_to_killed[name]

                # s
                s = name_to_s[name]

                # a
                a = name_to_a[name]

                r = 0
                s_ = None
                R += r

                if agent.type == PRAY:
                    pray_dqn.observe((s, a, r, s_))
                if agent.type == PREDATOR:
                    predator_dqn.observe((s, a, r, s_))
                if agent.type == HUMANOID:
                    humanoid_dqn.observe((s, a, r, s_))

        # replay
        if pray_dqn.steps > OBSERVE and self.train:
            print('Pray training...')
            pray_dqn.replay()
        if predator_dqn.steps > OBSERVE and self.train:
            print('Predator training...')
            predator_dqn.replay()
        if humanoid_dqn.steps > OBSERVE and self.train:
            print('Humanoid training...')
            humanoid_dqn.replay()

        # reset the killed dictionary
        self.name_to_killed = {}

        return R


    def debug(self, comments):
        # makes sure reverse dictionary corresponds to the original
        loc_to_name = {loc: name for name, loc in self.name_to_loc.items()}
        name_to_loc = {name: loc for loc, name in self.loc_to_name.items()}

        if not loc_to_name == self.loc_to_name:
            input(comments)
            print(loc_to_name)
            print(self.loc_to_name)
        if not name_to_loc == self.name_to_loc:
            input(comments)
            print(name_to_loc)
            print(self.name_to_loc)

    def debug2(self, comments):
        # makes sure state corresponds to name_to_agent
        for row in range(self.rows):
            for col in range(self.cols):
                occupier_type = self.state[row, col]
                if (row, col) in self.loc_to_name:
                    occupier_name = self.loc_to_name[(row, col)]
                    occupier = self.name_to_agent[occupier_name]
                    if occupier_type != occupier.type:
                        print(comments)
                        print(occupier_type)
                        print(occupier.type)
                        input('end')

    def debug3(self, comments):
        # makes sure name_to_loc and name_to_agent are in sync
        for name in self.alive:
            loc = self.name_to_loc[name]
            agent = self.name_to_agent[name]
            if loc != (agent.row, agent.col):
                print(comments)
                input('end')

    def debug4(self, comments):
        # makes sure dead_bodies is updated properly
        for dead_body in self.dead_bodies:
            row, col = dead_body[0]
            if self.state[row, col] != DEAD_BODY:
                print(comments)
                input('end')
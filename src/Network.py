from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from parameters import *
import os
import math


class Brain:
    def __init__(self, decision_number):
        # models
        self.model = self.create_model(decision_number)
        self.target_model = self.create_model(decision_number)

    def create_model(self, decision_number):

        # bodily state
        bodily_state = Input(shape=(5,))
        hidden1 = Dense(shape=256, activation='relu')(bodily_state)
        hidden2 = Dense(shape=1024, activation='relu')(hidden1)
        hidden3 = Dense(shape=1024, activation='relu')(hidden2)

        # visual input
        vision = Input(shape=(INPUT_SIZE, INPUT_SIZE, CHANNEL_NUMBER))
        conv1 = Conv2D(32, kernel_size=4, activation='relu')(vision)
        max_pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(64, kernel_size=4, activation='relu')(max_pool1)
        max_pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(32, kernel_size=4, activation='relu')(max_pool2)
        max_pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        flat = Flatten()(max_pool3)

        # merged dense layer
        merge = concatenate([hidden3, flat])
        hidden4 = Dense(1024, activation='relu')(merge)
        hidden5 = Dense(1024, activation='relu')(hidden4)
        output = Dense(decision_number, activation=None)(hidden5)

        # model
        model = Model(inputs=[vision, bodily_state], outputs=output)

        # plot graph
        plot_model(model, to_file='model_structure.png')

        # optimizer
        opt = Adam(lr=L_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        # compile
        model.compile(loss='mse', optimizer=opt)

        return model

    def print_summary(self):
        # summarize layers
        print(self.model.summary)

    def plot_graph(self):
        # plot graph
        plot_model(self.model, to_file='model_structure.png')

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=BATCH_SIZE, epochs=epochs, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.target_model.predict(s)
        else:
            return self.model.predict(s)

    def predict_one(self, s, target=False):
        return self.predict(s, target=target).flatten()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        print('target updated')


class Memory:
    def __init__(self, capacity):
        self.samples = []
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def is_full(self):
        return len(self.samples) >= self.capacity


class Network:
    def __init__(self, decision_number, type, train):
        self.steps = 0
        self.epsilon = MAX_EPSILON

        self.decision_number = decision_number

        self.brain = Brain(decision_number)
        self.memory = Memory(MEMORY_CAPACITY)
        self.train = train
        if not self.train:
            # create weights directory
            pray_path = os.path.join(os.path.join(weights_dir, pray_dir), weights_file)
            predator_path = os.path.join(os.path.join(weights_dir, predator_dir), weights_file)
            humanoid_path = os.path.join(os.path.join(weights_dir, humanoid_dir), weights_file)

            if type == PRAY and os.path.exists(pray_path):
                self.brain.model.load_weights(pray_path)
                print('weights loaded')
            if type == PREDATOR and os.path.exists(predator_path):
                self.brain.model.load_weights(predator_path)
                print('weights loaded')
            if type == HUMANOID and os.path.exists(humanoid_path):
                self.brain.model.load_weights(humanoid_path)
                print('weights loaded')

    def decide(self, visible, bodily_state):
        if not self.train:
            self.epsilon = 0.01
        if random.random() < self.epsilon:
            return random.randint(0, self.decision_number)
        else:
            s = self.convert(visible, bodily_state)
            return np.argmax(self.brain.predict_one(s))

    def observe(self, sample):
        # store data in (s, a, r, s') format
        self.memory.add(sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.update_target_model()

        # slowly decrease Epsilon based on our experience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        # for visual information
        no_state = np.zeros((INPUT_SIZE, INPUT_SIZE, CHANNEL_NUMBER))

        states = np.array([o[0] for o in batch])
        states_ = []
        for o in batch:
            vision, bodily_state = o[3]
            if vision is None:
                states_.append([no_state, bodily_state])
            else:
                states_.append([vision, bodily_state])
        states_ = np.array(states_)
        #states_ = np.array([([no_state, o[3][1]] if o[3][0] is None else o[3]) for o in batch])

        pred = self.brain.predict(states)
        pred_ = self.brain.predict(states_, target=True)

        x = np.zeros((batchLen, INPUT_SIZE, INPUT_SIZE, CHANNEL_NUMBER))
        y = np.zeros((batchLen, self.decision_number))

        for i in range(batchLen):
            o = batch[i]
            s = o[0]
            a = o[1]
            r = o[2]
            s_ = o[3]

            t = pred[i]
            if s_[0] is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * np.amax(pred_[i])

            x[i] = s
            y[i] = t
        self.brain.train(x, y)

        loss_val = self.brain.model.evaluate(x, y)
        print('Loss: %f' % loss_val)

    def convert(self, visible, bodily_state):
        # get visual information first
        vision = np.zeros((INPUT_SIZE, INPUT_SIZE, CHANNEL_NUMBER))

        for row in range(2 * VISIBLE_RADIUS + 1):
            for col in range(2 * VISIBLE_RADIUS + 1):
                if visible[row, col] == WATER:
                    vision[row, col, WATER_CHANNEL] = 1
                if visible[row, col] == FOOD1:
                    vision[row, col, FOOD1_CHANNEL] = 1
                if visible[row, col] == FOOD2:
                    vision[row, col, FOOD2_CHANNEL] = 1
                if visible[row, col] == FOOD3:
                    vision[row, col, FOOD3_CHANNEL] = 1
                if visible[row, col] == PRAY:
                    vision[row, col, PRAY_CHANNEL] = 1
                if visible[row, col] == PREDATOR:
                    vision[row, col, PREDATOR_CHANNEL] = 1
                if visible[row, col] == HUMANOID:
                    vision[row, col, HUMANOID_CHANNEL] = 1
                if visible[row, col] == DEAD_BODY:
                    vision[row, col, DEAD_BODY_CHANNEL] = 1

        # s is a list of visual information and bodily state

        return [vision, bodily_state]


import math
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from src.parameters import *
import os


class Brain:
    def __init__(self, decision_number):
        # models
        self.model = self.create_cnn(decision_number)
        self.target_model = self.create_cnn(decision_number)

    def create_cnn(self, decision_number):
        model = Sequential()

        model.add(Conv2D(
            filters=32,
            input_shape=(INPUT_SIZE, INPUT_SIZE, CHANNEL_NUMBER),
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        ))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        ))

        model.add(Flatten())

        model.add(Dense(units=decision_number))

        opt = Adam(lr=L_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='mse', optimizer=opt)

        return model

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=BATCH_SIZE, epochs=epochs, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.target_model.predict(s)
        else:
            return self.model.predict(s)

    def predict_one(self, s, target=False):
        return self.predict(s.reshape(1, INPUT_SIZE, INPUT_SIZE, CHANNEL_NUMBER), target=target).flatten()
        pass

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

    def decide(self, visible):
        if not self.train:
            self.epsilon = 0
        if random.random() < self.epsilon:
            return random.randint(0, self.decision_number - 1), 'Random'
        else:
            s = self.convert(visible)
            return np.argmax(self.brain.predict_one(s)), 'Decision'

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

        no_state = np.zeros((INPUT_SIZE, INPUT_SIZE, CHANNEL_NUMBER))

        states = np.array([o[0] for o in batch])
        states_ = np.array([(no_state if o[3] is None else o[3]) for o in batch])

        pred = self.brain.predict(states)
        pred_ = self.brain.predict(states_, target=True)
        #pTarget_ = self.brain.predict(states_, target=True)

        x = np.zeros((batchLen, INPUT_SIZE, INPUT_SIZE, CHANNEL_NUMBER))
        y = np.zeros((batchLen, self.decision_number))

        for i in range(batchLen):
            o = batch[i]
            s = o[0]
            a = o[1]
            r = o[2]
            s_ = o[3]

            t = pred[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * np.amax(pred_[i])
                #t[a] = r + GAMMA * pTarget_[i][np.argmax(pred_[i])]  # double DQN

            x[i] = s
            y[i] = t
        self.brain.train(x, y)

        loss_val = self.brain.model.evaluate(x, y)
        print('Loss: %f' % loss_val)

    def convert(self, visible):
        s = np.zeros((INPUT_SIZE, INPUT_SIZE, CHANNEL_NUMBER))

        for row in range(2 * VISIBLE_RADIUS + 1):
            for col in range(2 * VISIBLE_RADIUS + 1):
                if visible[row, col] == WATER:
                    s[row, col, WATER_CHANNEL] = 1
                if visible[row, col] == FOOD1:
                    s[row, col, FOOD1_CHANNEL] = 1
                if visible[row, col] == FOOD2:
                    s[row, col, FOOD2_CHANNEL] = 1
                if visible[row, col] == FOOD3:
                    s[row, col, FOOD3_CHANNEL] = 1
                if visible[row, col] == PRAY:
                    s[row, col, PRAY_CHANNEL] = 1
                if visible[row, col] == PREDATOR:
                    s[row, col, PREDATOR_CHANNEL] = 1
                if visible[row, col] == HUMANOID:
                    s[row, col, HUMANOID_CHANNEL] = 1
                if visible[row, col] == DEAD_BODY:
                    s[row, col, DEAD_BODY_CHANNEL] = 1
        return s


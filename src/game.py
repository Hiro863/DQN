import cv2
from environment import Environment
from DQN import Network
from parameters import *
import os


class Game:
    def __init__(self, world_map, num_foods, num_prays, num_predators, num_humanoids, max_turn, train=True):

        # maximum number of turns
        self.max_turn = max_turn

        # current turn
        self.turn = 0

        # environment
        self.environment = Environment(world_map, num_foods, num_prays, num_predators, num_humanoids, self.turn, train)

        # video frames
        self.video_frame = self.video_frame = np.zeros((GRID_SIZE * self.environment.rows, GRID_SIZE * self.environment.cols, 3))

        # different species as different players
        self.pray_dqn = Network(PRAY_DECISION_NUMBER, PRAY, train)
        self.predator_dqn = Network(PREDATOR_ACTION_NUMBER, PREDATOR, train)
        self.humanoid_dqn = Network(HUMANOID_ACTION_NUMBER, HUMANOID, train)

        # total reward per game
        self.R = 0

        # game setting
        self.num_foods = num_foods
        self.num_prays = num_prays
        self.num_predators = num_predators
        self.num_humanoids = num_humanoids

    def state_to_video(self):
        # convert state to video frame
        self.video_frame[:, :, :] = 0
        for row in range(self.environment.rows):
            for col in range(self.environment.cols):

                # water
                if self.environment.state[row, col] == WATER:
                    for i in range(3):
                        self.video_frame[row * GRID_SIZE: row * GRID_SIZE + GRID_SIZE,
                        col * GRID_SIZE: col * GRID_SIZE + GRID_SIZE, i] = COLORS[WATER][i] / 255

                # food 1
                if self.environment.state[row, col] == FOOD1:
                    for i in range(3):
                        self.video_frame[row * GRID_SIZE: row * GRID_SIZE + GRID_SIZE,
                        col * GRID_SIZE: col * GRID_SIZE + GRID_SIZE, i] = COLORS[FOOD1][i] / 255

                # food 2
                if self.environment.state[row, col] == FOOD2:
                    for i in range(3):
                        self.video_frame[row * GRID_SIZE: row * GRID_SIZE + GRID_SIZE,
                        col * GRID_SIZE: col * GRID_SIZE + GRID_SIZE, i] = COLORS[FOOD2][i] / 255

                # food 3
                if self.environment.state[row, col] == FOOD3:
                    for i in range(3):
                        self.video_frame[row * GRID_SIZE: row * GRID_SIZE + GRID_SIZE,
                        col * GRID_SIZE: col * GRID_SIZE + GRID_SIZE, i] = COLORS[FOOD3][i] / 255

                # pray
                if self.environment.state[row, col] == PRAY:
                    for i in range(3):
                        self.video_frame[row * GRID_SIZE: row * GRID_SIZE + GRID_SIZE,
                        col * GRID_SIZE: col * GRID_SIZE + GRID_SIZE, i] = COLORS[PRAY][i] / 255

                # predator
                if self.environment.state[row, col] == PREDATOR:
                    for i in range(3):
                        self.video_frame[row * GRID_SIZE: row * GRID_SIZE + GRID_SIZE,
                        col * GRID_SIZE: col * GRID_SIZE + GRID_SIZE, i] = COLORS[PREDATOR][i] / 255

                # humanoid
                if self.environment.state[row, col] == HUMANOID:
                    for i in range(3):
                        self.video_frame[row * GRID_SIZE: row * GRID_SIZE + GRID_SIZE,
                        col * GRID_SIZE: col * GRID_SIZE + GRID_SIZE, i] = COLORS[HUMANOID][i] / 255

                # dead body
                if self.environment.state[row, col] == DEAD_BODY:
                    for i in range(3):
                        self.video_frame[row * GRID_SIZE: row * GRID_SIZE + GRID_SIZE,
                        col * GRID_SIZE: col * GRID_SIZE + GRID_SIZE, i] = COLORS[DEAD_BODY][i] / 255

    def visualise(self, out):

        # convert to video
        self.state_to_video()

        # record
        if RECORD:
            video_frame = self.video_frame * 255
            video_frame = video_frame.astype('uint8')
            video_frame = cv2.resize(video_frame, (500, 250))
            out.write(video_frame)

        # show image
        # visualise
        cv2.namedWindow('Virtual World')
        cv2.moveWindow('Virtual World', 20, 20)
        cv2.imshow('Virtual World', self.video_frame[:, :, :])

    def check_terminate(self):
        # terminate if q is pressed
        if cv2.waitKey(PLAY_RATE) & 0xFF == ord('q'):
            return True
        elif self.max_turn < self.turn:
            return True
        elif not self.environment.alive:
            return True
        else:
            return False

    def terminate(self, out):
        # terminate
        out.release()
        cv2.destroyAllWindows()

    def save_weights(self, type):

        # for each species
        pray_path = os.path.join(weights_dir, pray_dir)
        predator_path = os.path.join(weights_dir, predator_dir)
        humanoid_path = os.path.join(weights_dir, humanoid_dir)

        # make directory
        if not os.path.exists(weights_dir):
            os.mkdir(weights_dir)
        if not os.path.exists(pray_path):
            os.mkdir(pray_path)
        if not os.path.exists(predator_path):
            os.mkdir(predator_path)
        if not os.path.exists(humanoid_path):
            os.mkdir(humanoid_path)

        # save weights
        if type == PRAY:
            self.pray_dqn.brain.model.save_weights(os.path.join(pray_path, weights_file))
        if type == PREDATOR:
            self.predator_dqn.brain.model.save_weights(os.path.join(predator_path, weights_file))
        if type == HUMANOID:
            self.humanoid_dqn.brain.model.save_weights(os.path.join(humanoid_path, weights_file))
        print('Weights saved')

    def run_game(self, visualise=False):
        # initialise
        self.turn = 0
        self.R = 0
        self.environment.refresh(self.num_foods, self.num_prays, self.num_predators, self.num_humanoids)

        # record
        fourcc = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')
        out = cv2.VideoWriter('video.avi', fourcc, 10, (500, 250), True)


        # initial state
        if visualise:
            self.visualise(out)


        # main game loop
        while True:

            print('Turn number: [%d] ----------------------------------------------------------' % self.turn)

            # update state
            self.environment.turn = self.turn
            self.R += self.environment.update_state(self.pray_dqn, self.predator_dqn, self.humanoid_dqn)

            # visualise
            if visualise:
                self.visualise(out)


            # turn update
            self.turn += 1

            # check for termination
            if self.check_terminate():
                self.terminate(out)
                break

        # save weights
        if self.pray_dqn.steps > OBSERVE:
            self.save_weights(PRAY)

        if self.predator_dqn.steps > OBSERVE:
            self.save_weights(PREDATOR)

        if self.humanoid_dqn.steps > OBSERVE:
            self.save_weights(HUMANOID)

        # return total reward
        return self.R, self.turn

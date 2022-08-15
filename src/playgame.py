import cv2
from src.game import Game
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from parameters import *
import os


def save_graph_score(R_list, turns_list, epsilon_list):

    fig = plt.figure()

    # average
    avr_R = []
    for R, turns in zip(R_list, turns_list):
        avr_R.append(float(R / turns))

    R_plot = fig.add_subplot(1, 1, 1)
    R_plot.set_xlabel('number of game turns')
    R_plot.set_ylabel('score')
    R_plot.plot(turns_list, R_list, 'b-')
    #R_plot.plot(turns_list, avr_R)

    epsilon_plot = R_plot.twinx()
    epsilon_plot.set_xlabel('number of game turns')
    epsilon_plot.set_ylabel('epsilon')
    epsilon_plot.plot(turns_list, epsilon_list, 'r-')


    fig.tight_layout()

    if not os.path.exists(graphs_dir):
        os.mkdir(graphs_dir)
    graphs_path = os.path.join(graphs_dir, 'score.png')
    plt.savefig(graphs_path)
    print('Graph saved')


def main():

    # game parameters

    img = cv2.imread('map.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.where(img == 75, -4, img)
    img = np.where(img == 150, -5, img)
    img = np.where(img == 221, -3, img)
    img = np.where(img == 29, 1, img)
    num_foods = NUMBER_OF_FOODS
    num_prays = NUMBER_OF_PRAYS
    num_predators = NUMBER_OF_PREDATORS
    num_humanoids = NUMBER_OF_HUMANOIDS
    max_turn = MAX_TURN

    # game results
    total_turn = 0
    R_list = []
    turn_list = []
    epsilon_list = []

    # number of games
    num_games = NUMBER_OF_GAMES

    if TRAIN:
        visualise = False
    else:
        visualise = True

    # define game
    game = Game(img, num_foods, num_prays, num_predators, num_humanoids, max_turn, train=TRAIN)

    # play the game
    for i in range(num_games):
        # play the game
        print('playing game number: %d/%d' % (i, num_games))
        R, turns = game.run_game(visualise=visualise)
        epsilon = game.pray_dqn.epsilon
        total_turn += turns

        # save results
        R_list.append(R)
        turn_list.append(total_turn)
        epsilon_list.append(epsilon)

        # print graph
        save_graph_score(R_list, turn_list, epsilon_list)
        print('Score: %d' % R)


if __name__ == '__main__':
    main()



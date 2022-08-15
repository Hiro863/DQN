environment = Environment()
neural_net = DQN()
def argmax(q_values):
    return value

def remember(s, a, r, s_):
    pass
import random
tuples = []
terminal = 0
GAMMA = 0


def replay():
    current_state, optimal_action, reward, next_state = random.sample(tuples)

    if current_state is terminal:
        Q = reward
    else:
        Q = GAMMA * max(neural_net(next_state))

    neural_net.train(x=current_state, y=Q)


while True:
    current_state = environment.get_state()


    q_values = neural_net(current_state)

    optimal_action = argmax(q_values)

    next_state, reward = environment.playgame(optimal_action)

    remember(current_state, optimal_action, reward, next_state)

    replay()



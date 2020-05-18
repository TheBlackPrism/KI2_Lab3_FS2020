import gym
import random
import dir_maker
import numpy as np
import argparse

from gym import wrappers
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class DQL:
    def __init__(self, environment_size, action_size, epsilon_decay, gamma, learning_rate):
        self.environment_size = environment_size
        self.action_size = action_size
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epochs = 1
        self.verbose = 0
        self.minibatch_size = 30
        self.memory = deque(maxlen=5000)
        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()

        model.add(Dense(64, input_dim=self.environment_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def add_memory(self, state, action, reward, state_prime, done):
        self.memory.append((state, action, reward, state_prime, done))

    def target_model_update(self):
        self.target_model.set_weights(self.model.get_weights())

    def select_action(self, s):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        q = self.model.predict(s)
        return np.argmax(q[0])

    def replay(self):
        """Vectorized method for experience replay"""
        minibatch = random.sample(self.memory, self.minibatch_size)
        minibatch = np.array(minibatch)
        not_done_indices = np.where(minibatch[:, 4] == False)
        y = np.copy(minibatch[:, 2])

        # If minibatch contains any non-terminal states, use separate update rule for those states
        if len(not_done_indices[0]) > 0:
            predict_sprime = self.model.predict(np.vstack(minibatch[:, 3]))
            predict_sprime_target = self.target_model.predict(np.vstack(minibatch[:, 3]))

            # Non-terminal update rule
            y[not_done_indices] += np.multiply(self.gamma,
                                               predict_sprime_target[not_done_indices,
                                                                     np.argmax(predict_sprime[not_done_indices, :][0],
                                                                               axis=1)][0])

        actions = np.array(minibatch[:, 1], dtype=int)
        y_target = self.model.predict(np.vstack(minibatch[:, 0]))
        y_target[range(self.minibatch_size), actions] = y
        self.model.fit(np.vstack(minibatch[:, 0]), y_target, epochs=self.epochs, verbose=self.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Deep Q learning algorithm to solve the Lunar Lander environment')
    parser.add_argument('-l', '--load',
                        help='File containing the model weights', const='weights.h5', nargs='?')
    parser.add_argument('-o', '--outdir',
                        help='Filepath to where the output will be stored (default sequential numbered folder)')
    parser.add_argument('-r', '--render',
                        help='If set, the environment will be rendered', default=False, const=True, nargs='?')
    parser.add_argument('-e', '--episodes',
                        help='Number of episodes the agent will make', default=200, type=int)
    parser.add_argument('-d', '--decay',
                        help='Sets the epsilon decay of the model (default = 0.9)', default=0.9, type=float)
    parser.add_argument('-lr', '--learning',
                        help='Sets the learning rate of the model (default = 0.001)', default=0.001, type=float)
    parser.add_argument('-g', '--gamma',
                        help='Sets the gamma of the model (default = 0.99)', default=0.99, type=float)
    parser.add_argument('-v', '--video',
                        help='If set records the video of the episodes', default=False, const=True, nargs='?')
    args = parser.parse_args()

    np.set_printoptions(precision=2)

    env = gym.make('LunarLander-v2')

    if args.outdir:
        out_dir = args.outdir
    else:
        out_dir = dir_maker.make_sequential_dir("DQL")

    print('\nSaving Results to: ' + str(out_dir) + "\n")

    if args.video:
        env = wrappers.Monitor(env, out_dir, video_callable=lambda episode_id: True, force=True)
    else:
        env = wrappers.Monitor(env, out_dir, video_callable=False, force=True)

    environment_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQL(environment_size, action_size, epsilon_decay=args.decay, gamma=args.gamma, learning_rate=args.learning)

    episodes = args.episodes

    with open(out_dir / "model_parameters.txt", "w") as f:
        f.write("Model Parameters:")
        f.write("\nepisodes:\t" + str(episodes))
        f.write("\nepsilon decay:\t" + str(agent.epsilon_decay))
        f.write("\ngamma:\t\t" + str(agent.gamma))
        f.write("\nlearning rate:\t" + str(agent.learning_rate))

    if args.load:
        agent.model.load_weights(args.load)
        agent.epsilon = 0

    # Cumulative reward
    reward_avg = deque(maxlen=100)

    for e in range(episodes):
        episode_reward = 0
        time = 0
        max_time_per_game = 1000
        state = env.reset()
        state = np.reshape(state, [1, environment_size])

        for time in range(max_time_per_game):
            if args.render:
                env.render()

            # Query next action from learner and perform action
            action = agent.select_action(state)
            state_prime, reward, done, info = env.step(action)

            # Time debuf to avoid that the algorithm floats over the target
            # if time > max_time_per_game / 2:
            #    reward -= (max_time_per_game / float(max_time_per_game - time)) * 20

            # Add cumulative reward
            episode_reward += reward

            # Reshape new state
            state_prime = np.reshape(state_prime, [1, environment_size])

            # Add experience to memory
            if not args.load:
                agent.add_memory(state, action, reward, state_prime, done)

            # Set current state to new state
            state = state_prime

            # Perform experience replay if memory length is greater than minibatch length
            if not args.load:
                if len(agent.memory) > agent.minibatch_size:
                    agent.replay()

            # If episode is done, exit loop
            if done:
                if not args.load:
                    agent.target_model_update()
                break

        # epsilon decay
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        # Running average of past 100 episodes
        reward_avg.append(episode_reward)
        print('episode: ', e, ' score: ', '%.2f' % episode_reward, ' avg_score: ', '%.2f' % np.average(
            reward_avg), ' frames: ', time, ' epsilon: ', '%.2f' % agent.epsilon)

    if not args.load:
        agent.model.save_weights(str(out_dir / 'weights.h5'))

    env.close()

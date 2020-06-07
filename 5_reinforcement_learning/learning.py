import gym
import time
import numpy as np
import cv2

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
import random
from collections import deque


SHOW_STATS_EVERY = 10
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 32
UPDATE_TARGET_EVERY = 50  # Co ile aktualizowac target_model
MODEL_NAME = 'model_rl'
EPISODES = 1000  # Ile gier ma byc wykonanych
SAVE_EVERY = 50

# Wymiary wejscia - przeskalowany obraz i liczba klatek
IN_X = 80
IN_Y = 105
FRAMES_NO = 3

# Exploracja
epsilon = 1
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.001


class DQNAgent:
    def __init__(self):

        # Model glowny
        self.in_shape = (IN_Y, IN_X, FRAMES_NO)
        self.model = self.create_model()

        # Target model
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # Experience replay
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.taget_update_counter = 0

    def create_model(self):
        """
        Stworzenie modelu przwidujacego q-wartosci na podstawie obserwcji: terazniejszej i poprzednich
        """
        model = Sequential()

        model.add(Conv2D(64, (8, 8), strides=(4, 4), input_shape=self.in_shape))
        model.add(Activation("relu"))

        model.add(Conv2D(64, (4, 4), strides=(2, 2)))
        model.add(Activation("relu"))

        model.add(Flatten())
        model.add(Dense(128, activation="relu"))

        model.add(Dense(env.action_space.n - 1, activation="linear"))  # 3 wyjscia - dla akcji 1, 2, 3
        model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255.0)[0]

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)  # Stworzenie batcha

        current_states = np.array([transition[0] for transition in minibatch]) / 255.0
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch]) / 255.0
        future_qs_list = self.target_model.predict(new_current_states)

        X = []  # Stan (obecny obraz oraz poprzednie)
        y = []  # Akcja o najwiekszej nagrodzie

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q  # ustalenie nowej q-value
            else:
                new_q = reward

            current_qs = current_qs_list[index]  # Lista q-wartosci dla wybranego stanu
            current_qs[action] = new_q  # poprawa q-wartosci dla wykonanej akcji

            X.append(current_state)  # obecne wejscie (obraz terazniejszy i poprzednie)
            y.append(current_qs)  # lista list q-wartosci (duza lista, wartosci dla kadej akcji)

        self.model.fit(np.array(X) / 255.0, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0)

        if terminal_state:
            self.taget_update_counter += 1

        if self.taget_update_counter > UPDATE_TARGET_EVERY:  # uaktualnianie target_model (by byl taki jak glowny)
            self.target_model.set_weights(self.model.get_weights())
            self.taget_update_counter = 0


env = gym.make('Breakout-v0')

ep_reward_list = []
observation = env.reset()

agent = DQNAgent()

for episode in range(EPISODES):
    prev_lives = 5
    episode_reward = 0
    step = 1
    prev_states = np.zeros((IN_Y, IN_X, FRAMES_NO))
    current_state = env.reset()
    done = False

    current_state = cv2.cvtColor(current_state, cv2.COLOR_BGR2GRAY)  # Zamiana na grayscale
    current_state = cv2.resize(current_state, (IN_X, IN_Y))

    # Poprzednie obserwacje - na poczatku ta sama powtorzona
    prev_states[:, :, 0] = current_state
    prev_states[:, :, 1] = current_state
    prev_states[:, :, 2] = current_state
    current_state = prev_states

    # Petla odpowiedzialna za wykonanie jednej gry
    while not done:

        # Wybranie akcji
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, env.action_space.n-1)

        # Wykonanie akcji
        observation, reward, done, info = env.step(action+1)

        # Utrata zycia - narzucenie akcji 1 i ujemna nagroda
        if prev_lives > info['ale.lives']:
            observation, reward, done, info = env.step(1)
            reward = -1.0

        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)  # Zamiana na grayscale
        observation = cv2.resize(observation, (IN_X, IN_Y))  # Przeskalowanie

        # Uaktualnieni poprzednich obserwacji
        prev_states[:, :, :-1] = prev_states[:, :, 1:]
        prev_states[:, :, -1] = observation
        new_state = prev_states

        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1
        episode_reward += reward
        prev_lives = info['ale.lives']

    if episode % SHOW_STATS_EVERY == 0 and episode > 0:
        ep_reward_list.append(episode_reward)

        average_reward = sum(ep_reward_list[-SHOW_STATS_EVERY:]) / len(ep_reward_list[-SHOW_STATS_EVERY:])
        min_reward = min(ep_reward_list[-SHOW_STATS_EVERY:])
        max_reward = max(ep_reward_list[-SHOW_STATS_EVERY:])

        print(f"episodes:{episode-SHOW_STATS_EVERY} - {episode}")
        print(f"average_reward:{average_reward}  min_reward: {min_reward}  max_reward: {max_reward}")
    if episode % SAVE_EVERY == 0 and episode > 0:
        agent.model.save(
            f'models/{MODEL_NAME}__t__{int(time.time())}.h5')
        print("model_saved")

    # Zmiany w eksploracji
    epsilon *= EPSILON_DECAY
    epsilon = max(MIN_EPSILON, epsilon)


agent.model.save(f'models/{MODEL_NAME}__t__{int(time.time())}.h5')




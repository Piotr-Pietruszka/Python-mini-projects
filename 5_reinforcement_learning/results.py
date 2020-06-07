# ==== AI - rezultaty ===

import gym
import time
import numpy as np
import cv2
from tensorflow.keras.models import load_model

IN_X = 80
IN_Y = 105
FRAMES_NO = 3

MODEL_NAME = 'models/model_rl__t__1585675913.h5'
model = load_model(MODEL_NAME)


env = gym.make('Breakout-v0')
current_state = env.reset()
done = False
t = 0

prev_states = np.zeros((IN_Y, IN_X, FRAMES_NO))
current_state = cv2.cvtColor(current_state, cv2.COLOR_BGR2GRAY)  # Zamiana na grayscale
current_state = cv2.resize(current_state, (IN_X, IN_Y))  # Przeskalowanie

# Poprzednie obserwacje - na poczatku ta sama powtorzona
prev_states[:, :, 0] = current_state
prev_states[:, :, 1] = current_state
prev_states[:, :, 2] = current_state

current_state = prev_states
prev_lives = 5
while not done:
    env.render()

    qs_list = model.predict(np.array(current_state).reshape(-1, *current_state.shape) / 255.0)[0]
    action = np.argmax(qs_list)

    if t == 0:
        observation, reward, done, info = env.step(1)
    else:
        observation, reward, done, info = env.step(action+1)  # +1, gdyz pomijana jest akcja 0

    if prev_lives > info['ale.lives']:
        # W przypadku utraty zycia - wymuszenie startu
        observation, reward, done, info = env.step(1)

    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)  # Zamiana na grayscale
    observation = cv2.resize(observation, (IN_X, IN_Y))  # Przeskalowanie

    # Uaktualnienie poprzednich obserwacji
    prev_states[:, :, :-1] = prev_states[:, :, 1:]
    prev_states[:, :, -1] = observation

    new_state = prev_states
    current_state = new_state

    time.sleep(0.05)
    prev_lives = info['ale.lives']
    t += 1




print(f"Liczba krokow w rozgrywce: {t}")



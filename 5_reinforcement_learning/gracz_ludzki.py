# ==== Gracz ludzki ===
# Sterowanie - strzalki, nowa rozgrywka - s

import gym
import time

player_action = 0


def key_press(key, mod):
    """
    Funkcja reagujaca na wcisniecie przycisku:
    Zmienia akcję do wykoania jesli wcisniety zostal jeden z przyciskow:
    strzalka w prawo (ruch w prawo)
    strzalka w lewo (ruch w lewo)
    s (start)
    """
    global player_action
    if key == 65363:
        player_action = 2
    if key == 65361:
        player_action = 3
    if key == 115:
        player_action = 1


def key_release(key, mod):
    """
    Funkcja reagujaca na zwolnienie przycisku.
    Jesli byl to przycisk odpowiedzialny za wykonywanie obecnej akcji,
    to jest ona ustawiana na 0
    """
    global player_action
    act = 0
    if key == 65363:
        act = 2
    if key == 65361:
        act = 3
    if key == 115:
        act = 1

    if player_action == act:
        player_action = 0


env = gym.make('Breakout-v0')

state = env.reset()
done = False
t = 0

# Petla symulacji
while not done:
    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release

    if t == 0:
        # Rozpoczecie gry
        state, reward, done, info = env.step(1)
        t = 1
    else:
        # Wykonanie kroku na podstawie akcji wybranej przez gracza
        state, reward, done, info = env.step(player_action)

    # Kroka pauza by przystosowac gre do ludzkiego gracza
    time.sleep(0.05)



# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import gym
import gym_donkeycar
import numpy as np
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    # SET UP ENVIRONMENT
    # You can also launch the simulator separately
    # in that case, you don't need to pass a `conf` object
    exe_path = "/home/mao/mysim/DonkeySimLinux/donkey_sim.x86_64"
    port = 9091

    conf = {"exe_path": exe_path, "port": port}

    env = gym.make("donkey-generated-track-v0", conf=conf)

    # PLAY
    obs = env.reset()
    for t in range(100):
        action = np.array([0.2, 0.1])  # drive straight with small speed
        # execute the action
        obs, reward, done, info = env.step(action)

    # Exit the scene
    env.close()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

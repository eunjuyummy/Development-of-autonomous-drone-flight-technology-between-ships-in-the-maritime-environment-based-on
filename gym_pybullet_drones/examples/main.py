"""Script demonstrating the use of `gym_pybullet_drones`' Gymnasium interface.

Class HoverAviary is used as a learning env for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python main.py

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning libraries `stable-baselines3`.

The results can be found in the 'result' folder.
Trained in different ways depending on file number
Even:PID, Odd:RL
"""
import os
import glob
import time
import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.BothCtrlAviary import BothCtrlAviary
from stable_baselines3.common.logger import configure
from gym_pybullet_drones.utils.utils import sync, str2bool
import matplotlib.pyplot as plt
from IPython import display
from datetime import datetime
import pdb
import math
import pybullet as p
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType, ActionType, ObservationType
from typing import Any, Dict
import torch as th
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from gym_pybullet_drones.envs.BothCtrlAviary import TensorboardCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import Image

DEFAULT_RECORD_VIDEO = True
DEFAULT_OUTPUT_FOLDER = 'results'
ACT_TYPE = ActionType.RPM
DEFAULT_NUM_DRONES = 1
DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_WAYPOINT = 0

def run(output_folder=DEFAULT_OUTPUT_FOLDER, record_video=DEFAULT_RECORD_VIDEO, ActionType=ACT_TYPE, num_drones=DEFAULT_NUM_DRONES, drone=DEFAULT_DRONES, input_num_waypoint=DEFAULT_NUM_WAYPOINT):
    #tmp_path = "./logs/"
    #new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    #### Check the environment's spaces ########################
    # ADD: input_num_waypoint @eunjuyummy 2023.09.18
    #env = gym.make("bothctrl-aviary-v0")
    #env = Monitor(env, output_folder)
    env = gym.make("bothctrl-aviary-v0", input_num_waypoint=DEFAULT_NUM_WAYPOINT) 
    #### Train the model #######################################
    #### RL ####################################################
    action_type = ActionType.RPM
    env.reset(act=action_type)
    model = PPO("MlpPolicy",
                env,
                verbose=1,
                tensorboard_log="./log/PPO_tensorboard/"
                )
    # video_recorder = VideoRecorderCallback(gym.make("bothctrl-aviary-v0"), render_freq=5000)
    model.learn(total_timesteps=1e6, callback=TensorboardCallback()) # Typically not enough
    #env.reset(act=action_type)
    print("Train RL model")
    #############################################################
    #### Test the model ########################################
    # reset(): Select action PID or RL
    # Use PID and RPM (RL) alternately
    for Try in range(4):
        #### PID ###################################################
        if Try % 2 == 0:
            action_type = ActionType.PID
            print("PID")
            duration_sec = 25
        #### RL ####################################################
        else:
            action_type = ActionType.RPM
            print("RL")
            duration_sec = 25
        ############################################################
        #### Initialize the controllers ############################
        if action_type == ActionType.PID:
            TARGET_POS, INIT_XYZS, INIT_RPYS, NUM_WP = env.calculate_target_positions()
            wp_counters = np.array([0 for i in range(num_drones)])
            if drone in [DroneModel.CF2X, DroneModel.CF2P]:
                ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]
            action = np.zeros((num_drones,4))
        ############################################################       
        obs, info = env.reset(seed=42, options={}, act=action_type)
        start = time.time()
        print("실행시간: ", duration_sec*env.CTRL_FREQ)
        for i in range(duration_sec*env.CTRL_FREQ):
            #### RL ####################################################
            if action_type == ActionType.RPM:
                action, _states = model.predict(obs,
                                                deterministic=True
                                                )
            obs, reward, terminated, truncated, info = env.step(action)
            #### PID ####################################################
            if action_type == ActionType.PID:
                for j in range(num_drones):
                    action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                         state=obs[j],
                                                                         target_pos=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2]]),
                                                                         target_rpy=INIT_RPYS[j, :]
                                                                        )
                # Go to the next way point and loop
                for j in range(num_drones):
                    wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0
            #############################################################
            # MOD: input_num_waypoint @eunjuyummy 2023.09.18
            # Save only the first attempt of RL and PID
            env.render()
            sync(i, start, env.CTRL_TIMESTEP)
            if terminated:
                obs = env.reset(seed=42, options={})
            # If the file number is even, PID If odd, RL
        env.make_gif()
    env.close()
    #############################################################

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script using HoverAviary')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--input_num_waypoint', default=DEFAULT_NUM_WAYPOINT,        type=int,           help='Number of waypoint (default: 2)', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))

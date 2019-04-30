import uuid
from threading import Thread
from time import sleep
from typing import Dict

import gym

from multiprocessing import Process, Manager

import roboschool.multiplayer
from datetime import datetime
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from roboschool import RoboschoolPong
from roboschool.gym_pong import PongSceneMultiplayer

from policy import SmallReactivePolicy

manager = Manager()


def play(policy: SmallReactivePolicy, player_num: int, video: bool, return_dict, server_name: str,
         max_steps=500):
    """
    Startup an agent to play ping pong
    :param max_steps: a maximum number of steps to run
    :param policy: a policy to act
    :param player_num: a player number
    :param video: if video of the game should be recorded
    :param return_dict: a dict to save results
    :param server_name: a name of the server hosting the game
    :return: None
    """

    env = gym.make("RoboschoolPong-v1")
    env.unwrapped.multiplayer(env, game_server_guid=server_name, player_n=player_num)

    pi = policy
    episode_n = 0
    score = 0
    for i in range(5):
        episode_n += 1
        obs = env.reset()

        if video:
            video_recorder = gym.wrappers.Monitor(env, "demo_pong_episode",
                                                  video_callable=lambda x: True, force=True)

        iteration_num = 0
        while 1:
            iteration_num += 1
            a = pi.act(obs)
            try:
                obs, rew, done, info = env.step(a)

            except:
                break
            score += rew

            if video:
                video_recorder.capture_frame()
            if done:
                break

            if iteration_num > max_steps:
                break

        if video:
            video_recorder.close()

        if iteration_num > max_steps:
            # env.close()
            break

    env.close()
    return_dict[player_num] = score


def play_single_process(policy_1: SmallReactivePolicy, policy_2: SmallReactivePolicy, return_dict,
                        max_steps=9500):
    """
    Startup a game of ping pong with 2 agents

    :param max_steps: a maximum amount of steps to execute
    :param policy_2: a policy of second agent
    :param policy_1: a policy of first agent
    :param return_dict: a dict to save results
    :return: None
    """

    game_1 = RoboschoolPong()
    scene = PongSceneMultiplayer()
    game_1.scene = scene
    game_1.player_n = 0
    game_2 = RoboschoolPong()
    game_2.scene = scene
    game_2.player_n = 1

    episode_n = 0
    score_1 = 0
    score_2 = 0
    for i in range(5):
        episode_n += 1
        scene.episode_restart()
        obs_1 = game_1.reset()
        obs_2 = game_2.reset()

        iteration_num = 0
        while 1:
            iteration_num += 1
            a_1 = policy_1.act(obs_1)
            try:
                game_1.apply_action(a_1)

            except:
                break

            a_2 = policy_2.act(obs_2)
            try:
                game_2.apply_action(a_2)
            except:
                break

            scene.global_step()
            obs_1, rew_1, done, info = game_1.step(a_1)
            obs_2, rew_2, done, info = game_2.step(a_2)
            game_1.render()

            score_1 += rew_1
            score_2 += rew_2

            if done:
                break

            if iteration_num > max_steps:
                break

        if iteration_num > max_steps:
            break

    return_dict[0] = score_1
    return_dict[1] = score_2


def run_sim_with_server(policy_1: SmallReactivePolicy, policy_2: SmallReactivePolicy) -> Dict:
    """
    Startup a server and perform game simulation
    :param policy_1: a policy for first agent
    :param policy_2: a policy for second agent
    :return: a dictionary with results of simulation
    """
    return_dict = manager.dict()
    server_name = uuid.uuid4().hex[:16].upper()
    server = Process(target=open_server, args=(server_name,), daemon=True)
    server.start()
    sleep(0.1)  # required to wait for the server, you can change the code of the server to account for this

    agent_1 = Thread(target=play, args=(policy_1, 0, True, return_dict, server_name), daemon=True)
    agent_2 = Thread(target=play, args=(policy_2, 1, False, return_dict, server_name), daemon=True)

    agent_1.start()
    agent_2.start()

    agent_1.join()
    agent_2.join()
    server.terminate()
    server.join()
    result = dict(return_dict)
    return result


def run_sim(policy_1: SmallReactivePolicy, policy_2: SmallReactivePolicy) -> Dict:
    """
    Perform a simulation of pin pong game with 2 agents
    :param policy_1: a policy for first agent
    :param policy_2: a policy for second agent
    :return: a dictionary with results of simulation
    """
    return_dict = manager.dict()
    server = Process(target=play_single_process, args=(policy_1, policy_2, return_dict), daemon=True)
    server.start()
    server.join()
    result = dict(return_dict)
    return result


def open_server(server_name: str):
    """
    Create a server for ping pong to process games between two agents
    :param server_name: a name of the server
    :return: None
    """
    game = roboschool.gym_pong.PongSceneMultiplayer()
    gameserver = roboschool.multiplayer.SharedMemoryServer(game, server_name, want_test_window=True)

    gameserver.serve_forever()


if __name__ == "__main__":
    run_sim(SmallReactivePolicy(), SmallReactivePolicy())

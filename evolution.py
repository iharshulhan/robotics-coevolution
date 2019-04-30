import concurrent
import random
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from typing import List
import pickle

import numpy as np
from collections import defaultdict

import pandas as pd

from simulation import run_sim
from policy import SmallReactivePolicy

metrics = pd.DataFrame(columns=['avg_wins', 'avg_score', 'top_wins', 'top_score'])


def add_noise(weights: np.array, mu: float = 0, sigma: float = 0.1) -> np.array:
    """
    Add random noise to nn weights
    :param sigma: standard deviation of noise
    :param mu: mean of noise
    :param weights: an array with weights
    :return: new array with noise
    """

    s = np.random.normal(mu, sigma, weights.shape)
    return weights + s


def perform_competition(agents: List[SmallReactivePolicy], k: int = 5,
                        tested_group_size: int = None) -> List[SmallReactivePolicy]:
    """
    Perform competitions against passed agents and return top-k performers
    :param tested_group_size: a size of tested subgroup, tested agents should be in the end of of the list
    :param k: how many agents to return
    :param agents: a list of agents to compete
    :return: top-k agents
    """

    scores = defaultdict(float)
    wins_against_baseline = defaultdict(int)
    n = len(agents)

    def points_from_comp(i, j):
        results = run_sim(agents[i], agents[j])
        if results[0] - results[1] > 1:
            scores[i] += 1 / ((n - 1) * 2)
            if i >= n - tested_group_size > j:
                wins_against_baseline[i] += 1
        elif results[1] - results[0] > 1:
            scores[j] += 1 / ((n - 1) * 2)
            if j >= n - tested_group_size > i:
                wins_against_baseline[j] += 1
        else:
            scores[j] += 0.333 / ((n - 1) * 2)
            scores[i] += 0.333 / ((n - 1) * 2)

    games = []

    for i in range(len(agents)):
        for j in range(len(agents)):
            if i != j:
                games.append((i, j))

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_results = [executor.submit(points_from_comp, *game) for game in games]
        concurrent.futures.wait(future_results)

    if tested_group_size:
        avg_score = 0
        avg_wins = 0
        top_score = 0
        top_wins = 0
        for i in range(n - tested_group_size, n):
            avg_score += scores[i]
            cur_wins = wins_against_baseline[i] / ((n - tested_group_size) * 2)
            avg_wins += cur_wins

            top_score = max(scores[i], top_score)
            top_wins = max(top_wins, cur_wins)

        print(f'Avg wins against orig. pop: {avg_wins / tested_group_size}')
        print(f'Avg score of evolved: {avg_score / tested_group_size}')
        print(f'Top wins against orig. pop: {top_wins}')
        print(f'Top score of evolved: {top_score}')

        metrics.loc[-1] = np.array([avg_wins / tested_group_size, avg_score / tested_group_size, top_wins, top_score])
        metrics.index += 1

    winners = np.array(sorted(scores.items(), key=lambda x: x[1], reverse=True)).astype(int)[:k, 0]
    return list(np.array(agents)[winners])


def test_new_variation(original: SmallReactivePolicy, mutated: SmallReactivePolicy,
                       agents: List[SmallReactivePolicy]) -> bool:
    """
    Test a mutated offspring against its original agent, by playing games with other agents
    :param original: an agent
    :param mutated: a mutated agent
    :param agents: a list of agents to enter competition
    :return: whether a mutated agent performs better
    """
    scores = defaultdict(float)
    n = len(agents)

    def points_from_comp(agent_1, agent_2, source_num, order_1, order_2):
        results = run_sim(agent_1, agent_2)

        if results[order_1] - results[order_2] > 1:
            scores[source_num] += 1 / (n * 2)

        elif 0 < results[order_2] - results[order_2] < 1:
            scores[source_num] += 0.333 / (n * 2)

    games = []

    for i in range(len(agents)):
        games.append((original, agents[i], 0, 0, 1))
        games.append((mutated, agents[i], 1, 0, 1))
        games.append((agents[i], original, 0, 1, 0))
        games.append((agents[i], mutated, 1, 1, 0))

    with ThreadPoolExecutor(max_workers=20) as executor:
        future_results = [executor.submit(points_from_comp, *game) for game in games]
        concurrent.futures.wait(future_results)

    return scores[1] > scores[0]


def randomly_change_policy(policy: SmallReactivePolicy, noise_std: float) -> SmallReactivePolicy:
    """
    Randomly change layers of policy network
    :param policy: a policy to change
    :return: a new instance of policy with randomly changed weights
    """
    new_policy = deepcopy(policy)

    num = random.randint(0, 5)
    if num == 0:
        new_policy.weights_final_b = add_noise(new_policy.weights_final_b, sigma=noise_std)
    if num == 1:
        new_policy.weights_final_w = add_noise(new_policy.weights_final_w, sigma=noise_std)
    if num == 2:
        new_policy.weights_dense2_b = add_noise(new_policy.weights_dense2_b, sigma=noise_std)
    if num == 3:
        new_policy.weights_dense1_b = add_noise(new_policy.weights_dense1_b, sigma=noise_std)
    if num == 4:
        new_policy.weights_dense2_w = add_noise(new_policy.weights_dense2_w, sigma=noise_std)
    if num == 5:
        new_policy.weights_dense1_w = add_noise(new_policy.weights_dense1_w, sigma=noise_std)

    return new_policy


def run_and_evolve(iterations: int = 20, experiment_num: int = 0, phase_num: int = 10):
    """
    Evolve agents and run simulations to select the best performers
    :param experiment_num: 0 to use all-vs-all competition, 1 for testing mutation against its parent
    :param iterations: number of evolving iterations to run
    :param phase_num: number of phases to tun
    :return:
    """
    try:
        with open('agents.pickle', 'rb') as f:
            agents = pickle.load(f)
    except:
        agents = [policy for policy in [randomly_change_policy(SmallReactivePolicy(), 1)] * 10]

    for phase in range(1, phase_num + 1):
        # Separate to different populations
        all_agents = agents.copy()
        random.shuffle(all_agents)
        test_agents = all_agents[:5]
        agents = agents[5:]

        start_time = datetime.now()
        for epoch in range(iterations):
            if experiment_num == 0:
                agents.extend([randomly_change_policy(agent, noise_std=2 / phase) for agent in agents])
                agents.extend([SmallReactivePolicy()])
                agents = perform_competition(agents)
            elif experiment_num == 1:
                for i, agent in enumerate(agents):
                    new_agent = randomly_change_policy(agent, noise_std=2 / phase)
                    if test_new_variation(agent, new_agent, test_agents):
                        agents[i] = new_agent

        print(datetime.now() - start_time)

        print(f'Phase {phase} comp: ')
        best = perform_competition(all_agents + agents, k=10, tested_group_size=5)
        metrics.to_csv('metrics2.csv')
        agents = best
        if phase % 10 == 0:
            with open(f'0agents_experiment_{experiment_num}_best_phase_{phase}.pickle', 'wb') as f:
                pickle.dump(best, f)


run_and_evolve(100, 1)

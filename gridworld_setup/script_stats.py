from __future__ import annotations
import warnings

import numpy as np
import pickle
import argparse

from tqdm import trange
from datetime import datetime

from learner import BayesianLearner
from bayesian_ToM.bayesian_teacher import AlignedBayesianTeacher, BayesianTeacher
from utils import *
from utils_viz import *

warnings.filterwarnings("ignore", category=RuntimeWarning)


def parse_args():
    parser = argparse.ArgumentParser('Training prediction model')
    parser.add_argument('--GRID_SIZE', type=int, default=15),
    parser.add_argument('--num_colors', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--max_obs', type=int, default=-1)
    parser.add_argument('--num_trials', type=int, default=200)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    # Parameters

    GRID_SIZE = args.GRID_SIZE
    GRID_SIZE_DEMO = args.GRID_SIZE * 3

    num_colors = args.num_colors

    if GRID_SIZE >= 15:
        rf_values_basic = [3, 5, 7]
    else:
        rf_values_basic = [3, 5]
    rf_values = np.array(rf_values_basic + [GRID_SIZE])
    rf_values_demo = np.array(rf_values_basic + [GRID_SIZE_DEMO])

    num_rf = len(rf_values)
    num_rf_demo = len(rf_values_demo)

    lambd = 0.01

    alpha = args.alpha
    def cost_fun(x: int, l: int) -> float:
        return alpha * (x / l)

    N = args.num_trials

    if args.max_obs == -1:
        max_obs = GRID_SIZE ** 2
    else:
        max_obs = args.max_obs

    save_folder = './outputs/eval_utility'
    date = datetime.now().strftime('%H%M%s')
    make_dirs(save_folder)
    save_filename = save_folder + f'/obs_{GRID_SIZE}_demo_{GRID_SIZE_DEMO}_norm_linear_{alpha}_max_obs_{max_obs}_num_trials_{N}_{date}.pickle'

    DICT_UTIL = {}
    DICT_UTIL[lambd] = {}
    DICT_UTIL['omniscient'] = {}
    DICT_UTIL['aligned'] = {}
    DICT_UTIL['utility_opt_non_adaptive'] = {}
    DICT_UTIL['reward_opt_non_adaptive'] = {}
    DICT_UTIL['uniform_sampling'] = {}
    DICT_UTIL['uniform_model'] = {}

    for rf_idx, receptive_field in enumerate(rf_values):
        for goal_color in range(num_colors):

            DICT_UTIL[lambd][goal_color, receptive_field] = []
            DICT_UTIL['aligned'][goal_color, receptive_field] = []
            DICT_UTIL['omniscient'][goal_color, receptive_field] = []
            DICT_UTIL['utility_opt_non_adaptive'][goal_color, receptive_field] = []
            DICT_UTIL['reward_opt_non_adaptive'][goal_color, receptive_field] = []
            DICT_UTIL['uniform_sampling'][goal_color, receptive_field] = []
            DICT_UTIL['uniform_model'][goal_color, receptive_field] = []

            for _ in trange(N):
                # print(f'Learner: rf={receptive_field} goal_color={IDX_TO_COLOR[goal_color+1]}')
                # Test teacher utility
                learner = BayesianLearner(goal_color=goal_color, receptive_field=receptive_field, grid_size=GRID_SIZE, env_type='MultiGoalsEnv')
                teacher = BayesianTeacher(env=learner.env, lambd=lambd, rf_values=rf_values_basic)
                aligned_teacher = AlignedBayesianTeacher(env=learner.env, rf_values=rf_values_basic)

                # Teacher observes the learner during one full episode on the first simple env
                ii = 0
                learner_pos_list = []
                learner_dir_list = []
                learner_action_list = []
                while not learner.terminated and ii < max_obs:
                    
                    agent_pos = learner.env.agent_pos
                    learner_pos_list.append(agent_pos)

                    agent_dir = learner.env.agent_dir
                    learner_dir_list.append(agent_dir)

                    teacher.update_knowledge(learner_pos=agent_pos, learner_dir=agent_dir, learner_step_count=ii)
                    aligned_teacher.update_knowledge(learner_pos=agent_pos, learner_dir=agent_dir, learner_step_count=ii)

                    traj = learner.play(size=1)
                    learner_action_list.append(traj[0])

                    teacher.observe(action=traj[0])
                    aligned_teacher.observe(action=traj[0])

                    ii += 1

                # Teacher use ToM to predict the utility of each demo for this particular learner --> select the more relevant demo
                learner = BayesianLearner(goal_color=goal_color, receptive_field=rf_values_demo[rf_idx], \
                                        grid_size=GRID_SIZE_DEMO, env_type='MultiRoomsGoalsEnv')
                aligned_teacher.init_env(learner.env)
                teacher.init_env(learner.env)
                
                # Compute all the demonstrations
                all_demos = []
                l_max = 0
                # Random demos
                for n_obj in range(3, 9):
                    demo = generate_random_demo(learner.env, n_obj=n_obj)
                    all_demos.append(demo)
                    l_max = np.max([len(demo), l_max])
                # Learner-specific demos
                for gc in range(num_colors):
                    for rf in rf_values_demo[:-1]:
                        demo = generate_demo(learner.env, rf, gc)
                        all_demos.append(demo)
                        l_max = np.max([len(demo), l_max])
                all_demos.append([]) # Full obs

                ## Rationality principle teacher
                selected_demo, demo_idx, predicted_best_utility, demos = teacher.select_demo(l_max, cost_fun, all_demos)

                # Learner "observes" the demo
                learner.observe(selected_demo)

                # Reset step count in the env
                learner.env.step_count = 0
                # Learner play after seen the demo
                _ = learner.play()

                utility = learner.reward - cost_fun(len(selected_demo), l_max)

                DICT_UTIL[lambd][goal_color, receptive_field].append(utility)
            
                ## Utility map        
                true_utility = np.zeros((num_colors, len(rf_values_demo), len(all_demos)))
                true_reward = np.zeros((num_colors, len(rf_values_demo), len(all_demos)))

                for gc in range(num_colors):
                    for ii, rf in enumerate(rf_values_demo):
                        for jj, demo in enumerate(all_demos):
                            reward = aligned_teacher.predicted_reward(demo, gc, ii)

                            true_reward[gc, ii, jj] = reward
                            true_utility[gc, ii, jj] = reward - cost_fun(len(demo), l_max)

                ## Aligned teacher
                weighted_utility = true_utility.copy()
                for gc in range(num_colors):
                    for ii in range(num_rf_demo):
                        weighted_utility[gc, ii, :] *= aligned_teacher.beliefs[gc, ii]
                selected_demo_idx = np.argmax(weighted_utility.sum(axis=0).sum(axis=0))

                utility = true_utility[goal_color, rf_idx, selected_demo_idx]
                DICT_UTIL['aligned'][goal_color, receptive_field].append(utility)

                ## Omnicient teacher
                selected_demo_idx = np.argmax(true_utility[goal_color, rf_idx, :])

                utility = true_utility[goal_color, rf_idx, selected_demo_idx]
                DICT_UTIL['omniscient'][goal_color, receptive_field].append(utility)
                
                ## Utility optimal non-adaptive but goal omniscient
                mean_util_per_demo = true_utility.mean(axis=0).mean(axis=0)
                selected_demo_idx = np.argmax(mean_util_per_demo)

                utility = true_utility[goal_color, rf_idx, selected_demo_idx]
                DICT_UTIL['utility_opt_non_adaptive'][goal_color, receptive_field].append(utility)

                ## Reward optimal non-adaptive but goal omniscient
                mean_reward_per_demo = true_reward.mean(axis=0).mean(axis=0)
                selected_demo_idx = np.argmax(mean_reward_per_demo)

                utility = true_utility[goal_color, rf_idx, selected_demo_idx]
                DICT_UTIL['reward_opt_non_adaptive'][goal_color, receptive_field].append(utility)

                ## Uniform sample
                selected_demo_idx = np.random.randint(0, len(all_demos))

                utility = true_utility[goal_color, rf_idx, selected_demo_idx]
                DICT_UTIL['uniform_sampling'][goal_color, receptive_field].append(utility)

                ## Uniform model
                pred_goal = np.random.randint(0, num_colors)
                pred_rf_idx = np.random.randint(0, len(rf_values))
                pred_rf = rf_values_demo[pred_rf_idx]

                selected_demo_idx = np.argmax(true_utility[pred_goal, pred_rf_idx, :])

                utility = true_utility[goal_color, rf_idx, selected_demo_idx]
                DICT_UTIL['uniform_model'][goal_color, receptive_field].append(utility)

            with open(save_filename, 'wb') as f:
                    pickle.dump(DICT_UTIL, f)
import argparse
import pickle
import json
from datetime import datetime
from tqdm import trange
import warnings

import sys
sys.path.append('.')
sys.path.append('./../')

from learner import BayesianLearner
from utils import *
from minigrid.core.constants import OBJECT_TO_IDX

warnings.filterwarnings("ignore", category=RuntimeWarning)

def parse_args():
    parser = argparse.ArgumentParser('Generate data to train ToMNet model')
    parser.add_argument('--num_train', '-n_train', type=int, default=100),
    parser.add_argument('--num_val', '-n_val', type=int, default=100),
    parser.add_argument('--num_test', '-n_test', type=int, default=10),
    parser.add_argument('--start_idx', '-start', type=int, default=0),
    parser.add_argument('--save_folder', '-save', type=str, default='./neural_network_ToM/data'),
    parser.add_argument('--grid_size', '-gs', type=int, default=15),
    parser.add_argument('--grid_size_demo', '-gs_demo', type=int, default=45),
    parser.add_argument('--rf_values_basic', '-rf_val', type=list, default=[3,5,7]),
    parser.add_argument('--num_colors', '-nc', type=int, default=4)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    config_dict = dict(num_train = args.num_train, 
                       num_val = args.num_val, 
                       num_test = args.num_test,
                       GRID_SIZE = args.grid_size,
                       GRID_SIZE_DEMO = args.grid_size_demo,
                       rf_values_basic = args.rf_values_basic,
                       start_idx = args.start_idx,
                       num_colors=args.num_colors)
    
    date = datetime.now().strftime("%m.%d.%Y")
    make_dirs(f'{args.save_folder}/dataset_{date}')

    with open(f'{args.save_folder}/dataset_{date}/dataset_config.json', 'w') as f:
        json.dump(config_dict, f)
    print(f"Save dataset config in {f'{args.save_folder}/dataset_{date}/dataset_config.json'}")
    
    num_data = [args.num_train, args.num_val, args.num_test]
    names = ['train', 'val', 'test']

    for ii, name in enumerate(names):

        make_dirs(f'{args.save_folder}/dataset_{date}/{name}')
           
        rf_values = args.rf_values_basic + [args.grid_size]
        rf_values_demo = args.rf_values_basic + [args.grid_size_demo]

        data_idx = args.start_idx

        for n in trange(num_data[ii]):
            for goal_color in range(args.num_colors):
                for rf_idx,receptive_field in enumerate(rf_values_demo):
                    for demo_idx, demo_rf in enumerate(rf_values_demo):

                        # Observation environemnt
                        learner = BayesianLearner(goal_color=goal_color, receptive_field=receptive_field, 
                                                grid_size=args.grid_size, env_type='MultiGoalsEnv', 
                                                num_colors=args.num_colors)
                        
                        images_obs_env = []
                        actions_obs_env = []
                        vis_mask = np.ones((args.grid_size, args.grid_size))
                        while not learner.terminated and learner.env.step_count < learner.env.max_steps:

                            grid = learner.env.grid.slice(0, 0, args.grid_size, args.grid_size)
                            image = grid.encode(vis_mask)
                            image[learner.env.agent_pos[0], learner.env.agent_pos[1]] = np.array([OBJECT_TO_IDX['agent'], 0 , 0])
                            images_obs_env.append(image)

                            traj = learner.play(size=1)

                            actions_obs_env.append(traj[0])

                        # Demonstration environment
                        receptive_field = rf_values_demo[rf_idx]
                        learner = BayesianLearner(goal_color=goal_color, receptive_field=receptive_field, 
                                                grid_size=args.grid_size_demo, env_type='MultiRoomsGoalsEnv',
                                                num_colors=args.num_colors)
                                            
                        # Reset env
                        learner.reset()
                        learner.change_receptive_field(receptive_field)

                        # Generate demo for predicted rf demo_rf (right goal_color)
                        demo = generate_demo(learner.env, demo_rf, goal_color)
                        
                        # Learner observes the demonstration
                        learner.observe(demo, render_mode=None)
                        images_demo = learner.render_frames_observation
                        if len(demo) > 0:
                            demo = [4] + demo 

                        # Reset step count in the env
                        learner.env.step_count = 0
                        # Learner play after observing the demo
                        
                        images_demo_env = []
                        actions_demo_env = []
                        vis_mask = np.ones((args.grid_size_demo, args.grid_size_demo))
                        while not learner.terminated and learner.env.step_count < learner.env.max_steps:
                            grid = learner.env.grid.slice(0, 0, args.grid_size_demo, args.grid_size_demo)
                            image = grid.encode(vis_mask)
                            image[learner.env.agent_pos[0], learner.env.agent_pos[1]] = np.array([OBJECT_TO_IDX['agent'], 0 , 0])
                            images_demo_env.append(image)

                            traj = learner.play(size=1)

                            actions_demo_env.append(traj[0])

                        size_init_traj = np.random.randint(0, len(actions_demo_env)-1)

                        query_state = images_demo_env[size_init_traj]
                        futur_traj = actions_demo_env[size_init_traj]

                        images_demo_env = images_demo_env[:size_init_traj]
                        actions_demo_env = actions_demo_env[:size_init_traj]

                        saving_filename = f'{args.save_folder}/dataset_{date}/{name}/data_{data_idx}.pickle'

                        with open(saving_filename, 'wb') as f:
                            pickle.dump(([(images_obs_env, actions_obs_env), (images_demo, demo), (images_demo_env, actions_demo_env)], futur_traj, query_state), f)

                        data_idx += 1

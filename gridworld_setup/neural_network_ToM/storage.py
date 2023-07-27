import numpy as np
import json
import os
import pickle

path = '/home/chetouani/Documents/STAGE_Clemence/ISIR_internship_ToM_gridworld/gridworld_setup/neural_network_ToM'

class Storage:

    def __init__(self, 
                 dataset_name: str,
                 max_length_obs: int,
                 max_length_demo_eval: int,
                 data_mode: str,
                 max_num_frames: int | None=None
                 ) -> None:
        
        self.dataset_name = dataset_name
        self.data_mode = data_mode
        self.max_length_obs = max_length_obs
        self.max_length_demo_eval = max_length_demo_eval

        self.max_num_frames = max_num_frames

        self.num_frames = 0
        self.data_paths = []
    
    def extract(self):
        config_data = json.load(open(path + f'/data/{self.dataset_name}/dataset_config.json'))
        strat_idx = config_data['start_idx']

        data_idx = strat_idx
        data_path = path + f'/data/{self.dataset_name}/{self.data_mode}/data_{data_idx}.pickle'
        
        while os.path.exists(data_path):
            self.data_paths.append(data_path)
            self.num_frames += 1
            data_idx += 1
            data_path = path + f'/data/{self.dataset_name}/{self.data_mode}/data_{data_idx}.pickle'
        
        dico = dict(
                    data_paths = self.data_paths,
                    max_length_obs = self.max_length_obs,
                    max_length_demo_eval = self.max_length_demo_eval
                    )

        return dico

    def reset(self):
        self.num_frames = 0
        self.data_paths = []
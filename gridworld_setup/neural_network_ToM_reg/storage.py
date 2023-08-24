import json
import os

global_path = "/home/chetouani/Documents/STAGE_Clemence/ISIR_internship_ToM_gridworld/gridworld_setup"
# global_path = '/gpfswork/rech/kcr/uxv44vw/clemence/ISIR_internship_ToM_gridworld/gridworld_setup'

path = f"{global_path}/neural_network_ToM_reg"


class Storage:
    def __init__(
        self,
        dataset_name: str,
        max_length_obs: int,
        max_length_demo: int,
        data_mode: str,
        max_num_frames: int | None = None,
        grid_size: int = 15,
        grid_size_demo: int = 45,
        num_channel: int = 3,
    ) -> None:
        self.dataset_name = dataset_name
        self.data_mode = data_mode
        self.max_length_obs = max_length_obs
        self.max_length_demo = max_length_demo

        self.max_num_frames = max_num_frames

        self.num_frames = 0
        self.data_paths = []

        self.grid_size = grid_size
        self.grid_size_demo = grid_size_demo

        self.num_channel = num_channel

    def extract(self):
        config_data = json.load(
            open(path + f"/data/{self.dataset_name}/dataset_config.json")
        )
        strat_idx = config_data["start_idx"]

        data_idx = strat_idx
        data_path = (
            path + f"/data/{self.dataset_name}/{self.data_mode}/data_{data_idx}.pickle"
        )

        while os.path.exists(data_path):
            self.data_paths.append(data_path)
            self.num_frames += 1
            data_idx += 1
            data_path = (
                path
                + f"/data/{self.dataset_name}/{self.data_mode}/data_{data_idx}.pickle"
            )

        dico = dict(
            data_paths=self.data_paths,
            max_length_obs=self.max_length_obs,
            max_length_demo=self.max_length_demo,
            grid_size=self.grid_size,
            grid_size_demo=self.grid_size_demo,
            num_channel=self.num_channel,
        )

        return dico

    def reset(self):
        self.num_frames = 0
        self.data_paths = []

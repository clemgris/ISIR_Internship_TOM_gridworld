import pickle
import numpy as np

from torch.utils.data import Dataset


def preprocess_data(
    data_path: str,
    max_length_obs: int,
    max_length_demo: int,
    grid_size: int,
    grid_size_demo: int,
    num_channel: int,
) -> tuple:
    data_list, reward = pickle.load(open(data_path, "rb"))

    trajectories = []
    max_lengths = [max_length_obs, max_length_demo]
    size = [grid_size, grid_size_demo]

    # Zero padded sequence of image concatenate with the agent action (action taken after the image screen)
    for kk, trajectory in enumerate(data_list):
        max_length = max_lengths[kk]
        frames, actions = trajectory

        assert len(frames) == len(actions)
        processed_frame = np.zeros((max_length, size[kk], size[kk], num_channel + 1))

        # Add zero-padding + concatenate the image and the action
        for ii, frame in enumerate(frames):
            if ii >= max_length:
                break
            processed_frame[ii, :, :, :-1] = frame
            processed_frame[ii, :, :, -1] = actions[ii]

        trajectories.append(processed_frame)

    return trajectories, reward


class ToMNetDataset(Dataset):
    def __init__(
        self,
        data_paths,
        max_length_obs,
        max_length_demo,
        grid_size: int,
        grid_size_demo: int,
        num_channel: int,
    ) -> None:
        self.data_paths = data_paths

        self.max_length_obs = max_length_obs
        self.max_length_demo = max_length_demo

        self.grid_size = grid_size
        self.grid_size_demo = grid_size_demo

        self.num_channel = num_channel

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, ind):
        data_path = self.data_paths[ind]
        [obs, demo], reward = preprocess_data(
            data_path,
            self.max_length_obs,
            self.max_length_demo,
            self.grid_size,
            self.grid_size_demo,
            self.num_channel,
        )

        return obs, demo, reward

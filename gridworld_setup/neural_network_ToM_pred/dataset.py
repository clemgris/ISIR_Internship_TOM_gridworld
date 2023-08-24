
import pickle
import numpy as np

from torch.utils.data import Dataset

def preprocess_data(data_path: str, 
                    max_length_obs: int, 
                    max_length_demo_eval: int
                    ) -> tuple:
    
    data_list, next_action, query_state = pickle.load(open(data_path, 'rb'))

    trajectories = []
    max_lengths = [max_length_obs, max_length_demo_eval]

    obs = data_list[0]
    demo, actions_demo = data_list[1]
    eval, actions_eval = data_list[2]
    demo_eval = (demo + eval, actions_demo + actions_eval)
                               
    # Zero padded sequence of image concatenate with the agent action (action taken after the image screen)
    for kk,trajectory in enumerate([obs, demo_eval]):
        max_length = max_lengths[kk]
        frames, actions = trajectory

        assert(len(frames) == len(actions))
        H, W, num_channel = frames[0].shape
        processed_frame = np.zeros((max_length, H, W, num_channel + 1))

        # Add zero-padding + concatenate the image and the action
        for ii, frame in enumerate(frames):
            if ii >= max_length:
                break
            processed_frame[ii, :, :, :-1] = frame
            processed_frame[ii, :, :, -1] = actions[ii]
        
        trajectories.append(processed_frame)

    return trajectories, next_action, query_state

class ToMNetDataset(Dataset):

    def __init__(self, 
                 data_paths, 
                 max_length_obs, 
                 max_length_demo_eval
                 ) -> None:
        self.data_paths = data_paths

        self.max_length_obs = max_length_obs
        self.max_length_demo_eval = max_length_demo_eval
    
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, ind):
        data_path = self.data_paths[ind]
        [obs, demo_eval], nex_action, query_state = preprocess_data(data_path,
                                                        self.max_length_obs,
                                                        self. max_length_demo_eval)

        return obs, demo_eval, query_state, nex_action
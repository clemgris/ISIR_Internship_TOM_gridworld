import torch.nn as nn
import torch as torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.append('/home/chetouani/Documents/STAGE_Clemence/ISIR_internship_ToM_gridworld/gridworld_setup')
sys.path.append('/home/chetouani/Documents/STAGE_Clemence/ISIR_internship_ToM_gridworld/gridworld_setup/neural_network_ToM')

from convLSTM import ConvLSTM

from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()

        self.conv_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_function(x)
        return x

class ResNetBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * ResNetBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * ResNetBlock.expansion),
        )

        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        if stride != 1 or in_channels != ResNetBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * ResNetBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * ResNetBlock.expansion)
            )

    def forward(self, x):
        x = x.double()
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x

class CharNet(nn.Module):
    def __init__(self, 
                 num_input: int, 
                 num_step: int,
                 num_output: int,
                 grid_size: int,
                 device: str='cuda'
                 ) -> None:
        
        super(CharNet, self).__init__()

        self.encoder = nn.Sequential(ResNetBlock(num_input, 4, 1).double().to(device),
                                    ResNetBlock(4, 8, 1).double().to(device),
                                    ResNetBlock(8, 16, 1).double().to(device),
                                    ResNetBlock(16, 32, 1).double().to(device),
                                    ResNetBlock(32, 32, 1).double().to(device),
                                    nn.ReLU().double().to(device),
                                    nn.BatchNorm2d(32).double().to(device), # [batch * num_step, output, H, W]
                                    nn.AvgPool2d(grid_size).double().to(device) # [batch * num_step, output, 1]
                                    )

        self.bn = nn.BatchNorm2d(32).double().to(device)
        self.relu = nn.ReLU().double().to(device)
        self.lstm = nn.LSTM(32, 64).double().to(device)
        self.avgpool = nn.AvgPool2d(grid_size).double().to(device)
        self.fc_final = nn.Linear(num_step * 64, num_output).double().to(device)
        self.hidden_size = 64

        self.device = device

    def init_hidden(self, input_dim: int) -> tuple:
        return (torch.zeros(1, input_dim, 64, device=self.device, dtype=torch.float64),
                torch.zeros(1, input_dim, 64, device=self.device, dtype=torch.float64))

    def forward(self, obs: torch.tensor) -> torch.tensor:
        # obs: [batch, num_past, num_step, H, W, channel]
        print('obs', obs.shape)
        obs = obs.permute(0, 1, 2, 5, 3, 4) # [batch, num_past, num_step, channel, H, W]
        batch_size, num_past, num_step, num_channel, H, W = obs.shape
        
        past_e_char = []
        for p in range(num_past):
            prev_h = self.init_hidden(batch_size)

            obs_past = obs[:, p] # [batch, num_step, channel, H, W]
            print(f'obs past {obs_past.shape} [batch, num_step, channel, H, W]')
            obs_past = obs_past.permute(1, 0, 2, 3, 4) # [num_step, batch, channel, H, W]
            obs_past = obs_past.reshape(-1, num_channel, H, W) # [batch * num_step, channel, H, W]
            print(f'obs_past {obs_past.shape} [batch * num_step, channel, H, W]')

            x = self.encoder(obs_past.double())
            print(f'x {x.shape}')

            x = x.view(num_step, batch_size, -1) # [num_step, batch, output]
            outs, _ = self.lstm(x, prev_h)
            outs = outs.permute(1, 0, 2) # [batch, num_step, output]
            print(f' outs {outs.shape} [batch, num_step, output]')
            
            # Mask outputs from zero-padding
            padded_obs_past = obs[:, p].reshape(batch_size, num_step, -1)
            mask = torch.any(padded_obs_past != 0, dim=-1).double()
            mask = mask.unsqueeze(-1)
            mask = mask.expand(-1, -1, outs.shape[-1])
            outs = outs * mask # [batch, num_step, output]
            print(f'outs {outs.shape} [batch, num_step, output]')
            
            outs = outs.reshape(batch_size, -1) # [batch, num_step * output]
            print(f'outs {outs.shape} [batch, num_step * output]')
            e_char_sum = self.fc_final(outs) # [batch, output]
            past_e_char.append(e_char_sum)

        # Sum e_char past traj
        past_e_char = torch.stack(past_e_char, dim=0)
        past_e_char_sum = sum(past_e_char)
        final_e_char = past_e_char_sum

        return final_e_char

class MentalNet(nn.Module):
    def __init__(self, num_input: int, 
                 num_step: int, 
                 num_output: int,
                 device: str='cuda'
                 ) -> None:
        
        super(MentalNet, self).__init__()
        
        self.num_ouput = num_output

        self.encoder = nn.Sequential(ResNetBlock(num_input, 4, 1).double().to(device),
                                        ResNetBlock(4, 8, 1).double().to(device),
                                        ResNetBlock(8, 16, 1).double().to(device),
                                        ResNetBlock(16, 32, 1).double().to(device),
                                        ResNetBlock(32, 32, 1).double().to(device),
                                        nn.ReLU().double().to(device),
                                        nn.BatchNorm2d(32).double().to(device)
                                        )
        
        self.relu = nn.ReLU().double().to(device)
        self.convlstm = ConvLSTM(32, [32], 3, 1).double().to(device)
        self.last_conv = nn.Conv2d(num_step * 32, num_output, kernel_size=3, padding=1, bias=False).double().to(device)
        
        self.device = device

    def init_hidden(self, input_dim: int) -> tuple:
            return (torch.zeros(1, input_dim, 32, device=self.device, dtype=torch.float64),
                    torch.zeros(1, input_dim, 32, device=self.device, dtype=torch.float64))

    def forward(self, obs: torch.tensor) -> torch.tensor:
        # obs: [batch, num_step, H, W, channel]
        init_obs = obs
        print(f'obs {obs.shape} [batch, num_step, H, W, channel]')
        obs = obs.permute(0, 1, 4, 2, 3) # [batch, num_step, channel, H, W]
        print(f'obs {obs.shape} [batch, num_step, channel, H, W]')
        batch_size, num_step, num_channel, H, W = obs.shape

        obs = obs.reshape(-1, num_channel, H, W) # [batch * num_step, channel, H, W]
        print(f'obs {obs.shape} [num_step * batch, channel, H, W]')
        
        x = self.encoder(obs.double())
        print(f'x {x.shape} [batch * num_step, output, H, W]')

        x = x.reshape(batch_size, num_step, -1, H, W) # [batch, num_step, ouput, H, W]
        print(f'x {x.shape} [batch, num_step, output, H, W]')

        outs, _ = self.convlstm(x) # [batch, num_step, ouput, H, W] ## WARNING MEMORY SPACE!!
        print(f'outs {outs.shape} ??')

        # Mask outputs from zero-padding
        padded_obs = init_obs.reshape(batch_size, num_step, -1)
        mask = torch.any(padded_obs != 0, dim=-1).double()
        print(f'mask {mask.shape}')
        mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        mask = mask.expand(-1, -1, 32, outs.size(3), outs.size(4))
        print(torch.any(mask != 1.))
        outs = outs * mask

        outs = outs.reshape(batch_size, -1, H, W) # [batch, num_step * output, H, W]
        
        # Last conv
        outs = self.last_conv(outs) # [batch, output, H, W]
        print('final shape', outs.shape)
        e_mental = self.relu(outs)
        
        return e_mental
    
class PredNet(nn.Module):
    def __init__(self, 
                 num_input: int,
                  num_step_obs: int,
                  num_steps_demo_eval: int,
                  num_output_char: int=8,
                  num_output_mental: int=32,
                  num_channel: int=3,
                  num_output_pred: int=6,
                  grid_size_obs: int=15,
                  grid_size_demo: int=45,
                  device: str='cuda',
                  num_types: int=4,
                  num_demo_types: int=4,
                  using_dist: bool=False
                  ) -> None:
        
        super(PredNet, self).__init__()

        self.with_mask = True

        self.num_types = num_types
        self.num_demo_types = num_demo_types
        self.num_output_pred = num_output_pred

        self.using_dist = using_dist

        self.device = device

        self.num_output_char = num_output_char
        self.num_output_mental = num_output_mental

        self.charnet = CharNet(num_input=num_input, num_step=num_step_obs, num_output=num_output_char, grid_size=grid_size_obs, device=device)
        self.mentalnet_traj = MentalNet(num_input=num_input, num_step=num_steps_demo_eval, num_output=num_output_mental, device=device)

        channel_in = num_output_mental + num_output_char + num_channel

        self.encoder = nn.Sequential(ResNetBlock(channel_in, 8, 1).double().to(device),
                                        ResNetBlock(8, 16, 1).double().to(device),
                                        ResNetBlock(16, 16, 1).double().to(device),
                                        ResNetBlock(16, 32, 1).double().to(device),
                                        ResNetBlock(32, 32, 1).double().to(device),
                                        nn.ReLU().double().to(device),
                                        nn.BatchNorm2d(32).double().to(device)
                                        )
        
        self.action_head = nn.Sequential(nn.Conv2d(32, 32, 1, 1).double(),
                                nn.ReLU().double(),
                                nn.Conv2d(32, 32, 1, 1).double(),
                                nn.ReLU().double(),
                                nn.AvgPool2d(grid_size_demo).double(),
                                nn.Flatten().double(),
                                nn.Linear(32, num_output_pred).double(),
                                nn.LogSoftmax(dim=1).double()
                                ).to(device)
            
    def forward(self, past_traj: torch.tensor, current_traj: torch.tensor, query_state: torch.tensor) -> tuple:

        batch_size, H, W, num_channel = query_state.shape
    
        # Past traj
        _, num_past, _, _, _, _ = past_traj.shape
        if num_past == 0:
            e_char = torch.zeros((batch_size, self.num_output_char, H, W, num_channel), device=self.device)
        else:
            e_char = self.charnet(past_traj)
            print('echar after model', e_char.shape)
            e_char_concat = e_char[..., None, None]
            e_char_concat = e_char_concat.repeat(1, 1, H, W) # [batch, num_output_char, H, W]
            print('echar after model', e_char.shape)


        # Current traj
        _, num_step, _, _, _ = current_traj.shape
        if num_step == 0:
            e_mental = torch.zeros((batch_size, self.num_output_mental, H, W, num_channel))
        else:
            e_mental = self.mentalnet_traj(current_traj)
            e_mental_concat = e_mental

        query_state_concat = query_state.permute(0, 3, 1, 2)

        print(e_char_concat.shape, e_mental_concat.shape, query_state_concat.shape)
        x_concat = torch.cat([e_char_concat, e_mental_concat, query_state_concat], axis=1) # [batch, num_output_char + num_output_mental + num_channel, H, W, num_channel]

        x = self.encoder(x_concat)
        action = self.action_head(x)

        return action, e_char, e_mental, query_state

    def train(self, data_loader: DataLoader, optim: Optimizer) -> dict:
        tot_loss = 0
        action_acc = 0
        metric = 0

        criterion_nll = nn.NLLLoss()

        for i, batch in enumerate(tqdm(data_loader)):

            past_traj, current_traj, query_state, target_action = batch
            
            past_traj = past_traj.float().to(self.device)
            current_traj = current_traj.float().to(self.device)
            query_state = query_state.float().to(self.device)
            target_action = target_action.long().to(self.device)

            pred_action, e_char, e_mental, query_state = self.forward(past_traj, current_traj, query_state)

            loss = criterion_nll(pred_action, target_action)
            
            # Backpropagation
            optim.zero_grad()

            loss.mean().backward()
            optim.step()

            pred_action_ind = torch.argmax(pred_action, dim=-1)
            tot_loss += loss.item()

            action_acc += (torch.sum(pred_action_ind == target_action).item() / len(target_action))

        dicts = dict(accuracy=action_acc / len(data_loader),
                     loss=tot_loss / len(data_loader))
        return dicts

    def evaluate(self, data_loader: DataLoader) -> dict:
        tot_loss = 0
        action_acc = 0

        criterion_nll = nn.NLLLoss()

        for i, batch in enumerate(tqdm(data_loader)):

            with torch.no_grad():

                past_traj, current_traj, query_state, target_action, true_idx_music = batch
                
                past_traj = past_traj.float().to(self.device)
                current_traj = current_traj.float().to(self.device)
                query_state = query_state.float().to(self.device)
                target_action = target_action.long().to(self.device)

                pred_action, e_char, e_mental, query_state = self.forward(past_traj, current_traj, query_state)
                loss = criterion_nll(pred_action, target_action)

            tot_loss += loss.item()

            pred_action_ind = torch.argmax(pred_action, dim=-1)
            # print('pred_action_ind', pred_action_ind, 'target_action', target_action)

            action_acc += (torch.sum(pred_action_ind == target_action).item() / len(target_action))
        
        dicts = dict(accuracy=action_acc / len(data_loader),
                     loss=tot_loss / len(data_loader))
        
        return dicts

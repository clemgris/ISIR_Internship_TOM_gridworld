import numpy as np
import os
import torch
import argparse
import json
from tqdm import tqdm

import torch.nn as nn
import torch as torch
import torch.nn.functional as F
from datetime import datetime
import torch.optim as optim
from storage import Storage
from dataset import ToMNetDataset
from torch.utils.data import DataLoader

import sys

sys.path.append(".")
sys.path.append("./../")

global_path = (
    "/gpfswork/rech/kcr/uxv44vw/clemence/ISIR_internship_ToM_gridworld/gridworld_setup"
)

from utils import make_dirs
from model import RegNet

from accelerate import metricelerator
from accelerate.utils import DistributedDataParallelKwargs

kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = accelerator(kwargs_handlers=[kwargs])


def parse_args():
    parser = argparse.ArgumentParser("Training prediction model")
    parser.add_argument("--n_epochs", "-e", type=int, default=10),
    parser.add_argument("--batch_size", "-b", type=int, default=3)
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data_filename", type=str, default="dataset_07.27.2023")
    parser.add_argument("--saving_name", type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    loading_path = f"{global_path}/neural_network_ToM_reg/data/{args.data_filename}"
    config = json.load(open(os.path.join(loading_path, "dataset_config.json")))

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cpu" or args.device == "cuda":
        device = args.device
    else:
        raise ValueError("Unknown device type")

    print(f"Working on device: {device} \n")

    # Dataset parameters
    grid_size, grid_size_demo = config["GRID_SIZE"], config["GRID_SIZE_DEMO"]

    print(f"Observation env size: {grid_size}")
    print(f"Demonstration env size: {grid_size_demo} \n")

    # Set the zero-padding --> model's size
    max_length_obs = grid_size * grid_size
    max_length_demo = +grid_size_demo * 10

    print(f"Maximal length on obs env: {max_length_obs}")
    print(f"Maximal length on demo env (demo): {max_length_demo} \n")

    # Load data
    train_storage = Storage(
        args.data_filename,
        data_mode="train",
        max_length_obs=max_length_obs,
        max_length_demo=max_length_demo,
    )
    train_data = train_storage.extract()

    val_storage = Storage(
        args.data_filename,
        data_mode="val",
        max_length_obs=max_length_obs,
        max_length_demo=max_length_demo,
    )
    val_data = val_storage.extract()

    test_storage = Storage(
        args.data_filename,
        data_mode="test",
        max_length_obs=max_length_obs,
        max_length_demo=max_length_demo,
    )
    test_data = test_storage.extract()

    train_dataset = ToMNetDataset(**train_data)
    print("Training data {}".format(len(train_data["data_paths"])))

    val_dataset = ToMNetDataset(**val_data)
    print("Validation data {}".format(len(val_data["data_paths"])))

    test_dataset = ToMNetDataset(**test_data)
    print("Test data {} \n".format(len(test_data["data_paths"])))

    # Saving weights and training config parameters
    if args.saving_name is None:
        date = datetime.now().strftime("%d.%m.%Y.%H.%M")
        saving_name = "_".join(("model", date))
    else:
        saving_name = args.saving_name
    make_dirs(f"./model_weights/{saving_name}")

    saving_path_loss = f"./model_weights/{saving_name}/prednet_model_best_loss.pt"
    config_saving_path_loss = f"./model_weights/{saving_name}/config_best_loss.json"

    saving_path_metric = f"./model_weights/{saving_name}/prednet_model_best_metric.pt"
    config_saving_path_metric = f"./model_weights/{saving_name}/config_best_metric.json"

    saving_path_dist = f"./model_weights/{saving_name}/prednet_model_best_dist.pt"
    config_saving_path_dist = f"./model_weights/{saving_name}/config_best_dist.json"

    outputs_saving_path = f"./model_weights/{saving_name}/outputs.json"

    # Training parameters
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    n_epochs = args.n_epochs

    # Load data and model
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    regnet = RegNet(
        num_input=4,
        num_step_obs=max_length_obs,
        num_steps_demo_eval=max_length_demo,
        grid_size_demo=grid_size_demo,
        grid_size_obs=grid_size,
        device=device,
    )

    optimizer = optim.Adam(regnet.parameters(), lr=learning_rate)

    regnet, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        regnet, optimizer, train_loader, val_loader, test_loader
    )

    is_main_process = accelerator.is_main_process

    # Training loop

    criterion_mae = nn.L1Loss()
    criterion_mse = nn.MSELoss()

    training_outputs = {}
    validation_outputs = {}
    test_outputs = {}

    best_model_metric = 0
    best_model_loss = 1e10
    for epoch in range(n_epochs):
        # Training loop
        tot_loss = 0
        metric = 0

        for i, batch in enumerate(tqdm(train_loader)):
            past_traj, demo, target_reward = batch

            past_traj = past_traj.double().to(device)
            demo = demo.double().to(device)
            target_reward = target_reward.long().to(device)

            pred_reward, e_char, e_mental = regnet(past_traj, demo)

            loss = criterion_mse(pred_reward, target_reward)

            # Backpropagation
            optimizer.zero_grad()

            # loss.mean().backward()
            accelerator.backward(loss.mean())
            optimizer.step()

            tot_loss += loss.item()

            metric += criterion_mae(pred_reward, target_reward).item()

        train_dict = dict(
            metric=metric / len(train_loader), loss=tot_loss / len(train_loader)
        )

        train_msg = "Train| Epoch {} Loss | {:.4f} | Metric | {:.4f} | ".format(
            epoch, train_dict["loss"], train_dict["metric"]
        )
        if is_main_process:
            training_outputs[epoch] = dict(
                loss=train_dict["loss"], metric=train_dict["metric"]
            )

        # Evaluate on the validation set
        tot_loss_val = 0
        metric_val = 0

        for i, batch in enumerate(tqdm(val_loader)):
            with torch.no_grad():
                past_traj, demo, target_reward = batch

                past_traj = past_traj.double().to(device)
                demo = demo.double().to(device)
                target_reward = target_reward.long().to(device)

                pred_reward, e_char, e_mental, query_state = regnet(past_traj, demo)
                loss = criterion_mae(pred_reward, target_reward)

            tot_loss_val += loss.item()
            metric_val += criterion_mae(pred_reward, target_reward).item()

        eval_val_dict = dict(
            metric=metric_val / len(val_loader), loss=tot_loss_val / len(val_loader)
        )

        eval_val_msg = "Val| Loss | {:.4f} | Metric | {:.4f} | ".format(
            eval_val_dict["loss"], eval_val_dict["metric"]
        )
        train_msg += eval_val_msg

        if is_main_process:
            print(train_msg)

            validation_outputs[epoch] = dict(
                loss=eval_val_dict["loss"], metric=eval_val_dict["metric"]
            )

            # Save best model based on the metric on the validation set
            if eval_val_dict["metric"] > best_model_metric:
                best_model_metric = eval_val_dict["metric"]

                torch.save(regnet.state_dict(), saving_path_metric)  # save model
                training_config = dict(
                    n_epochs=epoch,
                    batch_size=batch_size,
                    lr=learning_rate,
                    data_filename=args.data_filename,
                )

                with open(config_saving_path_metric, "w") as f:  # save config
                    json.dump(training_config, f)

            # Save best model based on the loss value on the validation set
            if eval_val_dict["loss"] < best_model_loss:
                best_model_loss = eval_val_dict["loss"]

                torch.save(regnet.state_dict(), saving_path_loss)  # save model
                training_config = dict(
                    n_epochs=epoch,
                    batch_size=batch_size,
                    lr=learning_rate,
                    data_filename=args.data_filename,
                )

                with open(config_saving_path_loss, "w") as f:  # save config
                    json.dump(training_config, f)

            # Save outputs
            dict_outputs = dict(train=training_outputs, val=validation_outputs)

            with open(outputs_saving_path, "w") as f:
                json.dump(dict_outputs, f)

    # Evaluation on the test set
    tot_loss_test = 0
    metric_test = 0

    for i, batch in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            past_traj, demo, target_reward = batch

            past_traj = past_traj.double().to(device)
            demo = demo.double().to(device)
            target_reward = target_reward.long().to(device)

            pred_reward, e_char, e_mental = regnet(past_traj, demo)
            loss = criterion_mae(pred_reward, target_reward)

        tot_loss_test += loss.item()
        metric_test += criterion_mae(pred_reward, target_reward).item()

    eval_test_dict = dict(
        metric=metric_test / len(test_loader), loss=tot_loss_test / len(test_loader)
    )

    eval_test_msg = "Eval on test| Epoch {} Loss | {:.4f} | Metric | {:.4f} | ".format(
        epoch, eval_test_dict["loss"], eval_test_dict["metric"]
    )

    if is_main_process:
        test_outputs[n_epochs - 1] = dict(
            loss=eval_test_dict["loss"], metric=eval_test_dict["metric"]
        )
        print(eval_test_msg)

        # Save outputs (MSE loss and metric)
        dict_outputs = dict(
            train=training_outputs, val=validation_outputs, test=test_outputs
        )

        with open(outputs_saving_path, "w") as f:
            json.dump(dict_outputs, f)

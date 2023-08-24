import numpy as np
import os
import torch
import argparse
import json

from datetime import datetime
import torch.optim as optim

from storage import Storage
from dataset import ToMNetDataset
from torch.utils.data import DataLoader

import sys

# global_path = '/home/chetouani/Documents/STAGE_Clemence/ISIR_internship_ToM_gridworld/gridworld_setup'
global_path = (
    "/gpfswork/rech/kcr/uxv44vw/clemence/ISIR_internship_ToM_gridworld/gridworld_setup"
)

sys.path.append(".")
sys.path.append("./../")

from utils import make_dirs

from model import PredNet

# from model import PredNet


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

    loading_path = f"{global_path}/neural_network_ToM/data/{args.data_filename}"
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
    max_length_demo_eval = grid_size_demo * grid_size_demo // 2  # + grid_size_demo * 2

    print(f"Maximal length on obs env: {max_length_obs}")
    print(f"Maximal length on demo env (demo + eval): {max_length_demo_eval} \n")

    # Load data
    train_storage = Storage(
        args.data_filename,
        data_mode="train",
        max_length_obs=max_length_obs,
        max_length_demo_eval=max_length_demo_eval,
    )
    train_data = train_storage.extract()

    val_storage = Storage(
        args.data_filename,
        data_mode="val",
        max_length_obs=max_length_obs,
        max_length_demo_eval=max_length_demo_eval,
    )
    val_data = val_storage.extract()

    test_storage = Storage(
        args.data_filename,
        data_mode="test",
        max_length_obs=max_length_obs,
        max_length_demo_eval=max_length_demo_eval,
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

    saving_path_acc = f"./model_weights/{saving_name}/prednet_model_best_acc.pt"
    config_saving_path_acc = f"./model_weights/{saving_name}/config_best_acc.json"

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

    prednet = PredNet(
        num_input=4,
        num_step_obs=max_length_obs,
        num_steps_demo_eval=max_length_demo_eval,
        grid_size_demo=grid_size_demo,
        grid_size_obs=grid_size,
        device=device,
    )

    optimizer = optim.Adam(prednet.parameters(), lr=learning_rate)

    # Training loop
    training_outputs = {}
    validation_outputs = {}
    test_outputs = {}

    best_model_acc = 0
    best_model_loss = 1e10
    for epoch in range(n_epochs):
        train_dict = prednet.train(train_loader, optimizer)
        train_msg = "Train| Epoch {} Loss | {:.4f} | Acc | {:.4f} | ".format(
            epoch, train_dict["loss"], train_dict["accuracy"]
        )
        training_outputs[epoch] = dict(
            loss=train_dict["loss"], accuracy=train_dict["accuracy"]
        )
        # Evaluate on the validation set
        eval_val_dict = prednet.evaluate(val_loader)
        eval_val_msg = "Val| Loss | {:.4f} | Acc | {:.4f} | ".format(
            eval_val_dict["loss"], eval_val_dict["accuracy"]
        )
        validation_outputs[epoch] = dict(
            loss=eval_val_dict["loss"], accuracy=eval_val_dict["accuracy"]
        )
        train_msg += eval_val_msg
        print(train_msg)

        # Save best model based on the accuracy on the validation set
        if eval_val_dict["accuracy"] > best_model_acc:
            best_model_acc = eval_val_dict["accuracy"]

            torch.save(prednet.state_dict(), saving_path_acc)  # save model
            training_config = dict(
                n_epochs=epoch,
                batch_size=batch_size,
                lr=learning_rate,
                data_filename=args.data_filename,
            )

            with open(config_saving_path_acc, "w") as f:  # save config
                json.dump(training_config, f)

        # Save best model based on the loss value on the validation set
        if eval_val_dict["loss"] < best_model_loss:
            best_model_loss = eval_val_dict["loss"]

            torch.save(prednet.state_dict(), saving_path_loss)  # save model
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

    # Evaluation
    eval_test_dict = prednet.evaluate(test_loader)
    eval_test_msg = "Eval on test| Epoch {} Loss | {:.4f} | Acc | {:.4f} | ".format(
        epoch, eval_test_dict["loss"], eval_test_dict["accuracy"]
    )
    test_outputs[n_epochs - 1] = dict(
        loss=eval_test_dict["loss"], accuracy=eval_test_dict["accuracy"]
    )
    print(eval_test_msg)

    # Save outputs (NLL loss and accuracy)
    dict_outputs = dict(
        train=training_outputs, val=validation_outputs, test=test_outputs
    )

    with open(outputs_saving_path, "w") as f:
        json.dump(dict_outputs, f)

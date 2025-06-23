import os
import json
import math
import torch
import argparse
import numpy as np
import pandas as pd

from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data import Dataset
from src.model import VBOCformerNeuralNetwork
from src.loss import ELBOLoss
from src.trainer import Trainer
from src.utils import set_seed, set_device

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def main(args, random_state):
    satellite = args.satellite
    cuda_index = args.cuda_index

    set_seed(random_state)

    dataset_path = f'./dataset'
    dataset_file = os.path.join(dataset_path, f'{satellite.lower()}.npz')
    np_data = np.load(dataset_file)
    dataset = {}
    for key in np_data:
        dataset[key] = np_data[key]
    dataset = pd.DataFrame.from_dict(dataset)
    features_train, features_validate_test, target_train, target_validate_test = train_test_split(dataset.iloc[:, :-1], dataset.iloc[:, [-1]], test_size=0.2, train_size=0.8,
                                                                                                  random_state=random_state, shuffle=True, stratify=None)
    features_validate, _, target_validate, _ = train_test_split(features_validate_test, target_validate_test, test_size=0.5, train_size=0.5,
                                                                random_state=random_state, shuffle=True, stratify=None)

    features_scaler = StandardScaler().fit(features_train)
    target_scaler = StandardScaler().fit(target_train)
    features_train_scaled = features_scaler.transform(features_train)
    features_validate_scaled = features_scaler.transform(features_validate)
    target_train_scaled = target_scaler.transform(target_train).reshape(-1)
    target_validate_scaled = target_scaler.transform(target_validate).reshape(-1)

    train_dataset = Dataset(Xs=features_train_scaled, ys=target_train_scaled)
    train_dataloader = DataLoader(dataset = train_dataset,
                                  batch_size = 64,
                                  shuffle = True,
                                  num_workers = 0,
                                  pin_memory = True,
                                  drop_last = False)
        
    validate_dataset = Dataset(Xs=features_validate_scaled, ys=target_validate_scaled)
    validate_dataloader = DataLoader(dataset = validate_dataset,
                                     batch_size = 64,
                                     shuffle = True,
                                     num_workers = 0,
                                     pin_memory = True,
                                     drop_last = False)
    
    device = set_device(cuda_index)

    with open(f"./optimized models/{satellite.lower()}.json", 'r') as f:
        parameters = json.load(f)
    hidden_layer_sizes = [160 *  2 ** i for i in range(int(math.log(parameters['d_model'] * 10 // 160, 2))-1, -1, -1)]
    model = VBOCformerNeuralNetwork(seq_len = len(dataset.columns) - 1,
                                    out_features = 1,
                                    hidden_layer_sizes = hidden_layer_sizes,
                                    d_model = parameters['d_model'],
                                    num_heads = parameters['num_heads'],
                                    num_encoder_layers = parameters['num_encoder_layers'],
                                    dim_feedforward = parameters['dim_feedforward'],
                                    dropout_rate = parameters['dropout_rate'],
                                    max_length = 5000,
                                    activation = F.relu,
                                    layer_norm_eps = 1e-5,
                                    encoder_norm = True,
                                    prior_pi = 0.5,
                                    prior_mu1 = 0.0,
                                    prior_mu2 = 0.0,
                                    prior_sigma1 = 0.1,
                                    prior_sigma2 = 0.4,
                                    mu_init = 0.0,
                                    sigma_init = -7.0,
                                    bias = True,
                                    frozen = False,
                                    norm_first = False)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(),
                            lr = 0.0001,
                            betas = (0.9, 0.999),
                            eps = 1e-8,
                            weight_decay = 0.01,
                            amsgrad  = True,
                            fused = True)
    
    loss_func = ELBOLoss(model = model,
                         inner_loss = nn.MSELoss(),
                         kl_weight = 0.0001,
                         num_samples = 10)
    
    config = {'epochs': 100,
              'save_best': True,
              'patience': 10,
              'device': device}
    trainer = Trainer(model = model,
                      optimizer = optimizer,
                      loss_func = loss_func,
                      config = config,
                      show_progressbar = True,
                      train_dataloader = train_dataloader,
                      validate_dataloader = validate_dataloader,)
    _, best_model_state_dict = trainer.train()
    model.load_state_dict(best_model_state_dict)
    model.to('cpu')
    model.unfreeze()
    model.eval()

    input = torch.rand_like(torch.tensor(features_train_scaled[0])).to(torch.float32).unsqueeze(0)
    
    if not os.path.exists(f'./trained models'):
        os.makedirs(f'./trained models')
    with torch.no_grad():
        torch.onnx.export(
            model = model,
            args = (input,),
            f = f'./trained models/{satellite.lower()}.onnx',
            export_params = True,
            verbose = True,
            input_names = ['features'],
            output_names = ['target'],
            opset_version  = 14,
            dynamic_axes = {
                'features': {0: 'batch_size'},
                'target': {0: 'batch_size'},
            }
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--satellite', type=str, choices=['MERIS', 'MODIS-Aqua', 'MODIS-Terra', 'SeaWiFS', 'SNPP-VIIRS'], required=True,
                        help='Select one of the following sensors: MERIS, MODIS-Aqua, MODIS-Terra, SeaWiFS, SNPP-VIIRS.')
    parser.add_argument('--cuda_index', type=int, required=False)
    args = parser.parse_args()

    global_seed = 0
    rstate = np.random.default_rng(seed=global_seed)
    random_state = rstate.integers(2**31 - 1)
    main(args, random_state)
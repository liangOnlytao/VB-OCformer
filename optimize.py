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

from hyperopt import hp, fmin, tpe, Trials
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def main(args, random_state):
    satellite = args.satellite
    cuda_index = args.cuda_index
    max_evals = args.max_evals

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

    space = {'d_model': hp.choice('d_model', [32, 64, 128, 256, 512]),
             'num_heads': hp.choice('num_heads', [2, 4, 8]),
             'num_encoder_layers': hp.choice('num_encoder_layers', list(range(1, 7))),
             'dim_feedforward': hp.choice('dim_feedforward', [128, 256, 512, 1024, 2048]),
             'dropout_rate': hp.choice('dropout_rate', [0.1, 0.2, 0.3])}
    
    def objective(hyperparameters, X_train=features_train_scaled, y_train=target_train_scaled, X_validate=features_validate_scaled, y_validate=target_validate_scaled):
        train_dataset = Dataset(Xs=X_train, ys=y_train)
        train_dataloader = DataLoader(dataset = train_dataset,
                                      batch_size = 64,
                                      shuffle = True,
                                      num_workers = 0,
                                      pin_memory = True,
                                      drop_last = False)
        
        validate_dataset = Dataset(Xs=X_validate, ys=y_validate)
        validate_dataloader = DataLoader(dataset = validate_dataset,
                                         batch_size = 64,
                                         shuffle = True,
                                         num_workers = 0,
                                         pin_memory = True,
                                         drop_last = False)
        
        device = set_device(cuda_index)

        hidden_layer_sizes = [160 * 2 ** i for i in range(int(math.log(hyperparameters['d_model'] * 10 // 160, 2))-1, -1, -1)]
        model = VBOCformerNeuralNetwork(seq_len = len(dataset.columns) - 1,
                                        out_features = 1,
                                        hidden_layer_sizes = hidden_layer_sizes,
                                        d_model = hyperparameters['d_model'],
                                        num_heads = hyperparameters['num_heads'],
                                        num_encoder_layers = hyperparameters['num_encoder_layers'],
                                        dim_feedforward = hyperparameters['dim_feedforward'],
                                        dropout_rate = hyperparameters['dropout_rate'],
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
                             num_samples = 1)
        
        config = {'epochs': 50,
                  'save_best': False,
                  'patience': 10,
                  'device': device}
        trainer = Trainer(model = model,
                          optimizer = optimizer,
                          loss_func = loss_func,
                          config = config,
                          show_progressbar = False,
                          train_dataloader = train_dataloader,
                          validate_dataloader = validate_dataloader)
        loss = trainer.train()

        return loss
    
    def optimize(max_evals):
        trials = Trials()
        parameter_best = fmin(fn = objective,
                              space = space,
                              algo = tpe.suggest,
                              max_evals = max_evals,
                              trials = trials,
                              rstate = np.random.default_rng(seed=random_state))
        
        return parameter_best
    
    parameter_best = optimize(max_evals)

    d_model = [32, 64, 128, 256, 512]
    num_heads = [2, 4, 8]
    num_encoder_layers = list(range(1, 7))
    dim_feedforward = [128, 256, 512, 1024, 2048]
    dropout_rate = [0.1, 0.2, 0.3]
    parameters = {'d_model': d_model[parameter_best['d_model']],
                  'num_heads': num_heads[parameter_best['num_heads']],
                  'num_encoder_layers': num_encoder_layers[parameter_best['num_encoder_layers']],
                  'dim_feedforward': dim_feedforward[parameter_best['dim_feedforward']],
                  'dropout_rate': dropout_rate[parameter_best['dropout_rate']]}
    
    if not os.path.exists(f'./optimized models'):
        os.makedirs(f'./optimized models')
    with open(f"./optimized models/{satellite.lower()}.json", 'w') as f:
        json.dump(parameters, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--satellite', type=str, choices=['MERIS', 'MODIS-Aqua', 'MODIS-Terra', 'SeaWiFS', 'SNPP-VIIRS'], required=True,
                        help='Select one of the following sensors: MERIS, MODIS-Aqua, MODIS-Terra, SeaWiFS, SNPP-VIIRS.')
    parser.add_argument('--cuda_index', type=int, required=False)
    parser.add_argument("--max_evals", type=int, required=True)
    args = parser.parse_args()

    global_seed = 0
    rstate = np.random.default_rng(seed=global_seed)
    random_state = rstate.integers(2**31 - 1)
    main(args, random_state)
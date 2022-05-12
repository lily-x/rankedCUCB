""" setup the three domains used for experiments

Lily Xu
September 2021 """

from adversary import EffortAdversary, RealEffortAdversary
import numpy as np
import pickle
import torch
import math

def get_wildlife(N, G, VERBOSE):
    raise Exception('cannot share wildlife domain in publicly released code due to sensitive nature of data, sorry!')

def get_synthetic(N, G, VERBOSE):
    """ fully synthetic dataset """
    # generate data
    num_features  = 4
    num_hids      = 100
    num_layers    = 10
    num_instances = 1
    num_samples   = 6

    data = generate_synthetic_data(N, num_features, num_hids,
                           num_layers, num_instances, num_samples,
                           attacker_w=-4.0)
    attacker_vals = data[2]

    # scale to [0, 1]
    attacker_vals -= attacker_vals.min()
    attacker_vals /= attacker_vals.max()

    # scale to [1, 11]
    attacker_vals *= 10
    attacker_vals += 1

    if VERBOSE:
        print('attacker vals', attacker_vals)
        print('---')

    adversary = EffortAdversary(attacker_vals)

    group_distrib = np.random.rand(G,N)
    # normalize sum to 1 per row
    group_distrib /= group_distrib.sum(axis=1)[:, None]

    print('group distribution')
    print(np.round(group_distrib, 2))

    return adversary, group_distrib



def generate_synthetic_data(num_targets, num_features, num_hids,
                       num_layers, num_instances, num_samples,
                       attacker_w=-4.0, noise=None):

    features = np.random.uniform(low=-10., high=10., size=(num_instances, num_targets, num_features))

    layer_list_attacker = []
    for i in range(num_layers):
        if i == 0:
            layer_list_attacker.append(torch.nn.Linear(num_features, num_hids))
        else:
            layer_list_attacker.append(torch.nn.Linear(num_hids, num_hids))
        layer_list_attacker.append(torch.nn.ReLU())
    layer_list_attacker.append(torch.nn.Linear(num_hids, 1))

    layer_list_defender = []
    for i in range(num_layers):
        if i == 0:
            layer_list_defender.append(torch.nn.Linear(num_features, num_hids))
        else:
            layer_list_defender.append(torch.nn.Linear(num_hids, num_hids))
        layer_list_defender.append(torch.nn.ReLU())
    layer_list_defender.append(torch.nn.Linear(num_hids, 1))

    attacker_model = torch.nn.Sequential(*layer_list_attacker)
    attacker_values = attacker_model(torch.from_numpy(features).float()).squeeze() * 10.
    if noise:
        attacker_values = attacker_values + torch.normal(mean=0.0, std=noise)
    # for numerical stability
    attacker_values = attacker_values - torch.max(attacker_values)

    defender_model = torch.nn.Sequential(*layer_list_defender)
    defender_values = defender_model(torch.from_numpy(features).float()).squeeze()
    # defender values are negative
    defender_values = defender_values - torch.max(defender_values)
    defender_values = defender_values / -torch.min(defender_values)

    return (features / math.sqrt(1. / 12. * 20 ** 2), defender_values.detach().numpy(), attacker_values.detach().numpy(), attacker_w)

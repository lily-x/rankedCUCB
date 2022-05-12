""" Lily Xu
read in result CSVs to process avg and stdev

September 2021
"""

import pandas as pd
import numpy as np
from scipy import stats
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('tkagg')

from driver import smooth

# domain, N, B, G, lamb, max_seed = 'wildlife', 25, 5, 4, 0.3, 31
domain, N, B, G, lamb, max_seed = 'synthetic', 40, 10, 5, 0.3, 31


T = 500
num_discretization = 10

lamb = 0.1
smooth_weight = 0.7 # for exponential moving average

seeds = range(1, max_seed)

ranked      = {}
ranked_conf = {}
naive       = {}
lizard      = {}
random      = {}
opt         = {}

dir = f'results/{domain}'
results = {'ranked': ranked, 'ranked_conf': ranked_conf, 'naive': naive, 'lizard': lizard, 'random': random, 'opt': opt}

for alg in results.values():
    alg['obj']      = np.zeros((T, len(seeds)))
    alg['reward']   = np.zeros((T, len(seeds)))
    alg['priority'] = np.zeros((T, len(seeds)))

for i, seed in enumerate(seeds):
    in_file = f'{dir}/ranked_bandits_N{N}_B{B}_G{G}_T{T}_lamb{lamb}_discre{num_discretization}_s{seed}.csv'
    print(f'reading {in_file}')
    df = pd.read_csv(in_file)
    for name, alg in results.items():
        alg['obj'][:, i]      = smooth(df[f'{name}_obj'], smooth_weight)
        alg['reward'][:, i]   = smooth(df[f'{name}_reward'], smooth_weight)
        alg['priority'][:, i] = smooth(df[f'{name}_priority'], smooth_weight)

dict_out = {}
for name, alg in results.items():
    dict_out[f'{name}_obj_avg']      = alg['obj'].mean(axis=1)
    dict_out[f'{name}_obj_sem']      = stats.sem(alg['obj'], axis=1)
    dict_out[f'{name}_reward_avg']   = alg['reward'].mean(axis=1)
    dict_out[f'{name}_reward_sem']   = stats.sem(alg['reward'], axis=1)
    dict_out[f'{name}_priority_avg'] = alg['ranked'].mean(axis=1)
    dict_out[f'{name}_priority_sem'] = stats.sem(alg['ranked'], axis=1)

out_file = f'{dir}/results_{domain}_N{N}_B{B}_G{G}_T{T}_lamb{lamb}_discre{num_discretization}.csv'
print(f'writing {out_file}')
df_out = pd.DataFrame(dict_out)
df_out.index.name = 't'  # index column name
df_out.to_csv(out_file)

x_vals = np.arange(T)
plt.figure(figsize=(10,5))
plt.suptitle(domain)
for i, metric in enumerate(['obj', 'reward', 'ranked']):
    plt.subplot(1,3,i+1)
    for name, alg in results.items():
        y_vals = dict_out[f'{name}_{metric}_avg']
        y_err  = dict_out[f'{name}_{metric}_sem']
        plt.plot(x_vals, y_vals, label=name)
        plt.fill_between(x_vals, y_vals-y_err, y_vals+y_err, alpha=.3)
    plt.legend()
    plt.title(metric)

plt.savefig(f'{dir}/plot_{domain}_N{N}_B{B}_G{G}_T{T}_lamb{lamb}_discre{num_discretization}.png')
plt.show()

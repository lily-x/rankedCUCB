""" given results, plot Pareto frontier

Lily Xu
September 2021 """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
mpl.use('tkagg')

fig, ax = plt.subplots()

domain, N, G, B = 'synthetic', 40, 5, 10
# domain, N, G, B = 'wildlife',   25, 4, 5

lamb_vals = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]

seeds = np.arange(1, 30)

methods = ['priority', 'opt', 'lizard']
method_vals = {}

# num last values to average over
n_avg_over = 10

for method in methods:
    method_vals[f'{method}_reward_avg']      = np.zeros(len(lamb_vals))
    method_vals[f'{method}_reward_stderr']   = np.zeros(len(lamb_vals))
    method_vals[f'{method}_priority_avg']    = np.zeros(len(lamb_vals))
    method_vals[f'{method}_priority_stderr'] = np.zeros(len(lamb_vals))

    for i, lamb in enumerate(lamb_vals):
        reward_vals = []
        priority_vals = []
        for seed in seeds:
            file_in = f'results/{domain}/ranked_cucb_N{N}_B{B}_G{G}_T500_lamb{lamb}_discre10_s{seed}.csv'

            results = pd.read_csv(file_in)

            # take average of last X values
            reward   = np.mean(results[f'{method}_reward'].iloc[-n_avg_over:])
            priority = np.mean(results[f'{method}_priority'].iloc[-n_avg_over:])

            reward_vals.append(reward)
            priority_vals.append(priority)

        method_vals[f'{method}_reward_avg'][i]      = np.mean(reward_vals)
        method_vals[f'{method}_priority_avg'][i]    = np.mean(priority_vals)
        method_vals[f'{method}_reward_stderr'][i]   = stats.sem(reward_vals)
        method_vals[f'{method}_priority_stderr'][i] = stats.sem(priority_vals)


dict_out = {}
dict_out['lamb_vals'] = lamb_vals
for method in methods:
    dict_out[f'{method}_reward_avg']      = method_vals[f'{method}_reward_avg']
    dict_out[f'{method}_reward_stderr']   = method_vals[f'{method}_reward_stderr']
    dict_out[f'{method}_priority_avg']    = method_vals[f'{method}_priority_avg']
    dict_out[f'{method}_priority_stderr'] = method_vals[f'{method}_priority_stderr']

data_out = pd.DataFrame(dict_out)
data_out.set_index('lamb_vals')
data_out.to_csv(f'pareto_frontier_{domain}_N{N}_B{B}_G{G}.csv', index_label='i')



for i, lamb in enumerate(lamb_vals):
    for method in methods:
        ax.annotate(lamb, (method_vals[f'{method}_reward_avg'][i], method_vals[f'{method}_priority_avg'][i]))
# plot standard err
plt.scatter(method_vals['ranked_cucb_reward_avg'], method_vals['ranked_cucb_priority_avg'], s=20*method_vals['ranked_cucb_reward_stderr'], c='orange', label='rankedCUCB')
plt.scatter(method_vals['lizard_reward_avg'], method_vals['lizard_priority_avg'], s=20*method_vals['lizard_reward_stderr'], c='g', label='lizard')
plt.scatter(method_vals['opt_reward_avg'], method_vals['opt_priority_avg'], s=20*method_vals['opt_reward_stderr'], c='r', label='opt')

plt.title(f'Pareto frontier - {domain}')
plt.xlabel('reward')
plt.ylabel('priority')
plt.legend()

plt.savefig(f'figures/pareto_frontier_{domain}_N{N}_B{B}_G{G}', bbox_inches='tight')
plt.show()

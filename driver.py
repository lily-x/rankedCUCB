""" run experiments

Lily Xu
July 2021 """

from ranked_cucb import RankedCUCB
from ranked_linear import RankedLinear
import numpy as np
import pandas as pd
from datetime import datetime
import sys, os
import argparse
import pickle
from get_domain import get_wildlife, get_synthetic

import matplotlib.pyplot as plt

if not os.path.exists('figures'):
    os.makedirs('figures')


def get_random(n_random=100):
    """ calculate random obj/reward/priority """
    max_effort = 1

    random_obj      = np.zeros(n_random)
    random_reward   = np.zeros(n_random)
    random_priority = np.zeros(n_random)
    random_kendall  = np.zeros(n_random)
    for j in range(n_random):
        beta = np.random.rand(N)
        beta /= np.sum(beta) # sum to budget
        beta *= budget

        # ensure we never exceed effort = 1 on any target
        while len(np.where(beta > max_effort)[0]) > 0:
            excess_idx = np.where(beta > 1)[0][0]
            excess = beta[excess_idx] - max_effort

            beta[excess_idx] = max_effort

            # add "excess" amount of effort randomly on other targets
            add = np.random.uniform(size=N - 1)
            add = (add / np.sum(add)) * excess

            beta[:excess_idx] += add[:excess_idx]
            beta[excess_idx+1:] += add[excess_idx:]

        obj, reward, priority = ranked_cucb.evaluate_true_objective(beta)
        kendall = ranked_cucb.evaluate_kendall_tau(reward)

        random_obj[j]      = obj
        random_reward[j]   = np.sum(reward)
        random_priority[j] = priority

    return random_obj, random_reward, random_priority, random_kendall


def cum_avg(vals):
    """ smoothing option: cumulative average """
    return np.cumsum(vals) / (np.arange(len(vals)) + 1)


def smooth(rewards, weight=0.7):
    """ smoothed exponential moving average """
    prev = rewards[0]
    smoothed = np.zeros(len(rewards))
    for i, val in enumerate(rewards):
        smoothed_val = prev * weight + (1 - weight) * val
        smoothed[i] = smoothed_val
        prev = smoothed_val

    return smoothed


def make_plot(ranked, naive, lizard, random, opt, xvals, xlabel, ylabel, filename,
    smooth_plot=False):
    """ generate plots """
    if smooth_plot:
        weight = 0.9
        ranked      = smooth(ranked, weight)
        naive     = smooth(naive, weight)
        lizard    = smooth(lizard, weight)
        filename  = filename + '-smooth'
    plt.figure()
    plt.plot(xvals, ranked, c='blue', label='ranked')
    plt.plot(xvals, naive, c='purple', label='naive_rank')
    plt.plot(xvals, lizard, c='green', label='lizard')
    plt.plot(xvals, random * np.ones(len(xvals)), c='pink', label='random')
    plt.plot(xvals, opt * np.ones(len(xvals)), c='red', label='opt')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'$N={N}$, $G={G}$, $\lambda = {lamb}$, $T={T}$, $B={budget}$')
    plt.legend(loc='best')
    plt.savefig(f'figures/{prefix}-N{N}-G{G}-lamb{lamb}-T{T}-B{budget}-{filename}.png', bbox_inches='tight')
    if local:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets', '-N', help='number of targets', type=int, default=20)
    parser.add_argument('--horizon', '-T', help='time horizon', type=int, default=500)
    parser.add_argument('--budget',  '-B', help='patrol budget', type=int, default=3)
    parser.add_argument('--groups',  '-G', help='number of groups', type=int, default=3)
    parser.add_argument('--discretization', '-d', help='num discretization', type=int, default=10)
    parser.add_argument('--lamb', '-l', help='lambda', type=float, default=0.4)
    parser.add_argument('--seed', '-s', help='random seed', type=int, default=0)
    parser.add_argument('--verbose', '-V', help='if True, then verbose output (default False)', action='store_true')
    parser.add_argument('--local', '-L', help='if True, running locally (default False)', action='store_true')
    parser.add_argument('--domain', '-D', help='domain to run on {synthetic, wildlife}', type=str, default='synthetic')
    parser.add_argument('--prefix', '-p', help='prefix for file writing', type=str, default='')

    args = parser.parse_args()
    local = args.local

    if local:
        # if running on local computer, need this to prevent matplotlib error
        import matplotlib as mpl
        mpl.use('tkagg')

    N      = args.targets   # num targets
    T      = args.horizon   # num timesteps
    budget = args.budget    # budget
    G      = args.groups    # num groups
    lamb   = args.lamb      # lambda tuning parameter
    seed   = args.seed      # random seed
    domain = args.domain
    prefix = args.prefix
    num_discretization = args.discretization # number of discretizations

    np.random.seed(seed)

    assert 0 <= lamb <= 1

    VERBOSE = args.verbose

    optimal = None

    if domain == 'synthetic':
        adversary, group_distrib = get_synthetic(N, G, VERBOSE)
    elif domain == 'wildlife':
        adversary, group_distrib = get_wildlife(N, G, VERBOSE)
    else:
        raise Exception(f'domain {domain} not implemented')

    ranked_cucb = RankedLinear(N, G, num_discretization, lamb, group_distrib,
            adversary, T, optimal, budget, VERBOSE=VERBOSE)

    ranked_obj, ranked_reward, ranked_priority, ranked_beta, ranked_kendall = ranked_cucb.run_alg(approach='RankedCUCB')

    ranked_cucb.gamma_conf = True
    # baseline: naive ranking (added with IJCAI submission)
    naive_obj, naive_reward, naive_priority, naive_beta, naive_kendall = ranked_cucb.run_alg(approach='naive_rank')

    # baseline: LIZARD
    lizard_obj, lizard_reward, lizard_priority, lizard_beta, lizard_kendall = ranked_cucb.run_alg(approach='LIZARD')

    # baseline: random
    random_obj, random_reward, random_priority, random_kendall = get_random()
    random_obj      = np.mean(random_obj)
    random_reward   = np.mean(random_reward)
    random_priority = np.mean(random_priority)
    random_kendall  = np.mean(random_kendall)

    # baseline: optimal -- calculate optimal obj/reward/priority
    opt_obj, opt_reward, opt_priority, opt_beta, opt_kendall = ranked_cucb.evaluate_optimal_arm()


    ranked_obj_i = np.argmax(ranked_obj)
    print('top ranked obj', ranked_obj_i, ranked_obj[ranked_obj_i])
    print('top ranked obj beta', ranked_beta[ranked_obj_i])

    # save out results CSV
    now = datetime.now()
    str_time = now.strftime('%d-%m-%Y_%H:%M:%S')

    prefix = ''
    out_dir = f'results/{domain}'
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

    filename = f'{out_dir}/{prefix}ranked_cucb_N{N}_B{budget}_G{G}_T{T}_lamb{lamb}_discre{num_discretization}_s{seed}.csv'
    with open(filename, 'w') as f:
        df_out = pd.DataFrame({'ranked_obj': ranked_obj, 'ranked_reward': ranked_reward, 'ranked_priority': ranked_priority,
            'naive_obj': naive_obj, 'naive_reward': naive_reward, 'naive_priority': naive_priority,
            'lizard_obj': lizard_obj, 'lizard_reward': lizard_reward, 'lizard_priority': lizard_priority,
            'random_obj': np.ones(T) * random_obj, 'random_reward': np.ones(T) * random_reward, 'random_priority': np.ones(T) * random_priority,
            'opt_obj': np.ones(T) * opt_obj, 'opt_reward': np.ones(T) * opt_reward, 'opt_priority': np.ones(T) * opt_priority})
        df_out.index.name = 't'  # index column name
        df_out.to_csv(filename)


    # save out plots
    if local:
        # EXPERIMENT 1
        # objective over time - just changing t, like in AAAI-21 paper
        make_plot(ranked_obj, naive_obj, lizard_obj, random_obj, opt_obj, np.arange(T),
                'Timestep $t$', 'Objective value', 'obj-time')

        # EXPERIMENT 2
        # reward over time - just changing t
        make_plot(ranked_reward, naive_reward, lizard_reward, random_reward, opt_reward, np.arange(T),
                'Timestep $t$', 'Reward', 'reward-time')

        # priority over time - just changing t
        make_plot(ranked_priority, naive_priority, lizard_priority, random_priority, opt_priority, np.arange(T),
                'Timestep $t$', 'Priority', 'priority-time')

        make_plot(ranked_kendall, naive_kendall, lizard_kendall, random_kendall, opt_kendall, np.arange(T),
                'Timestep $t$', 'Kendall Tau', 'priority-kendall')


        make_plot(ranked_obj, naive_obj, lizard_obj, np.mean(random_obj), opt_obj, np.arange(T),
                'Timestep $t$', 'Objective value', 'obj-time', smooth_plot=True)
        make_plot(ranked_reward, naive_reward, lizard_reward, np.mean(random_reward), opt_reward, np.arange(T),
                'Timestep $t$', 'Reward', 'reward-time', smooth_plot=True)
        make_plot(ranked_priority, naive_priority, lizard_priority, np.mean(random_priority), opt_priority, np.arange(T),
                'Timestep $t$', 'Priority', 'priority-time', smooth_plot=True)

    # plot cumulative averages
    make_plot(cum_avg(ranked_obj), cum_avg(naive_obj), cum_avg(lizard_obj), np.mean(random_obj), opt_obj, np.arange(T),
            'Timestep $t$', 'Objective value', 'obj-time-cumavg')
    make_plot(cum_avg(ranked_reward), cum_avg(naive_reward), cum_avg(lizard_reward), np.mean(random_reward), opt_reward, np.arange(T),
            'Timestep $t$', 'Reward', 'reward-time-cumavg')
    make_plot(cum_avg(ranked_priority), cum_avg(naive_priority), cum_avg(lizard_priority), np.mean(random_priority), opt_priority, np.arange(T),
            'Timestep $t$', 'Priority', 'priority-time-cumavg')

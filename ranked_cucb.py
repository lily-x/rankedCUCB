""" implement RankedCUCB algorithm

Lily Xu
July 2021 """

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from abc import ABCMeta, abstractmethod
import sys

class RankedCUCB:
    def __init__(self, N, G, num_discretization, lamb, group_distrib,
            adversary, T, optimal, budget,
            increasingness=True, use_features=False, history=None, VERBOSE=True,
            gamma_conf=False):
        """
        params
        ------
        targets (num_targets, num_features)

        assume true_rank is in numerical order
        """
        # assert len(true_rank) == G
        assert len(group_distrib) == G
        for g in range(G):
            assert len(group_distrib[g]) == N
            assert np.abs(np.sum(group_distrib[g]) - 1) < 1e-5
        assert type(num_discretization) is int
        assert 0 <= lamb <= 1, f'lamb is {lamb}'

        self.N = N # num targets
        self.G = G # num groups
        self.delta = 1 / num_discretization
        self.lamb = lamb # priority tuning parameter
        self.true_rank = np.arange(N) # true prioritization ranking
        self.group_distrib = group_distrib # group distribution/density across targets (dict)
        self.Gamma_priority = None
        self.Gamma = None

        self.adversary = adversary
        self.T = T # num timesteps
        self.optimal = optimal
        self.increasingness = increasingness  # whether we want to build in that reward function is increasing - monotonicity
        self.use_features = use_features

        self.budget = budget

        self.VERBOSE = VERBOSE

        self.B           = [[] for _ in range(self.N)]  # list of active arms (decomposed)
        self.cum_rewards = [{} for _ in range(self.N)]  # cumulative reward
        self.n           = [{} for _ in range(self.N)]  # number of pulls

        self.n_w_history = [{} for _ in range(self.N)]  # number of pulls INCL historical data

        self.L = np.ones(self.N)  # Lipschitz constant in each dimension

        self.t = 0 # current timestep

        # initialize
        self.num_discretization = num_discretization
        eff_levels = np.linspace(0, 1, num_discretization+1)
        # eliminating floating point glitches
        eff_levels = np.round(eff_levels, 3)
        self.eff_levels = eff_levels

        self.reset()

        # integrate features by computing distance in feature space
        # (== distance in reward functions)
        self.dist = np.zeros((self.N, self.N))
        if self.use_features:
            for i1 in range(self.N):
                for i2 in range(self.N):
                    if i1 == i2: continue
                    self.dist[i1, i2] = self.adversary.compute_pwl_distance(i1, i2)

        self.t_uncover       = np.full(self.T, np.nan)  # track which arms were uncovered at each timestep
        self.t_ucb           = np.full(self.T, np.nan)
        self.exploit_rewards = np.zeros(self.T)
        self.true_reward     = np.zeros(self.T)
        self.mip_UCB         = np.zeros(self.T)

        # gamma_conf: whether to use the updated conf radius (incorporating gamma) that we derived
        self.gamma_conf = gamma_conf

        distance_diff = np.zeros(self.N)
        for i in range(self.N):
            for g in range(self.G - 1):
                for h in range(g+1, self.G):
                    distance_diff[i] += max(0, self.group_distrib[h][i] - self.group_distrib[g][i])
        self.distance_diff = distance_diff


    ############################################################
    # universal methods
    ############################################################

    def reset(self):
        for i in range(self.N):
            for eff in self.eff_levels:
                if eff not in self.n[i]:
                    self.B[i].append(eff)
                self.n[i][eff] = 0
                self.n_w_history[i][eff] = 0
                self.cum_rewards[i][eff] = 0

    def get_epsilon(self):
        """ decaying epsilon (attenuation) """
        return (self.t+1)**(-1/3)  # t+1 to avoid issues with t=0

    def get_ucb(self, i, eff):
        """ current UCB estimate for (target, effort) """
        # eliminate floating point glitches
        eff = np.round(eff, 3)

        assert 0 <= eff <= 1

        if eff == 0:
            mu = 0.
        elif self.n_w_history[i][eff] == 0:
            mu = 1.
        else:
            mu = self.cum_rewards[i][eff] / self.n_w_history[i][eff]

        conf = self.conf(i, eff)
        return self.mu_and_conf(mu, conf, i)

    def mu_and_conf(self, mu, conf, i, approach):
        """ keeping this here because it's called in two places and the first time around i didn't update get_ucb ... """
        if approach in ['LIZARD', 'naive_rank']:
            return mu + conf

        elif approach == 'RankedCUCB':
            if self.Gamma[i] >= 0:
                return mu + conf
            else:
                return mu - conf

        return mu + conf

    def calculate_group_effort(self, beta):
        """ calculate total effort allocated to each group
        based on UCB estimates """
        effective_effort = np.zeros(self.N)
        # use UCB estimates for each (target, effort) to understand effective effort
        for i in range(self.N):
            effective_effort[i] = self.get_ucb(i, beta[i])  # UCB at target i with effort beta[i]

        # calculate how much effective effort is exerted to each group
        group_effort = np.zeros(self.G)
        for g in range(self.G):
            group_effort[g] = np.sum(effective_effort * self.group_distrib[g])

        return group_effort

    def calculate_ranking(self, beta):
        """ rank according to beta effort
        returns a list that is a permutation of 1..N """

        group_effort = self.calculate_group_effort(beta)
        ranking = np.flip(np.argsort(group_effort))

        return ranking

    def calculate_true_reward(self, beta):
        """ using the ground-truth probabilities, calculate the true reward of an action beta """

        reward = np.zeros(self.N)
        for i in range(self.N):
            reward_prob = self.adversary.visit_target(i, beta[i])
            reward[i] = reward_prob
        return reward


    ############################################################
    # abstract methods
    ############################################################
    def evaluate_true_objective(self, beta):
        """ calculate the objective earned by executing beta
        true value based on true probabilities
        """
        if np.sum(beta) > self.budget + 0.0001:
            raise Exception('beta exceeds budget! sum = {}, beta = {}'.format(np.sum(beta), beta))

        # true reward
        reward = self.calculate_true_reward(beta)

        # true priority
        priority = 0
        for i in range(self.N):
            for g in range(self.G - 1): # pairwise comparisons of each pair of groups
                for h in range(g+1, self.G):
                    priority += reward[i] * max(0, self.group_distrib[h][i] - self.group_distrib[g][i])
        priority /= (self.G) * (self.G - 1) / 2.
        priority *= -self.N

        # true objective
        obj = (self.lamb * np.sum(reward)) + ((1-self.lamb) * priority)

        return obj, reward, priority

    @abstractmethod
    def ranking_distance(self, beta):
        """
        true_rank position we assume to be sorted 1..G """
        pass

    @abstractmethod
    def evaluate_obj(self, reward):
        """ evaluate the objective earned by executing beta to achieve empirical reward 'reward' """
        pass

    @abstractmethod
    def evaluate_priority(self, reward):
        """ evaluate the prioritization earned by executing beta to achieve empirical reward 'reward' """
        # TODO
        pass

    @abstractmethod
    def run_alg(self, approach):
        """
        return the TRUE (using ground-truth probabilities) objective, reward, and priority values """
        pass


    ############################################################
    # general helpers
    ############################################################

    def get_random_arm(self, num_random=1):
        """ pick random 'arms' corresponding to coverage """
        random_beta = np.random.uniform(0, 1, size=(num_random, self.N))

        # normalize to sum to budget
        random_beta = random_beta.T / random_beta.sum(axis=1) * self.budget
        random_beta = random_beta.T

        max_effort = 1.

        # ensure we never exceed effort = 1 on any target
        for i in range(num_random):
            while len(np.where(random_beta[i,:] > max_effort)[0]) > 0:
                excess_idx = np.where(random_beta[i,:] > max_effort)[0][0]
                excess = random_beta[i, excess_idx] - max_effort

                random_beta[i, excess_idx] = max_effort

                # add "excess" amount of effort randomly on other targets
                add = np.random.uniform(size=self.N - 1)
                add = (add / np.sum(add)) * excess

                random_beta[i, :excess_idx] += add[:excess_idx]
                random_beta[i, excess_idx+1:] += add[excess_idx:]

        return random_beta


    ############################################################
    # LIZARD functions
    ############################################################

    def zoom_step(self, approach, display=False):
        """ execute single step of zooming algorithm
        returns arm that was selected """
        beta = self.selection_rule(approach)

        # pull arm
        rewards = np.zeros(self.N)
        for i in range(self.N):
            eff = beta[i]

            # get reward, with optimistic bound based on distance from arm center
            reward_prob = self.adversary.visit_target(i, beta[i])
            observed_reward = np.random.choice([0, 1], p=[1-reward_prob, reward_prob])

            rewards[i] = reward_prob

            self.n_w_history[i][eff] += 1
            self.n[i][eff] += 1
            self.cum_rewards[i][eff] += observed_reward

        self.true_reward[self.t] = rewards.mean()
        if self.VERBOSE:
            print('             true reward {:.4f}'.format(self.true_reward[self.t]))

        return beta, rewards


    @abstractmethod
    def solve_MP(self, reward, approach, epsilon=True):
        pass

    def selection_rule(self, approach):
        """ selection rule
        returns arm that was selected

        approach='LIZARD': ILP from AAAI-21 paper
        ='RankedCUCB': modified ILP using Gamma[i] coefficients
        ='naive_rank': objective with naive ranking for "priority" component (no learning)
        """

        pre_index = {}
        index = {}
        # compute pre-indexes
        for i in range(self.N):
            for j, eff in enumerate(self.B[i]):
                eff = self.B[i][j] # keep in case of floating point error

                if eff == 0:
                    mu = 0.
                elif self.n_w_history[i][eff] == 0:
                    mu = 1. #1. / 2.    # TODO: should this be the total observed average? or 1/2?
                else:
                    mu = self.cum_rewards[i][eff] / self.n_w_history[i][eff]

                conf = self.conf(i, eff)

                # need to do minus conf radius if gamma is negative
                pre_index[(i,j)] = self.mu_and_conf(mu, conf, i, approach)

                if np.isnan(pre_index[(i,j)]):
                    raise Exception(f'uh oh! nan! pre_index {pre_index[(i,j)]} at {(i,j)}, mu {mu}, conf {conf}')

        use_pre_index = {}

        # compute indexes - with feature distance
        for i1 in range(self.N):
            for j1, eff1 in enumerate(self.B[i1]):
                eff1 = self.B[i1][j1]  # used to prevent floating point issues

                use_pre_index[(i1, j1)] = '-'

                # monotonicity: zero equals zero assumption
                # with 0 effort == 0 reward assumption, set uncertainty to 0
                if self.increasingness:
                    if eff1 == 0:
                        index[(i1, j1)] = 0.
                        continue

                min_pre = pre_index[(i1, j1)]

                if self.use_features:
                    loop_over = range(self.N)
                else:
                    loop_over = [i1]

                for i2 in loop_over:
                    for j2, eff2 in enumerate(self.B[i1]):
                        eff2 = self.B[i2][j2]  # used to prevent floating point issues

                        if self.increasingness:
                            dist = max(0, eff1 - eff2) * self.L[i1]
                        else:
                            dist = abs(eff1 - eff2) * self.L[i1]
                        influenced_dist = pre_index[(i2, j2)] + dist + self.dist[i1, i2]
                        if influenced_dist < min_pre:
                            min_pre = influenced_dist
                            if abs(j1 - j2) > 1e-1:
                                use_pre_index[(i1, j1)] = (i1, j2)
                            if abs(i1 - i2) > 1e-1:
                                use_pre_index[(i1, j1)] = '{} @@@@@@'.format((i2, j2))
                            else:
                                if min_pre == 0:
                                    print('weird! j1 {}, j2 {}, eff1 {:.2f}, eff2 {:.2f} dist {:.2f}'.format(j1, j2, eff1, eff2, dist))

                index[(i1, j1)] = min_pre
                if np.isnan(min_pre):
                    raise Exception(f'uh oh! nan! index {(i1, j1)}, min_pre {min_pre}, pre {pre_index[(i2, j2)]}, dist {dist}, other dist {self.dist[i1, i2]}')

        model, x = self.solve_MP(index, approach)

        if self.VERBOSE:
            print(' --- round {:4.0f}, arm UCB {:.3f}'.format(self.t, model.objVal))

        print_pulls = ''
        print_zero_pulls = ''

        for i in range(self.N):
            for j, eff in enumerate(self.B[i]):
                eff = self.B[i][j]

                # put * next to arms we pull
                star = '*' if x[i][j].x == 1 else ' '

                # put ! next to any UCBs with violations (UCB lower than true mu)
                true_mu = self.adversary.pwl[i].get_reward(eff)
                star2 = '!' if true_mu > index[(i,j)] else ' '

                n    = self.n_w_history[i][eff]
                mu   = self.cum_rewards[i][eff] / max(1, n)
                conf = self.conf(i, eff)

                out = '({:2.0f}, {:2.0f}) n {:3.0f}, eff {:.4f}, mu {:.3f}, true mu {:.3f}, conf {:.3f}, pre-I {:.3f}, I {:.3f} || {} {} {}'.format(
                                    i, j, n, eff, mu, true_mu, conf,
                                    pre_index[(i,j)], index[(i,j)],
                                    star, use_pre_index[(i,j)], star2)

                if n == 0:
                    print_zero_pulls += out + '\n'
                else:
                    print_pulls += out + '\n'

        if self.VERBOSE:
            print(print_pulls)
            print(print_zero_pulls)

        self.t_ucb[self.t-1] = model.objVal

        arm = np.full(self.N, np.nan)

        # convert x to beta
        for i in range(self.N):
            for j, eff in enumerate(self.B[i]):
                if abs(x[i][j].x - 1) < 1e-2:
                    arm[i] = self.B[i][j]

            assert not np.isnan(arm[i]), 'MIP x[{}] vals are {}'.format(i, [x[i][j].x for j in range(len(self.B[i]))])

        return arm

    def get_optimal_arm(self, approach):
        """ selection rule
        returns arm that was selected

        approach='LIZARD': ILP from AAAI-21 paper
        ='RankedCUCB': modified ILP using Gamma[i] coefficients
        ='naive_rank': objective with naive ranking for "priority" component (no learning)

        budget b

            # GIVEN VARIABLES
            # B[i]
            # y in B[i]
            # L[i]
            # r[i](y)
            # mu[i](y)
            # M
        """
        true_reward = {}
        for i in range(self.N):
            for j in range(len(self.B[i])):
                true_reward[(i,j)] = self.adversary.visit_target(i, self.B[i][j])

        model, x = self.solve_MP(true_reward, approach, epsilon=False)

        arm = np.full(self.N, np.nan)

        # convert x to beta
        for i in range(self.N):
            for j, eff in enumerate(self.B[i]):
                if abs(x[i][j].x - 1) < 1e-2:
                    arm[i] = self.B[i][j]

            assert not np.isnan(arm[i]), 'MIP x[{}] vals are {}'.format(i, [x[i][j].x for j in range(len(self.B[i]))])

        return arm


    def evaluate_optimal_arm(self):
        """ compute the arm that optimizes the MP and evaluate objective, priority, and reward """
        opt_arm = self.get_optimal_arm(approach='RankedCUCB')

        opt_obj, opt_reward, opt_priority = self.evaluate_true_objective(opt_arm)
        opt_kendall = self.evaluate_kendall_tau(opt_reward)

        return opt_obj, np.sum(opt_reward), opt_priority, opt_arm, opt_kendall


    def generic_r(self, num_pulls):
        eps = .1
        r = np.sqrt(-np.log(eps) / (2. * max(1, num_pulls)))

        return r

    @abstractmethod
    def conf(self, i, eff):
        pass

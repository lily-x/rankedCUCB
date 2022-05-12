""" implement rankedCUCB bandit algorithm

Lily Xu
July 2021 """

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import sys
from ranked_cucb import RankedCUCB


def compute_target_rank(group_distrib):
    """ for naive prioritization baseline, precompute the rank metric for each target
    uses the rank metric but does not include reward coefficient. no learning """
    n_groups  = len(group_distrib)
    n_targets = len(group_distrib[0])
    target_metrics = np.zeros(n_targets)

    for i in range(n_targets):
        for g in range(n_groups - 1):
            for h in range(g+1, n_groups):
                target_metrics[i] += group_distrib[h][i] - group_distrib[g][i]

    return target_metrics


class RankedLinear(RankedCUCB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calculate_gamma()

    def conf(self, i, eff):
        """ confidence radius of a given arm
        i = target
        eff = effort """
        assert eff <= 1.

        if eff in self.n[i]:
            num_pulls = self.n[i][eff]
        else:
            num_pulls = 0

        if self.gamma_conf:
            r =  np.sqrt((3. * np.log(self.T) * self.Gamma[i]**2) / (2. * (num_pulls + 1)))
        else:
            eps = .1
            r = np.sqrt(-np.log(eps) / (2. * max(1, num_pulls)))

        return r

    def ranking_distance(self, beta):
        """ score is between -1 and 0, where 0 is perfect concordance and -1 is perfect disconcordance
        distance self.true_rank vs. beta_rank
        true_rank position we assume to be sorted 1..G """

        distance = 0

        beta_rank = self.calculate_ranking(beta)
        beta_group_effort = self.calculate_group_effort(beta)

        # position of each group g in the ranking
        beta_rank_position = np.zeros(self.G)
        for g in range(self.G):
            beta_rank_position[g] = np.where(beta_rank == g)[0][0]

        # pairwise comparisons of each pair of groups
        for g in range(self.G - 1):
            for h in range(g+1, self.G):
                if (beta_rank_position[g] < beta_rank_position[h]): # if pairwise agreement
                    continue
                distance += abs(beta_group_effort[g] - beta_group_effort[h])

        # divide by number of pairs, (G choose 2)
        distance /= (self.G) * (self.G - 1) / 2.

        return distance


    def evaluate_true_objective(self, beta):
        """ calculate the objective earned by executing beta
        true value based on true probabilities
        """
        if np.sum(beta) > self.budget + 0.0001:
            raise Exception('beta exceeds budget! sum = {}, beta = {}'.format(np.sum(beta), beta))

        # true reward
        reward = self.calculate_true_reward(beta)

        # true prioritization
        priority = self.evaluate_priority(reward)

        # true objective
        obj = (self.lamb * np.sum(reward)) + ((1-self.lamb) * priority)

        return obj, reward, priority

    def evaluate_obj(self, reward):
        """ evaluate the objective earned by executing beta to achieve empirical reward 'reward' """
        priority = self.evaluate_priority(reward)
        obj = (self.lamb * np.sum(reward)) + ((1 - self.lamb) * priority)
        return obj

    def evaluate_priority(self, reward):
        """ evaluate the prioritization by executing beta to achieve empirical reward 'reward' """

        priority = 0
        for g in range(self.G - 1): # pairwise comparisons of each pair of groups
            for h in range(g+1, self.G):
                inner = 0
                for i in range(self.N):
                    inner += reward[i] * (self.group_distrib[h][i] - self.group_distrib[g][i])
                priority += inner
        priority = (-self.N * priority) / ((self.G) * (self.G - 1) / 2.)
        return priority


    def evaluate_kendall_tau(self, reward):
        """ evaluate the objective w.r.t. Kendall tau distance
        (even though we cannot directly solve for that with the MP) """

        kendall_tau = 0
        for g in range(self.G - 1):
            for h in range(g+1, self.G):
                g_benefit = np.dot(reward, self.group_distrib[g])
                h_benefit = np.dot(reward, self.group_distrib[h])
                if g_benefit > h_benefit:
                    kendall_tau += 1
                else:
                    kendall_tau -= 1

        kendall_tau /= ((self.G) * (self.G - 1) / 2.)

        priority = self.N * kendall_tau

        return priority


    def calculate_gamma(self):
        """ calculate Gamma[i] coefficients for objective
        note this is independent of beta, so this can be precomputed """
        Gamma = np.zeros(self.N)
        Gamma_priority = np.zeros(self.N) # only prioritization component so that we can multiply with epsilon

        for i in range(self.N):
            dist_sum = 0

            # pairwise comparisons of each pair of groups
            for g in range(self.G - 1):
                for h in range(g+1, self.G):
                    dist_sum += self.group_distrib[h][i] - self.group_distrib[g][i]

            # divide by number of pairs, (G choose 2)
            dist_sum /= (self.G) * (self.G - 1) / 2.

            Gamma[i] = self.lamb - (1-self.lamb) * self.N * dist_sum
            Gamma_priority[i] = self.N * dist_sum

        self.Gamma = Gamma
        self.Gamma_priority = Gamma_priority

        if np.all(self.Gamma >= 0):
            print('Gamma good', np.sum(self.Gamma < 0), self.N)
        else:
            print('   BAD:', np.sum(self.Gamma < 0), '/', self.N, 'of Gamma are negative')

        return Gamma


    def run_alg(self, approach='RankedCUCB'):
        """
        return the TRUE (using ground-truth probabilities) objective, reward, and prioritization values

        approach:
        'RankedCUCB' : our approach
        'LIZARD'     : LIZARD baseline
        'naive_rank' : naive ranked baseline
        """
        assert approach in ['RankedCUCB', 'LIZARD', 'naive_rank'], f'approach {approach} not implemented'

        self.reset()

        # empirical observed reward, objective, prioritization
        all_beta     = np.zeros((self.T, self.N))
        all_reward   = np.zeros((self.T, self.N))
        all_obj      = np.zeros(self.T)
        all_priority = np.zeros(self.T)

        # true reward, objective, rank priority (based on ground-truth probability)
        all_true_reward  = np.zeros((self.T, self.N))
        all_true_obj     = np.zeros(self.T)
        all_true_priority = np.zeros(self.T)
        all_true_kendall  = np.zeros(self.T)

        self.t = 0

        for t in range(self.T):
            if self.VERBOSE:
                print('-----------------------------------')
                print(' t = {}'.format(t))
                print('-----------------------------------')
            else:
                if t % 50 == 0:
                    print(' t = {}'.format(t))
            zoom_display = True if self.VERBOSE and t % 10 else False

            # compute UCBs for all arms
            # select superarm to max reward
            beta, reward = self.zoom_step(approach) # LIZARD MP

            obj      = self.evaluate_obj(reward)
            priority = self.evaluate_priority(reward)
            true_obj, true_reward, true_priority = self.evaluate_true_objective(beta)

            true_kendall = self.evaluate_kendall_tau(true_reward)

            all_reward[t, :] = reward
            all_beta[t, :]   = beta
            all_obj[t]       = obj
            all_priority[t]  = priority

            all_true_obj[t]       = true_obj
            all_true_priority[t]  = true_priority
            all_true_reward[t, :] = true_reward
            all_true_kendall[t]   = true_kendall

            if self.VERBOSE and zoom_display:
                print(' round {}, beta = {}'.format(t, beta))
                print('  true reward sum {:.3f}, true obj {:.3f}'.format(np.sum(true_reward), true_obj))

            self.t += 1

        all_reward = np.sum(all_reward, axis=1)
        all_true_reward = np.sum(all_true_reward, axis=1)

        if self.VERBOSE:
            print('\nbeta')
            for t in range(self.T):
                print('  {} {} {:.3f}'.format(t, np.round(all_beta[t, :], 3), all_beta[t, :].sum()))

            print('\nreward per pull')
            for i in range(self.N):
                print('----')
                print('  {}'.format(i))
                for eff in sorted(self.cum_rewards[i].keys()):
                    true_mu = self.adversary.pwl[i].get_reward(eff)
                    mu = self.cum_rewards[i][eff] / max(self.n_w_history[i][eff], 1)
                    print('    n {:3.0f}, eff {:.4f}, mu {:.3f}, true mu {:.3f}, conf {:.3f}'.format(
                        self.n_w_history[i][eff], eff, mu, true_mu, self.conf(i, eff)))

            percent_wrong = (len(np.where(self.mip_UCB < self.true_reward)[0]) / self.T) * 100
            print('\nUCB wrong {:.2f}% of the time'.format(percent_wrong))

        return all_true_obj, all_true_reward, all_true_priority, all_beta, all_true_kendall


    ############################################################
    # LIZARD functions
    ############################################################

    def solve_MP(self, reward, approach, epsilon=True):
        """ reward can either be the UCB index or the true reward,
        if we want to solve optimally with the ground truth """
        M = 1e6    # big M

        model = gp.Model('milp')

        # silence output
        model.setParam('OutputFlag', 0)

        # x: indicator saying pull arm j at target i
        x = [[model.addVar(vtype=GRB.BINARY, name='x_{}_{}'.format(i, j))
                for j in range(len(self.B[i]))] for i in range(self.N)]

        # objective
        if approach == 'LIZARD':
            # LIZARD objective: no prioritization
            model.setObjective(gp.quicksum([x[i][j] * reward[(i,j)]
                for i in range(self.N) for j in range(len(self.B[i]))]), GRB.MAXIMIZE)
        elif approach == 'naive_rank':
            # baseline: naive rank objective
            target_rank = compute_target_rank(self.group_distrib)
            model.setObjective(
                gp.quicksum([x[i][j] * reward[(i,j)] * target_rank[i]
                for i in range(self.N) for j in range(len(self.B[i]))]), GRB.MAXIMIZE)

        elif approach == 'RankedCUCB':
            if epsilon:
                use_eps = 1 - self.get_epsilon()
            else:
                # prioritization objective, without epsilon
                # used to solve for the optimal action knowing ground truth
                use_eps = 1

            model.setObjective(gp.quicksum([x[i][j] * reward[(i,j)] * (self.lamb - ((1-self.lamb) * use_eps * self.Gamma_priority[i]))
                for i in range(self.N) for j in range(len(self.B[i]))]), GRB.MAXIMIZE)

        else:
            raise NotImplementedError



        model.addConstrs((gp.quicksum(x[i][j] for j in range(len(self.B[i]))) == 1
                            for i in range(self.N)), 'one_per_target') # pull one arm per target

        model.addConstr(gp.quicksum([x[i][j] * self.B[i][j]
                            for i in range(self.N) for j in range(len(self.B[i]))]) <= self.budget, 'budget')  # stay in budget

        model.optimize()

        if model.status != GRB.OPTIMAL:
            raise Exception('Uh oh! Model status is {}'.format(model.status))

        return model, x

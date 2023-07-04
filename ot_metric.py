
import itertools
import ot
import numpy as np
import pandas as pd


class OTFRM:
    def __init__(self, method='exact', p=2, cost_metric='cosine', numItermax=100000, entreg=.1):
        """ Args: 
                method: the method to solve OT problem.
                p: the coefficient in the OT cost (i.e., the p in p-Wasserstein).
                cost_metric: the metric to measure the distance in feature space, only support cosine now.
                numIntermax: the maximum number of iterations before stopping the optimization algorithm if it has not converged.
                entreg: the strength of entropy regularization for sinkhorn.
        """
        assert method in ['exact', 'sinkhorn', 'asgd'], "Method {} is invalid!".format(method)
        assert cost_metric =='cosine', "Only support cosine metric to ground feature cost at present!"
        self.method = method
        self.p = p
        self.cost_metric = cost_metric
        self.numItermax = numItermax
        self.entreg = entreg

    def ot_sim_calcul(self, embs1, embs2):
        """ Calculate similarity of two different classes (/tasks) based on their p-Wasserstein distance.
            Args: embs1, embs2 : embedding representation array of two classes.
                cost_metric: the distance metric of cost matrix of p-Wasserstein, only 'cosine' is supported at present.
        """
        # calculate cost matrix
        C = ot.dist(embs1, embs2, metric=self.cost_metric) # cost matrix
        if self.p > 1: C = pow(C, self.p)
        # solve the OT problem, obtain the optimal coupling matrix
        if self.method == 'exact':
            P = ot.emd(ot.unif(C.shape[0]), ot.unif(C.shape[1]), C, numItermax=self.numItermax) # optimal coupling matrix pi*
        elif self.method == 'sinkhorn':
            P = ot.sinkhorn(ot.unif(C.shape[0]), ot.unif(C.shape[1]), C, reg=self.entreg, numItermax=1000)
        elif self.method == 'asgd':
            P = ot.stochastic.solve_semi_dual_entropic(ot.unif(C.shape[0]), ot.unif(C.shape[1]), C, reg=self.entreg, method='ASGD', numItermax=10000, log=False)
        # calculate p-Wasserstein distance
        ot_loss = (P * C).sum()
        pwdist = pow(ot_loss, 1/self.p)
        # define ot similarity = 1 - p-Wasserstein distance, for cosine cost metric
        ot_sim = 1 - pwdist # Notice: only cosine space have: sim = 1 - dist
        return ot_sim

    def otfrm(self, all_tasks_embs_dict):
        """ OT-based Feature Representation Measure proposed on MASSA.
            Args: all_tasks_embs_dict: embedding representation of all tasks in shape {'task_name: (num_samples, 512)}.
            Return: OTFRM of all tasks in a dict.
        """
        # Firstly calculate of possible combinations of tasks, store and then index to avoid repeated calculation
        inter_tasks_comb_sim_dict = {}
        inter_tasks_combinations = itertools.combinations(all_tasks_embs_dict.keys(), 2)
        for task_name1, task_name2 in inter_tasks_combinations:
            inter_task_sim = self.ot_sim_calcul(all_tasks_embs_dict[task_name1], all_tasks_embs_dict[task_name2])
            inter_tasks_comb_sim_dict[(task_name1, task_name2)] = inter_task_sim
            print('Inter OT-cosine similarity between {} and {}: {:.4f}'.format(task_name1, task_name2, inter_task_sim))

        # Calculate otfrm of each task
        all_tasks_otfrm_dict = {}
        for task_name, task_embs in all_tasks_embs_dict.items():
            # numerator: average of cosine similarities of all unique pairs in a class, across all classes
            task_intra_dist = ot.dist(task_embs, task_embs, metric=self.cost_metric).mean()
            task_intra_sim = 1 - task_intra_dist

            # denominator: average of OT-cosine similarities of unique pairs between samples in the class and samples out of the class
            complement_tasks = list(all_tasks_embs_dict.keys())
            complement_tasks.remove(task_name)
            task_inter_sim_list = []
            for complement_task in complement_tasks:
                try:
                    task_inter_sim_list.append(inter_tasks_comb_sim_dict[(task_name, complement_task)])
                except KeyError as InverseOrderError:
                    task_inter_sim_list.append(inter_tasks_comb_sim_dict[(complement_task, task_name)])
            task_inter_sim = np.mean(task_inter_sim_list)

            # OTFRM: numerator / denominator
            otfrm = task_intra_sim / task_inter_sim
            all_tasks_otfrm_dict[task_name] = otfrm
            print(" -> Task-{} intra-sim: {:4f}, inter-sim: {:.4f}, OTFRM: {:.4f}".format(task_name, task_intra_sim, task_inter_sim, otfrm))

        return all_tasks_otfrm_dict


class OTDistance:
    def __init__(self, method='exact', p=2, cost_metric='cosine', numItermax=100000, entreg=.1):
        """ Args: 
                method: the method to solve OT problem.
                p: the coefficient in the OT cost (i.e., the p in p-Wasserstein).
                cost_metric: the metric to measure the distance in feature space.
                numItermax: the maximum number of iterations before stopping the optimization algorithm if it has not converged.
                entreg: the strength of entropy regularization for sinkhorn.
        """
        assert method in ['exact', 'sinkhorn', 'asgd'], "Method {} is invalid!".format(method)
        self.method = method
        self.p = p
        self.cost_metric = cost_metric
        self.numItermax = numItermax
        self.entreg = entreg

    def ot_dist_calcul(self, embs1, embs2):
        """ Calculate similarity of two different classes (/tasks) based on their p-Wasserstein distance.
            Args: embs1, embs2 : embedding representation array of two classes.
                cost_metric: the distance metric of cost matrix of p-Wasserstein, only 'cosine' is supported at present.
        """
        # calculate cost matrix
        C = ot.dist(embs1, embs2, metric=self.cost_metric) # cost matrix
        if self.p > 1: C = pow(C, self.p)
        # solve the OT problem, obtain the optimal coupling matrix
        if self.method == 'exact':
            P = ot.emd(ot.unif(C.shape[0]), ot.unif(C.shape[1]), C, numItermax=self.numItermax) # optimal coupling matrix pi*
        elif self.method == 'sinkhorn':
            P = ot.sinkhorn(ot.unif(C.shape[0]), ot.unif(C.shape[1]), C, reg=self.entreg, numItermax=1000)
        elif self.method == 'asgd':
            P = ot.stochastic.solve_semi_dual_entropic(ot.unif(C.shape[0]), ot.unif(C.shape[1]), C, reg=self.entreg, method='ASGD', numItermax=10000, log=False)
        # calculate p-Wasserstein distance
        ot_loss = (P * C).sum()
        pwdist = pow(ot_loss, 1/self.p)
        return pwdist

    def transfer_dist(self, pt_emb, all_tasks_embs_dict):
        """ Calculate pairwise cosine distance.
            Args: 
                pt_emb: embedding representation of pretraining.
                all_tasks_embs_dict: embedding representation of all tasks in shape {task_name: (num_samples, emb_dim)}.
            Return: OT distance of all tasks in a dict.
        """
        all_tasks_name = all_tasks_embs_dict.keys()

        distance_df = pd.DataFrame(columns=[*all_tasks_name], index=range(1))
        # Inter-tasks distances
        for task_name in all_tasks_name:
            inter_task_dist = self.ot_dist_calcul(pt_emb, all_tasks_embs_dict[task_name])
            distance_df[task_name].iloc[0] = inter_task_dist
            print('Inter distance between pretraining and {}: {:.4f}'.format(task_name, inter_task_dist))

        return distance_df
        
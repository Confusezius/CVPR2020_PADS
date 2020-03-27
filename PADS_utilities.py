"""=============================================================================================="""
################### LIBRARIES ###################
import warnings
warnings.filterwarnings("ignore")

import collections
import os, sys, numpy as np, argparse, imp, datetime, time, pickle as pkl, random, json
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
import pandas as pd

import torch, torch.nn as nn, torch.nn.functional as F
import auxiliaries as aux
import datasets as data

import netlib as netlib
import losses as losses
import evaluate as eval

import faiss

from scipy.spatial import distance
from sklearn.preprocessing import normalize
from sklearn import metrics
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')




"""=============================================================================================="""
def include_pads_args(parser):
    ### Train-Validation Setup
    parser.add_argument('--train_val_split_by_class',  action='store_true',
                        help='Split the training data either by class (call this flag) or per class (dont call it).')
    parser.add_argument('--train_val_split',           default=0.85,   type=float,
                        help='Percentage of training data that is retained for training. The remainder is used to set up the validation set.')

    ###
    parser.add_argument('--policy_n_support',          default=30,   type=int,
                        help='Resolution of the support grid, i.e. number of bins between the support interval --policy_support_limit')

    parser.add_argument('--policy_support_limit',     nargs='+',    default=[0.1,1.4],      type=float,
                        help='Limit of the sampling distribution support. Excluding small values removes negatives that might be too hard.')

    parser.add_argument('--policy_lr',                default=0.01, type=float,
                        help='Learning rate of sampling policy.')

    parser.add_argument('--policy_sample_freq',      default=30,   type=int,
                        help='Number of iteration (M in paper) to update the network before computing the validation rewward metrics and updating the policy')

    parser.add_argument('--policy_action_values',   nargs='+',    default=[0.5,1,1.5],  type=float,
                        help='[alpha,1,beta] - values to update the sampling distribution bins by. Updates are done multiplicatively.')

    parser.add_argument('--policy_size',            nargs='+',    default=[128, 128],   type=int,
                        help='Size of the utilized policy. Values in the list denote number of neurons, length the number of layers.')

    parser.add_argument('--policy_winit_type',      default='none', type=str,
                        help='(Optional) Method to initialize policy weights with. Options: none, he_n.')

    parser.add_argument('--policy_dont_use_a2c',    action='store_true',
                        help='If set, PPO + A2C will be used.')

    parser.add_argument('--policy_baseline_ema_gamma', default=0.05, type=float,
                        help='Exponential moving average decay. Relevant if no A2C is used, but a moving average instead.')

    parser.add_argument('--policy_reward_update_w', default=0.99, type=float)
    parser.add_argument('--policy_ppo_ratio', default=0.2, type=float)
    parser.add_argument('--policy_no_baseline',     action='store_true',
                        help='Number of episodes to include for reinforce-type training.')

    ###
    parser.add_argument('--policy_include_train_metrics', action='store_true')
    parser.add_argument('--policy_run_avgs_no_avg', action='store_true')
    parser.add_argument('--policy_run_avgs',        nargs='+',    default=[2,8,16,32],  type=int,
                        help='Running averages of state metrics defined in --policy_state_metrics to be included in the policy input state.')

    parser.add_argument('--policy_state_metrics',   nargs='+',    default=['recall','nmi','dists'], type=str,
                        help='Metrics to include into the policy input state. Available options: recall, nmi & dists (intra- and interclass distances).')

    parser.add_argument('--policy_metric_history',  default=10,   type=int,
                        help='Number of validation metrics steps to be included in the input state. High values incorporate validation metrics from old policy updates/network states.')


    ###
    parser.add_argument('--policy_reward_type',    default=2, type=int,
                        help='Target reward metric to be optimized on the validation set. 0,1,2 denote nmi,recall,recall+nmi as reward signal.')
    # parser.add_argument('--policy_reward_metrics',  nargs='+',    default=['recall-0'], type=str,
    #                     help='Target reward metric to be optimized on the validation set. Options are: recall-N and nmi. Multiple combinations are allowed.')

    ###
    parser.add_argument('--policy_parameter_history',     default=1, type=int,           help='History of sampling distribution values to be included into the state. Default only includes previous parameters.')
    parser.add_argument('--policy_scramble_freq',         default=100000000,   type=int, help='Optional: After the given number of epochs, training/validation are re-scrambled. Not used by default.')
    parser.add_argument('--policy_scramble_dataloaders',  action='store_true',           help='Optional: Needs to be set if training/validation set scrambling should be performed.')

    ###
    parser.add_argument('--policy_mode', default='ppo', type=str, help='RL mode to use. Options: ppo for Proximal Policy Optimization (PPO) or reinforce for REINFORCE.')
    parser.add_argument('--policy_old_update_iter',       default=3, type=int,           help='PPO-Specific Parameter: Number of iterations before updating the old policy again.')

    ###
    parser.add_argument('--policy_sup_metric_collect',  default=-1,              type=int,           help='Optional: If one wishes to utilize episodes with T>1, set this value to the appropr. scale. ')
    parser.add_argument('--policy_training_delay',  default=0,               type=int,           help='Number of epochs to train with the initial distribution before turning on PADS. Default runs PADS directly.')
    parser.add_argument('--policy_init_distr',      default='uniform_low',   type=str,           help='Type of initial distribution to use. Default is uniform_low, which places high probabilities between [0.3 and 0.7].\
                                                                                                      Other options are: random, uniform_high, uniform_avg, uniform_low, uniform_low_and_tight, uniform_lowish_and_tight and the\
                                                                                                      respective normal variants. You may also set uniform/low and set the mean/std in --policy_init_params.')
    parser.add_argument('--policy_init_params',     nargs='+', default=[0.5,0.04],   type=float, help='Custom initial distribution parameters for either normal or uniform initial distributions:\
                                                                                                      e.g. [mu, sig] = [0.5, 0.04] for normal.')

    parser.add_argument('--policy_log_distr_plots', action='store_true',  help='Plot Sampling Distribution Progression as Pyplot.')
    parser.add_argument('--policy_merge_oobs',      action='store_true',  help='Self-Regularisation Pt.1: Values below lower intertval bound are controlled together (same sampling bin).')
    parser.add_argument('--policy_include_pos',     action='store_true',  help='Self-Regularisation Pt.2: Include positives into negative sample selection. Excludes positive==anchor.')
    parser.add_argument('--policy_include_same',    action='store_true',  help='Self-Regularisation Pt.3: Specifically include positive==anchor in negatives as well.')
    parser.add_argument('--policy_no_self_reg',     action='store_true',  help='Perform NO Self-Regularisation (notable performance drop!).')
    return parser


"""=============================================================================================="""
class PADS():
    def __init__(self, general_pars, state_maker_pars, policy_pars, sampler_pars, val_pars):
        """
        Base-Class for PADS:
        Input:
            general_pars:       Dictionary of basic parameters for the PADS-class, primarily concerning how and when the policy is updated.
            state_maker_pars:   Dictionary of parameters relating to the input state construction.
            policy_pars:        Dictionary of parameters regarding the setup of the underlying policy network.
            sampler_pars:       Dictionary of parameters to set up a parametric sampler, which is used to sample data for the DML task.
            val_pars:           Dictionary of parameters regarding the validation set, primarily which metrics to compute and which to use for state and reward.
        """
        ##############################################################################################################
        self.log_distr_plots    = general_pars['log_distr_plots']
        self.sup_metric_collect = general_pars['sup_metric_collect']

        ####### Key Input Parameters to update the underlying policy and sampling distribution. ######
        # PPO & Method parameters; exponential moving average decay for REINFORCE methods.
        self.use_ppo           = general_pars['mode']=='ppo'
        self.old_policy_update = general_pars['old_policy_update']
        self.ema_alpha         = general_pars['ema_alpha']
        self.ppo_ratio         = general_pars['ppo_ratio']

        # Counter for the number of calls to this class and the number of calls to wait before activating the policy.
        self.call_count = 0
        self.call_delay = general_pars['call_delay']

        # Counter for the number of class done. If this passes the sampling frequency threshold, the policy is updated/val. metrics are collected.
        self.train_iter_before_update = 0
        self.sample_frequency         = general_pars['sample_frequency']

        # Logging elemebts: Use wandb-based logging, the logging instance to use, and the environment name.
        self.wandb_log, self.logger, self.logname = general_pars['wandb_log'], general_pars['logger'], general_pars['logname']

        # List of relevant metrics, i.e. validation reward, history of parameters or history of metrics.
        self.coll_rewards         = []
        self.parameter_collect    = []
        self.val_metric_collect, self.reward_metric_collect = [],[]

        # Length is determined by the threshold values defined below.
        self.metric_history     = general_pars['metric_history']
        self.parameter_history  = general_pars['parameter_history']

        self.reward_update_w    = general_pars['reward_update_w']

        ###############################################################################################################
        ####### Set up a input-state maker instance. ########
        self.state_maker      = StateMaker(**state_maker_pars)


        ####### Create the underlying policy and its optimizer. ###########
        policy_pars['input_dim'] = self.state_maker.state_dim
        self.policy              = Policy(**policy_pars)
        if self.use_ppo:
            self.old_policy       = Policy(**policy_pars)
            self.old_policy.load_state_dict(self.policy.state_dict())
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=general_pars['policy_lr'])


        ####### Set up a parametric sampler instance, which will be passed to the DML criterion. #######
        self.param_sampler    = Parametric_Sampler(**sampler_pars)

        ####### Set up an instance for validation-metric computation. ########
        self.calc_val_metrics = Compute_Complete_Metrics(**val_pars)
        # self.calc_val_metrics = Compute_Validation_Metrics(**val_pars)



        ###############################################################################################################
        ########### Create a summary text for this instance. #####
        learned_sampler_summary  = '\n*****\n'
        learned_sampler_summary += 'Policy Summary:\nState Dim: {} | Running Avgs: {} | Metric History: {}\nReward Type: {} | Activation Delay: {} epochs\nIter until update: {}'.format(
                                    self.state_maker.state_dim, self.state_maker.running_averages, self.metric_history, val_pars['reward_type'], self.call_delay, self.sample_frequency)
        learned_sampler_summary += '\n*****'
        print(learned_sampler_summary)

        self.sub_episode_rewards = []


    def assign_sampler(self, criterion):
        """
        Assign the internal parameteric sampler to the criterion used to optimize the DML network.
        Updates to this internal sampler will also automatically translate to the criterion sampler.

        Input:
            criterion: Instance of the criterion-classes in <losses.py>.

        Returns:
            Nothing!
        """
        criterion.sampler = self.param_sampler


    def update(self, criterion, model, training_iter, epoch):
        """
        Function to update the internal sampling policy, use the model to compute the validation metrics and log the relevant information.

        Input:
            criterion:
            model:
            training_iter:
            epoch:

        Returns:
            Nothing!
        """

        ##################################################################
        _ = self.policy.cuda()
        if self.use_ppo:
            self.old_policy.cuda()

        ##################################################################
        import numpy as np
        if epoch>=self.call_delay:
            if self.train_iter_before_update in [self.sample_frequency//(self.sup_metric_collect+1)*(ik+1) for ik in range(self.sup_metric_collect)] and self.sup_metric_collect>0 and self.train_iter_before_update!=self.sample_frequency:
                reward_metric, _ = self.calc_val_metrics.compute(model)
                self.sub_episode_rewards.append(reward_metric)
                print(self.sub_episode_rewards)

            if self.train_iter_before_update==self.sample_frequency:
                ### Reset the iteration counter:
                self.train_iter_before_update=0


                ######## CREATE THE CURRENT INPUT STATE ###########
                ### Compute validation metrics for the reward signal and the input state
                reward_metric, validation_metrics = self.calc_val_metrics.compute(model)
                if len(self.sub_episode_rewards):
                    self.sub_episode_rewards.append(reward_metric)
                    reward_metric       = np.sum([0.7**(len(self.sub_episode_rewards)-i-1)*sub_rew for i,sub_rew in enumerate(self.sub_episode_rewards)])
                    self.sub_episode_rewards = []

                self.val_metric_collect.append(validation_metrics)
                self.reward_metric_collect.append(reward_metric)

                ### If not enough steps have been performed, we fill the state with copies of current metrics:
                import numpy as np
                # for the validation metrics
                while len(self.val_metric_collect)<np.clip(self.metric_history,2,None):
                    self.val_metric_collect.append(validation_metrics)

                # for the sampling distribution values
                self.parameter_collect.append(criterion.sampler.distr)
                while len(self.parameter_collect)<self.parameter_history:
                    self.parameter_collect.append(criterion.sampler.distr)

                ### Create the input state!
                state = self.state_maker.make_state(self.val_metric_collect[-int(self.metric_history):], self.parameter_collect[-int(self.parameter_history):], training_iter)
                state = torch.from_numpy(state).type(torch.cuda.FloatTensor)


                ######## UPDATE THE POLICY ###########
                if self.call_count>0:
                    ### Get the log-action-probabilities for the previous input state to compute the loss for.
                    policy_value, action_weights   = self.policy(self.prev_state)
                    action_probs                   = torch.nn.functional.softmax(action_weights, dim=1)
                    log_action_probs               = torch.nn.functional.log_softmax(action_weights, dim=1)
                    ### log-probabilities for only the actions that were used (ua):
                    log_action_probs_ua = log_action_probs[:, self.used_actions.reshape(-1), range(action_weights.shape[-1])]

                    ### Reward signal: Sign of reward metric change:
                    policy_reward = np.sign(self.reward_metric_collect[-1] - self.reward_metric_collect[-2])

                    ### Compute the Advantage for the policy gradient.
                    if not self.policy.use_a2c:
                        self.coll_rewards.append(policy_reward)
                        policy_advantage = policy_reward-self.exp_mov_avg(self.coll_rewards[:-1], alpha=self.ema_alpha)
                    else:
                        policy_reward    = policy_reward+self.reward_update_w*self.policy(state)[0].detach()
                        policy_advantage = policy_reward-policy_value.detach()

                    ### Compute the respective policy loss for either PPO or REINFORCE.
                    if self.use_ppo:
                        old_log_action_probs_ua = torch.nn.functional.log_softmax(self.old_policy(self.prev_state)[1], dim=1)[:, self.used_actions.reshape(-1), range(action_weights.shape[-1])]
                        ratio = torch.exp(log_action_probs_ua - old_log_action_probs_ua.detach())
                        loss  = torch.min(ratio*policy_advantage, torch.clamp(ratio, 1-self.ppo_ratio, 1+self.ppo_ratio)*policy_advantage)
                        loss  = - torch.mean(loss)
                    else:
                        loss  = - torch.mean(policy_advantage*log_action_probs_ua)

                    ### Value function loss if A2C is used:
                    if self.policy.use_a2c: loss = loss + 0.5*torch.nn.MSELoss()(policy_value, torch.tensor(self.reward_metric_collect[-1]).type(torch.cuda.FloatTensor))

                    ### Compute Gradients!
                    self.policy_optim.zero_grad()
                    loss.backward()

                    ### Log the relevant policy gradients to evaluate the behaviour.
                    grads    = np.concatenate([p.grad.detach().cpu().numpy().flatten() for p in self.policy.parameters() if p.grad is not None])
                    grad_l2, grad_max = np.mean(np.sqrt(np.mean(np.square(grads)))), np.mean(np.max(np.abs(grads)))
                    self.logger.progress_saver[self.logname].log('Gradient L2', grad_l2,  group='Gradient L2')
                    self.logger.progress_saver[self.logname].log('Gradient Max',grad_max, group='Gradient Max')

                    ### Update the policy!
                    self.policy_optim.step()


                ######## UPDATE THE SAMPLING DISTRIBUTION WITH CURRENT STATE! ###########
                with torch.no_grad():
                    actions = self.policy.act(state)
                distr_adjustments = self.policy.convert(actions)
                criterion.sampler.update_sample_distr(distr_adjustments)

                ### Remember the actions and the state used!
                self.prev_state   = state
                self.used_actions = actions


                ### Create a summary plot of the sampling distribution. Optionally log it online as well.
                if self.wandb_log:
                    criterion.sampler.log_sampling_distr()
                    self.logger.update(all=True)


                ### Finally, if PPO is used, we update the old policy:
                if self.use_ppo:
                    if self.call_count%self.old_policy_update==0 and self.call_count>1:
                        self.old_policy.load_state_dict(self.policy.state_dict())

                ###
                self.call_count += 1

            self.train_iter_before_update += 1


    def exp_mov_avg(self, vals,alpha):
        if len(vals):
            ema = vals[0]
            for x in vals[1:]:
                ema = (1-alpha)*ema + alpha*x
        else:
            ema = 0
        return ema








"""==============================================="""
class StateMaker():
    def __init__(self, n_support, max_iter, state_metrics=['recall', 'nmi', 'dists'], num_v_classes=100, include_train_metrics=False,
                 running_averages=[2,8,32], k_vals=[1,2,4,8], metric_history=2, parameter_history=1, no_avg=False):
        #The current input state comprises: computed mean/std of k-values, the sampl. distr. params, the relative iteration number
        #as well as inter/intra-class distances.
        self.state_metrics, self.max_iter = state_metrics, max_iter
        self.state_dim, self.state_style = 0,{}
        self.running_averages = running_averages
        self.k_vals, self.n_support = k_vals, n_support
        self.no_avg = no_avg


        ###
        has_recall_stats = 'recall' in self.state_metrics
        has_nmi_stats    = 'nmi'    in self.state_metrics
        has_dists_stats  = 'dists'  in self.state_metrics
        has_svd_stats    = 'svd'    in self.state_metrics

        ###
        mul = len(running_averages)+metric_history
        if has_recall_stats:
            self.state_dim += mul*len(k_vals)
        if has_nmi_stats:
            self.state_dim += mul
        if has_svd_stats:
            self.state_dim += mul
        if has_dists_stats:
            self.state_dim += mul*2

        ###
        if include_train_metrics:
            self.state_dim += self.state_dim

        #Include Distribution Parameters
        if parameter_history==-1: parameter_history=metric_history
        self.parameter_history = parameter_history

        self.state_dim           += (n_support-1)*parameter_history
        #Include Progress Vale
        self.state_dim           += 1

        ###
        self.state_container   = {'metric_history':None, 'distr_params':None, 'progress':0,
                                  'running_averages':[collections.deque(maxlen=running_average) for running_average in running_averages]}


    def make_state(self, metric_history, criterion_params, it_num):
        self.state_container['metric_history'] = metric_history
        self.state_container['distr_params']   = criterion_params
        self.state_container['progress']       = it_num

        for k in range(len(self.running_averages)):
            self.state_container['running_averages'][k].append(metric_history[-1])

        ###
        state = []

        ###
        state.append(np.array(self.state_container['metric_history']).reshape(-1))

        ###
        if self.no_avg:
            state.append(np.array([run_avg[0] for run_avg in self.state_container['running_averages']]).reshape(-1))
        else:
            state.append(np.array([np.mean(np.array(list(run_avg)),axis=0) for run_avg in self.state_container['running_averages']]).reshape(-1))

        ###
        state.append(np.array(self.state_container['distr_params']).reshape(-1))

        ###
        state.append(np.array([self.state_container['progress']/self.max_iter]))

        return np.concatenate(state).reshape(1, -1)






"""==============================================="""
def scramble_train_val(training_dataloader, validation_dataloader, LearnedSamplerInstance,
                       train_val_split, split_by_class):
    image_dict, validation_image_dict = training_dataloader.dataset.image_dict, validation_dataloader.dataset.image_dict

    if split_by_class:
        max_key = int(np.max(list(image_dict.keys())))
        validation_image_dict = {key+max_key+1:item for key,item in validation_image_dict.items()}
        image_dict.update(validation_image_dict)
    else:
        image_dict = {x: np.array(list(set(list(image_dict[x]) + list(validation_image_dict[x])))) for x in set(validation_image_dict).union(validation_image_dict)}

    image_dict_keys = list(image_dict.keys())

    if split_by_class:
        train_val_split = int(len(image_dict_keys)*train_val_split)
        train = list(np.random.choice(image_dict_keys, train_val_split, replace=False) )
        val   = [key for key in image_dict_keys if key not in train]
        train_image_dict, val_image_dict = {key:image_dict[key] for key in train}, {key:image_dict[key] for key in val}
    else:
        train_image_dict, val_image_dict = {},{}
        for key in image_dict_keys:
            train_ixs = np.random.choice(len(image_dict[key]), int(len(image_dict[key])*train_val_split), replace=False)
            val_ixs   = np.array([x for x in range(len(image_dict[key])) if x not in train_ixs])
            train_image_dict[key] = np.array(image_dict[key])[train_ixs]
            val_image_dict[key]   = np.array(image_dict[key])[val_ixs]


    training_dataloader.dataset.image_dict   = train_image_dict
    training_dataloader.dataset.init_setup()
    validation_dataloader.dataset.image_dict = val_image_dict
    validation_dataloader.dataset.init_setup()

    LearnedSamplerInstance.data = validation_dataloader





"""==============================================="""
class Compute_Complete_Metrics():
    def __init__(self, dataloader, logger, logname, k_vals=[1,2,4,8], reward_type=2, state_metrics=['recall','nmi','dists','svd'], wandb_log=False):
        if not isinstance(dataloader, list): dataloader = [dataloader]
        self.data       = dataloader[0]
        self.train_data = dataloader[1] if len(dataloader)==2 else None

        self.num_classes   = len(np.unique(self.data.dataset.avail_classes))

        self.state_metrics = state_metrics

        self.k_vals        = k_vals

        self.logger, self.logname = logger, logname

        self.container   = []
        self.wandb_log   = wandb_log

        self.reward_type = reward_type #Set to 0 if nmi only, 1 if recall-0 only.

    def compute(self, net):
        torch.cuda.empty_cache()

        with torch.no_grad():
            embeds, labels     = [],[]
            for i,inp in enumerate(self.data):
                emb = net(inp[1].type(torch.cuda.FloatTensor))
                embeds.append(emb.detach().cpu().numpy().tolist())
                labels.append(inp[0].detach().cpu().numpy())
            embeds   = np.vstack(embeds).astype('float32')
            labels   = np.hstack(labels).reshape(-1,1)

            state_metrics  = []
            reward_metrics = []

            ########### VAL
            for i,mode in enumerate(self.state_metrics):
                metrics = self.compute_metric(mode)(embeds, labels)
                if mode=='recall':
                    if self.reward_type==2 or self.reward_type==1:
                        reward_metrics.append(metrics[0])
                if mode=='nmi':
                    if self.reward_type==2 or self.reward_type==0:
                        reward_metrics.append(metrics[0])
                state_metrics.append(metrics)

                ### Log Computed Metrics
                if mode=='recall':
                    for i,k_val in enumerate(self.k_vals):
                        self.logger.progress_saver[self.logname].log(mode+'@{}'.format(k_val), metrics[i], group=mode)
                else:
                    for sub_reward in metrics:
                        self.logger.progress_saver[self.logname].log(mode, sub_reward, group=mode)

                torch.cuda.empty_cache()

        if self.train_data is not None:
            embeds, labels = [], []
            with torch.no_grad():
                for i,inp in enumerate(self.train_data):
                    emb = net(inp[1].type(torch.cuda.FloatTensor))
                    embeds.append(emb.detach().cpu().numpy().tolist())
                    labels.append(inp[0].detach().cpu().numpy())
                embeds   = np.vstack(embeds).astype('float32')
                labels   = np.hstack(labels).reshape(-1,1)

            for i,mode in enumerate(self.state_metrics):
                metrics = self.compute_metric(mode)(embeds, labels)
                state_metrics.append(metrics)

                ### Log Computed Metrics
                if mode=='recall':
                    for i,k_val in enumerate(self.k_vals):
                        self.logger.progress_saver[self.logname].log('T-'+mode+'@{}'.format(k_val), metrics[i], group='T-'+mode)
                else:
                    for sub_reward in metrics:
                        self.logger.progress_saver[self.logname].log('T-'+mode, sub_reward, group='T-'+mode)

                torch.cuda.empty_cache()


        state_metrics = [x for y in state_metrics for x in y]
        return np.mean(reward_metrics), state_metrics


    def compute_metric(self, mode):
        if mode=='recall':
            return self.recall
        elif mode=='dists':
            return self.dists
        elif mode=='nmi':
            return self.nmi
        elif mode=='svd':
            return self.svd
        elif mode=='log_svd':
            return self.log_svd
        else:
            raise Exception('Reward mode {} not available!'.format(mode))

    def dists(self, embeddings, labels):
        embed_label_locs = []
        for lab in np.unique(labels):
            embed_label_locs.append(np.where(labels==lab)[0])

        coms, intra_dists = [],[]

        for loc in embed_label_locs:
            c_dists = distance.cdist(embeddings[loc], embeddings[loc], 'cosine')
            c_dists = np.sum(c_dists)/(len(c_dists)**2-len(c_dists))
            intra_dists.append(c_dists)

            com   = normalize(np.mean(embeddings[loc],axis=0).reshape(1,-1)).reshape(-1)
            coms.append(com)

        mean_inter_dist = distance.cdist(np.array(coms), np.array(coms), 'cosine')
        mean_inter_dist = np.sum(mean_inter_dist)/(len(mean_inter_dist)**2-len(mean_inter_dist))

        rel_embed_dist = np.mean(intra_dists)/mean_inter_dist

        return [np.mean(intra_dists), mean_inter_dist]

    def recall(self, embeddings, labels):
        recall_vals        = self.k_vals
        if not isinstance(embeddings, list):
            ### FOR EVERY QUERY-LESS Dataset
            faiss_search_index = faiss.IndexFlatL2(embeddings.shape[-1])
            faiss_search_index.add(embeddings)
            _, k_nn = faiss_search_index.search(embeddings, int(np.max(recall_vals)+1))
            k_nn_labels = labels.reshape(-1)[k_nn[:,1:]]
        else:
            ### FOR IN-SHOP Dataset
            faiss_search_index = faiss.IndexFlatL2(embeddings[1].shape[-1])
            faiss_search_index.add(embeddings[1])
            _, k_nn = faiss_search_index.search(embeddings[0], int(np.max(recall_vals)+1))
            k_nn_labels = embeddings[1].reshape(-1)[k_nn]

        #
        recall_at_ks = []
        for k in recall_vals:
            recall_at_k = np.sum([1 for label, recalled_pred_labels in zip(labels, k_nn_labels) if label in recalled_pred_labels[:k]])/len(labels)
            recall_at_ks.append(recall_at_k)
        #
        return recall_at_ks

    def nmi(self, embeddings, labels):
        if isinstance(embeddings, list):
            embeddings = np.concatenate(embeddings, axis=0)
            labels     = np.concatenate(labels, axis=0)

        faiss_search_index  = faiss.IndexFlatL2(embeddings.shape[-1])
        faiss_cluster_index = faiss.Clustering(embeddings.shape[-1], self.num_classes)
        faiss_cluster_index.n_iter, faiss_cluster_index.min_points_per_centroid, faiss_cluster_index.max_points_per_centroid = 20,1,1000000000
        #
        faiss_cluster_index.train(embeddings, faiss_search_index)
        embedding_centroids = faiss.vector_float_to_array(faiss_cluster_index.centroids).reshape(self.num_classes, embeddings.shape[-1])
        #
        faiss_search_index  = faiss.IndexFlatL2(embedding_centroids.shape[-1])
        faiss_search_index.add(embedding_centroids)
        _, centroids_cluster_labels = faiss_search_index.search(embeddings, 1)
        #
        NMI = metrics.cluster.normalized_mutual_info_score(centroids_cluster_labels.reshape(-1), labels.reshape(-1))
        #
        return [NMI]

    def svd(self, embeddings, labels=None):
        from sklearn.decomposition import TruncatedSVD
        from scipy.stats import entropy
        n_comp   = embeddings.shape[-1]-1
        svd      = TruncatedSVD(n_components=n_comp, n_iter=7, random_state=42)
        svd.fit(embeddings)
        s        = np.clip(svd.singular_values_, 1e-6, None)
        s_norm   = s/np.sum(s)
        uniform  = np.ones(n_comp)/(n_comp)
        return [entropy(s_norm, uniform)]

    def log_svd(self, embeddings, labels=None):
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=embeddings.shape[-1]-1, n_iter=7, random_state=42)
        svd.fit(train_emb)
        s   = np.clip(svd.singular_values_, 1e-6)
        s_norm = log(s/np.sum(s))
        return list(s_norm)




"""==============================================="""
class Policy(torch.nn.Module):
    def __init__(self, input_dim, output_dim, policy_size, action_values, use_a2c=False, w_init='none'):
        super(Policy,self).__init__()

        self.input_dim, self.output_dim, self.policy_size = input_dim, output_dim, policy_size
        self.action_values, self.use_a2c = np.array(action_values), use_a2c

        #####
        self.network = []
        for i,n_hidden in enumerate(self.policy_size):
            self.network.extend([nn.Linear(int(input_dim), int(n_hidden)), nn.ReLU()])
            input_dim = n_hidden
        self.network = nn.Sequential(*self.network)

        #####
        self.action_weights = nn.Linear(self.policy_size[-1], len(self.action_values)*self.output_dim)
        if self.use_a2c:
            self.value = nn.Linear(self.policy_size[-1],1)

        #####
        self.w_init = w_init
        netlib.initialize_weights(self, w_init)

    def forward(self, x):
        x = self.network(x)
        action_weights = self.action_weights(x).view(x.size(0),len(self.action_values),self.output_dim)
        if self.use_a2c:
            value = self.value(x).view(x.size(0),1)
        else:
            value = None
        return value, action_weights

    def act(self, x):
        with torch.no_grad():
            _,action_weights  = self(x)
            action_probs_bs = F.softmax(action_weights, dim=1)
            actions_to_take = []
            for action_probs in action_probs_bs.detach().cpu().numpy():
                sub_acts = []
                for i in range(action_probs.shape[-1]):
                    select_p = action_probs[:,i]
                    if np.isnan(select_p).any():
                        select_p[np.where(np.isnan(select_p))[0]]=0
                        if np.sum(select_p)==0: select_p = np.ones(len(select_p))
                        select_p = select_p/select_p.sum()
                    sub_acts.append(np.random.choice(len(action_probs), p=select_p))
                actions_to_take.append(sub_acts)
                # actions_to_take.append([np.random.choice(len(action_probs), p=action_probs[:,i]) for i in range(action_probs.shape[-1])])
            actions_to_take = np.array(actions_to_take).reshape(-1)
        return actions_to_take

    def convert(self, action_idxs):
        return self.action_values[action_idxs]





"""==============================================="""
class Parametric_Sampler():
    def __init__(self, n_support, support_limit, save_path=None, wandb_log=False,
                 init_distr='random', merge_oobs=False, include_pos=False, include_same=False):
        self.n_support, self.support_limit = n_support, support_limit
        self.init_distr, self.save_path, self.wandb_log = init_distr, save_path, wandb_log

        self.support = np.linspace(support_limit[0], support_limit[1], self.n_support)

        self.no_norm_distr = self.give_distr(init_distr)
        self.distr         = self.norm(self.no_norm_distr)

        self.distr_collect = {'support_limit':support_limit, 'n_support':n_support, 'progression':[], 'nonorm_progression':[]}

        self.merge_oobs   = merge_oobs
        self.include_pos  = include_pos
        self.include_same = include_same


    def give_distr(self, name):
        if name   == 'random':
            distr_to_init = np.array([1.]*(self.n_support-1))
        elif name == 'distance':
            distr_to_init = self.probmode(self.support, self.n_support, upper_lim=0.5, mode='distance')
        elif name == 'uniform_low_and_tight':
            distr_to_init = self.probmode(self.support, self.n_support, mu=0.4, sig=0.1, mode='uniform')
        elif name == 'uniform_lowish_and_tight':
            distr_to_init = self.probmode(self.support, self.n_support, mu=0.6, sig=0.1, mode='uniform')
        elif name == 'heavyside_low':
            distr_to_init = self.probmode(self.support, self.n_support, mu=0.5,  mode='heavyside')
        elif name == 'uniform_low':
            distr_to_init = self.probmode(self.support, self.n_support, mu=0.5, sig=0.2, mode='uniform')
        elif name == 'uniform_avg':
            distr_to_init = self.probmode(self.support, self.n_support, mu=1.,  sig=0.2, mode='uniform')
        elif name == 'uniform_high':
            distr_to_init = self.probmode(self.support, self.n_support, mu=1.5, sig=0.2, mode='uniform')
        elif name == 'normal_low':
            distr_to_init = self.probmode(self.support, self.n_support, mu=0.5, sig=0.05, mode='gaussian')
        elif name == 'normal_avg':
            distr_to_init = self.probmode(self.support, self.n_support, mu=1.1,  sig=0.04, mode='gaussian')
        elif name == 'normal_high':
            distr_to_init = self.probmode(self.support, self.n_support, mu=1.6, sig=0.04, mode='gaussian')
        else:
            raise Exception('Init. Distr. >> {} << not available!'.format(name))
        return distr_to_init

    def norm(self, distr):
        return distr/np.sum(distr)

    def pdist(self, A):
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min = 0)
        return res.clamp(min = 0).sqrt()

    def update_sample_distr(self, distr_change):
        self.no_norm_distr = np.clip(self.no_norm_distr*distr_change, 1e-25, 1e25)
        self.distr         = self.norm(self.no_norm_distr)

        import copy
        self.distr_collect['progression'].append(copy.deepcopy(self.distr))
        self.distr_collect['nonorm_progression'].append(copy.deepcopy(self.no_norm_distr))
        pkl.dump(self.distr_collect, open(self.save_path+'/distr_progression.pkl','wb'))

    def give(self, batch, labels, gt_labels=None):
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()

        positives, negatives, anchors = [],[],[]
        distances     = self.pdist(batch.detach())

        p_assigns     = np.sum((distances.cpu().numpy().reshape(-1)>self.support[1:-1].reshape(-1,1)).T,axis=1).reshape(distances.shape)
        sample_ps     = self.distr[p_assigns]

        if not self.merge_oobs:
            outside_support_lim = (distances.cpu().numpy().reshape(-1)<self.support_limit[0]) * (distances.cpu().numpy().reshape(-1)>self.support_limit[1])
            outside_support_lim = outside_support_lim.reshape(distances.shape)
            sample_ps[outside_support_lim] = 0

        bs = len(batch)
        for i in range(bs):
            neg = labels!=labels[i]; pos = labels==labels[i]
            anchors.append(i)

            if not self.include_same:
                pos[i] = 0
            positives.append(np.random.choice(np.where(pos)[0]))

            if self.include_pos:
                sample_p = sample_ps[i]
                sample_p = sample_p/sample_p.sum()
                negatives.append(np.random.choice(bs,p=sample_p))
            else:
                sample_p = sample_ps[i][neg]
                sample_p = sample_p/sample_p.sum()
                negatives.append(np.random.choice(np.arange(bs)[neg],p=sample_p))

        sampled_triplets = [[a,p,n] for a,p,n in zip(anchors, positives, negatives)]
        return sampled_triplets



    def log_sampling_distr(self):
        import wandb
        import numpy as np
        wandb.log({'Dist. Distr.':     wandb.Histogram(np_histogram=(np.array(self.distr),np.array(self.support)))})
        wandb.log({'Log Dist. Distr.': wandb.Histogram(np_histogram=(np.log(np.clip(np.array(self.distr), 1e-20, None))-np.log(1e-20),np.array(self.support)))})

    def gaussian(self, x, mu=1, sig=1):
        return 1/np.sqrt(2*np.pi*sig)*np.exp(-(x-mu)**2/(2*sig**2))

    def uniform(self, x, mu=1, sig=0.25):
        sp = x.copy()
        sp[(x>=mu-sig) * (x<=mu+sig)] = 1
        sp[(x<mu-sig) + (x>mu+sig)]   = 0
        return sp

    def distance(self, x, upper_lim):
        sp = x.copy()
        sp[x<=upper_lim] = 1
        sp[x>=upper_lim] = 0
        return sp

    def heavyside(self, x, mu):
        sp = x.copy()
        sp[x<=mu] = 1
        sp[x>mu]  = 0
        return sp

    def probmode(self, support_space, support, mu=None, sig=None, upper_lim=None, mode='uniform'):
        if mode=='uniform':
            probdist = self.uniform(support_space[:-1]+2/support, mu=mu, sig=sig)
        elif mode=='gaussian':
            probdist = self.gaussian(support_space[:-1]+2/support, mu=mu, sig=sig)
        elif mode=='distance':
            probdist = self.distance(support_space[:-1]+2/support, upper_lim)
        elif mode=='heavyside':
            probdist = self.heavyside(support_space[:-1]+2/support, mu=mu)
        probdist = np.clip(probdist, 1e-15, 1)
        probdist = probdist/probdist.sum()
        return probdist

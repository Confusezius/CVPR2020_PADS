"""
Main training function to train DML with/without PADS. Utilizes a set of key arguments, which are augmented with PADS-arguments from <PADS_utilities.py>.
"""

"""==================================================================================================="""
################### LIBRARIES ###################
import warnings
warnings.filterwarnings("ignore")

import os, sys, numpy as np, argparse, imp, datetime, time, pickle as pkl, random, json, collections
# os.chdir('/media/karsten_dl/QS/Data/Dropbox/Projects/Confusezius_git/RL_Sampler_DML_V2')
# os.chdir('/home/karsten_dl/Dropbox/Projects/Confusezius_git/RL_Sampler_DML')
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
import pandas as pd

import torch, torch.nn as nn
import auxiliaries as aux
import datasets as data

# sys.path.insert(0,os.getcwd()+'/models')
import netlib as netlib
import losses as losses
import evaluate as eval

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import PADS_utilities as rl


"""==================================================================================================="""
################### INPUT ARGUMENTS ###################
parser = argparse.ArgumentParser()

####### Main Parameter: Dataset to use for Training
parser.add_argument('--dataset',      default='cub200',   type=str, help='Dataset to use.')

### General Training Parameters
parser.add_argument('--lr',                default=0.00001,  type=float, help='Learning Rate for network parameters.')
parser.add_argument('--n_epochs',          default=150,      type=int,   help='Number of training epochs.')
parser.add_argument('--kernels',           default=8,        type=int,   help='Number of workers for pytorch dataloader.')
parser.add_argument('--bs',                default=112 ,     type=int,   help='Mini-Batchsize to use.')
parser.add_argument('--samples_per_class', default=4,        type=int,   help='Number of samples in one class drawn before choosing the next class. Set to >1 for losses other than ProxyNCA.')
parser.add_argument('--seed',              default=1,        type=int,   help='Random seed for reproducibility.')
parser.add_argument('--scheduler',         default='step',   type=str,   help='Type of learning rate scheduling. Currently: step & exp.')
parser.add_argument('--gamma',             default=0.3,      type=float, help='Learning rate reduction after tau epochs.')
parser.add_argument('--decay',             default=0.0004,   type=float, help='Weight decay for optimizer.')
parser.add_argument('--tau',               default=[150],nargs='+',type=int,help='Stepsize before reducing learning rate.')

##### Loss-specific Settings
parser.add_argument('--loss',         default='triplet',    type=str,   help='Choose between TripletLoss, ProxyNCA, ...')
parser.add_argument('--sampling',     default='learned',    type=str,   help='For triplet-based losses: Modes of Sampling: random, semihard, distance.')
### MarginLoss
parser.add_argument('--margin',       default=0.2,          type=float, help='Margin for Triplet Loss')
parser.add_argument('--beta_lr',      default=0.0005,       type=float, help='Learning Rate for class margin parameters in MarginLoss')
parser.add_argument('--beta',         default=1.2,          type=float, help='Initial Class Margin Parameter in Margin Loss')
parser.add_argument('--nu',           default=0,            type=float, help='Regularisation value on betas in Margin Loss.')
parser.add_argument('--beta_constant',                      action='store_true')
### ProxyNCA
parser.add_argument('--proxy_lr',     default=0.00001,     type=float, help='Learning Rate for Proxies in ProxyNCALoss.')
### NPair L2 Penalty
parser.add_argument('--l2npair',      default=0.02,        type=float, help='Learning Rate for Proxies in ProxyNCALoss.')

##### Evaluation Settings
parser.add_argument('--k_vals',       nargs='+', default=[1,2,4,8], type=int, help='Recall @ Values.')

##### Network parameters
parser.add_argument('--embed_dim',    default=128,         type=int,   help='Embedding dimensionality of the network. Note: dim=128 or 64 is used in most papers.')
parser.add_argument('--arch',         default='resnet50',  type=str,   help='Choice of loss function. Alternative Option: ProxyNCA')
parser.add_argument('--not_pretrained',                    action='store_true')

##### Setup Parameters
parser.add_argument('--gpu',          default=0,           type=int,   help='Random seed for reproducibility.')
parser.add_argument('--no_weights',                        action='store_true')
parser.add_argument('--savename',     default='group_plus_seed',   type=str,   help='Appendix to save folder name if any special information is to be included.')
### Paths to datasets and storage folder
parser.add_argument('--source_path',  default=os.getcwd()+'/../../Datasets', type=str, help='Path to training data.')
parser.add_argument('--save_path',    default=os.getcwd()+'/Training_Results', type=str, help='Where to save everything.')

### Wandb Log Arguments
parser.add_argument('--wandb_log',            action='store_true')
parser.add_argument('--wandb_project_name',   default='RL-Sampler-NovelSet',  type=str,   help='Appendix to save folder name if any special information is to be included.')
parser.add_argument('--wandb_group',          default='SampleGroup',  type=str,   help='Appendix to save folder name if any special information is to be included.')

### Include PADS input flags
parser = rl.include_pads_args(parser)

##### Read in parameters
opt = parser.parse_args()




"""==================================================================================================="""
### The following setting is useful when logging to wandb and running multiple seeds per setup:
### By setting the savename to <group_plus_seed>, the savename will instead comprise the wandb_group and the seed!
if opt.savename=='group_plus_seed':
    if opt.wandb_log:
        opt.savename = opt.wandb_group+'_s{}'.format(opt.seed)
    else:
        opt.savename = ''

### If wandb-logging is turned on, initialize the wandb-run here:
if opt.wandb_log:
    import wandb
    wandb.init(project=opt.wandb_project_name, group=opt.wandb_group, name=opt.savename, dir=opt.save_path)
    wandb.config.update(opt)


"""==================================================================================================="""
full_training_start_time = time.time()



"""==================================================================================================="""
opt.source_path += '/'+opt.dataset
opt.save_path   += '/'+opt.dataset


#Adjust the Recall-Values for SOP to Literature values:
if opt.dataset=='online_products':
    opt.k_vals = [1,10,100,1000]

#Assert that the construction of the batch makes sense, i.e. the division into class-subclusters.
assert not opt.bs%opt.samples_per_class, 'Batchsize needs to fit number of samples per class for distance sampling and margin/triplet loss!'

opt.pretrained = not opt.not_pretrained
opt.use_learned_sampler = opt.sampling=='learned'


"""==================================================================================================="""
################### GPU SETTINGS ###########################
os.environ["CUDA_DEVICE_ORDER"]   ="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(opt.gpu)


"""==================================================================================================="""
#################### SEEDS FOR REPROD. #####################
torch.backends.cudnn.deterministic=True; np.random.seed(opt.seed); random.seed(opt.seed)
torch.manual_seed(opt.seed); torch.cuda.manual_seed(opt.seed); torch.cuda.manual_seed_all(opt.seed)


"""==================================================================================================="""
##################### NETWORK SETUP ##################
#NOTE: Networks that can be used: 'bninception, resnet50, resnet101, alexnet...'
#>>>>  see import pretrainedmodels; pretrainedmodels.model_names
opt.device = torch.device('cuda')
model      = netlib.networkselect(opt)
# if opt.wandb_log: wandb.watch(model)

print('{} Setup for {} with {} sampling on {} complete with #weights: {}'.format(opt.loss.upper(), opt.arch.upper(), opt.sampling.upper(), opt.dataset.upper(), aux.gimme_params(model)))

_          = model.to(opt.device)
to_optim   = [{'params':model.parameters(),'lr':opt.lr,'weight_decay':opt.decay}]


"""============================================================================"""
#################### DATALOADER SETUPS ##################
dataloaders      = data.give_dataloaders(opt.dataset, opt)
opt.num_classes  = len(dataloaders['training'].dataset.avail_classes)


"""============================================================================"""
#################### CREATE LOGGING FILES ###############
sub_loggers = ['Train', 'Test', 'Model Grad']
if opt.use_learned_sampler:
    sub_loggers += ['RL-Policy', 'RL-Policy Grad', 'Val']
LOG = aux.LOGGER(opt, sub_loggers=sub_loggers, start_new=True, log_to_wandb=opt.wandb_log)


"""============================================================================"""
#################### LOSS SETUP ####################
if opt.use_learned_sampler: opt.sampling = 'random'
criterion, to_optim = losses.loss_select(opt.loss, opt, to_optim)
if opt.use_learned_sampler: opt.sampling = 'learned'
_ = criterion.to(opt.device)


"""============================================================================"""
############################################# vvv RL_SAMPLER vvv ##################################################
if opt.use_learned_sampler:
    # rl_sub_loggers = ['RL-Policy', 'RL-Policy Grad', 'Val']
    # RL_LOG = aux.LOGGER(opt, sub_loggers=rl_sub_loggers, start_new=False, log_to_wandb=opt.wandb_log)

    general_pars     = {'policy_lr':opt.policy_lr, 'logger':LOG, 'logname':'RL-Policy', 'old_policy_update':opt.policy_old_update_iter,
                        'metric_history':opt.policy_metric_history, 'mode':opt.policy_mode, 'parameter_history':opt.policy_parameter_history, 'ppo_ratio': opt.policy_ppo_ratio,
                        'call_delay':opt.policy_training_delay,  'sample_frequency':opt.policy_sample_freq, 'sup_metric_collect': opt.policy_sup_metric_collect,
                        'wandb_log':opt.wandb_log, 'ema_alpha': opt.policy_baseline_ema_gamma, 'reward_update_w': opt.policy_reward_update_w,
                        'log_distr_plots':opt.policy_log_distr_plots}
    #
    state_maker_pars = {'n_support':opt.policy_n_support, 'max_iter':opt.n_epochs*len(dataloaders['training']), 'num_v_classes':len(dataloaders['validation'].dataset.avail_classes),
                        'state_metrics':  opt.policy_state_metrics,   'running_averages':opt.policy_run_avgs, 'no_avg': opt.policy_run_avgs_no_avg, 'include_train_metrics': opt.policy_include_train_metrics,
                        'k_vals':   opt.k_vals,           'metric_history':opt.policy_metric_history, 'parameter_history':opt.policy_parameter_history}
    #
    policy_pars      = {'output_dim':   opt.policy_n_support-1,   'policy_size':opt.policy_size,
                        'action_values':opt.policy_action_values, 'use_a2c':not opt.policy_dont_use_a2c,
                        'w_init':       opt.policy_winit_type}
    #
    sampler_pars     = {'n_support': opt.policy_n_support,  'support_limit':opt.policy_support_limit,   'include_same':opt.policy_include_same,
                        'init_distr':opt.policy_init_distr, 'include_pos':opt.policy_include_pos,
                        'save_path': LOG.save_path,      'wandb_log':opt.wandb_log, 'merge_oobs':opt.policy_merge_oobs}

    #
    val_pars         = {'logger':LOG, 'logname':'Val', 'reward_type':opt.policy_reward_type, 'dataloader':dataloaders['validation'] if not opt.policy_include_train_metrics else [dataloaders['validation'], dataloaders['evaluation']],
                        'k_vals':opt.k_vals, 'state_metrics':opt.policy_state_metrics, 'wandb_log':opt.wandb_log}


    ####
    imp.reload(rl)
    PADS = rl.PADS(general_pars=general_pars, state_maker_pars=state_maker_pars, policy_pars=policy_pars,
                   sampler_pars=sampler_pars, val_pars=val_pars)
    _    = PADS.policy.to(opt.device)
    PADS.assign_sampler(criterion)
############################################# ^^^ RL_SAMPLER ^^^ ##################################################


"""============================================================================"""
#################### OPTIM SETUP ####################
optimizer    = torch.optim.Adam(to_optim)
scheduler    = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.tau, gamma=opt.gamma)


"""============================================================================"""
################### SCRIPT MAIN ##########################
print('\n-----\n')

iter_count = 0
for epoch in range(opt.n_epochs):
    opt.epoch = epoch
    ### Scheduling Changes specifically for cosine scheduling
    if opt.scheduler!='none': print('Running with learning rates {}...'.format(' | '.join('{}'.format(x) for x in scheduler.get_lr())))


    """======================================="""
    if opt.use_learned_sampler and opt.policy_scramble_dataloaders and epoch%opt.policy_scramble_freq==0 and epoch!=0:
        print('Scrambled!')
        rl.scramble_train_val(dataloaders['training'], dataloaders['validation'], PADS, opt.train_val_split, opt.train_val_split_by_class)


    """======================================="""
    ### Train one epoch
    start = time.time()
    _ = model.train()

    loss_collect = []
    grad_l2, grad_max = [], []
    data_iterator = tqdm(dataloaders['training'], desc='Epoch {} Training...'.format(epoch))
    for i,(class_labels, input) in enumerate(data_iterator):
        features  = model(input.to(opt.device))

        loss      = criterion(features, class_labels)
        optimizer.zero_grad()
        loss.backward()

        ### Compute Model Gradients and log them!
        grads              = np.concatenate([p.grad.detach().cpu().numpy().flatten() for p in model.parameters() if p.grad is not None])
        grad_l2.append(np.mean(np.sqrt(np.mean(np.square(grads)))))
        grad_max.append(np.mean(np.max(np.abs(grads))))

        ### Update network weights!
        optimizer.step()

        ###
        loss_collect.append(loss.item())

        ###
        iter_count += 1

        ###
        if opt.use_learned_sampler:
            PADS.update(criterion, model, iter_count, epoch)

        if i==len(dataloaders['training'])-1: data_iterator.set_description('Epoch (Train) {0}: Mean Loss [{1:.4f}]'.format(epoch, np.mean(loss_collect)))


    ####
    LOG.progress_saver['Model Grad'].log('Grad L2',  np.mean(grad_l2),  group='L2')
    LOG.progress_saver['Model Grad'].log('Grad Max', np.mean(grad_max), group='Max')
    LOG.progress_saver['Train'].log('epochs', epoch)
    LOG.progress_saver['Train'].log('loss', np.mean(loss_collect))
    LOG.progress_saver['Train'].log('time', np.round(time.time()-start, 4))


    """======================================="""
    ### Evaluate -
    _ = model.eval()
    if opt.dataset in ['cars196', 'cub200', 'online_products']:
        eval_params = {'dataloader':dataloaders['testing'], 'model':model, 'opt':opt}
    elif opt.dataset=='in-shop':
        eval_params = {'query_dataloader':dataloaders['query'], 'gallery_dataloader':dataloaders['gallery'], 'model':model, 'opt':opt}
    elif opt.dataset=='vehicle_id':
        eval_params = {'dataloaders':[dataloaders['testing_set1'], dataloaders['testing_set2'], dataloaders['testing_set3']], 'model':model, 'opt':opt}
    eval_params['epoch'] = epoch

    if opt.use_learned_sampler: eval_params['aux_store'] = PADS.policy.state_dict()
    eval.evaluate(opt.dataset, LOG, save=True, **eval_params)

    LOG.update(all=True)


    """======================================="""
    ### Learning Rate Scheduling Step
    if opt.scheduler != 'none':
        scheduler.step()

    print('\n-----\n')




"""======================================================="""
### CREATE A SUMMARY TEXT FILE
summary_text = ''
full_training_time = time.time()-full_training_start_time
summary_text += 'Training Time: {} min.\n'.format(np.round(full_training_time/60,2))

summary_text += '---------------\n'
for sub_logger in LOG.sub_loggers:
    metrics       = LOG.graph_writer[sub_logger].ov_title
    summary_text += '{} metrics: {}\n'.format(sub_logger.upper(), metrics)

with open(opt.save_path+'/training_summary.txt','w') as summary_file:
    summary_file.write(summary_text)

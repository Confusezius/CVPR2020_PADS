import os, sys, numpy as np, argparse, imp, datetime, time, pickle as pkl, random, json, collections
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch, torch.nn as nn

#Custom Libraries
import datasets as data
import netlib as netlib
import auxiliaries as aux


"""==============================="""
#Name of folder containing the network checkpoint.
network   = 'CUB_PADS_R50'
#Path to above folder - this setup assumes that the full folder is stored in the same directory as this script.
netfolder = 'CVPR2020_TrainingResults/CUB/R50'
#Load network and setup parameters, which are stored in a Namespace.
opt       = pkl.load(open(netfolder+'/'+network+'/'+'hypa.pkl','rb'))
#Load network passed on the resp. parameters and load with trained weights.
model     = netlib.networkselect(opt)
model.load_state_dict(torch.load(netfolder+'/'+network+'/checkpoint_Class.pth.tar')['state_dict'])


"""================================"""
os.environ["CUDA_DEVICE_ORDER"]   ="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"


"""==============================="""
#Get dataloaders, primarily the one for the test set. For that, point to the folder that contains the datasets:
opt.source_path = '<path_to_dataset>/'+opt.dataset
dataloaders      = data.give_dataloaders(opt.dataset, opt)
opt.num_classes  = len(dataloaders['training'].dataset.avail_classes)
opt.device       = torch.device('cuda')


"""================================"""
#Compute test metrics - note that weights were stored at optimal R@1 performance.
_ = model.to(opt.device)
_ = model.eval()
start = time.time()
image_paths = np.array(dataloaders['testing'].dataset.image_list)
with torch.no_grad():
    evaltypes       = ['Class']
    metrics, labels = aux.eval_metrics_one_dataset(model, dataloaders['testing'], device=opt.device, k_vals=opt.k_vals, opt=opt, evaltypes=evaltypes)
    ###
    full_result_str = ''
    for evaltype in evaltypes:
        result_str = ', '.join('@{0}: {1:.4f}'.format(k,rec) for k,rec in zip(opt.k_vals, metrics[evaltype]['Recall@k']))
        result_str = '{0}-embed: NMI [{1:.4f}] | F1 [{2:.4f}] | Recall [{3}]'.format(evaltype, metrics[evaltype]['NMI'], metrics[evaltype]['F1'], result_str)
        full_result_str += result_str+'\n'

print(full_result_str)

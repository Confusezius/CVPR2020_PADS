"""
Main evaluation functions that embeddeds all test images and computes metrics described in auxiliaries.py.
"""

"""=================================================================================================================="""
################### LIBRARIES ###################
import warnings
warnings.filterwarnings("ignore")

import os, sys, numpy as np, argparse, imp, datetime, time, pickle as pkl, random, json, csv
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
import pandas as pd

import torch, torch.nn as nn
import auxiliaries as aux
import datasets as data

import netlib
import losses as losses

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


"""=================================================================================================================="""
def evaluate(dataset, LOG, **kwargs):
    if dataset in ['cars196', 'cub200', 'online_products']:
        ret = evaluate_one_dataset(LOG, **kwargs)
    else:
        raise Exception('No implementation for dataset {} available!')

    return ret



"""========================================================="""
from scipy.spatial import distance
from sklearn.preprocessing import normalize

def distance_measure(embeddings, labels):
    embedding_locs = []
    for lab in np.unique(labels):
        embedding_locs.append(np.where(labels==lab)[0])

    coms, intra_dists = [],[]

    for loc in embedding_locs:
        c_dists = distance.cdist(embeddings[loc], embeddings[loc], 'cosine')
        c_dists = np.sum(c_dists)/(len(c_dists)**2-len(c_dists))
        intra_dists.append(c_dists)
        com   = normalize(np.mean(embeddings[loc],axis=0).reshape(1,-1)).reshape(-1)
        coms.append(com)

    mean_inter_dist = distance.cdist(np.array(coms), np.array(coms), 'cosine')
    mean_inter_dist = np.sum(mean_inter_dist)/(len(mean_inter_dist)**2-len(mean_inter_dist))
    # rel_embed_dist = np.mean(intra_dists)/mean_inter_dist
    mean_intra_dist = np.mean(intra_dists)

    return mean_intra_dist, mean_inter_dist



"""========================================================="""
def evaluate_one_dataset(LOG, dataloader, model, opt, spliteval=True, evaltypes=['Class'], save=True, give_return=False, aux_store=None, epoch=0, monitor_distances=True):
    start = time.time()
    image_paths = np.array(dataloader.dataset.image_list)
    with torch.no_grad():
        metrics, labels = aux.eval_metrics_one_dataset(model, dataloader, device=opt.device, spliteval=spliteval, k_vals=opt.k_vals, opt=opt, evaltypes=evaltypes)

        ###
        full_result_str = ''
        for evaltype in evaltypes:
            result_str = ', '.join('@{0}: {1:.4f}'.format(k,rec) for k,rec in zip(opt.k_vals, metrics[evaltype]['Recall@k']))
            result_str = '{0}-embed: NMI [{1:.4f}] | F1 [{2:.4f}] | Recall [{3}]'.format(evaltype, metrics[evaltype]['NMI'], metrics[evaltype]['F1'], result_str)
            full_result_str += result_str+'\n'

        ###
        if LOG is not None :
            for evaltype in evaltypes:
                if save:
                    if (evaltype+'_Recall' not in LOG.progress_saver['Test'].groups.keys()) or metrics[evaltype]['Recall@k'][0]>np.max(LOG.progress_saver['Test'].groups[evaltype+'_Recall']['Recall @ 1']['content']):
                        aux.set_checkpoint(model, opt, LOG.progress_saver, LOG.prop.save_path+'/checkpoint_{}.pth.tar'.format(evaltype), aux=aux_store)
                        aux.recover_closest_one_dataset(metrics[evaltype]['Features'], image_paths, LOG.prop.save_path+'/sample_recoveries.png')

                LOG.progress_saver['Test'].log('NMI',    metrics[evaltype]['NMI'],   group=evaltype+'_NMI')
                LOG.progress_saver['Test'].log('F1',     metrics[evaltype]['F1'],    group=evaltype+'_F1')
                for k_val, recall_val in zip(opt.k_vals, metrics[evaltype]['Recall@k']):
                    LOG.progress_saver['Test'].log('Recall @ {}'.format(k_val), recall_val, group=evaltype+'_Recall')

                if monitor_distances:
                    intra_dist, inter_dist = distance_measure(metrics[evaltype]['Features'], labels)
                    LOG.progress_saver['Test'].log('Intraclass',intra_dist,group=evaltype+'_Distances')
                    LOG.progress_saver['Test'].log('Interclass',inter_dist,group=evaltype+'_Distances')


        LOG.progress_saver['Test'].log('Epochs', epoch, group='Epochs')
        LOG.progress_saver['Test'].log('Time',   np.round(time.time()-start, 4), group='Time')


    print(full_result_str)
    if give_return:
        return metrics
    else:
        None

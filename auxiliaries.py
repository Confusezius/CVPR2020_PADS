"""
Contains utility functions to compute standard DML metrics such as Recall, NMI or F1.
Also some other QOL stuff that is helpful and the main Data-Logger class.
"""



"""============================================================================================================="""
######## LIBRARIES #####################
import warnings
warnings.filterwarnings("ignore")

import numpy as np, os, sys, pandas as pd, csv
import torch, torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import faiss
from sklearn import metrics
from sklearn import cluster

import losses as losses
import datetime
import pickle as pkl



"""============================================================================================================="""
################# ACQUIRE NUMBER OF WEIGHTS #################
def gimme_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


################# SAVE TRAINING PARAMETERS IN NICE STRING #################
def gimme_save_string(opt):
    varx = vars(opt)
    base_str = ''
    for key in varx:
        base_str += str(key)
        if isinstance(varx[key],dict):
            for sub_key, sub_item in varx[key].items():
                base_str += '\n\t'+str(sub_key)+': '+str(sub_item)
        else:
            base_str += '\n\t'+str(varx[key])
        base_str+='\n\n'
    return base_str



def f1_score(model_generated_cluster_labels, target_labels, feature_coll, computed_centroids):
    from scipy.special import comb

    d = np.zeros(len(feature_coll))
    for i in range(len(feature_coll)):
        d[i] = np.linalg.norm(feature_coll[i,:] - computed_centroids[model_generated_cluster_labels[i],:])

    labels_pred = np.zeros(len(feature_coll))
    for i in np.unique(model_generated_cluster_labels):
        index = np.where(model_generated_cluster_labels == i)[0]
        ind = np.argmin(d[index])
        cid = index[ind]
        labels_pred[index] = cid


    N = len(target_labels)

    # cluster n_labels
    avail_labels = np.unique(target_labels)
    n_labels     = len(avail_labels)

    # count the number of objects in each cluster
    count_cluster = np.zeros(n_labels)
    for i in range(n_labels):
        count_cluster[i] = len(np.where(target_labels == avail_labels[i])[0])

    # build a mapping from item_id to item index
    keys     = np.unique(labels_pred)
    num_item = len(keys)
    values   = range(num_item)
    item_map = dict()
    for i in range(len(keys)):
        item_map.update([(keys[i], values[i])])


    # count the number of objects of each item
    count_item = np.zeros(num_item)
    for i in range(N):
        index = item_map[labels_pred[i]]
        count_item[index] = count_item[index] + 1

    # compute True Positive (TP) plus False Positive (FP)
    tp_fp = 0
    for k in range(n_labels):
        if count_cluster[k] > 1:
            tp_fp = tp_fp + comb(count_cluster[k], 2)

    # compute True Positive (TP)
    tp = 0
    for k in range(n_labels):
        member = np.where(target_labels == avail_labels[k])[0]
        member_ids = labels_pred[member]

        count = np.zeros(num_item)
        for j in range(len(member)):
            index = item_map[member_ids[j]]
            count[index] = count[index] + 1

        for i in range(num_item):
            if count[i] > 1:
                tp = tp + comb(count[i], 2)

    # False Positive (FP)
    fp = tp_fp - tp

    # compute False Negative (FN)
    count = 0
    for j in range(num_item):
        if count_item[j] > 1:
            count = count + comb(count_item[j], 2)

    fn = count - tp

    # compute F measure
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    beta = 1
    F = (beta*beta + 1) * P * R / (beta*beta * P + R)

    return F




"""============================================================================================================="""
def eval_metrics_one_dataset(model, test_dataloader, device, k_vals=[1,2,4,8], spliteval=True, evaltypes=['Class'], epoch=0, opt=None):
    torch.cuda.empty_cache()

    _ = model.eval()
    n_classes = len(test_dataloader.dataset.avail_classes)

    feature_colls = {evaltype:[] for evaltype in evaltypes}

    ### For all test images, extract features
    with torch.no_grad():
        target_labels, feature_coll = [],[]
        final_iter = tqdm(test_dataloader, desc='Computing {} Set(s) of Evaluation Metrics...'.format(len(evaltypes)))
        image_paths= [x[0] for x in test_dataloader.dataset.image_list]
        for idx,inp in enumerate(final_iter):
            input_img,target = inp[-1], inp[0]
            target_labels.extend(target.numpy().tolist())
            out = model(input_img.to(device))
            for evaltype in evaltypes:
                if 'Combined' in evaltype:
                    weights = [float(x) for x in evaltype.split('-')[1:]]
                    feature_colls[evaltype].extend(torch.nn.functional.normalize(torch.cat([weights[0]*out['Class'],weights[1]*out['Aux']], dim=-1), dim=-1).cpu().detach().numpy().tolist())
                else:
                    if isinstance(out, dict):
                        feature_colls[evaltype].extend(out[evaltype].cpu().detach().numpy().tolist())
                    else:
                        feature_colls[evaltype].extend(out.cpu().detach().numpy().tolist())
        target_labels = np.hstack(target_labels).reshape(-1,1)

        computed_metrics = {evaltype:{} for evaltype in evaltypes}

        for evaltype in evaltypes:
            feature_coll = np.vstack(feature_colls[evaltype]).astype('float32')

            torch.cuda.empty_cache()
            ### Set CPU Cluster index
            cpu_cluster_index = faiss.IndexFlatL2(feature_coll.shape[-1])
            kmeans            = faiss.Clustering(feature_coll.shape[-1], n_classes)
            kmeans.niter = 20
            kmeans.min_points_per_centroid = 1
            kmeans.max_points_per_centroid = 1000000000

            ### Train Kmeans
            kmeans.train(feature_coll, cpu_cluster_index)
            computed_centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(n_classes, feature_coll.shape[-1])

            ### Assign feature points to clusters
            faiss_search_index = faiss.IndexFlatL2(computed_centroids.shape[-1])
            faiss_search_index.add(computed_centroids)
            _, model_generated_cluster_labels = faiss_search_index.search(feature_coll, 1)

            ### Compute NMI
            NMI = metrics.cluster.normalized_mutual_info_score(model_generated_cluster_labels.reshape(-1), target_labels.reshape(-1))


            ### Recover max(k_vals) nearest neighbours to use for recall computation
            faiss_search_index  = faiss.IndexFlatL2(feature_coll.shape[-1])
            faiss_search_index.add(feature_coll)
            _, k_closest_points = faiss_search_index.search(feature_coll, int(np.max(k_vals)+1))
            k_closest_classes   = target_labels.reshape(-1)[k_closest_points[:,1:]]

            ### Compute Recall
            recall_all_k = []
            for k in k_vals:
                recall_at_k = np.sum([1 for target, recalled_predictions in zip(target_labels, k_closest_classes) if target in recalled_predictions[:k]])/len(target_labels)
                recall_all_k.append(recall_at_k)

            ### Compute F1 Score
            F1 = f1_score(model_generated_cluster_labels, target_labels, feature_coll, computed_centroids)

            computed_metrics[evaltype] = {'F1':F1, 'NMI':NMI, 'Recall@k':recall_all_k, 'Features':feature_coll}
    return computed_metrics, target_labels



"""============================================================================================================="""
####### RECOVER CLOSEST EXAMPLE IMAGES #######
def recover_closest_one_dataset(feature_matrix_all, image_paths, save_path, n_image_samples=10, n_closest=3):
    image_paths = np.array([x[0] for x in image_paths])
    sample_idxs = np.random.choice(np.arange(len(feature_matrix_all)), n_image_samples)

    faiss_search_index = faiss.IndexFlatL2(feature_matrix_all.shape[-1])
    faiss_search_index.add(feature_matrix_all)
    _, closest_feature_idxs = faiss_search_index.search(feature_matrix_all, n_closest+1)

    sample_paths = image_paths[closest_feature_idxs][sample_idxs]

    f,axes = plt.subplots(n_image_samples, n_closest+1)
    for i,(ax,plot_path) in enumerate(zip(axes.reshape(-1), sample_paths.reshape(-1))):
        ax.imshow(np.array(Image.open(plot_path)))
        ax.set_xticks([])
        ax.set_yticks([])
        if i%(n_closest+1):
            ax.axvline(x=0, color='g', linewidth=13)
        else:
            ax.axvline(x=0, color='r', linewidth=13)
    f.set_size_inches(10,20)
    f.tight_layout()
    f.savefig(save_path)
    plt.close()



"""============================================================================================================="""
################## SET NETWORK TRAINING CHECKPOINT #####################
def set_checkpoint(model, opt, progress_saver, savepath, aux=None):
    torch.save({'state_dict':model.state_dict(), 'opt':opt, 'progress':progress_saver, 'aux':aux}, savepath)





"""============================================================================================================="""
################## WRITE TO CSV FILE #####################
class CSV_Writer():
    def __init__(self, save_path):
        self.save_path = save_path
        self.written         = []
        self.n_written_lines = {}

    def log(self, group, segments, content):
        if group not in self.n_written_lines.keys():
            self.n_written_lines[group] = 0

        with open(self.save_path+'_'+group+'.csv', "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            if group not in self.written: writer.writerow(segments)
            for line in content:
                writer.writerow(line)
                self.n_written_lines[group] += 1

        self.written.append(group)



################## PLOT SUMMARY IMAGE #####################
class InfoPlotter():
    def __init__(self, save_path, title='Training Log', figsize=(25,19)):
        self.save_path = save_path
        self.title     = title
        self.figsize   = figsize
        self.colors    = ['r','g','b','y','m','c','orange','darkgreen','lightblue']

    def make_plot(self, base_title, title_append, sub_plots, sub_plots_data):
        sub_plots = list(sub_plots)
        if 'epochs' not in sub_plots:
            x_data = range(len(sub_plots_data[0]))
        else:
            x_data = range(sub_plots_data[np.where(np.array(sub_plots)=='epochs')[0][0]][-1]+1)

        self.ov_title = [(sub_plot,sub_plot_data) for sub_plot, sub_plot_data in zip(sub_plots,sub_plots_data) if sub_plot.lower() not in ['epoch','epochs','time']]
        self.ov_title = [(x[0],np.max(x[1])) if 'loss' not in x[0].lower() else (x[0],np.min(x[1])) for x in self.ov_title]
        self.ov_title = title_append +': '+ '  |  '.join('{0}: {1:.4f}'.format(x[0],x[1]) for x in self.ov_title)
        sub_plots_data = [x for x,y in zip(sub_plots_data, sub_plots) if y.lower() not in ['epochs']]
        sub_plots      = [x for x in sub_plots if x.lower() not in ['epochs']]

        plt.style.use('ggplot')
        f,ax = plt.subplots(1)
        ax.set_title(self.ov_title, fontsize=22)
        for i,(data, title) in enumerate(zip(sub_plots_data, sub_plots)):
            ax.plot(x_data, data, '-{}'.format(self.colors[i]), linewidth=1.7, label=base_title+' '+title)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.tick_params(axis='both', which='minor', labelsize=18)
        ax.legend(loc=2, prop={'size': 16})
        f.set_size_inches(self.figsize[0], self.figsize[1])
        f.savefig(self.save_path+'_'+title_append+'.svg')
        plt.close()


################## GENERATE LOGGING FOLDER/FILES #######################
def set_logging(opt):
    checkfolder = opt.save_path+'/'+opt.savename
    if opt.savename == '':
        date = datetime.datetime.now()
        time_string = '{}-{}-{}-{}-{}-{}'.format(date.year, date.month, date.day, date.hour, date.minute, date.second)
        checkfolder = opt.save_path+'/{}_{}_'.format(opt.dataset.upper(), opt.arch.upper())+time_string
    counter     = 1
    while os.path.exists(checkfolder):
        checkfolder = opt.save_path+'/'+opt.savename+'_'+str(counter)
        counter += 1
    os.makedirs(checkfolder)
    opt.save_path = checkfolder

    with open(opt.save_path+'/Parameter_Info.txt','w') as f:
        f.write(gimme_save_string(opt))
    pkl.dump(opt,open(opt.save_path+"/hypa.pkl","wb"))


class Progress_Saver():
    def __init__(self):
        self.groups = {}

    def log(self, segment, content, group=None):
        if group is None: group = segment
        if group not in self.groups.keys():
            self.groups[group] = {}

        if segment not in self.groups[group].keys():
            self.groups[group][segment] = {'content':[],'saved_idx':0}

        self.groups[group][segment]['content'].append(content)


class LOGGER():
    def __init__(self, opt, sub_loggers=[], prefix=None, start_new=True, log_to_wandb=False):
        self.prop   = opt
        self.prefix = '{}_'.format(prefix) if prefix is not None else ''
        self.sub_loggers = sub_loggers

        ### Make Logging Directories
        if start_new: set_logging(opt)

        ### Set Graph and CSV writer
        self.csv_writer, self.graph_writer, self.progress_saver = {},{},{}
        for sub_logger in sub_loggers:
            csv_savepath = opt.save_path+'/CSV_Logs'
            if not os.path.exists(csv_savepath): os.makedirs(csv_savepath)
            self.csv_writer[sub_logger]     = CSV_Writer(csv_savepath+'/Data_{}{}'.format(self.prefix, sub_logger))

            prgs_savepath = opt.save_path+'/Progression_Plots'
            if not os.path.exists(prgs_savepath): os.makedirs(prgs_savepath)
            self.graph_writer[sub_logger]   = InfoPlotter(prgs_savepath+'/Graph_{}{}'.format(self.prefix, sub_logger))

            self.progress_saver[sub_logger] = Progress_Saver()

        ### WandB Init
        self.save_path    = opt.save_path
        self.log_to_wandb = log_to_wandb

    def update(self, *sub_loggers, all=False):
        wandb_content = []

        if all: sub_loggers = self.sub_loggers

        for sub_logger in list(sub_loggers):
            for group in self.progress_saver[sub_logger].groups.keys():
                pgs      = self.progress_saver[sub_logger].groups[group]
                segments = pgs.keys()
                per_seg_saved_idxs = [pgs[segment]['saved_idx'] for segment in segments]
                per_seg_contents     = [pgs[segment]['content'][idx:] for segment,idx in zip(segments, per_seg_saved_idxs)]
                per_seg_contents_all = [pgs[segment]['content'] for segment,idx in zip(segments, per_seg_saved_idxs)]

                #Adjust indexes
                for content,segment in zip(per_seg_contents, segments):
                    self.progress_saver[sub_logger].groups[group][segment]['saved_idx'] += len(content)

                tupled_seg_content = [list(seg_content_slice) for seg_content_slice in zip(*per_seg_contents)]

                self.csv_writer[sub_logger].log(group, segments, tupled_seg_content)
                if 'epoch' not in group.lower():
                    self.graph_writer[sub_logger].make_plot(sub_logger, group, segments, per_seg_contents_all)

                for i,segment in enumerate(segments):
                    if 'epoch' not in segment:
                        if group == segment:
                            name = sub_logger+': '+group.title()
                        else:
                            name = sub_logger+': '+group.title()+': '+segment.title()
                        wandb_content.append((name,per_seg_contents[i]))

        if self.log_to_wandb:
            import wandb

            commit=False
            for i,item in enumerate(wandb_content):
                if i==len(wandb_content)-1: commit=True

                if isinstance(item[1], list):
                    for j,sub_item in enumerate(item[1]):
                        wandb.log({item[0]:sub_item}, commit=commit)
                else:
                    wandb.log({item[0]:item[0]}, commit=commit)




"""================================================================================================="""
### Container to use with latent space separation
def run_kmeans(features, n_cluster):
    n_samples, dim = features.shape
    kmeans = faiss.Kmeans(dim, n_cluster)
    kmeans.n_iter, kmeans.min_points_per_centroid, kmeans.max_points_per_centroid = 20,5,1000000000
    kmeans.train(features)
    _, cluster_assignments = kmeans.index.search(features,1)
    return cluster_assignments



"""================================================================================================="""
### Adjust Parameters for different loss classes
def adjust_pars(loss_pars, opt, ix=0, mode='class'):
    pars_to_check = ['nu', 'beta', 'beta_lr', 'beta_constant', 'embed_dim', 'margin', 'loss', 'sampling', 'num_classes', 'proxy_lr']
    ref           = [mode+'_'+x for x in pars_to_check]

    dopt, lopt = vars(opt), vars(loss_pars)

    for loss_key, class_key in zip(pars_to_check, ref):
        lopt[loss_key] = dopt[class_key]

    loss_pars.lr = opt.lr




"""============================================================================================================="""
### Generate Network Graph
def save_graph(opt, model):
    inp = torch.randn((1,3,224,224)).to(opt.device)
    network_output = model(inp)
    if isinstance(network_output, dict): network_output = network_output['Class']

    from graphviz import Digraph
    def make_dot(var, savename, params=None):
        """
        Generate a symbolic representation of the network graph.
        """
        if params is not None:
            assert all(isinstance(p, Variable) for p in params.values())
            param_map = {id(v): k for k, v in params.items()}

        node_attr = dict(style='filled',
                         shape='box',
                         align='left',
                         fontsize='6',
                         ranksep='0.1',
                         height='0.6',
                         width='1')
        dot  = Digraph(node_attr=node_attr, format='svg', graph_attr=dict(size="40,10", rankdir='LR', rank='same'))
        seen = set()

        def size_to_str(size):
            return '('+(', ').join(['%d' % v for v in size])+')'

        def add_nodes(var):
            replacements  = ['Backward', 'Th', 'Cudnn']
            color_assigns = {'Convolution':'orange',
                             'ConvolutionTranspose': 'lightblue',
                             'Add': 'red',
                             'Cat': 'green',
                             'Softmax': 'yellow',
                             'Sigmoid': 'yellow',
                             'Copys':   'yellow'}
            if var not in seen:
                op1 = torch.is_tensor(var)
                op2 = not torch.is_tensor(var) and str(type(var).__name__)!='AccumulateGrad'

                text = str(type(var).__name__)
                for rep in replacements:
                    text = text.replace(rep, '')
                color = color_assigns[text] if text in color_assigns.keys() else 'gray'

                if 'Pool' in text: color = 'lightblue'

                if op1 or op2:
                    if hasattr(var, 'next_functions'):
                        count = 0
                        for i, u in enumerate(var.next_functions):
                            if str(type(u[0]).__name__)=='AccumulateGrad':
                                if count==0: attr_text = '\nParameter Sizes:\n'
                                attr_text += size_to_str(u[0].variable.size())
                                count += 1
                                attr_text += ' '
                        if count>0: text += attr_text


                if op1:
                    dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
                if op2:
                    dot.node(str(id(var)), text, fillcolor=color)

                seen.add(var)

                if op1 or op2:
                    if hasattr(var, 'next_functions'):
                        for u in var.next_functions:
                            if u[0] is not None:
                                if str(type(u[0]).__name__)!='AccumulateGrad':
                                    dot.edge(str(id(u[0])), str(id(var)))
                                    add_nodes(u[0])
                    if hasattr(var, 'saved_tensors'):
                        for t in var.saved_tensors:
                            dot.edge(str(id(t)), str(id(var)))
                            add_nodes(t)

        add_nodes(var.grad_fn)
        dot.save(savename)
        return dot

    if not os.path.exists(opt.save_path):
        raise Exception('No save folder {} available!'.format(opt.save_path))

    viz_graph = make_dot(network_output, opt.save_path+"/Network_Graphs"+"/{}_network_graph".format(opt.arch))
    viz_graph.format = 'svg'
    viz_graph.render()

    torch.cuda.empty_cache()
    # print('Done.')
    # if view: viz_graph.view()

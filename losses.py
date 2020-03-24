"""
Main DML objectives used in the paper.
"""

"""================================================================================================="""
import warnings
warnings.filterwarnings("ignore")

import torch, random, itertools as it, numpy as np, faiss, random
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from PIL import Image

import numpy as np



"""================================================================================================="""
def loss_select(loss, opt, to_optim):
    if loss=='triplet':
        loss_params  = {'margin':opt.margin, 'sampling_method':opt.sampling}
        criterion    = TripletLoss(**loss_params)
    elif loss=='marginloss':
        loss_params  = {'margin':opt.margin, 'nu': opt.nu, 'beta':opt.beta,
                        'n_classes':opt.num_classes, 'sampling_method':opt.sampling,
                        'beta_constant':opt.beta_constant}
        criterion    = MarginLoss(**loss_params)
        to_optim    += [{'params':criterion.parameters(), 'lr':opt.beta_lr, 'weight_decay':0}]
    else:
        raise Exception('Loss {} not available!'.format(loss))
    return criterion, to_optim





"""================================================================================================="""
### Sampler() holds all possible triplet sampling options: random, SemiHardNegative & Distance-Weighted.
class Sampler():
    def __init__(self, method='random'):
        self.method = method
        if method=='semihard':
            self.give = self.semihardsampling
        elif method=='distance':
            self.give = self.distanceweightedsampling
        elif method=='random':
            self.give = self.randomsampling

    def randomsampling(self, batch, labels):
        """
        This methods finds all available triplets in a batch given by the classes provided in labels, and randomly
        selects batch.batchsize triplets.
        """
        if isinstance(labels, torch.Tensor): labels = labels.detach().numpy()
        unique_classes = np.unique(labels)
        indices        = np.arange(len(batch))
        class_dict     = {i:indices[labels==i] for i in unique_classes}

        sampled_triplets = [list(it.product([x],[x],[y for y in unique_classes if x!=y])) for x in unique_classes]
        sampled_triplets = [x for y in sampled_triplets for x in y]

        sampled_triplets = [[x for x in list(it.product(*[class_dict[j] for j in i])) if x[0]!=x[1]] for i in sampled_triplets]
        sampled_triplets = [x for y in sampled_triplets for x in y]

        #NOTE: The number of possible triplets is given by #unique_classes*(2*(samples_per_class-1)!)*(#unique_classes-1)*samples_per_class
        sampled_triplets = random.sample(sampled_triplets, batch.shape[0])
        return sampled_triplets

    def semihardsampling(self, batch, labels, gt_labels=None):
        if isinstance(labels, torch.Tensor): labels = labels.detach().numpy()
        bs = batch.size(0)
        #Return distance matrix for all elements in batch (BSxBS)
        distances = self.pdist(batch.detach()).detach().cpu().numpy()

        positives, negatives = [], []
        anchors = []
        for i in range(bs):
            l, d = labels[i], distances[i]
            anchors.append(i)
            #1 for batchelements with label l
            neg = labels!=l; pos = labels==l
            #0 for current anchor
            pos[i] = False

            #Find negatives that violate triplet constraint semi-negatives
            neg_mask = np.logical_and(neg,d<d[np.where(pos)[0]].max())
            #Find positives that violate triplet constraint semi-hardly
            pos_mask = np.logical_and(pos,d>d[np.where(neg)[0]].min())

            if pos_mask.sum()>0:
                positives.append(np.random.choice(np.where(pos_mask)[0]))
            else:
                positives.append(np.random.choice(np.where(pos)[0]))

            if neg_mask.sum()>0:
                negatives.append(np.random.choice(np.where(neg_mask)[0]))
            else:
                negatives.append(np.random.choice(np.where(neg)[0]))

        sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]
        return sampled_triplets

    def pdist(self, A, eps = 1e-4):
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min = 0)
        return res.clamp(min = eps).sqrt()

    def distanceweightedsampling(self, batch, labels, gt_labels=None, lower_cutoff=0.5, upper_cutoff=1.4):
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
        bs = batch.shape[0]

        distances    = self.pdist(batch.detach()).clamp(min=lower_cutoff)

        def inverse_sphere_distances(batch, dist, labels, anchor_label, orig_labels=None, anchor_orig_label=None):
            bs,dim       = len(dist),batch.shape[-1]

            #negated log-distribution of distances of unit sphere in dimension <dim>
            log_q_d_inv = ((2.0 - float(dim)) * torch.log(dist) - (float(dim-3) / 2) * torch.log(1.0 - 0.25 * (dist.pow(2))))
            log_q_d_inv[np.where(labels==anchor_label)[0]] = 0

            q_d_inv     = torch.exp(log_q_d_inv - torch.max(log_q_d_inv)) # - max(log) for stability
            q_d_inv[np.where(labels==anchor_label)[0]] = 0

            ### NOTE: Cutting of values with high distances made the results slightly worse.
            # q_d_inv[np.where(dist>upper_cutoff)[0]]    = 0

            q_d_inv = q_d_inv/q_d_inv.sum()
            return q_d_inv.detach().cpu().numpy()

        positives, negatives = [],[]
        labels_visited = []
        anchors = []

        for i in range(bs):
            neg = labels!=labels[i]; pos = labels==labels[i]
            q_d_inv = inverse_sphere_distances(batch, distances[i], labels, labels[i])
            #Sample positives randomly
            pos[i] = 0
            positives.append(np.random.choice(np.where(pos)[0]))
            #Sample negatives by distance
            negatives.append(np.random.choice(bs,p=q_d_inv))

        sampled_triplets = [[a,p,n] for a,p,n in zip(list(range(bs)), positives, negatives)]
        return sampled_triplets



"""================================================================================================="""
### Standard Triplet Loss, finds triplets in Mini-batches.
class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1, sampling_method='random', size_average=False):
        """
        Args:
            margin:             Triplet Margin.
            triplets_per_batch: A batch allows for multitudes of triplets to use. This gives the number
                                if triplets to sample from.
        """
        super(TripletLoss, self).__init__()
        self.margin             = margin
        self.size_average       = size_average
        self.sampler            = Sampler(method=sampling_method)

    def triplet_distance(self, anchor, positive, negative):
        return torch.nn.functional.relu((anchor-positive).pow(2).sum()-(anchor-negative).pow(2).sum()+self.margin)

    def forward(self, batch, labels, gt_labels=None):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels:  nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
            sampled_triplets: Optional: Provided pre-sampled triplets
        """
        if gt_labels is not None:
            sampled_triplets = self.sampler.give(batch, labels, gt_labels)
        else:
            sampled_triplets = self.sampler.give(batch, labels)
        loss             = torch.stack([self.triplet_distance(batch[triplet[0],:],batch[triplet[1],:],batch[triplet[2],:]) for triplet in sampled_triplets])

        if self.size_average:
            return torch.mean(loss)
        else:
            return torch.sum(loss)






"""================================================================================================="""
### MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.
class MarginLoss(torch.nn.Module):
    def __init__(self, margin=0.2, nu=0, beta=1.2, n_classes=100, beta_constant=False, sampling_method='distance'):
        """
        Args:
            margin:             Triplet Margin.
            nu:                 Regularisation Parameter for beta values if they are learned.
            beta:               Class-Margin values.
            n_classes:          Number of different classes during training.
        """
        super(MarginLoss, self).__init__()
        self.margin             = margin
        self.n_classes          = n_classes
        self.beta_constant     = beta_constant

        self.beta_val = beta
        self.beta = beta if beta_constant else torch.nn.Parameter(torch.ones(n_classes)*beta)

        self.nu                 = nu

        self.sampling_method    = sampling_method
        self.sampler            = Sampler(method=sampling_method)


    def forward(self, batch, labels, gt_labels=None):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels: nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()

        if callable(self.sampler):
            sampled_triplets = self.sampler(batch, labels, gt_labels)
        else:
            sampled_triplets = self.sampler.give(batch, labels, gt_labels)

        d_ap, d_an = [],[]
        for triplet in sampled_triplets:
            train_triplet = {'Anchor': batch[triplet[0],:], 'Positive':batch[triplet[1],:], 'Negative':batch[triplet[2]]}

            pos_dist = ((train_triplet['Anchor']-train_triplet['Positive']).pow(2).sum()+1e-8).pow(1/2)
            neg_dist = ((train_triplet['Anchor']-train_triplet['Negative']).pow(2).sum()+1e-8).pow(1/2)

            d_ap.append(pos_dist)
            d_an.append(neg_dist)
        d_ap, d_an = torch.stack(d_ap), torch.stack(d_an)

        if self.beta_constant:
            beta = self.beta
        else:
            beta = torch.stack([self.beta[labels[triplet[0]]] for triplet in sampled_triplets]).type(torch.cuda.FloatTensor)

        pos_loss = torch.nn.functional.relu(d_ap-beta+self.margin)
        neg_loss = torch.nn.functional.relu(beta-d_an+self.margin)

        pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.)).type(torch.cuda.FloatTensor)

        if pair_count == 0.:
            loss = torch.sum(pos_loss+neg_loss)
        else:
            loss = torch.sum(pos_loss+neg_loss)/pair_count

        # if self.nu: loss = loss + beta_regularisation_loss.type(torch.cuda.FloatTensor)

        return loss

"""
Dataloaders for CUB200-2011, CARS196 and Stanford Online Products.
"""

"""==================================================================================================="""
################### LIBRARIES ###################
import warnings
warnings.filterwarnings("ignore")

import numpy as np, os, sys, pandas as pd, csv, copy
import torch, torch.nn as nn, matplotlib.pyplot as plt, random

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import pretrainedmodels.utils as utils
import auxiliaries as aux




"""==================================================================================================="""
################ FUNCTION TO RETURN ALL DATALOADERS NECESSARY ####################
def give_dataloaders(dataset, opt):
    ### ImageNet Properties
    opt.mean, opt.std, opt.input_space, opt.input_range = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 'RGB', [0,1]
    if 'class_samples_per_class' in vars(opt).keys():
        opt.samples_per_class = opt.class_samples_per_class

    if opt.dataset=='cub200':
        datasets = give_CUB200_datasets(opt)
    elif opt.dataset=='cars196':
        datasets = give_CARS196_datasets(opt)
    elif opt.dataset=='online_products':
        datasets = give_OnlineProducts_datasets(opt)
    else:
        raise Exception('No Dataset >{}< available!'.format(dataset))

    dataloaders = {}
    for key,dataset in datasets.items():
        if dataset is not None:
            is_val = dataset.is_validation
            dataloaders[key] = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, num_workers=opt.kernels, shuffle=not is_val, pin_memory=True, drop_last=not is_val)

    return dataloaders




"""==================================================================================================="""
################# FUNCTIONS TO RETURN TRAIN/VAL PYTORCH DATASETS FOR CUB200, CARS196 AND STANFORD ONLINE PRODUCTS ####################################
def give_CUB200_datasets(opt):
    """
    This function generates a training and testing dataloader for Metric Learning on the CUB-200-2011 dataset.
    For Metric Learning, the dataset is sorted by name, and the first halt used for training while the last half is used for testing.
    So no random shuffling of classes.
    """
    image_sourcepath  = opt.source_path+'/images'
    image_classes     = sorted([x for x in os.listdir(image_sourcepath) if '._' not in x], key=lambda x: int(x.split('.')[0]))
    conversion        = {int(x.split('.')[0]):x.split('.')[-1] for x in image_classes}
    image_list        = {int(key.split('.')[0]):sorted([image_sourcepath+'/'+key+'/'+x for x in os.listdir(image_sourcepath+'/'+key) if '._' not in x]) for key in image_classes}
    image_list        = [[(key,img_path) for img_path in image_list[key]] for key in image_list.keys()]
    image_list        = [x for y in image_list for x in y]

    image_dict    = {}
    for key, img_path in image_list:
        key = key-1
        if not key in image_dict.keys():
            image_dict[key] = []
        image_dict[key].append(img_path)

    keys = sorted(list(image_dict.keys()))
    # random.shuffle(keys)
    #Following "Deep Metric Learning via Lifted Structured Feature Embedding", we use the first half of classes for training.
    train,test      = keys[:len(keys)//2], keys[len(keys)//2:]

    if opt.sampling=='learned':
        if opt.train_val_split_by_class:
            train_val_split = int(len(train)*opt.train_val_split)
            train, val      = train[:train_val_split], train[train_val_split:]
            train_image_dict, val_image_dict, test_image_dict = {key:image_dict[key] for key in train}, {key:image_dict[key] for key in val}, {key:image_dict[key] for key in test}
        else:
            train_image_dict, val_image_dict = {},{}

            for key in train:
                # train_ixs = np.random.choice(len(image_dict[key]), int(len(image_dict[key])*opt.train_val_split), replace=False)
                train_ixs   = np.array(list(set(np.round(np.linspace(0,len(image_dict[key])-1,int(len(image_dict[key])*opt.train_val_split)))))).astype(int)
                val_ixs     = np.array([x for x in range(len(image_dict[key])) if x not in train_ixs])
                train_image_dict[key] = np.array(image_dict[key])[train_ixs]
                val_image_dict[key]   = np.array(image_dict[key])[val_ixs]
    else:
        train_image_dict = {key:image_dict[key] for key in train}

    test_image_dict = {key:image_dict[key] for key in test}



    train_dataset = BaseTripletDataset(train_image_dict, opt, samples_per_class=opt.samples_per_class)
    test_dataset  = BaseTripletDataset(test_image_dict,  opt, is_validation=True)
    eval_dataset  = BaseTripletDataset(train_image_dict, opt, is_validation=True)
    train_dataset.conversion = conversion
    test_dataset.conversion  = conversion
    eval_dataset.conversion  = conversion

    if opt.sampling!='learned':
        return {'training':train_dataset, 'testing':test_dataset, 'evaluation':eval_dataset}
    else:
        val_dataset   = BaseTripletDataset(val_image_dict,   opt, is_validation=True)
        val_dataset.conversion   = conversion
        return {'training':train_dataset, 'validation':val_dataset, 'testing':test_dataset, 'evaluation':eval_dataset}



def give_CARS196_datasets(opt):
    """
    This function generates a training and testing dataloader for Metric Learning on the CARS-196 dataset.
    For Metric Learning, the dataset is sorted by name, and the first halt used for training while the last half is used for testing.
    So no random shuffling of classes.
    """
    image_sourcepath  = opt.source_path+'/images'
    image_classes = sorted([x for x in os.listdir(image_sourcepath)])
    conversion    = {i:x for i,x in enumerate(image_classes)}
    image_list    = {i:sorted([image_sourcepath+'/'+key+'/'+x for x in os.listdir(image_sourcepath+'/'+key)]) for i,key in enumerate(image_classes)}
    image_list    = [[(key,img_path) for img_path in image_list[key]] for key in image_list.keys()]
    image_list    = [x for y in image_list for x in y]

    image_dict    = {}
    for key, img_path in image_list:
        if not key in image_dict.keys():
            image_dict[key] = []
        image_dict[key].append(img_path)


    keys = sorted(list(image_dict.keys()))
    # random.shuffle(keys)
    #Following "Deep Metric Learning via Lifted Structured Feature Embedding", we use the first half of classes for training.
    train,test      = keys[:len(keys)//2], keys[len(keys)//2:]

    if opt.sampling=='learned':
        if opt.train_val_split_by_class:
            train_val_split = int(len(train)*opt.train_val_split)
            train, val      = train[:train_val_split], train[train_val_split:]
            train_image_dict, val_image_dict, test_image_dict = {key:image_dict[key] for key in train}, {key:image_dict[key] for key in val}, {key:image_dict[key] for key in test}
        else:
            train_image_dict, val_image_dict = {},{}
            for key in train:
                train_ixs = np.random.choice(len(image_dict[key]), int(len(image_dict[key])*opt.train_val_split), replace=False)
                val_ixs   = np.array([x for x in range(len(image_dict[key])) if x not in train_ixs])
                train_image_dict[key] = np.array(image_dict[key])[train_ixs]
                val_image_dict[key]   = np.array(image_dict[key])[val_ixs]
            test_image_dict = {key:image_dict[key] for key in test}
        val_dataset   = BaseTripletDataset(val_image_dict,   opt, is_validation=True)
        val_dataset.conversion   = conversion
    else:
        train_image_dict, test_image_dict = {key:image_dict[key] for key in train}, {key:image_dict[key] for key in test}
        val_dataset = None

    train_dataset = BaseTripletDataset(train_image_dict, opt, samples_per_class=opt.samples_per_class)
    test_dataset  = BaseTripletDataset(test_image_dict,  opt, is_validation=True)
    eval_dataset  = BaseTripletDataset(train_image_dict, opt, is_validation=True)

    train_dataset.conversion = conversion
    test_dataset.conversion  = conversion
    eval_dataset.conversion  = conversion

    return {'training':train_dataset, 'validation':val_dataset, 'testing':test_dataset, 'evaluation':eval_dataset}



def give_OnlineProducts_datasets(opt):
    image_sourcepath  = opt.source_path+'/images'
    training_files = pd.read_table(opt.source_path+'/Info_Files/Ebay_train.txt', header=0, delimiter=' ')
    test_files     = pd.read_table(opt.source_path+'/Info_Files/Ebay_test.txt', header=0, delimiter=' ')


    conversion, super_conversion = {},{}
    for class_id, path in zip(training_files['class_id'],training_files['path']):
        conversion[class_id] = path.split('/')[0]
    for super_class_id, path in zip(training_files['super_class_id'],training_files['path']):
        conversion[super_class_id] = path.split('/')[0]
    for class_id, path in zip(test_files['class_id'],test_files['path']):
        conversion[class_id] = path.split('/')[0]

    train_image_dict, test_image_dict, super_train_image_dict  = {},{},{}
    for key, img_path in zip(training_files['class_id'],training_files['path']):
        key = key-1
        if not key in train_image_dict.keys():
            train_image_dict[key] = []
        train_image_dict[key].append(image_sourcepath+'/'+img_path)

    for key, img_path in zip(test_files['class_id'],test_files['path']):
        key = key-1
        if not key in test_image_dict.keys():
            test_image_dict[key] = []
        test_image_dict[key].append(image_sourcepath+'/'+img_path)

    for key, img_path in zip(training_files['super_class_id'],training_files['path']):
        key = key-1
        if not key in super_train_image_dict.keys():
            super_train_image_dict[key] = []
        super_train_image_dict[key].append(image_sourcepath+'/'+img_path)


    train_keys  = list(train_image_dict.keys())

    # if opt.train_val_split_by_class:
    if opt.sampling=='learned':
        train_val_split = int(len(train_keys)*opt.train_val_split)
        train, val  = train_keys[:train_val_split], train_keys[train_val_split:]
        train_image_dict, val_image_dict = {key:train_image_dict[key] for key in train}, {key:train_image_dict[key] for key in val}
        val_dataset         = BaseTripletDataset(val_image_dict,   opt, is_validation=True)
        val_dataset.conversion         = conversion
    else:
        val_dataset = None
    # else:
    #     train_image_dict_temp, val_image_dict_temp = {},{}
    #     for key in train_keys:
    #         print(len(train_image_dict[key]))
    #         train_ixs = np.random.choice(len(train_image_dict[key]), int(len(train_image_dict[key])*opt.train_val_split), replace=False)
    #         val_ixs   = np.array([x for x in range(len(train_image_dict[key])) if x not in train_ixs])
    #         train_image_dict_temp[key] = np.array(image_dict[key])[train_ixs]
    #         val_image_dict_temp[key]   = np.array(image_dict[key])[val_ixs]

    super_train_dataset = BaseTripletDataset(super_train_image_dict, opt, is_validation=True)
    train_dataset       = BaseTripletDataset(train_image_dict, opt, samples_per_class=opt.samples_per_class)
    test_dataset        = BaseTripletDataset(test_image_dict,  opt, is_validation=True)
    eval_dataset        = BaseTripletDataset(train_image_dict, opt, is_validation=True)

    super_train_dataset.conversion = super_conversion
    train_dataset.conversion       = conversion
    test_dataset.conversion        = conversion
    eval_dataset.conversion        = conversion

    return {'training':train_dataset, 'validation':val_dataset, 'testing':test_dataset, 'evaluation':eval_dataset, 'super_evaluation':super_train_dataset}



"""==================================================================================================="""
################## BASIC PYTORCH DATASET USED FOR ALL DATASETS ##################################
class BaseTripletDataset(Dataset):
    def __init__(self, image_dict, opt, samples_per_class=8, is_validation=False):
        self.is_validation = is_validation

        self.pars        = opt
        self.image_dict  = image_dict

        self.samples_per_class = samples_per_class

        #####
        self.init_setup()

        ##### Option 2: Use Mean/Stds on which the networks were trained
        if 'bninception' in opt.arch:
            normalize = transforms.Normalize(mean=[0.502, 0.4588, 0.4078],std=[0.0039, 0.0039, 0.0039])
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        transf_list = []
        if not self.is_validation:
            transf_list.extend([transforms.RandomResizedCrop(size=224), transforms.RandomHorizontalFlip(0.5)])
        else:
            transf_list.extend([transforms.Resize(256), transforms.CenterCrop(224)])

        transf_list.extend([transforms.ToTensor(),
                            normalize])
        self.transform = transforms.Compose(transf_list)


    def init_setup(self):
        self.n_files     = np.sum([len(self.image_dict[key]) for key in self.image_dict.keys()])

        self.avail_classes    = sorted(list(self.image_dict.keys()))
        self.image_dict    = {i:self.image_dict[key] for i,key in enumerate(self.avail_classes)}
        self.avail_classes = sorted(list(self.image_dict.keys()))

        if not self.is_validation:
            #Select current class to sample images from up to <samples_per_class>
            self.current_class   = np.random.randint(len(self.avail_classes))
            self.classes_visited = [self.current_class, self.current_class]
            self.n_samples_drawn = 0

        # if self.is_validation or self.samples_per_class==1:
        self.image_list = [[(x,key) for x in self.image_dict[key]] for key in self.image_dict.keys()]
        self.image_list = [x for y in self.image_list for x in y]

        # self.sample_probs = np.ones(len(self.image_list))/len(self.image_list)

        self.is_init = True


    def ensure_3dim(self, img):
        if len(img.size)==2:
            img = img.convert('RGB')
        return img


    def __getitem__(self, idx):
        if self.is_init:
            self.current_class = self.avail_classes[idx%len(self.avail_classes)]
            self.is_init = False

        if not self.is_validation:

            if self.samples_per_class==1:
                return (self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0]))))

            if self.n_samples_drawn==self.samples_per_class:
                #Once enough samples per class have been drawn, we choose another class to draw samples from.
                #Note that we ensure with self.classes_visited that no class is chosen if it had been chosen
                #previously or one before that.
                counter = copy.deepcopy(self.avail_classes)
                for prev_class in self.classes_visited:
                    if prev_class in counter: counter.remove(prev_class)

                self.current_class   = counter[idx%len(counter)]
                self.classes_visited = self.classes_visited[1:]+[self.current_class]
                self.n_samples_drawn = 0


            class_sample_idx = idx%len(self.image_dict[self.current_class])
            self.n_samples_drawn += 1

            out_img = self.transform(self.ensure_3dim(Image.open(self.image_dict[self.current_class][class_sample_idx])))
            if 'bninception' in self.pars.arch:
                out_img = out_img[range(3)[::-1],:]
            return (self.current_class,out_img)
        else:
            out_img = self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0])))
            if 'bninception' in self.pars.arch:
                out_img = out_img[range(3)[::-1],:]
            return (self.image_list[idx][-1], out_img)

    def __len__(self):
        return self.n_files

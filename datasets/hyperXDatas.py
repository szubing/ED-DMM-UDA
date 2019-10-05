import torch
import torch.utils.data
import numpy as np
from scipy import io
import random
import os
from tqdm import tqdm

def get_indian_src(split, batch_size=50, exp_dict=None):
    img, gt, _, _, label_values, _ = get_dataset_crossSense('Indiana',
                                target_folder='./cross_scene_data/')
    if exp_dict['src_train_size'] != -1:
        if exp_dict['sample_already']:
            train_gt, test_gt = get_sample(exp_dict['src_dataset'], exp_dict['src_train_size'], exp_dict['run'])
        else:
            train_gt, test_gt = sample_gt(gt, exp_dict['src_train_size'], mode='fixed_withone')
            save_sample(train_gt, test_gt, exp_dict['src_dataset'], exp_dict['src_train_size'], exp_dict['run'])
        if split == 'train':
            HyperXDataset = HyperX(img, train_gt, exp_dict['src_dataset'], exp_dict['patch_size'], exp_dict['flip_aug'], exp_dict['rotation_aug'])
        elif split == 'val':
            HyperXDataset = HyperX(img, test_gt, exp_dict['src_dataset'], exp_dict['patch_size'], exp_dict['flip_aug'], exp_dict['rotation_aug'])
    else:
        HyperXDataset = HyperX(img, gt, exp_dict['src_dataset'], exp_dict['patch_size'], exp_dict['flip_aug'], exp_dict['rotation_aug'])
    
    indian_data_src_loader = torch.utils.data.DataLoader(dataset=HyperXDataset,
                               batch_size=batch_size,
                               shuffle=True)
    
    return indian_data_src_loader
    
def get_indian_tgt(split, batch_size=50, exp_dict=None):
    _, _, img, gt, label_values, _ = get_dataset_crossSense('Indiana',
                                target_folder='./cross_scene_data/')
    if exp_dict['tgt_train_size'] != -1:
        if exp_dict['sample_already']:
            train_gt, test_gt = get_sample(exp_dict['tgt_dataset'], exp_dict['tgt_train_size'], exp_dict['run'])
        else:
            train_gt, test_gt = sample_gt(gt, exp_dict['tgt_train_size'], mode='fixed_withone')
            save_sample(train_gt, test_gt, exp_dict['tgt_dataset'], exp_dict['tgt_train_size'], exp_dict['run'])
        if split == 'train':
            HyperXDataset = HyperX(img, train_gt, exp_dict['tgt_dataset'], exp_dict['patch_size'], exp_dict['flip_aug'], exp_dict['rotation_aug'])
        elif split == 'val':
            HyperXDataset = HyperX(img, gt, exp_dict['tgt_dataset'], exp_dict['patch_size'], exp_dict['flip_aug'], exp_dict['rotation_aug'])
    else:
        HyperXDataset = HyperX(img, gt, exp_dict['tgt_dataset'], exp_dict['patch_size'], exp_dict['flip_aug'], exp_dict['rotation_aug'])
    
    if split == 'train_supervised':
        if exp_dict['sample_already']:
            train_gt, test_gt = get_sample(exp_dict['tgt_dataset'], exp_dict['tgt_train_size_supervised'], exp_dict['run'])
        else:
            train_gt, test_gt = sample_gt(gt, exp_dict['tgt_train_size_supervised'], mode='fixed_withone')
            save_sample(train_gt, test_gt, exp_dict['tgt_dataset'], exp_dict['tgt_train_size_supervised'], exp_dict['run'])
        HyperXDataset = HyperX(img, train_gt, exp_dict['tgt_dataset'], exp_dict['patch_size'], exp_dict['flip_aug'], exp_dict['rotation_aug'])
    if split == 'test_supervised':
        if exp_dict['sample_already']:
            train_gt, test_gt = get_sample(exp_dict['tgt_dataset'], exp_dict['tgt_train_size_supervised'], exp_dict['run'])
        else:
            train_gt, test_gt = sample_gt(gt, exp_dict['tgt_train_size_supervised'], mode='fixed_withone')
            save_sample(train_gt, test_gt, exp_dict['tgt_dataset'], exp_dict['tgt_train_size_supervised'], exp_dict['run'])
        HyperXDataset = HyperX(img, test_gt, exp_dict['tgt_dataset'], exp_dict['patch_size'], exp_dict['flip_aug'], exp_dict['rotation_aug'])
        
    indian_data_tgt_loader = torch.utils.data.DataLoader(dataset=HyperXDataset,
                               batch_size=batch_size,
                               shuffle=True)
    
    return indian_data_tgt_loader
    
def get_pavia_src(split, batch_size=50, exp_dict=None):
    img, gt, _, _, label_values, _ = get_dataset_crossSense('Pavia',
                                target_folder='./cross_scene_data/')
    # _, _, img, gt, label_values, _ = get_dataset_crossSense('Pavia',
                                # target_folder='./cross_scene_data/')
    
    if exp_dict['src_train_size'] != -1:
        if exp_dict['sample_already']:
            train_gt, test_gt = get_sample(exp_dict['src_dataset'], exp_dict['src_train_size'], exp_dict['run'])
        else:
            train_gt, test_gt = sample_gt(gt, exp_dict['src_train_size'], mode='fixed_withone')
            save_sample(train_gt, test_gt, exp_dict['src_dataset'], exp_dict['src_train_size'], exp_dict['run'])
        if split == 'train':
            HyperXDataset = HyperX(img, train_gt, exp_dict['src_dataset'], exp_dict['patch_size'], exp_dict['flip_aug'], exp_dict['rotation_aug'])
        elif split == 'val':
            HyperXDataset = HyperX(img, test_gt, exp_dict['src_dataset'], exp_dict['patch_size'], exp_dict['flip_aug'], exp_dict['rotation_aug'])
    else:
        HyperXDataset = HyperX(img, gt, exp_dict['src_dataset'], exp_dict['patch_size'], exp_dict['flip_aug'], exp_dict['rotation_aug'])
    
    pavia_data_src_loader = torch.utils.data.DataLoader(dataset=HyperXDataset,
                               batch_size=batch_size,
                               shuffle=True)
    
    return pavia_data_src_loader
    
def get_pavia_tgt(split, batch_size=50, exp_dict=None):
    _, _, img, gt, label_values, _ = get_dataset_crossSense('Pavia',
                                target_folder='./cross_scene_data/')
    # img, gt, _, _, label_values, _ = get_dataset_crossSense('Pavia',
                                # target_folder='./cross_scene_data/')
    
    if exp_dict['tgt_train_size'] != -1:
        if exp_dict['sample_already']:
            train_gt, test_gt = get_sample(exp_dict['tgt_dataset'], exp_dict['tgt_train_size'], exp_dict['run'])
        else:
            train_gt, test_gt = sample_gt(gt, exp_dict['tgt_train_size'], mode='fixed_withone')
            save_sample(train_gt, test_gt, exp_dict['tgt_dataset'], exp_dict['tgt_train_size'], exp_dict['run'])
        if split == 'train':
            HyperXDataset = HyperX(img, train_gt, exp_dict['tgt_dataset'], exp_dict['patch_size'], exp_dict['flip_aug'], exp_dict['rotation_aug'])
        elif split == 'val':
            HyperXDataset = HyperX(img, gt, exp_dict['tgt_dataset'], exp_dict['patch_size'], exp_dict['flip_aug'], exp_dict['rotation_aug'])
    else:
        HyperXDataset = HyperX(img, gt, exp_dict['tgt_dataset'], exp_dict['patch_size'], exp_dict['flip_aug'], exp_dict['rotation_aug'])
    
    if split == 'train_supervised':
        if exp_dict['sample_already']:
            train_gt, test_gt = get_sample(exp_dict['tgt_dataset'], exp_dict['tgt_train_size_supervised'], exp_dict['run'])
        else:
            train_gt, test_gt = sample_gt(gt, exp_dict['tgt_train_size_supervised'], mode='fixed_withone')
            save_sample(train_gt, test_gt, exp_dict['tgt_dataset'], exp_dict['tgt_train_size_supervised'], exp_dict['run'])
        HyperXDataset = HyperX(img, train_gt, exp_dict['tgt_dataset'], exp_dict['patch_size'], exp_dict['flip_aug'], exp_dict['rotation_aug'])
    if split == 'test_supervised':
        if exp_dict['sample_already']:
            train_gt, test_gt = get_sample(exp_dict['tgt_dataset'], exp_dict['tgt_train_size_supervised'], exp_dict['run'])
        else:
            train_gt, test_gt = sample_gt(gt, exp_dict['tgt_train_size_supervised'], mode='fixed_withone')
            save_sample(train_gt, test_gt, exp_dict['tgt_dataset'], exp_dict['tgt_train_size_supervised'], exp_dict['run'])
        HyperXDataset = HyperX(img, test_gt, exp_dict['tgt_dataset'], exp_dict['patch_size'], exp_dict['flip_aug'], exp_dict['rotation_aug'])
    
    pavia_data_tgt_loader = torch.utils.data.DataLoader(dataset=HyperXDataset,
                               batch_size=batch_size,
                               shuffle=True)
    
    return pavia_data_tgt_loader

def get_sh_src(split, batch_size=50, exp_dict=None):
    img, gt, _, _, label_values, _ = get_dataset_crossSense('shanghai-hangzhou',
                                target_folder='./cross_scene_data/')
    if exp_dict['src_train_size'] != -1:
        if exp_dict['sample_already']:
            train_gt, test_gt = get_sample(exp_dict['src_dataset'], exp_dict['src_train_size'], exp_dict['run'])
        else:
            train_gt, test_gt = sample_gt(gt, exp_dict['src_train_size'], mode='fixed_withone')
            save_sample(train_gt, test_gt, exp_dict['src_dataset'], exp_dict['src_train_size'], exp_dict['run'])
        if split == 'train':
            HyperXDataset = HyperX(img, train_gt, exp_dict['src_dataset'], exp_dict['patch_size'], exp_dict['flip_aug'], exp_dict['rotation_aug'])
        elif split == 'val':
            HyperXDataset = HyperX(img, test_gt, exp_dict['src_dataset'], exp_dict['patch_size'], exp_dict['flip_aug'], exp_dict['rotation_aug'])
    else:
        HyperXDataset = HyperX(img, gt, exp_dict['src_dataset'], exp_dict['patch_size'], exp_dict['flip_aug'], exp_dict['rotation_aug'])
    
    sh_data_src_loader = torch.utils.data.DataLoader(dataset=HyperXDataset,
                               batch_size=batch_size,
                               shuffle=True)
    
    return sh_data_src_loader
    
def get_sh_tgt(split, batch_size=50, exp_dict=None):
    _, _, img, gt, label_values, _ = get_dataset_crossSense('shanghai-hangzhou',
                                target_folder='./cross_scene_data/')
    if exp_dict['tgt_train_size'] != -1:
        if exp_dict['sample_already']:
            train_gt, test_gt = get_sample(exp_dict['tgt_dataset'], exp_dict['tgt_train_size'], exp_dict['run'])
        else:
            train_gt, test_gt = sample_gt(gt, exp_dict['tgt_train_size'], mode='fixed_withone')
            save_sample(train_gt, test_gt, exp_dict['tgt_dataset'], exp_dict['tgt_train_size'], exp_dict['run'])
        if split == 'train':
            HyperXDataset = HyperX(img, train_gt, exp_dict['tgt_dataset'], exp_dict['patch_size'], exp_dict['flip_aug'], exp_dict['rotation_aug'])
        elif split == 'val':
            HyperXDataset = HyperX(img, gt, exp_dict['tgt_dataset'], exp_dict['patch_size'], exp_dict['flip_aug'], exp_dict['rotation_aug'])
    else:
        HyperXDataset = HyperX(img, gt, exp_dict['tgt_dataset'], exp_dict['patch_size'], exp_dict['flip_aug'], exp_dict['rotation_aug'])
    
    if split == 'train_supervised':
        if exp_dict['sample_already']:
            train_gt, test_gt = get_sample(exp_dict['tgt_dataset'], exp_dict['tgt_train_size_supervised'], exp_dict['run'])
        else:
            train_gt, test_gt = sample_gt(gt, exp_dict['tgt_train_size_supervised'], mode='fixed_withone')
            save_sample(train_gt, test_gt, exp_dict['tgt_dataset'], exp_dict['tgt_train_size_supervised'], exp_dict['run'])
        HyperXDataset = HyperX(img, train_gt, exp_dict['tgt_dataset'], exp_dict['patch_size'], exp_dict['flip_aug'], exp_dict['rotation_aug'])
    if split == 'test_supervised':
        if exp_dict['sample_already']:
            train_gt, test_gt = get_sample(exp_dict['tgt_dataset'], exp_dict['tgt_train_size_supervised'], exp_dict['run'])
        else:
            train_gt, test_gt = sample_gt(gt, exp_dict['tgt_train_size_supervised'], mode='fixed_withone')
            save_sample(train_gt, test_gt, exp_dict['tgt_dataset'], exp_dict['tgt_train_size_supervised'], exp_dict['run'])
        HyperXDataset = HyperX(img, test_gt, exp_dict['tgt_dataset'], exp_dict['patch_size'], exp_dict['flip_aug'], exp_dict['rotation_aug'])
    
    sh_data_tgt_loader = torch.utils.data.DataLoader(dataset=HyperXDataset,
                               batch_size=batch_size,
                               shuffle=True)
    return sh_data_tgt_loader
    
def get_dataset_crossSense(dataset_name, target_folder='./cross_scene_data/'):
    folder = target_folder + dataset_name + '/'
    if dataset_name == 'Indiana':
        #load the image
        cubeData = io.loadmat(folder + 'DataCube.mat')
        img_1 = cubeData['DataCube1']
        img_2 = cubeData['DataCube2']
        gt_1 = cubeData['gt1']
        gt_2 = cubeData['gt2']
        label_values = cubeData['class_name']
        color_map = cubeData['color_map']
    elif dataset_name == 'Pavia':
        #load the image
        cubeData = io.loadmat(folder + 'DataCube.mat')
        img_1 = cubeData['DataCube1']
        img_2 = cubeData['DataCube2']
        gt_1 = cubeData['gt1']
        gt_2 = cubeData['gt2']
        label_values = cubeData['class_name']
        color_map = cubeData['color_map']
    elif dataset_name == 'shanghai-hangzhou':
        #load the image
        cubeData = io.loadmat(folder + 'DataCube.mat')
        img_1 = cubeData['DataCube1']
        img_2 = cubeData['DataCube2']
        gt_1 = cubeData['gt1']
        gt_2 = cubeData['gt2']
        label_values = cubeData['class_name']
        color_map = cubeData['color_map']
    else:
        raise ValueError("{} dataset is unknown.".format(dataset_name))
    return img_1, gt_1, img_2, gt_2, label_values, color_map
#######################
def sample_gt(gt, train_size, mode='fixed_withone'):
    train_gt = np.zeros_like(gt)
    test_gt = np.copy(gt)
    if train_size < 0:
        raise('Error: train_size less than zero')
    if train_size == 0:
        return np.zeros_like(gt), gt
    train_size = int(train_size)
    
    if mode == 'fixed_withtwo':
        #print("Sampling {} with train size = {} in total".format(mode, train_size))
        indices = np.nonzero(gt)
        X = list(zip(*indices)) # x,y features
        train_indices = random.sample(X, train_size)
        index = tuple(zip(*train_indices))
        train_gt[index] = gt[index]
        test_gt[index] = 0
    
    elif mode == 'fixed_withone':
        #print("Sampling {} with train size = {} in each class".format(mode, train_size))
        train_indices = []
        for c in np.unique(gt):
            if c == 0:
                continue
            indices = np.nonzero(gt == c)
            X = list(zip(*indices)) # x,y features
            train_indices += random.sample(X, train_size)
        index = tuple(zip(*train_indices))
        train_gt[index] = gt[index]
        test_gt[index] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt
###################################### torch datasets
class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, dataset_name, patch_size=5, flip_argument=True, rotated_argument=True):
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        self.patch_size = patch_size
        self.flip_augmentation = flip_argument
        self.rotated_augmentation = rotated_argument
        self.name = dataset_name
        
        p = self.patch_size // 2
        # add padding
        if self.patch_size > 1:
            self.data = np.pad(self.data, ((p,p),(p,p),(0,0)), mode='constant')
            self.label = np.pad(self.label, p, mode='constant')
        else:
            self.flip_argument = False
            self.rotated_argument = False
        self.indices = []
        for c in np.unique(self.label):
            if c == 0:
                continue
            c_indices = np.nonzero(self.label == c)
            X = list(zip(*c_indices))
            self.indices += X
        ## shuffle the index
        np.random.shuffle(self.indices)

    def resetGt(self, gt):
        self.label = gt
        p = self.patch_size // 2
        # add padding
        if self.patch_size > 1:
            self.label = np.pad(gt, p, mode='constant')
            
        self.indices = []
        for c in np.unique(self.label):
            if c == 0:
                continue
            c_indices = np.nonzero(self.label == c)
            X = list(zip(*c_indices))
            self.indices += X
            
    @staticmethod
    def flip(*arrays):
        #horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        # if horizontal:
            # arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays
    
    # dengbin
    @staticmethod
    def rotated(*arrays):
        p = np.random.random()
        if p < 0.25:
            arrays = [np.rot90(arr) for arr in arrays]
        elif p < 0.5:
            arrays = [np.rot90(arr, 2) for arr in arrays]
        elif p < 0.75:
            arrays = [np.rot90(arr, 3) for arr in arrays]
        else:
            pass
        return arrays

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        if self.flip_augmentation and self.patch_size > 1:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
        if self.rotated_augmentation and self.patch_size > 1:
            # Perform data rotated augmentation (only on 2D patches) #dengbin 20181018
            data, label = self.rotated(data, label)
        
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)

        # Extract the center label if needed
        if self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            #data = data[:, 0, 0]
            label = label[0, 0]

        return data, label-1
#########################################
def get_sample(dataset_name, sample_size, run):
    sample_file = './trainTestSplit/' + dataset_name + '/sample' + str(sample_size) + '_run' + str(run) + '.mat'
    data = io.loadmat(sample_file)
    train_gt = data['train_gt']
    test_gt = data['test_gt']
    return train_gt, test_gt

def save_sample(train_gt, test_gt, dataset_name, sample_size, run):
    sample_dir = './trainTestSplit/' + dataset_name + '/'
    if not os.path.isdir(sample_dir):
        os.makedirs(sample_dir)
    sample_file = sample_dir + 'sample' + str(sample_size) + '_run' + str(run) + '.mat'
    io.savemat(sample_file, {'train_gt':train_gt, 'test_gt':test_gt})
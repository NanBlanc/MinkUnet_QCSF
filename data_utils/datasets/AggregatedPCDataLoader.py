import numpy as np
import warnings
import os
from torch.utils.data import Dataset
from pcd_utils.pcd_transforms import *
import MinkowskiEngine as ME
import torch
import json
import OSToolBox as ost

warnings.filterwarnings('ignore')

class AggregatedPCDataLoader(Dataset):
    def __init__(self, root,  split='train', dataset_name="SimQC", resolution=0.05, intensity_channel=False, ignore_labels=None):
        self.root = root
        self.dataset_name = dataset_name
        self.resolution = resolution
        self.intensity_channel = intensity_channel
        self.ignore_labels=ignore_labels
        
        assert (split == 'train' or split == 'validation' or split =='test')
        self.split = split
        
        #find the right name for validation folder
        split_path = self.root+"/"+self.split 
        if split=='validation':
            if os.path.isdir(self.root+"/validation"):
                split_path = self.root+"/validation"
            elif os.path.isdir(self.root+"/val"):
                split_path = self.root+"/val"
               
        self.datapath_list(split_path)
        print('The size of %s data is %d'%(split,len(self.points_datapath)))


    def datapath_list(self,split_path):
        self.points_datapath = ost.getFileBySubstr(split_path,'.ply')

    def transforms(self, points):
        if self.split == 'train':
            theta = torch.FloatTensor(1,1).uniform_(0, 2*np.pi).item()
            scale_factor = torch.FloatTensor(1,1).uniform_(0.95, 1.05).item()
            rot_mat = np.array([[np.cos(theta),
                                    -np.sin(theta), 0],
                                [np.sin(theta),
                                    np.cos(theta), 0], [0, 0, 1]])

            points[:, :3] = np.dot(points[:, :3], rot_mat) * scale_factor
            return points
        else:
            return points

    def __len__(self):
        return len(self.points_datapath)
    
    def read_ply(self, file_name):
        data = ost.read_ply(file_name)
        cloud_x = data['x']
        cloud_y = data['y']
        cloud_z = data['z']
        labels = (data['class']).astype(np.int32)
        intensity = data['intensity']/1024.1268310546875
        intensity = intensity.reshape(len(intensity), 1)
        labels = labels.reshape(len(labels), 1)
        return(np.c_[cloud_x, cloud_y, cloud_z], labels, intensity)
    
    def _get_item(self, index):
        if self.split != 'test':
            # print(self.points_datapath[index])
            points_set, labels, intensity = self.read_ply(self.points_datapath[index])            
            points_set = np.c_[points_set,intensity]
            
            # remove unlabeled points
            #REMOVING IGNORE LABELS IS NOT A GOOD IDEA
            # if self.ignore_labels is not None:
            #     for label_toignore in self.ignore_labels:
            #         unlabeled = labels[:,0] == 0
            #         points_set = np.c_[points_set,intensity]
            #         # remove unlabeled points
            #         labels = np.delete(labels, unlabeled, axis=0)
            #         points_set = np.delete(points_set, unlabeled, axis=0)
                    
            #remap labels to learning values
            points_set[:, :3] = self.transforms(points_set[:, :3])
            
            if not self.intensity_channel:
                points_set = points_set[:, :3]   
            return points_set, labels.astype(np.int32)
        else:
            points_set = self.read_ply_unlabel(self.points_datapath[index])
            return points_set

    def __getitem__(self, index):
        return self._get_item(index)

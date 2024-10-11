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
    def __init__(self, root,  split='train', dataset_name="SimQC", resolution=0.05, use_intensity=False, max_intensity=None, ignore_labels=None):
        self.root = root
        self.dataset_name = dataset_name
        self.resolution = resolution
        self.use_intensity = use_intensity
        self.max_intensity = max_intensity 
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

    def qcsf_transforms(self, points):
        # translation :
        translation_xy=np.random.uniform(-20,0,2)
        translation_xyz=np.append(translation_xy,np.random.uniform(-10,10))
        points+=translation_xyz
        
        # rotation :
        theta = np.random.uniform(0, 2*np.pi)
        rot_mat = np.array([[np.cos(theta),-np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0], 
                            [0, 0, 1]])
        points[:, :3] = np.dot(points[:, :3], rot_mat)
        
        # jittering : 
        sigma=0.05
        clip=0.2
        points += np.clip(sigma * np.random.randn(points.shape[0],points.shape[1]), -1*clip, clip)    
        return points
    
    def qcsf_intensity_transform(self, intensity_values, maximum):
        sigma=maximum/100
        clip=sigma*4
        intensity_values += np.clip(sigma * np.random.randn(intensity_values.shape[0],intensity_values.shape[1]), -1*clip, clip)
        normalized=intensity_values/maximum
        return normalized

    def __len__(self):
        return len(self.points_datapath)
    
    def read_ply(self, file_name):
        data = ost.read_ply(file_name)
        cloud_x = data['x']
        cloud_y = data['y']
        cloud_z = data['z']
        labels = (data['class']).astype(np.int32)
        intensity=[]
        if self.use_intensity :
            intensity = data['intensity']
            intensity = intensity.reshape(len(intensity), 1)
        labels = labels.reshape(len(labels), 1)
        return(np.c_[cloud_x, cloud_y, cloud_z], labels, intensity)
    
    def _get_item(self, index):
        if self.split != 'test':
            # print(self.points_datapath[index])
            points, labels, intensity = self.read_ply(self.points_datapath[index])            
            
            #transforms
            points = self.qcsf_transforms(points)
            
            if not self.use_intensity :
                return points, labels.astype(np.int32)
            
            #transform int
            intensity=self.qcsf_intensity_transform(intensity,self.max_intensity)
            points_set = np.c_[points,intensity]
            return points_set, labels.astype(np.int32)
            
            # remove unlabeled points
            #REMOVING IGNORE LABELS IS NOT A GOOD IDEA
            # if self.ignore_labels is not None:
            #     for label_toignore in self.ignore_labels:
            #         unlabeled = labels[:,0] == 0
            #         points_set = np.c_[points_set,intensity]
            #         # remove unlabeled points
            #         labels = np.delete(labels, unlabeled, axis=0)
            #         points_set = np.delete(points_set, unlabeled, axis=0)
                    
        else:
            # print(self.points_datapath[index])
            points, labels, intensity = self.read_ply(self.points_datapath[index])            
            if not self.use_intensity :
                return points, labels.astype(np.int32)
            
            #normalisation intensity
            intensity/=self.max_intensity
            points_set = np.c_[points,intensity]
            return points_set, labels.astype(np.int32),self.points_datapath[index]
        

    def __getitem__(self, index):
        return self._get_item(index)

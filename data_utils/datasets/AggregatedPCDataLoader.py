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
    def __init__(self, root,  split='train', dataset_name="SimQC", resolution=0.05, use_intensity=False, max_intensity=None, use_transform=True, ignore_labels=None, dataset_usage=None):
        self.root = root
        self.dataset_name = dataset_name
        self.resolution = resolution
        self.use_intensity = use_intensity
        self.max_intensity = max_intensity 
        self.ignore_labels=ignore_labels
        self.dataset_usage=dataset_usage
        self.use_transform=use_transform
        
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
        
        tmp_path_files=ost.getFileBySubstr(split_path,'.ply')
        if self.dataset_usage is None or self.dataset_usage==1 or self.split!= "train" :
            self.points_datapath = tmp_path_files
        else :
            print("INFO : dataset_usage detected, only using :", self.dataset_usage,"% of dataset")
            ind_full=np.load(ost.getFileBySubstr(self.root,'index_selection')[0],allow_pickle=True)
            ind_selected=ind_full[:int(ind_full.shape[0]*self.dataset_usage)]
            self.points_datapath =[tmp_path_files[i] for i in ind_selected]   

    def transform(self, points):
        ##point drop
        #random point drop
        points=ost.randomDrop(points,0.1)
        #cuboid drop
        points=ost.randomCuboidDrop(points,2,6,4,1)
        
        ##position
        # translation :
        translation_xy=np.random.uniform(-20,0,2)
        translation_xyz=np.append(translation_xy,np.random.uniform(-10,10))
        points[:,:3]+=translation_xyz
        #scene flip
        points=ost.randomFlip(points)
        # rotation :
        theta = np.random.uniform(0, 2*np.pi)
        points=ost.rotationZ(points, theta)
        # jittering : 
        points=ost.jittering(points,0.05)
        
        ##features
        #intensity augment
        if self.use_intensity:
            points=ost.featureAugmentation(points,3,self.max_intensity)
        return points

    def __len__(self):
        return len(self.points_datapath)
    
    def _get_item(self, index):
        if self.split != 'test':
            # print(self.points_datapath[index])
            cloud = ost.readPly(self.points_datapath[index])            
            
            # transforms
            if self.use_transform:
                cloud = self.transform(cloud)
            
            
            #reshape labels as (nb_po,1)
            labels=cloud[:,4].astype(np.int32) if self.use_intensity else cloud[:,4].astype(np.int32)
            labels=cloud[:,4].astype(np.int32).reshape(cloud.shape[0],1)
                           
            if not self.use_intensity :
                labels=cloud[:,4].astype(np.int32).reshape(cloud.shape[0],1)
                return cloud[:,:3], labels
            else :
                labels=cloud[:,4].astype(np.int32).reshape(cloud.shape[0],1)
                return cloud[:,:4], labels
            
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
            cloud = ost.readPly(self.points_datapath[index])            
            if not self.use_intensity :
                labels=cloud[:,4].astype(np.int32).reshape(cloud.shape[0],1)
                return cloud[:,:3], labels, self.points_datapath[index]
            else :
                #normalisation intensity
                cloud[:,3]/=self.max_intensity
                labels=cloud[:,4].astype(np.int32).reshape(cloud.shape[0],1)
                return cloud[:,:4], labels, self.points_datapath[index]
        

    def __getitem__(self, index):
        return self._get_item(index)

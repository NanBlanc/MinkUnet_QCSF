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
            
            #transforms
            cloud = self.transform(cloud)
            
            #reshape labels as (nb_po,1)
            labels=cloud[:,4].astype(np.int32) if self.use_intensity else cloud[:,4].astype(np.int32)
            labels=cloud[:,4].astype(np.int32).reshape(cloud.shape[0],1)
                           
            if not self.use_intensity :
                labels=cloud[:,3].astype(np.int32).reshape(cloud.shape[0],1)
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
                labels=cloud[:,3].astype(np.int32).reshape(cloud.shape[0],1)
                return cloud[:,:3], labels, self.points_datapath[index]
            else :
                #normalisation intensity
                cloud[:,3]/=self.max_intensity
                labels=cloud[:,4].astype(np.int32).reshape(cloud.shape[0],1)
                return cloud[:,:4], labels, self.points_datapath[index]
        

    def __getitem__(self, index):
        return self._get_item(index)

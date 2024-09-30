import numpy as np
import torch
import torchvision
from PIL import Image
from math import ceil, floor
import argparse
from numpy import inf
import MinkowskiEngine as ME
import os
from utils import *
from data_utils.datasets.AggregatedPCDataLoader import AggregatedPCDataLoader as data_loader
from data_utils.collations import numpy_to_sparse_tensor
import open3d as o3d
from tqdm import tqdm
from data_utils import data_map_SimQC
from data_utils.ioueval import iouEval
from collections import defaultdict
import OSToolBox as ost
import time



def model_pipeline_validation(model, data, args):
    eval = iouEval(n_classes=args.num_classes, ignore=0)
    for iter_n, (x_coord, x_feats, x_label, _, _) in enumerate(tqdm(data)):
        x, y = numpy_to_sparse_tensor(x_coord, x_feats, x_label)

        if 'UNet' in args.sparse_model:
            y = y[:, 0]
        else:
            y = torch.from_numpy(np.asarray(y))
            y = y[:, 0]

        h = model['model'](x)
        z = model['classifier'](h)

        y = y.cuda() if args.use_cuda else y

        # accumulate accuracy
        pred = z.max(dim=1)[1]
        eval.addBatch(pred.long().cpu().numpy(), y.long().cpu().numpy())
        
        del x,y,pred,h,z
        torch.cuda.empty_cache()
        
        
    acc = eval.getacc()
    mean_iou, class_iou = eval.getIoU()
    # return the epoch mean loss
    return acc, mean_iou, class_iou


def test_pipeline(model, test_loader, args):
    evaluator = iouEval(n_classes=args.num_classes)

    for iter_n, batch in enumerate(tqdm(test_loader)):
        
        p_coord, p_feats, p_label, fname = batch
        
        x,y = numpy_to_sparse_tensor(p_coord, p_feats , p_label)
     
        y = y[:,0]

        h = model['model'](x)
        z = model['classifier'](h)
        pred = z.max(dim=1)[1]
        
        pred_np=pred.long().cpu().numpy()
        evaluator.addBatch(pred_np, y.long().cpu().numpy())
        
        if args.save_inference:
            pred_np.tofile(args.log_dir+"/"+ost.pathLeaf(fname[0])+".bin")
        
        del x,pred,h,z
    
    miou, iou= [a.cpu().numpy() for a in evaluator.getIoU()]
    acc=evaluator.getacc().cpu().numpy()
    tp, fp, fn, conf = [a.cpu().numpy() for a in evaluator.getStats()]
    with open(args.log_dir+"/metrics.txt","w") as f :
        f.write("OA "+ str(acc)+"\n")
        f.write("mIoU "+ str(miou)+"\n")
        for metric,name in zip([iou,tp,fp,fn],["IoU","TP","FP","FN"]) :
            for i,val in enumerate(metric) :
                f.write(name+"_"+data_map_SimQC.labels[i]+" "+str(val)+"\n")
    
    np.savetxt(args.log_dir+"/confusion_matrix.txt",conf,fmt="%d")
        




def run_inference(model, args):
    dataset=data_loader(root=args.data_dir,  split=args.split, dataset_name=args.dataset_name, resolution=args.sparse_resolution, intensity_channel=args.use_intensity)

    # create the data loader for train and validation data
    collate_function = SparseCollation(args.sparse_resolution, args.split, args.num_points)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_function, shuffle=False, num_workers=0)
    test_pipeline(model, test_loader, args)
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SparseSimCLR')

    parser.add_argument('--dataset-name', type=str, default='SimQC',                                                                help='Name of dataset (default: ParisLille3D')
    parser.add_argument('--data-dir', type=str, default='/home/reza/PHD/Data/SimQC',                                                help='Path to dataset (default: /home/reza/PHD/Data/Parislille3D/fps_knn')
    parser.add_argument('--use-cuda', action='store_true', default=True,                                                            help='using cuda (default: True')
    parser.add_argument('--split', type=str, default='test',                                                                        help='dataset split (default: test)')
    parser.add_argument('--num-classes', type=int, default=4,                                                                       help='Number of classes in the dataset')
    parser.add_argument('--device-id', type=int, default=0,                                                                         help='GPU device id (default: 0')
    parser.add_argument('--feature-size', type=int, default=128,                                                                    help='Feature output size (default: 128')
    parser.add_argument('--num-points', type=int, default=80000,                                                                    help='Number of points sampled from point clouds (default: 80000')
    parser.add_argument('--sparse-resolution', type=float, default=0.05,                                                            help='Sparse tensor resolution (default: 0.01')
    parser.add_argument('--sparse-model', type=str, default='MinkUNet',                                                             help='Sparse model to be used (default: MinkUNet')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',                                                           help='input inference batch-size')
    parser.add_argument('--save-inference', action='store_true', default=True,                                               help='save the inference as point clouds (default: False')
    parser.add_argument('--use-intensity', action='store_true', default=True,                                                       help='use points intensity')
    parser.add_argument('--ignore-labels', type=str, default="5",                                                                   help='str of ignore labels sperated by commas ex : --ignore-labels="1,2,3"')
    parser.add_argument('--log-dir', type=str, default='/home/reza/PHD/Sum24/SimQC/MinkUNet/logs/inference',                        help='logging directory (default: checkpoint)')
    parser.add_argument('--checkpoint', type=str, default='/home/reza/PHD/Sum24/SimQC/MinkUNet/logs/train_1/bestepoch4_model.pt',     help='path of checkpoint to use')

    
    args = parser.parse_args()
    args.ignore_labels = args.ignore_labels.split(',')
    print("IGNORE LABELS NOT WELL CODED" )
    
    if args.save_inference:
        args.log_dir=ost.createDirIncremental(args.log_dir)
        print("Saving inference at :",args.log_dir)


    if args.use_cuda:
        dtype = torch.cuda.FloatTensor
        device = torch.device("cuda")
        torch.cuda.set_device(0)
        print('Found GPU')
    else:
        dtype = torch.FloatTensor
        device = torch.device("DID NOT FOUND ANY GPU, running on CPU")

    # set_deterministic()

    # define backbone architecture
    #get core
    minkunet = get_model(args, dtype)
    minkunet.eval() #set nn.Module in eval mode (for batchnorm and dropout for example)
    #get head
    classifier = get_classifier_head(args, dtype)
    classifier.eval() #set nn.Module in eval mode (for batchnorm and dropout for example)


    core_path=args.checkpoint
    head_path=ost.pathBranch(args.checkpoint)+"/"+ost.pathLeaf(args.checkpoint)+"_head.pt"
    if os.path.isfile(args.checkpoint) & os.path.isfile(head_path):
        print("Found core:",core_path,"\nFound head :", head_path)
    else:
        print("Can not find checkpoints at "+core_path)
        import sys
        sys.exit()    
    
    checkpoint = torch.load(core_path)
    minkunet.load_state_dict(checkpoint['model'])
    epoch_core = checkpoint['epoch']
    
    checkpoint = torch.load(head_path)
    classifier.load_state_dict(checkpoint['model'])
    epoch_head = checkpoint['epoch']
    
    print("Core of epoch",epoch_core," and head of epoch", epoch_head," loaded")
    
    model = {'model': minkunet.cuda(), 'classifier': classifier.cuda()}
    start = time.time()
    run_inference(model, args)

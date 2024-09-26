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
from data_utils.collations import numpy_to_sparse_tensor
import open3d as o3d
from tqdm import tqdm
from data_utils import data_map_KITTI360, data_map_ParisLille3D, data_map_Toronto3D
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


def model_pipeline_test(model, data, args):
    eval = iouEval(n_classes=args.num_classes, ignore=0)
    pc = []
    labels = []
    preds = []
    for iter_n, (x_coord, x_feats, inv_inds, real_pc) in enumerate(tqdm(data)):
        x = numpy_to_sparse_tensor(x_coord, x_feats)
        h = model['model'](x)
        z = model['classifier'](h)
        pred = z.max(dim=1)[1]
        pc.append(real_pc[0])
        preds.append(pred.cpu().numpy()[inv_inds[0]])
        del x,pred,h,z

    pc = np.vstack(pc)
    preds = np.hstack(preds)
    return pc, preds


def run_inference(model, args):
    data_val = data_loader(root=args.data_dir, split=args.split, dataset_name=args.dataset_name,
                                                pre_training=False, resolution=args.sparse_resolution, orig=args.orig)

    # create the data loader for train and validation data
    val_loader = torch.utils.data.DataLoader(
        data_val,
        batch_size=args.batch_size,
        collate_fn=SparseCollation(args.sparse_resolution, args.split, inf),
        shuffle=True,
    )
    if args.dataset_name == "KITTI360":
        labels = data_map_KITTI360.labels
    elif args.dataset_name == "ParisLille3D":
        labels = data_map_ParisLille3D.labels
    elif args.dataset_name == "Toronto3D":
        labels = data_map_Toronto3D.labels
    # retrieve validation loss
    pc, preds = model_pipeline_test(model, val_loader, args)
    # print(f'\nModel Acc.: {model_acc}\tModel mIoU: {model_miou}\n\n- Per Class mIoU:')
    # for class_ in range(model_class_iou.shape[0]):
        # print(f'\t{labels[class_]}: {model_class_iou[class_].item()}')
    ost.write_ply('test100_org.ply', [pc,np.int32(preds)], ['x','y','z','c'])
    # prob = np.vstack(prob)
    # print(np.nanmean(prob, axis=0))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SparseSimCLR')

    parser.add_argument('--dataset-name', type=str, default='KITTI360',
                        help='Name of dataset (default: ParisLille3D')
    parser.add_argument('--data-dir', type=str, default='/home/reza/PHD/Data/KITTI360/orig',
                        help='Path to dataset (default: /home/reza/PHD/Data/Parislille3D/fps_knn')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='using cuda (default: True')
    parser.add_argument('--split', type=str, default='test',
                        help='dataset split (default: test)')
    parser.add_argument('--num-classes', type=int, default=16,
                        help='Number of classes in the dataset')
    parser.add_argument('--device-id', type=int, default=0,
                        help='GPU device id (default: 0')
    parser.add_argument('--percentage-labels', type=float, default=1.0,
                        help='Percentage of labels used for training (default: 1.0')
    parser.add_argument('--feature-size', type=int, default=128,
                        help='Feature output size (default: 128')
    parser.add_argument('--sparse-resolution', type=float, default=0.05,
                        help='Sparse tensor resolution (default: 0.01')
    parser.add_argument('--sparse-model', type=str, default='MinkUNet',
                        help='Sparse model to be used (default: MinkUNet')
    parser.add_argument('--use-normals', action='store_true', default=False,
                        help='use points normals (default: False')
    parser.add_argument('--log-dir', type=str, default='checkpoint/fine_tune_orig',
                        help='logging directory (default: checkpoint/fine_tune)')
    parser.add_argument('--best', type=str, default='lastepoch59',
                        help='best loss or accuracy over training (default: bestloss)')
    parser.add_argument('--checkpoint', type=str, default='fine_tune_orig',
                        help='model checkpoint (default: fine_tune)')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input inference batch-size')
    parser.add_argument('--visualize-pcd', action='store_true', default=False,
                        help='visualize inference point cloud (default: False')
    parser.add_argument('--orig', action='store_true', default=True,
                        help='visualize inference point cloud (default: False')
    parser.add_argument('--inference', action='store_true', default=True,
                        help='visualize inference point cloud (default: False')



    args = parser.parse_args()

    if args.use_cuda:
        dtype = torch.cuda.FloatTensor
        device = torch.device("cuda")
        torch.cuda.set_device(0)
        print('GPU')
    else:
        dtype = torch.FloatTensor
        device = torch.device("cpu")

    set_deterministic()

    # define backbone architecture
    minkunet = get_model(args, dtype)
    minkunet.eval()

    classifier = get_classifier_head(args, dtype)
    classifier.eval()

    model_filename = f'{args.best}_model_{args.checkpoint}.pt'
    classifier_filename = f'{args.best}_model_head_{args.checkpoint}.pt'
    print(model_filename, classifier_filename)
    # load pretained weights
    if os.path.isfile(f'{args.log_dir}/{args.dataset_name}/{args.percentage_labels}/{model_filename}') and os.path.isfile(f'{args.log_dir}/{args.dataset_name}/{args.percentage_labels}/{classifier_filename}'):
       checkpoint = torch.load(f'{args.log_dir}/{args.dataset_name}/{args.percentage_labels}/{model_filename}')
       minkunet.load_state_dict(checkpoint['model'])
       epoch = checkpoint['epoch']

       checkpoint = torch.load(f'{args.log_dir}/{args.dataset_name}/{args.percentage_labels}/{classifier_filename}')
       classifier.load_state_dict(checkpoint['model'])

       print(f'Loading model: {args.checkpoint}, from epoch: {epoch}')
    else:
       print('Trained model not found!')
       import sys
       sys.exit()
    
    model = {'model': minkunet.cuda(), 'classifier': classifier.cuda()}
    start = time.time()
    run_inference(model, args)

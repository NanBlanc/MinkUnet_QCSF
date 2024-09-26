import numpy as np
import torch
import torchvision
from PIL import Image
from math import ceil, floor
from numpy import inf
import MinkowskiEngine as ME
import os
from utils import *
from data_utils.collations import numpy_to_sparse_tensor
import open3d as o3d
from tqdm import tqdm
from data_utils import data_map_SimQC
from data_utils.ioueval import iouEval


def model_pipeline(model, data, num_classes):
    eval = iouEval(n_classes=num_classes, ignore=0)
    # prob = []
    for iter_n, (x_coord, x_feats, x_label) in enumerate(tqdm(data)):
        x, y = numpy_to_sparse_tensor(x_coord, x_feats, x_label)

        y = y[:, 0]
        h = model['model'](x)
        z = model['classifier'](h)
        y = y.cuda()

        # accumulate accuracy
        pred = z.max(dim=1)[1]
        
        # prob.append(confidenc_prob(pred, z, args.num_classes))
        eval.addBatch(pred.long().cpu().numpy(), y.long().cpu().numpy())


    acc = eval.getacc()
    mean_iou, class_iou = eval.getIoU()
    # return the epoch mean loss
    return acc, mean_iou, class_iou


def run_inference(model, test_loader, dataset_name, num_classes):

    labels = data_map_SimQC.labels
    # retrieve validation loss
    model_acc, model_miou, model_class_iou = model_pipeline(model, test_loader, num_classes)
    print(f'\nModel Acc.: {model_acc}\tModel mIoU: {model_miou}\n\n- Per Class mIoU:')
    for class_ in range(model_class_iou.shape[0]):
        print(f'\t{labels[class_]}: {model_class_iou[class_].item()}')
    return model_acc, model_miou, model_class_iou
    # prob = np.vstack(prob)
    # print(np.nanmean(prob, axis=0))


def inference(dataset_name, num_classes, model, model_head, test_loader):
   
    network = {'model': model.cuda(), 'classifier': model_head.cuda()}
    model_acc, model_miou, model_class_iou = run_inference(network, test_loader, dataset_name, num_classes)
    return model_acc, model_miou, model_class_iou
    
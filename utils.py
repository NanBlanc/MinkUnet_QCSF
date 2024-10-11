import MinkowskiEngine as ME
import numpy as np
from data_utils.collations import SparseCollation
from data_utils.datasets.AggregatedPCDataLoader import AggregatedPCDataLoader as data_loader
from models.minkunet import *
from models.blocks import SegmentationClassifierHead
import torch

sparse_models = {
    'MinkUNet': MinkUNet,
}

data_class = {
    'SimQC': 4,

}

latent_features = {
    'SparseResNet14': 512,
    'SparseResNet18': 1024,
    'SparseResNet34': 2048,
    'SparseResNet50': 2048,
    'SparseResNet101': 2048,
    'MinkUNet': 96,
    'MinkUNetSMLP': 96,
    'MinkUNet14': 96,
    'MinkUNet18': 1024,
    'MinkUNet34': 2048,
    'MinkUNet50': 2048,
    'MinkUNet101': 2048,
}

def set_deterministic():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

def list_parameters(models):
    optim_params = []
    for model in models:
        optim_params += list(models[model].parameters())

    return optim_params

def get_model(args, dtype):
    return sparse_models[args.sparse_model](
        in_channels=4 if args.use_intensity else 3,
        out_channels=latent_features[args.sparse_model],
    )#.type(dtype)

def get_classifier_head(args, dtype):
    if 'UNet' in args.sparse_model:
        return SegmentationClassifierHead(
                in_channels=latent_features[args.sparse_model], out_channels=data_class[args.dataset_name]
            )#.type(dtype)
    else:
        return ClassifierHead(
                in_channels=latent_features[args.sparse_model], out_channels=data_class[args.dataset_name]
            )#.type(dtype)

def get_optimizer(optim_params, args):
    if 'UNet' in args.sparse_model:
        optimizer = torch.optim.SGD(optim_params, lr=args.lr, momentum=0.9, weight_decay=args.decay_lr)
    else:
        optimizer = torch.optim.Adam(optim_params, lr=args.lr, weight_decay=args.decay_lr)

    return optimizer

# def get_class_weights(dataset):
#     weights = list(content.values()) if dataset == 'SemanticKITTI' else list(content_indoor.values())

#     weights = torch.from_numpy(np.asarray(weights)).float()
#     if torch.cuda.is_available():
#         weights = weights.cuda()

#     return weights

def write_summary(writer, summary_id, report, epoch):
    writer.add_scalar(summary_id, report, epoch)

def get_dataset(args):
    #ajouter get test dataset
    data_train = data_loader(root=args.data_dir, split='train', dataset_name=args.dataset_name, resolution=args.sparse_resolution, use_intensity=args.use_intensity, max_intensity=args.max_intensity, ignore_labels=args.ignore_labels)
    data_test = data_loader(root=args.data_dir, split='validation', dataset_name=args.dataset_name, resolution=args.sparse_resolution, use_intensity=args.use_intensity, max_intensity=args.max_intensity, ignore_labels=args.ignore_labels)
    return data_train, data_test

def get_data_loader(data_train, data_test, args):
    collate_fn = None

    collate_fn_train = SparseCollation(args.sparse_resolution, 'train', args.num_points)
    if args.inference:
        test_split = 'test'
    else:
        test_split = 'validation'
    collate_fn_test = SparseCollation(args.sparse_resolution, test_split, args.num_points)
    train_loader = torch.utils.data.DataLoader(
        data_train,
        batch_size=args.batch_size,
        collate_fn=collate_fn_train,
        shuffle=True,
        num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        data_test,
        batch_size=args.batch_size,
        collate_fn=collate_fn_test,
        shuffle=True,
        num_workers=0
    )

    return train_loader, test_loader

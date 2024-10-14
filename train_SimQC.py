from trainer.aggregated_pc_trainer import AggregatedPCTrainer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from data_utils import data_map_SimQC as simQC
import torch
from utils import *
import argparse
from numpy import inf
import MinkowskiEngine as ME

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SparseSimCLR')
    
    #INPUT
    parser.add_argument('--data-dir', type=str, default='/home/reza/PHD/Data/SimQC_sample',                        help='Path to dataset (default: ./Datasets/SimQC')
    # parser.add_argument('--data-dir', type=str, default='/home/reza/PHD/Data/SimQC',                        help='Path to dataset (default: ./Datasets/SimQC')
    
    #OUTPUT
    parser.add_argument('--log-dir', type=str, default='/home/reza/PHD/Sum24/SimQC/MinkUNet/logs/train',  help='logging directory (default: checkpoint)')
    
    #SHOULD STAY LIKE THAT
    parser.add_argument('--dataset-name', type=str, default='SimQC',                                    help='Name of dataset (default: SimQC')
    parser.add_argument('--num_classes', type=int, default=4,                                            help='Number of classes in the dataset')
    parser.add_argument('--batch-size', type=int, default=24, metavar='N',                               help='input training batch-size')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',                                    help='number of training epochs (default: 15)')
    parser.add_argument('--lr', type=float, default=2.4e-1,                                               help='learning rate (default: 2.4e-1')
    parser.add_argument("--decay-lr", default=1e-4, action="store", type=float,                           help='Learning rate decay (default: 1e-4')
    parser.add_argument('--use-cuda', action='store_true', default=True,                                    help='using cuda (default: True')
    parser.add_argument('--device-id', type=int, default=0,                                               help='GPU device id (default: 0')
    parser.add_argument('--feature-size', type=int, default=128,                                         help='Feature output size (default: 128')
    parser.add_argument('--sparse-resolution', type=float, default=0.05,                                  help='Sparse tensor resolution (default: 0.05')
    parser.add_argument('--num-points', type=int, default=80000,                                         help='Number of points sampled from point clouds (default: 80000')
    parser.add_argument('--sparse-model', type=str, default='MinkUNet',                                   help='Sparse model to be used (default: MinkUNet')
    parser.add_argument('--linear-eval', action='store_true', default=False,                              help='Fine-tune or linear evaluation (default: False')
    parser.add_argument('--load-checkpoint', action='store_true', default=False,                          help='load checkpoint (default: True')
    parser.add_argument('--load-epoch', type=str, default='lastepoch199',                                 help='model checkpoint (default: classifier_checkpoint)')
    parser.add_argument('--accum-steps', type=int, default=1,                                            help='Number steps to accumulate gradient')
    parser.add_argument('--inference', action='store_true', default=False,                               help='visualize inference point cloud (default: False')
    parser.add_argument('--use-intensity', action='store_true', default=True,                             help='use points intensity')
    parser.add_argument('--max-intensity', type=float, default=1025,                                    help='max valued of intensity used to normalize')
    parser.add_argument('--ignore-labels', type=str, default=4,                                        help='str of ignore labels sperated by commas ex : --ignore-labels="1,2,3"')
    parser.add_argument('--nb-val-batches', type=int, default=20,                                        help='int of maximum nb of batches to run for a val')
    args = parser.parse_args()


    if args.use_cuda:
        dtype = torch.cuda.FloatTensor
        device = torch.device("cuda")
        print('GPU')
    else:
        dtype = torch.FloatTensor
        device = torch.device("cpu")
    
    #Pas sur que deterministic is good 
    # set_deterministic()

    data_train, data_test = get_dataset(args)
    train_loader, test_loader = get_data_loader(data_train, data_test, args)
    
    w=np.array(list(simQC.labels_weights.values()))
    criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(w).float())

    model = get_model(args, dtype)
    model_head = get_classifier_head(args, dtype)
    
    #IF PMULTIPLE GPUS :
    # if torch.cuda.device_count() > 1:

        # model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
        # model_head = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model_head)

        # model_agg_pc = AggregatedPCTrainer(model, model_head, criterion, train_loader, test_loader, args)
        # trainer = Trainer(gpus=-1, accelerator='ddp', check_val_every_n_epoch=args.epochs, max_epochs=args.epochs, accumulate_grad_batches=args.accum_steps)
        # trainer.fit(model_agg_pc)
    
    #IF ONLY ONE GPU
    # else:
    
    model_agg_pc = AggregatedPCTrainer(model, model_head, criterion, train_loader, test_loader, args)
    trainer = Trainer(gpus=[0], max_epochs=args.epochs, accumulate_grad_batches=args.accum_steps,num_sanity_val_steps=0,limit_val_batches=args.nb_val_batches,logger=False)
    trainer.fit(model_agg_pc)

    #INFERENCE
    #model_acc, model_miou, model_class_iou = inference(self.params.dataset_name, self.params.num_classes, self.model, self.model_head, self.val_loader)

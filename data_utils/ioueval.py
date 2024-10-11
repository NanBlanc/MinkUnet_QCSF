import sys
import torch
import numpy as np

class iouEval:
  def __init__(self, n_classes, ignore=None):
    # classes
    self.n_classes = n_classes
    self.accum_loss = []
    
    # What to include and ignore from the means
    if ignore != None:
        self.ignore = torch.tensor(ignore).long()
        self.include = torch.tensor(
            [n for n in range(self.n_classes) if n not in self.ignore]).long()
    else:
        self.ignore = ignore
        self.include =  torch.tensor(
            [n for n in range(self.n_classes)]).long()
    print("[IOU EVAL] IGNORE: ", self.ignore)
    print("[IOU EVAL] INCLUDE: ", self.include)

    # get device
    self.device = torch.device('cpu')
    if torch.cuda.is_available():
      self.device = torch.device('cuda')

    # reset the class counters
    self.reset()
    
  def num_classes(self):
    return self.n_classes

  def delete(arr: torch.Tensor, ind: int, dim: int) -> torch.Tensor:
    skip = [i for i in range(arr.size(dim)) if i != ind]
    indices = [slice(None) if i != dim else skip for i in range(arr.ndim)]
    return arr.__getitem__(indices)
  
  def reset(self):
    self.accum_loss = []
    self.conf_matrix = torch.zeros(
        (self.n_classes, self.n_classes), device=self.device).long()

  def addLoss(self, batch_loss):
      self.accum_loss.append(batch_loss)

  def addBatch(self, x, y):  # x=preds, y=targets
    # to tensor
    x_row = torch.from_numpy(x).to(self.device).long()
    y_row = torch.from_numpy(y).to(self.device).long()

    # sizes should be matching
    x_row = x_row.reshape(-1)  # de-batchify
    y_row = y_row.reshape(-1)  # de-batchify

    # check
    assert(x_row.shape == x_row.shape)

    # idxs are labels and predictions
    idxs = torch.stack([x_row, y_row], dim=0)

    # ones is what I want to add to conf when I
    ones = torch.ones((idxs.shape[-1]), device=self.device).long()

    # make confusion matrix (cols = gt, rows = pred)
    self.conf_matrix = self.conf_matrix.index_put_(
        tuple(idxs), ones, accumulate=True)

  def getStats(self):
    # remove fp from confusion on the ignore classes cols
    conf = self.conf_matrix.clone().double()

    if self.ignore != None:
        row_exclude=self.ignore
        conf = torch.cat((conf[:row_exclude],conf[row_exclude+1:]))
        column_exclude=self.ignore
        conf = torch.cat((conf[:,:column_exclude],conf[:,column_exclude+1:]),dim=1)

    
    # get the clean stats
    tp = conf.diag()
    fp = conf.sum(dim=1) - tp
    fn = conf.sum(dim=0) - tp
    return tp, fp, fn, conf

  def getIoU(self):
    tp, fp, fn, _ = self.getStats()
    intersection = tp
    union = tp + fp + fn + 1e-15
    iou = intersection / union
    iou_mean = (intersection[self.include] / union[self.include]).mean()
    return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

  def getacc(self):
    tp, fp, fn, _ = self.getStats()
    total_tp = tp.sum()
    total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
    acc_mean = total_tp / total
    return acc_mean  # returns "acc mean"

  def getloss(self):
      return np.mean(self.accum_loss)
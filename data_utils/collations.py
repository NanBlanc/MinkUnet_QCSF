import numpy as np
import MinkowskiEngine as ME
import torch

def array_to_sequence(batch_data):
        return [ row for row in batch_data ]

def array_to_torch_sequence(batch_data):
    return [ torch.from_numpy(row).float() for row in batch_data ]

def numpy_to_sparse_tensor(p_coord, p_feats, p_label=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_coord = ME.utils.batched_coordinates(array_to_sequence(p_coord), dtype=torch.float32)
    p_feats = ME.utils.batched_coordinates(array_to_torch_sequence(p_feats), dtype=torch.float32)[:, 1:]

    if p_label is not None:
        p_label = ME.utils.batched_coordinates(array_to_torch_sequence(p_label), dtype=torch.float32)[:, 1:]
    
        return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            ), p_label.cuda()

    return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            )

def point_set_to_coord_feats(point_set, labels, resolution, num_points, deterministic=False):
    p_feats = point_set.copy()
    p_coord = np.round(point_set[:, :3] / resolution)
    p_coord -= p_coord.min(0, keepdims=1)

    _, mapping = ME.utils.sparse_quantize(coordinates=p_coord, return_index=True)
    if len(mapping) > num_points:
        if deterministic:
            # for reproducibility we set the seed
            np.random.seed(42)
        mapping = np.random.choice(mapping, num_points, replace=False)

    return p_coord[mapping], p_feats[mapping], labels[mapping]


def collate_points_to_sparse_tensor(pi_coord, pi_feats, pj_coord, pj_feats):
    # voxelize on a sparse tensor
    points_i = numpy_to_sparse_tensor(pi_coord, pi_feats)
    points_j = numpy_to_sparse_tensor(pj_coord, pj_feats)

    return points_i, points_j



class SparseCollation:
    def __init__(self, resolution, split, num_points=80000):
        self.resolution = resolution
        self.num_points = num_points
        self.split = split

    def __call__(self, list_data):
        #Si test
        if self.split != "test":
            
            points_set, labels = list(zip(*list_data))
    
            points_set = np.asarray(points_set)
            labels = np.asarray(labels)

            p_feats = []
            p_coord = []
            p_label = []

            for points, label in zip(points_set, labels):
                coord, feats, label_, = point_set_to_coord_feats(points, label, self.resolution, self.num_points, True)
                p_feats.append(feats)
                p_coord.append(coord)
                p_label.append(label_)
    
            p_feats = np.asarray(p_feats)
            p_coord = np.asarray(p_coord)
            p_label = np.asarray(p_label)

    
            return p_coord, p_feats, p_label
        
        #si validation
        else:
            points_set = list(zip(*list_data))
            points_set = np.asarray(points_set)

            p_feats = []
            p_coord = []

            for points in points_set:
                coord, feats = point_set_to_coord_feats_inv(points, None, self.resolution, self.num_points, True)
                p_feats.append(feats)
                p_coord.append(coord)

    
            p_feats = np.asarray(p_feats)
            p_coord = np.asarray(p_coord)
    
    
            return p_coord, p_feats

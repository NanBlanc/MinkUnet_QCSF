import OSToolBox as ost
import numpy as np
import os
import numpy as np
from scipy.spatial import cKDTree
from data_utils import data_map_KITTI360

def read_ply(file_name):
    data = ost.read_ply(file_name)
    cloud_x = data['x']
    cloud_y = data['y']
    cloud_z = data['z']
    cluster = (data['c']).astype(np.int32)
    # UTM_OFFSET = [627285, 4841948, 0]
    # cloud_x = cloud_x - UTM_OFFSET[0]
    # cloud_y = cloud_y - UTM_OFFSET[1]
    # cloud_z = cloud_z - UTM_OFFSET[2]
    return(np.c_[cloud_x, cloud_y, cloud_z], cluster)


def read_ply_unlabeled(file_name):
    data = ost.read_ply(file_name)
    cloud_x = data['x']
    cloud_y = data['y']
    cloud_z = data['z']  
    UTM_OFFSET = [627285, 4841948, 0]
    cloud_x = cloud_x - UTM_OFFSET[0]
    cloud_y = cloud_y - UTM_OFFSET[1]
    cloud_z = cloud_z - UTM_OFFSET[2]

    return(np.c_[cloud_x, cloud_y, cloud_z])

points_datapath = []
seq_ids = [ '02' ]
split = "validation"
root = "/home/reza/PHD/Data/Toronto3D/orig"
for seq in seq_ids:
    point_seq_path = os.path.join(root, split, 'sequences', seq)
    point_seq_bin = os.listdir(point_seq_path)
    point_seq_bin.sort()
    points_datapath += [ os.path.join(point_seq_path, point_file) for point_file in point_seq_bin ]


pc = np.c_[read_ply("Toronto_Val_01_finetune.ply")]
# pc1 = read_ply_unlabel('/home/reza/PHD/Data/Parislille3D/orig_others/test/sequences/05/ajaccio_2.ply')
# pc2 = read_ply_unlabel('/home/reza/PHD/Data/Parislille3D/orig_others/test/sequences/06/ajaccio_57.ply')
# pc3 = read_ply_unlabel('/home/reza/PHD/Data/Parislille3D/orig_others/test/sequences/07/dijon_9.ply')
# xyz_new = np.c_[xyz,np.zeros((xyz.shape[0],1))]



# def assign_labels(pc, points_datapath):
    # Build KD-trees for fast nearest neighbor search
pc_tree = cKDTree(pc[:, :3])
# learning_map_inv = data_map_KITTI360.learning_map_inv
for f, file_name in enumerate(points_datapath):
    print(f)
    pc_q = read_ply_unlabeled(file_name)
    # Find the indices of the nearest neighbors in PC for each point in pc1, pc2, and pc3
    _, pc_q_indices = pc_tree.query(pc_q, k=1)
    # _, pc2_indices = pc_tree.query(pc2_coords, k=1)
    # _, pc3_indices = pc_tree.query(pc3_coords, k=1)

    # Extract labels from PC for the corresponding indices
    pc_q_labels = pc[pc_q_indices, 3]
    # pc2_labels = pc[pc2_indices, 3]
    # pc3_labels = pc[pc3_indices, 3]
    # pc_q_labels = np.vectorize(learning_map_inv.get)(pc_q_labels)
    # np.save(label_filename,pc_q_labels)

    # Add the labels to the original pc1, pc2, and pc3
    pc_q = np.c_[pc_q, pc_q_labels]
    # labeled_pc2 = np.column_stack((pc2_coords, pc2_labels))
    # labeled_pc3 = np.column_stack((pc3_coords, pc3_labels))
    ost.write_ply(file_name[:-4] + '_labeled.ply', pc_q, ['x','y','z','c'])
    # return labeled_pc1, labeled_pc2, labeled_pc3


# labels_datapath = []
# seq_ids = ["08","18"]
# split = "test"
# root = "/home/reza/PHD/Data/KITTI360/fps_knn"
# for seq in seq_ids:
#     point_seq_path = os.path.join(root, split, 'sequences_labeled', seq)
#     point_seq_bin = os.listdir(point_seq_path)
#     point_seq_bin.sort()
#     labels_datapath += [ os.path.join(point_seq_path, point_file) for point_file in point_seq_bin ]
# path = "/home/reza/PHD/Data/KITTI360/fps_knn/test/preds/"
# for f, file_name in enumerate(labels_datapath):
#     _, labels = read_ply(file_name)
#     save_name = "{0:0=4d}".format(int(file_name.split('/')[-2])) + '_' + file_name.split('/')[-1][:-12] + '.npy'
#     np.save(path+save_name, labels.astype(np.uint8))

# labels1 = np.load("/home/reza/PHD/Data/KITTI360/fps_knn/test/labels/0008_0000000002_0000000245.npy")
# labeled_pc1, labeled_pc2, labeled_pc3 = assign_labels(pc, pc1, pc2, pc3)

# ost.write_ply('ajaccio_2_label_100_org.ply', labeled_pc1, ['x','y','z','c'])
# ost.write_ply('ajaccio_57_label_100_org.ply', labeled_pc2, ['x','y','z','c'])
# ost.write_ply('dijon_9_label_100_org.ply', labeled_pc3, ['x','y','z','c'])


# import OSToolBox as ost
# import numpy as np
# import os


# def read_label(file_name):
#     data = ost.read_ply(file_name)
#     cluster = (data['c']).astype(np.int32)
#     return cluster


# ajaccio_2_label_1 = read_label('/home/reza/PHD/Data/Parislille3D/orig_others/test/labels/ajaccio_2_label_01_org.ply')
# ajaccio_57_label_1 = read_label('/home/reza/PHD/Data/Parislille3D/orig_others/test/labels/ajaccio_57_label_01_org.ply')
# dijon_9_label_1 = read_label('/home/reza/PHD/Data/Parislille3D/orig_others/test/labels/dijon_9_label_01_org.ply')


# output_file = "dijon_9.txt"
# np.savetxt(output_file, dijon_9_label_1, fmt='%d', delimiter='\n')
    
# output_file = "ajaccio_57.txt"
# np.savetxt(output_file, ajaccio_57_label_1, fmt='%d', delimiter='\n')

# output_file = "ajaccio_2.txt"
# np.savetxt(output_file, ajaccio_2_label_1, fmt='%d', delimiter='\n')
    
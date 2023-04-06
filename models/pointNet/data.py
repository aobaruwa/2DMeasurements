from torch.utils.data.dataloader import Dataset
import numpy as np
import torch

class PcDataset(Dataset):
    def __init__(self, infile, num_points):
        """ 
        pc_file: numpy array of point_clouds and their landmarks: each data point is a  
                 concatenation of [pc_array, landmark_array, boolean_masks]
        """
        self.pc_data = infile
        self.num_points = num_points
        with np.load(infile, allow_pickle=True) as pc_data:
            self.xs = torch.from_numpy(pc_data['xs'])
            self.ys = torch.from_numpy(pc_data['ys'])
            self.masks = torch.from_numpy(pc_data['masks'])

    def __getitem__(self, idx):
        norm_data_point = self.xs[idx]
        return norm_data_point.squeeze()[:self.num_points], self.ys[idx].squeeze(), self.masks[idx].squeeze()

    def __len__(self):
        return len(self.ys)

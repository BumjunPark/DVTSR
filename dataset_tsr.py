import torch.utils.data as data
import torch
import h5py


class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        self.data0_path = file_path[0]
        self.data1_path = file_path[1]
        self.label_path = file_path[2]

    def __getitem__(self, index):
        d0 = h5py.File(self.data0_path, 'r')
        d1 = h5py.File(self.data1_path, 'r')
        l = h5py.File(self.label_path, 'r')
        self.data0 = d0.get("data")
        self.data1 = d1.get("data")
        self.label = l.get("data")

        return torch.from_numpy(self.data0[index, :, :, :]).float(), torch.from_numpy(
            self.data1[index, :, :, :]).float(), torch.from_numpy(
            self.label[index, :, :, :]).float()

    def __len__(self):
        hf = h5py.File(self.data0_path, 'r')
        temp = hf.get("data")
        return temp.shape[0]

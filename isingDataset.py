from torch.utils.data import Dataset


class IsingDataset(Dataset):

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        return x

    def __len__(self):
        return len(self.data)

    # data = np.load('ising/cfg_x016_b0100.npy')
    #
    # def __init__(self, ising_dir, transform=None):
    #     self.ising_dir = ising_dir
    #     self.transform = transform
    #
    #     all_data = os.listdir(ising_dir)
    #     self.total_ising_data = all_data
    #
    # def __len__(self):
    #     return len(self.data)
    #
    # def __getitem__(self, idx):
    #     # data_loc = os.path.join(self.ising_dir, self.total_ising_data[idx])
    #
    #     ising_data = self.data[idx]
    #     tensor_data = self.transform(ising_data)
    #     return tensor_data

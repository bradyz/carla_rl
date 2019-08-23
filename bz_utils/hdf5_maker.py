"""
Turns a list of dicts into an hdf5 file, like Pandas to_csv().
All items must be numpy ndarrays.
"""
import h5py

from pathlib import Path

from torch.utils.data import Dataset


def save_hdf5(dataset_path, data_list, dtypes):
    data = {key: list() for key in dtypes.keys()}

    for row in data_list:
        for key in data:
            data[key].append(row[key])

    with h5py.File(dataset_path, mode='w') as h:
        for key, val in data.items():
            chunks = tuple([1] + list(val[0].shape))

            h.create_dataset(key, data=val, dtype=dtypes[key], chunks=chunks)


class HDF5Dataset(Dataset):
    def __init__(self, dataset_dir):
        super().__init__()

        self.root_dir = dataset_dir
        self.data_map = dict()
        self.index_map = dict()
        self.keys = None

        for file_path in Path(dataset_dir).glob('**/*.hdf5'):
            data = h5py.File(str(file_path), 'r')
            keys = list(sorted((data.keys())))
            index = len(self.index_map)

            if self.keys is None:
                self.keys = keys
            else:
                assert keys == self.keys

            for i in range(len(data[keys[0]])):
                self.data_map[index + i] = data
                self.index_map[index + i] = i

    def __len__(self):
        return len(file_path)

    def __getitem__(self, index):
        i = self.index_map[index]
        data = self.data_map[index]

        return {k: data[k][i] for k in self.keys}

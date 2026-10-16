import torch
import numpy as np
import copy
import h5py
import os

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, split):

        super().__init__()
        self.split = split

        self.data = {sample['id']: sample for sample in self.split.data}
        self.ids = list(self.data.keys())
        self.num_folds = self.split.num_folds
        self.current_iter = None

    def get_subset(self, iteration, mode):


        subset = self.copy()
        subset.current_iter = iteration

        subset.data = {sample['id']: sample for sample in self.split.data if sample['folds'][iteration] == mode}
        subset.ids = list(subset.data.keys())


        if hasattr(subset, 'child_datasets'):
            for dataset_name, dataset in subset.child_datasets.items():
                assert not hasattr(dataset, 'child_datasets'), f'BaseDataset.get_subset() does not support multiple levels of child datasets. Dataset name: {dataset_name}'
                subset.child_datasets[dataset_name] = dataset.get_subset(iteration, mode)


        if len(subset.ids) == 0:
            return None

        return subset

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        raise NotImplementedError

    @staticmethod
    def _sample(bag, sample_n, pad, sample_indices):

        if sample_n is None:
            sample_n = len(bag)

        if len(bag) < sample_n:
            if pad:

                mask = torch.cat([torch.ones(len(bag)), torch.zeros(sample_n - len(bag))])

                if isinstance(bag, torch.Tensor):
                    padding = torch.zeros((sample_n - len(bag), ) + bag.shape[1:])
                    baglet = torch.cat([bag, padding], dim=0)
                elif isinstance(bag, list):
                    padding = [None] * (sample_n - len(bag))
                    baglet = bag + padding
                elif isinstance(bag, np.ndarray):
                    padding = np.zeros((sample_n - len(bag), ) + bag.shape[1:])
                    baglet = np.concatenate([bag, padding], axis=0)
                else:
                    raise NotImplementedError(f'Bag type {type(bag)} not implemented')
            else:
                mask = torch.ones(len(bag))
                baglet = bag
        elif len(bag) >= sample_n:
            mask = torch.ones(sample_n)
            if sample_indices is None:
                sample_indices = torch.randperm(len(bag))[:sample_n].tolist()
            baglet = bag[sample_indices]

            if isinstance(bag, torch.Tensor):
                baglet = torch.stack(baglet, dim=0)
            elif isinstance(bag, list):
                pass
            elif isinstance(bag, np.ndarray):
                baglet = np.stack(baglet, axis=0)

        assert isinstance(baglet, type(bag)), f'Baglet type {type(baglet)} does not match bag type {type(bag)}'

        mask = mask.bool()

        return baglet, mask, sample_indices

    def collate_fn(self, batch):

        collated = {}

        for key in batch[0].keys():
            collated[key] = self.recursive_collate([d[key] for d in batch])


        collated['current_iter'] = self.current_iter

        return collated

    def recursive_collate(self, items):

        try:

            if isinstance(items[0], torch.Tensor):
                return torch.stack(items, dim=0)
            elif isinstance(items[0], np.ndarray):
                return np.stack(items, axis=0)
            elif isinstance(items[0], dict):
                keys = items[0].keys()
                return {key: self.recursive_collate([d[key] for d in items]) for key in keys}
            elif isinstance(items[0], list):
                return [self.recursive_collate([d[i] for d in items]) for i in range(len(items[0]))]
            else:
                return items
        except Exception as e:
            print(f'Recursive collation failed with error: {e}')
            print(f'Attempted to collate items with shapes: {[item.shape if isinstance(item, torch.Tensor) or isinstance(item, np.ndarray) else item for item in items]}')

    @staticmethod
    def load_h5(load_path, keys = None):

        assert isinstance(keys, list) or keys is None, 'keys must be a list or None'
        assert os.path.exists(load_path), f'File {load_path} does not exist'

        try:
            with h5py.File(load_path, 'r') as file:
                if keys is None:
                    keys = list(file.keys())

                assets = {key: file[key][:] for key in keys}
                attributes = {key: dict(file[key].attrs) for key in keys}
        except Exception as e:
            raise Exception(f'\033[91mError loading h5 file at {load_path}\033[0m')

        return assets, attributes

    def copy(self):

        return copy.deepcopy(self)

    def get_datasampler(self, sampler = 'random'):

        if sampler == 'sequential':
            return torch.utils.data.SequentialSampler(self)
        elif sampler == 'random':
            return torch.utils.data.RandomSampler(self)
        elif isinstance(sampler, dict) and sampler['type'] == 'weighted':
            raise NotImplementedError(f'WeightedRandomSampler not implemented yet.')
        else:
            raise NotImplementedError(f'Sampler type {sampler} not implemented')

    def get_dataloader(self, current_iter, mode, batch_size = None):


        subset_dataset = self.get_subset(current_iter, mode)
        if subset_dataset is None:
            return None



        return torch.utils.data.DataLoader(subset_dataset,
                                            batch_size = len(subset_dataset) if batch_size is None else batch_size,
                                            sampler = subset_dataset.get_datasampler('random'),
                                            num_workers = 1,
                                            collate_fn = subset_dataset.collate_fn)
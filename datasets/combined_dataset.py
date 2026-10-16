from datasets.base_dataset import BaseDataset


class CombinedDataset(BaseDataset):
    def __init__(self, child_datasets)  :

        super().__init__(list(child_datasets.values())[0].split)


        for dataset_name, dataset in child_datasets.items():
            assert set(dataset.ids) == set(self.ids), f'All child datasets must have the same IDs. Dataset {dataset_name} has different IDs.'
        self.child_datasets = child_datasets

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        if isinstance(idx, (list, tuple)):
            raise NotImplementedError(
                "CombinedDataset does not support list-style index. "
                "Use DataLoader + collate_fn instead."
            )


        if hasattr(idx, "item"):
            idx = idx.item()

        sample_id = self.ids[idx]


        sample_batch = {
            "id": sample_id,
            "modalities": {}
        }


        for name, dataset in self.child_datasets.items():
            data = dataset[idx]
            sample_batch["modalities"][name] = data


            assert data["id"] == sample_id, f"Dataset {name} returned mismatched ID at idx {idx}: {data['id']} != {sample_id}"

        return sample_batch
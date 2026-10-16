import torch
from datasets.base_dataset import BaseDataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from datasets.transform_pipelines import TransformPipeline

class OmicsDataset(BaseDataset):
    def __init__(self, split,
                 omics_embeddings_dirs,
                 feature_transforms: TransformPipeline = None):


        super().__init__(split)


        self.omics_df = pd.read_csv(omics_embeddings_dirs, index_col=0)

        self.feature_transforms = feature_transforms or TransformPipeline()


        missing = [i for i in self.ids if i not in self.omics_df.index]

        assert len(missing) == 0, f"Missing omics for IDs: {missing}"

    def __getitem__(self, idx):


        sample_id = self.ids[idx]

        x = self.omics_df.loc[sample_id].values.astype(np.float32)


        x = self.feature_transforms(x)


        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)


        return {
            'id': sample_id,
            'features': x
        }

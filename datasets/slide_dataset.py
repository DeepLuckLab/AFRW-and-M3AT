from datasets.base_dataset import BaseDataset
import torch
import os
import numpy as np
from IPython.display import display
from datasets.transform_pipelines import TransformPipeline

class SlideDataset(BaseDataset):
    def __init__(self,
                 split,
                 load_from,
                 bag_size = None,
                 pad = False,
                 slide_merge_mode="list",
                 feature_transforms: TransformPipeline = None,
                 coord_transforms: TransformPipeline = None):


        super().__init__(split)


        assert slide_merge_mode in ["concat", "list"]
        self.slide_merge_mode = slide_merge_mode


        self.load_from = load_from if isinstance(load_from, list) else [load_from]


        self.bag_size = bag_size
        self.pad = pad

        self.feature_transforms = feature_transforms or TransformPipeline()
        self.coord_transforms = coord_transforms or TransformPipeline()


        self.slide_paths = self._scan_slide_paths(self.load_from)


    @staticmethod
    def _scan_slide_paths(dirs):
        paths = {}
        for root in dirs:

            if not os.path.exists(root):
                print(f"[WARN] Dataset path not found: {root}")
                continue


            for f in os.listdir(root):
                if f.endswith(".h5"):
                    slide_id = os.path.splitext(f)[0]
                    paths[slide_id] = os.path.join(root, f)

        return paths


    def _load_slide_h5(self, slide_id):

        if slide_id not in self.slide_paths:
            raise FileNotFoundError(f"Slide {slide_id} not found in {self.load_from}.")

        path = self.slide_paths[slide_id]

        data, attrs = self.load_h5(path, keys=["features", "coords"])
        patch_size = None

        if "patch_size_level0" in attrs["coords"]:
            patch_size = attrs["coords"]["patch_size_level0"]

        return {
            "features": data["features"],
            "coords": data["coords"],
            "patch_size_level0": patch_size
        }


    def _load_and_merge_slides(self, slide_ids):

        slide_assets = []


        for sid in slide_ids:
            data = self._load_slide_h5(sid)
            slide_assets.append(data)


        if self.slide_merge_mode == "concat":

            features = [np.concatenate([a["features"] for a in slide_assets], axis=0)]
            coords = [np.concatenate([a["coords"] for a in slide_assets], axis=0)]
            attributes = [{"patch_size_level0": a["patch_size_level0"]} for a in slide_assets]

            return {
                "features": features,
                "coords": coords,
                "attributes": attributes
            }

        elif self.slide_merge_mode == "list":

            features_list = [a["features"] for a in slide_assets]
            coords_list = [a["coords"] for a in slide_assets]
            attributes_list = [{"patch_size_level0": a["patch_size_level0"]} for a in slide_assets]

            return {
                "features": features_list,
                "coords": coords_list,
                "attributes": attributes_list
            }

        else:

            raise ValueError(f"Unknown slide_merge_mode: {self.slide_merge_mode}")


    def _apply_transforms(self, assets):

        assets["features"] = self.feature_transforms(assets["features"])

        assets["coords"]   = self.coord_transforms(assets["coords"])
        return assets


    def _apply_bag_sampling(self, assets):


        sampled_features = []
        sampled_coords = []
        sampled_masks = []


        if self.bag_size is None:
            for feats in assets["features"]:
                sampled_features.append(feats)
                sampled_masks.append(np.ones(len(feats), dtype=bool))
            assets["mask"] = sampled_masks
            return assets


        for feats, coords in zip(assets["features"], assets["coords"]):


            baglet, mask, sample_indices = self._sample(
                bag=feats,
                sample_n=self.bag_size,
                pad=self.pad,
                sample_indices=None
            )


            sampled_features.append(baglet)
            sampled_masks.append(mask)


            if coords is not None:
                sampled_coords.append(coords[sample_indices])
            else:
                sampled_coords.append(None)


        assets["features"] = sampled_features
        assets["coords"] = sampled_coords
        assets["mask"] = sampled_masks
        return assets


    def _assemble_output(self, sample_id, slide_ids, assets):


        out = {"id": sample_id,
               "paths": [self.slide_paths[s] for s in slide_ids],
               "attributes": assets["attributes"],

               "features": [torch.from_numpy(x).float() for x in assets["features"]],
               "coords": [torch.from_numpy(x).float() for x in assets["coords"]]}


        if "mask" in assets:
            out["mask"] = [m for m in assets["mask"]]
        return out


    def __getitem__(self, idx):


        sample_id = self.ids[idx] if isinstance(idx, int) else idx


        slide_ids = self.data[sample_id]['slide_id']


        if not slide_ids:
            display(self.data)
            raise RuntimeError(f"No slides found for case ID {sample_id} in split with {len(self.data)} samples.")


        assets = self._load_and_merge_slides(slide_ids)

        assets = self._apply_transforms(assets)

        assets = self._apply_bag_sampling(assets)


        return self._assemble_output(sample_id, slide_ids, assets)
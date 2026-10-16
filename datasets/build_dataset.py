from datasets.slide_dataset import SlideDataset
from datasets.omics_dataset import OmicsDataset
from datasets.label_dataset import LabelDataset
from datasets.combined_dataset import CombinedDataset


def build_multimodal_embeddings(split, task_name,
                               slide_embeddings_dirs,
                               slide_merge_mode,
                               omics_embeddings_dirs,
                               bag_size, modalities):

    children = {}

    if "slide" in modalities:
        children["slide"] = slide_embeddings_dataset(split=split,
                                                     slide_embeddings_dirs=slide_embeddings_dirs,
                                                     slide_merge_mode=slide_merge_mode,
                                                     bag_size=bag_size)

    if "omics" in modalities:
        children["omics"] = omics_dataset(
            split,
            omics_embeddings_dirs
        )

    children["labels"] = labels_dataset(split, task_name)

    return CombinedDataset(children)


def slide_embeddings_dataset(split, slide_embeddings_dirs, slide_merge_mode, bag_size = None):

    if isinstance(slide_embeddings_dirs, str):
        slide_embeddings_dirs = [slide_embeddings_dirs]

    return SlideDataset(split=split,
                        load_from=list(set(slide_embeddings_dirs)),
                        slide_merge_mode = slide_merge_mode,
                        bag_size=bag_size,
                        pad = False)



def omics_dataset(split, omics_embeddings_dirs):
    return OmicsDataset(split, omics_embeddings_dirs)



def labels_dataset(split, task):

    for prefix in ["OS", "PFS", "DSS", "DFS"]:
        if task.startswith(prefix):
            return LabelDataset(split, task_names = [task], extra_attrs = [f"{task}_event", f"{task}_months"], dtype = 'int')


    return LabelDataset(split, task_names = [task], dtype = 'int')
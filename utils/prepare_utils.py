import os
import numpy as np
import torch
from torch import nn
from utils.get_split_task_utils import get_split_and_task_info
from datasets.build_dataset import build_multimodal_embeddings
from losses.nll_loss import NLLSurvLoss
from losses.cox_loss import CoxLoss
from sklearn.utils.class_weight import compute_class_weight
from utils.load_init_model import init_model_fun
from utils.load_attack_model import init_attack_model_fun
from utils.checkpoint_utils import save_model_architecture
import random
from numpy.random import MT19937, RandomState, SeedSequence

def set_seed(seed=112, disable_cudnn=False):


    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    random.seed(seed)

    rs = RandomState(
        MT19937(SeedSequence(seed)))

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


    if not disable_cudnn:

        torch.backends.cudnn.benchmark = False

        torch.backends.cudnn.deterministic = True
    else:

        torch.backends.cudnn.enabled = False

def _prepare_split_dataset(args):

    split_path = args.split

    task_config = args.task_config


    split, task_info = get_split_and_task_info(split_path, task_config)

    split.save(os.path.join(args.results_dir, "split.csv"), row_divisor="slide_id")

    valid_modalities = {"slide", "omics", "all"}
    if args.modalities not in valid_modalities:
        raise ValueError(f"Unknown modalities setting: {args.modalities}")

    modalities = []


    if args.modalities in ("slide", "all"):
        modalities.append("slide")
        slide_embeddings_dirs = args.slide_embeddings_dirs
        slide_merge_mode = args.slide_merge_mode
    else:
        slide_embeddings_dirs = None
        slide_merge_mode = None


    if args.modalities in ("omics", "all"):
        modalities.append("omics")
        omics_embeddings_dirs = args.omics_embeddings_dirs
    else:
        omics_embeddings_dirs = None



    dataset = build_multimodal_embeddings(
        split=split,
        task_name=task_info["task_col"],
        slide_embeddings_dirs=slide_embeddings_dirs,
        slide_merge_mode = slide_merge_mode,
        omics_embeddings_dirs=omics_embeddings_dirs,
        bag_size=args.bag_size,
        modalities=modalities,
    )
    return split, task_info, modalities, dataset


def _prepare_dataloader(args):
    pass


def _prepare_loss_function(args, task_info, split):


    task_name = task_info["task_col"]


    if task_info["task_type"] == "survival":
        return NLLSurvLoss(alpha=0.5, eps=1e-7, reduction="mean")


    if args.balanced:

        fold_weights = {
            fold: compute_class_weight(
                "balanced",
                classes=np.array(sorted(split.unique_classes(task_name))),
                y=split.y(task_name, fold, "train"),
            )
            for fold in range(split.num_folds)
        }

        return {
            fold: nn.CrossEntropyLoss(weight=torch.from_numpy(weights).float())
            for fold, weights in fold_weights.items()
        }


    else:
        return nn.CrossEntropyLoss()


def _prepare_the_model(args, task_info):


    args.num_classes = len(task_info["label_dict"])


    model_kwargs = {
        "input_feature_dim": args.input_feature_dim,
        "embed_dim": args.embed_dim,
        "num_classes": args.num_classes,
    }


    init_model = init_model_fun(args.model_name, **model_kwargs)


    init_model = init_model.to(args.device)


    results_dir = getattr(args, 'results_dir', None)
    if results_dir:
        save_model_architecture(init_model, os.path.join(args.results_dir, f"model.txt"))


    return init_model

def _prepare_the_attack(args):

    if args.adversarial_training:
        model_kwargs = {
            "eps": args.adv_eps,
            "alpha": args.adv_alpha,
            "steps": args.adv_steps,
        }
        init_attatck_model = init_attack_model_fun(perturb_config=args.perturb_config, **model_kwargs)

        cfg = init_attatck_model.at_model.config
        eps = getattr(cfg, "eps", "none")
        alpha = getattr(cfg, "alpha", "none")
        steps = getattr(cfg, "steps", "none")

        filename = f"{args.perturb_config}_e{eps}_a{alpha}_s{steps}_config.txt"
        results_dir = getattr(args, 'results_dir', None)
        if results_dir:
            output_file = os.path.join(args.results_dir, filename)
            with open(output_file, "w") as f:
                f.write(f"init attack {args.perturb_config} model config: {cfg.__dict__}\n")
            print(f"init attatck {args.perturb_config} model config: {cfg.__dict__}")

        return init_attatck_model

    return None


def _prepare_optimizer(args):
    if args.optimizer_type == "AdamW":

        optimizer_config = {
            "type": "AdamW",
            "base_lr": args.base_learning_rate,
            "weight_decay": args.weight_decay,
        }
        return optimizer_config

    else:

        raise NotImplementedError(
            f'Optimizer type {args.optimizer_type} not yet implemented. Please choose from "AdamW".'
        )


def _prepare_scheduler(args):

    if args.scheduler_type == "cosine":
        scheduler_config = {
            "type": "cosine",
            "eta_min": 1e-8,
            "step_on": "accumulation-step",
        }
        return scheduler_config
    else:
        return None


def prepare_function(args):

    split, task_info, modalities, dataset = _prepare_split_dataset(args)

    loss_fn = _prepare_loss_function(args, task_info, split)

    init_model = _prepare_the_model(args, task_info)

    init_attack_model = _prepare_the_attack(args)

    optimizer_config = _prepare_optimizer(args)

    scheduler_config = _prepare_scheduler(args)

    task_name = task_info["task_col"]
    task_type = task_info["task_type"]


    model_kwargs = {
        "task_name": task_name,
        "task_type": task_type,
        "model_name": args.model_name,
        "dataset": dataset,
        "modalities": modalities,
        "slide_encoder": init_model,
        "attack_model": init_attack_model,
        "precision": torch.float32,
        "num_classes": len(task_info["label_dict"]),
        "loss": loss_fn,
        "optimizer_config": optimizer_config,
        "scheduler_config": scheduler_config,
        "device": args.device,
        "return_raw_attention": args.return_raw_attention,
        "return_intermediate_feats": args.return_intermediate_feats,
        "attn_mask": args.attn_mask,
        "adv_training_types": args.adv_training_types if args.adversarial_training else None,
    }


    return model_kwargs, dataset

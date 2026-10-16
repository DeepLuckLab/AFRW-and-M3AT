from utils.core.adversarial_wrapper import AdversarialWrapper
from utils.core.standard_wrapper import StandardWrapper
from utils.checkpoint_utils import (
    save_model_architecture,
    pick_checkpoint,
    load_checkpoint,
    save_checkpoint,
    freeze,
)
from metrics.statistical_metrics import (
    get_loss,
    get_95_ci,
    get_mean_se,
    bootstrap,
    save_fold_prob_pred,
)
from metrics.classification_metrics import (
    auc_roc_plot,
    confusion_matrix_plot,
    precision_recall_plot,
    classification_metrics_report,
)
from metrics.survival_metrics import calculate_risk, survival_metrics_report
import os
import numpy as np
import torch
from torch.optim import AdamW, SGD, Adam
import time
from tqdm import trange
from tqdm import tqdm
import json

def train_val_main(args, model_kwargs, dataset):


    precision = model_kwargs.get("precision", torch.float32)

    optimizer_config = model_kwargs.get("optimizer_config", None)

    scheduler_config = model_kwargs.get("scheduler_config", None)

    num_folds = dataset.num_folds

    for current_iter in range(num_folds):


        print(
            "###################################################################################"
        )
        print(f"Training: Fold {current_iter + 1} of {num_folds}...")
        print(f"Adversarial training by {args.adv_training_types}" if args.adversarial_training else f"Standard training by {args.task_type}")
        print(
            "###################################################################################"
        )

        dataloaders = {
            mode: dataset.get_dataloader(current_iter, mode, batch_size=args.batch_size)
            for mode in ["train", "val", "test"]
        }


        if args.adversarial_training:
            model = AdversarialWrapper(**model_kwargs)
        else:
            model = StandardWrapper(**model_kwargs)


        optimizer = _init_optimizer(optimizer_config, model)

        scheduler = _init_scheduler(args, scheduler_config, optimizer, dataloaders)

        try:

            grad_scaler = torch.amp.GradScaler(
                "cuda", enabled=(precision == torch.float32)
            )
        except:

            grad_scaler = torch.cuda.amp.GradScaler(
                enabled=(precision == torch.float32)
            )


        train_losses = []
        val_losses = []

        best_val_loss = 1e4

        early_stop_counter = 0

        stop_training = False


        if args.view_progress == "bar":
            loop = trange(args.num_epochs, desc="Epochs")
        elif args.view_progress == "verbose":
            loop = range(args.num_epochs)
        else:
            raise ValueError(
                f"view_progress must be 'bar' or 'verbose', got {args.view_progress} instead."
            )


        if args.checkpoint_strategy == "best-val-loss" and dataloaders["val"] is None:
            raise ValueError(
                "checkpoint_strategy='best-val-loss' requires a validation dataloader, "
                "but none was provided.\n\n"
                "Please set one of the following instead:\n"
                "  • checkpoint_strategy='every-K'   (e.g., every-1, every-5)\n"
                "  • checkpoint_strategy='last-K'    (e.g., last-3, last-1)\n\n"
                "Tip: 'best-val-loss' only works when a validation set exists to compute validation loss."
            )


        for current_epoch in loop:

            for mode in ["train", "val"]:

                if dataloaders[mode] is None and mode=="val":
                    mode = "test"

                if args.view_progress == "bar":
                    loop.set_description(f"Epoch {current_epoch} {mode}")


                start = time.time()


                metrics = _loop_over_batches(
                    args,
                    loop,
                    mode,
                    model,
                    optimizer,
                    scheduler,
                    scheduler_config,
                    dataloaders,
                    precision,
                    grad_scaler,
                )


                end = time.time()


                with open(os.path.join(args.results_dir, "progress.txt"), "a") as f:
                    f.write(
                        f"DONE: Fold {current_iter + 1}/{num_folds} | {mode} | Epoch {current_epoch}\n"
                    )


                if args.view_progress == "verbose":
                    print(
                        f"Finished epoch = {current_epoch} in {end - start:.2f} seconds"
                    )


                avg_loss = metrics["avg_loss"]

                if mode == "train":
                    train_losses.append(avg_loss)


                if mode in ["val", "test"]:
                    if args.checkpoint_strategy == "best-val-loss" and mode == "val":
                        val_losses.append(avg_loss)

                        (
                            new_best_loss,
                            best_val_loss,
                            early_stop_counter,
                            stop_training,
                        ) = _early_stopping(
                            args=args,
                            metrics=metrics,
                            current_epoch=current_epoch,
                            best_val_loss=best_val_loss,
                            early_stop_counter=early_stop_counter,
                        )

                        if new_best_loss:
                            ckpt_dir = os.path.join(
                                args.results_dir, "checkpoints", f"fold_{current_iter}"
                            )
                            os.makedirs(ckpt_dir, exist_ok=True)
                            ckpt_path = os.path.join(
                                ckpt_dir, f"epoch_{current_epoch}.pt"
                            )
                            save_checkpoint(model, args.checkpoint_strategy, ckpt_path)
                    else:
                        save_conditions = [
                            args.checkpoint_strategy.startswith("every-")
                            and (current_epoch + 1)
                            % int(args.checkpoint_strategy.split("-")[1])
                            == 0,
                            args.checkpoint_strategy.startswith("last-")
                            and (
                                    (current_epoch + 1)
                                    >= max(
                                1,
                                args.num_epochs
                                - int(args.checkpoint_strategy.split("-")[1])
                                + 1,
                            )
                            ),
                        ]

                        if any(save_conditions):
                            ckpt_dir = os.path.join(
                                args.results_dir, "checkpoints", f"fold_{current_iter}"
                            )
                            os.makedirs(ckpt_dir, exist_ok=True)
                            ckpt_path = os.path.join(
                                ckpt_dir, f"epoch_{current_epoch}.pt"
                            )
                            save_checkpoint(model, args.checkpoint_strategy, ckpt_path)


            if args.early_stopping and stop_training:
                print(
                    f"Early stopping triggered at epoch {current_epoch + 1}. Exiting training loop for fold {current_iter + 1}."
                )
                break


        get_loss(args, current_iter, train_losses, val_losses)


    return None


def _loop_over_batches(
        args,
        loop,
        mode,
        model,
        optimizer,
        scheduler,
        scheduler_config,
        dataloaders,
        precision,
        grad_scaler,
):

    if mode == "train":
        model.train()
        context_manager = torch.enable_grad()
    elif mode in ["val", "test"]:
        model.eval()
        context_manager = torch.inference_mode()

        if args.adversarial_training:
            context_manager = torch.no_grad()
    else:
        raise ValueError('mode must be either "train", "val", or "test".')


    all_losses = []
    all_info = []
    num_samples_processed = 0
    num_gradient_steps = 0


    with context_manager:

        for batch_idx, batch in enumerate(dataloaders[mode]):
            num_samples_processed += len(batch["id"])

            with torch.autocast(
                device_type="cuda", dtype=precision, enabled=precision != torch.float32
            ):

                loss, info = model(batch, output="loss")


                loss = loss / args.gradient_accumulation
                assert isinstance(
                    loss, torch.Tensor
                ), f"Loss must be a tensor, got {loss} instead"
                assert isinstance(
                    info, list
                ), f"Info must be a list on CPU, got {info} instead"


            all_losses.append(loss.item())
            all_info.extend(info)


            if mode == "train":

                grad_scaler.scale(loss).backward()

                if (batch_idx + 1) % args.gradient_accumulation == 0:
                    grad_scaler.step(optimizer)
                    current_scale = grad_scaler.get_scale()
                    grad_scaler.update()
                    optimizer_skipped = (grad_scaler.get_scale() < current_scale)
                    optimizer.zero_grad()
                    num_gradient_steps += 1


                    if scheduler_config is not None and scheduler is not None:
                        if (
                            scheduler_config
                            and scheduler_config["step_on"] == "accumulation-step"
                            and not optimizer_skipped
                        ):
                            scheduler.step()
            else:

                if (batch_idx + 1) % args.gradient_accumulation == 0:
                    num_gradient_steps += 1


            avg_loss = sum(all_losses)/num_gradient_steps


            if (
                args.view_progress == "bar"
                and (batch_idx + 1) % args.gradient_accumulation == 0
            ):
                loop.set_postfix(
                    num_batches=f"{batch_idx + 1}/{len(dataloaders[mode])}",
                    num_samples=num_samples_processed,
                    avg_loss=f"{avg_loss:.4f}",
                )


    if scheduler_config is not None and scheduler is not None:
        if (
            mode == "train"
            and scheduler_config
            and scheduler_config["step_on"] == "epoch"
            and not optimizer_skipped
        ):
            scheduler.step()

    metrics = {
        "avg_loss": avg_loss,
        "all_losses": all_losses,
        "all_info": all_info,
    }

    return metrics


def evaluation(args, model_kwargs, dataset, mode, clean_data):


    all_labels_across_folds = []
    all_probs_across_folds = []
    all_preds_across_folds = []
    all_scores_across_folds = []


    task_name = model_kwargs.get("task_name", None)
    task_type = model_kwargs.get("task_type", None)
    precision = model_kwargs.get("precision", torch.float32)
    model_kwargs['clean_data'] = clean_data

    num_folds = dataset.num_folds
    if model_kwargs['clean_data']:
        save_key = 'clean'
    else:
        save_key = f'adv_e{args.adv_eps}_a{args.adv_alpha}_s{args.adv_steps}'


    loop = trange(num_folds)
    for current_iter in loop:

        eval_dataloader = dataset.get_dataloader(current_iter, mode, batch_size=1)
        if eval_dataloader is None:
            return None

        loop.set_description(
            f"Running {mode} split on {len(eval_dataloader.dataset)} samples"
        )


        checkpoint_dir = os.path.join(
            args.results_dir, "checkpoints", f"fold_{current_iter}"
        )
        ckpt_path = pick_checkpoint(checkpoint_dir)


        if args.adversarial_training:
            model = AdversarialWrapper(**model_kwargs)
        else:
            model = StandardWrapper(**model_kwargs)

        model = load_checkpoint(model, ckpt_path)
        freeze(model)


        labels, probs, preds, case_ids = _accumulate_probs_preds(
            args, task_type, task_name, eval_dataloader, model, precision
        )


        if mode == "test":
            save_fold_prob_pred(
                args.results_dir,
                current_iter,
                case_ids,
                labels,
                probs,
                preds,
                task_type,
                save_key
            )


        if len(eval_dataloader.dataset) == 1 or num_folds == 1:

            all_labels_across_folds.append(labels)
            all_probs_across_folds.append(probs)
            all_preds_across_folds.append(preds)
        else:

            per_fold_save_dir = os.path.join(
                args.results_dir, f"{save_key}_{mode}_metrics", f"fold_{current_iter}"
            )
            scores = _compute_metrics(args, task_type, labels, probs, per_fold_save_dir)
            all_scores_across_folds.append(scores)


    summary = _finalize_metrics(
        args,
        mode,
        task_type,
        all_labels_across_folds,
        all_probs_across_folds,
        all_scores_across_folds,
        save_key
    )


    with open(os.path.join(args.results_dir, f"{save_key}_{mode}_metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    return summary


def single_evaluation(args, batch, checkpoint_dir, model_kwargs, clean_data):


    task_name = model_kwargs.get("task_name", None)
    task_type = model_kwargs.get("task_type", None)
    precision = model_kwargs.get("precision", torch.float32)
    model_kwargs['clean_data'] = clean_data

    ckpt_path = pick_checkpoint(checkpoint_dir)

    if args.adversarial_training:
        model = AdversarialWrapper(**model_kwargs)
    else:
        model = StandardWrapper(**model_kwargs)

    model = load_checkpoint(model, ckpt_path)
    freeze(model)

    case_ids, label, prob, pred, attention_raw, slide_features = _single_probs_preds(args, batch, task_type, task_name, model, precision)

    return case_ids, label, prob, pred, attention_raw, slide_features

def _early_stopping(
        args,
        metrics,
        current_epoch,
        best_val_loss,
        early_stop_counter,
        patience=5,
        stop_epoch=10,
):


    val_loss = metrics.get("avg_loss", None)


    new_best_loss = False

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        new_best_loss = True
    else:

        early_stop_counter += 1


    stop_training = False

    if args.early_stopping:

        if early_stop_counter >= patience and (current_epoch + 1) >= stop_epoch:
            stop_training = True


    return new_best_loss, best_val_loss, early_stop_counter, stop_training


def _accumulate_probs_preds(args, task_type, task_name, dataloader, model, precision):


    case_ids_all = []
    labels_all = []
    probs_all = []
    preds_all = []

    context_manager = torch.inference_mode()
    if args.adversarial_training:
        context_manager = torch.no_grad()


    with context_manager, torch.autocast(
        device_type="cuda", dtype=precision, enabled=precision != torch.float32
    ):
        for batch in dataloader:

            case_ids = batch["id"]
            case_ids_all.append(str(case_ids[0]))

            if task_type == "classification":
                label = batch["modalities"]["labels"][task_name].cpu().int().numpy().tolist()[0]
                logits = model(batch, output="logits")
                prob = torch.softmax(logits[0], dim=0).cpu().detach().numpy()
                pred = torch.argmax(logits, dim=-1).cpu().detach().int().numpy().tolist()[0]
                labels_all.append(label)
                probs_all.append(prob)
                preds_all.append(pred)

            elif task_type == "survival":
                label = {
                    "survival_event": batch["modalities"]["labels"]["extra_attrs"][
                        f"{task_name}_event"
                    ],
                    "survival_time": batch["modalities"]["labels"]["extra_attrs"][f"{task_name}_months"],
                }
                logits = model(batch, output="logits")
                risk = calculate_risk(logits[0]).cpu().detach().numpy()
                labels_all.append(label)
                probs_all.append(risk)

    if task_type == "classification":
        labels_all = np.array(labels_all)
        probs_all = np.array(probs_all)
        preds_all = np.array(preds_all)
    elif task_type == "survival":
        labels_all = {
            k: np.array([v[k][0] for v in labels_all]) for k in labels_all[0].keys()
        }
        probs_all = np.array(probs_all)
        preds_all = None

    return labels_all, probs_all, preds_all, case_ids_all


def _single_probs_preds(args, batch, task_type, task_name, model, precision):

    context_manager = torch.inference_mode()
    if args.adversarial_training:
        context_manager = torch.no_grad()

    attention_raw = None
    slide_features = None

    with context_manager, torch.autocast(
        device_type="cuda", dtype=precision, enabled=precision != torch.float32
    ):
        case_ids = batch["id"]
        if task_type == "classification":
            label = batch["modalities"]["labels"][task_name].cpu().int().numpy().tolist()[0]
            logits = model(batch, output="logits")
            if args.return_raw_attention:
                attention_raw = model(batch, output="attention_raw")
            if args.return_ori_feats:
                slide_features = model(batch, output="features")
            prob = torch.softmax(logits[0], dim=0).cpu().detach().numpy()
            pred = torch.argmax(logits, dim=-1).cpu().detach().int().numpy().tolist()[0]
            return case_ids, label, prob, pred, attention_raw, slide_features
        elif task_type == "survival":
            label = {
                "survival_event": batch["modalities"]["labels"]["extra_attrs"][
                    f"{task_name}_event"
                ],
                "survival_time": batch["modalities"]["labels"]["extra_attrs"][f"{task_name}_months"],
            }
            logits = model(batch, output="logits")
            if args.return_raw_attention:
                attention_raw = model(batch, output="attention_raw")
            if args.return_ori_feats:
                slide_features = model(batch, output="features")
            pred = torch.argmax(logits, dim=-1).cpu().detach().int().numpy().tolist()[0]
            risk = calculate_risk(logits[0]).cpu().detach().numpy()
            return case_ids, label, risk, pred, attention_raw, slide_features
        else:
            raise RuntimeError("Task type not supported.")


def _compute_metrics(args, task_type, labels, probs, save_dir):

    if task_type == "classification":


        auc_roc_plot(
            labels,
            probs,
            args.num_classes,
            saveto=os.path.join(save_dir, "roc_curves.png"),
        )

        confusion_matrix_plot(
            labels,
            probs,
            args.num_classes,
            saveto=os.path.join(save_dir, "confusion_matrices.png"),
        )

        precision_recall_plot(
            labels,
            probs,
            args.num_classes,
            saveto=os.path.join(save_dir, "pr_curves.png"),
        )

        scores = classification_metrics_report(
            labels,
            probs,
            args.num_classes,
            saveto=os.path.join(save_dir, "metrics.json"),
        )

        return scores["overall"]
    elif task_type == "survival":


        scores = survival_metrics_report(
            labels["survival_event"],
            labels["survival_time"],
            probs,
            saveto=os.path.join(save_dir, "metrics.json"),
        )

        return scores

    return None


def _finalize_metrics(
    args, mode, task_type, labels_across_folds, probs_across_folds, scores_across_folds, save_key
):


    if len(labels_across_folds) > 0:

        bootstraps = bootstrap(
            labels_across_folds, probs_across_folds, args.num_bootstraps
        )

        if task_type == "classification":

            scores_across_folds = [
                classification_metrics_report(labels, probs, args.num_classes)[
                    "overall"
                ]
                for labels, probs in tqdm(
                    bootstraps, desc=f"Computing {args.num_bootstraps} bootstraps"
                )
            ]
        elif task_type == "survival":

            scores_across_folds = [
                survival_metrics_report(
                    labels["survival_event"], labels["survival_time"], probs
                )
                for labels, probs in tqdm(
                    bootstraps, desc=f"Computing {args.num_bootstraps} bootstraps"
                )
            ]


        folder_path = os.path.join(args.results_dir, f"{save_key}_{mode}_metrics")
        os.makedirs(folder_path, exist_ok=True)

        for idx, metrics_dict in enumerate(scores_across_folds):
            folder_path_curr = os.path.join(folder_path, f"bootstrap_{idx}")
            os.makedirs(folder_path_curr, exist_ok=True)

            file_path = os.path.join(folder_path_curr, "metrics.json")
            with open(file_path, "w") as f:
                json.dump(metrics_dict, f, indent=4)


        return get_95_ci(scores_across_folds)
    else:

        return get_mean_se(scores_across_folds)


def _init_optimizer(optimizer_config, model):
    optimizer_type = optimizer_config["type"]
    extra_kwargs = {
        k: v
        for k, v in optimizer_config.items()
        if k not in ["type", "get_param_groups", "param_group_args", "base_lr"]
    }

    if "get_param_groups" in optimizer_config:
        param_groups = optimizer_config["get_param_groups"](
            model, **optimizer_config["param_group_args"]
        )
        assert isinstance(
            param_groups, list
        ), "get_param_groups must return a list of dictionaries."
        assert (
            len(param_groups) > 0
        ), "get_param_groups must return a non-empty list of dictionaries."
    else:
        param_groups = model.parameters()

    if optimizer_type.lower() == "adam":
        return Adam(param_groups, optimizer_config["base_lr"], **extra_kwargs)
    elif optimizer_type.lower() == "sgd":
        return SGD(param_groups, optimizer_config["base_lr"], **extra_kwargs)
    elif optimizer_type.lower() == "adamw":
        return AdamW(param_groups, optimizer_config["base_lr"], **extra_kwargs)
    else:
        raise NotImplementedError(f"Optimizer {optimizer_type} not implemented.")


def _init_scheduler(args, scheduler_config, optimizer, dataloaders):
    if scheduler_config is not None:
        assert isinstance(scheduler_config["type"], str)
        if scheduler_config["type"] == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config["mode"],
                factor=scheduler_config["factor"],
                patience=scheduler_config["patience"],
            )
        elif scheduler_config["type"] == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config["step_size"],
                gamma=scheduler_config["gamma"],
            )
        elif scheduler_config["type"] == "cosine":
            assert (
                args.gradient_accumulation == 1
            ), "CosineAnnealingLR scheduler is not compatible with gradient accumulation."
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=(
                    args.num_epochs
                    if scheduler_config["step_on"] == "epoch"
                    else len(dataloaders["train"]) * args.num_epochs
                ),
                eta_min=scheduler_config["eta_min"],
            )
        elif scheduler_config["type"] == "cosine_warm_restart":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=scheduler_config["T_0"],
                T_mult=scheduler_config["T_mult"],
                eta_min=scheduler_config["eta_min"],
            )
        else:
            raise NotImplementedError(
                f"Scheduler type {scheduler_config['type']} not implemented."
            )
    else:
        return None

python get_split_args.py \
    --split /test_task/k=all.tsv \
    --task_config /test_task/config.yaml \
    --num_folds 5 \
    --cv_type kfold_test \
    --test_frac 0.1 \
    --val_frac 0.1 \
    --seed 1



CUDA_VISIBLE_DEVICES=0 python main_args.py \
    --task_type classification \
    --model_name abmil \
    --train_source test_task \
    --mode test \
    --seed 1 \
    --split /test_task/k=all.tsv \
    --task_config /test_task/config.yaml \
    --slide_merge_mode list \
    --slide_embeddings_dirs /features \
    --results_dir_root /debug_results \
    --modalities slide \
    --input_feature_dim 1024 \
    --embed_dim 512 \
    --bag_size None \
    --balanced \
    --base_learning_rate 1e-5 \
    --weight_decay 1e-5 \
    --optimizer_type AdamW \
    --scheduler_type cosine \
    --gradient_accumulation 1 \
    --num_epochs 300 \
    --batch_size 1 \
    --checkpoint_strategy best-val-loss \
    --early_stopping \
    --view_progress bar \
    --adversarial_training \
    --perturb_config pgd_at \
    --adv_training_types at \
    --adv_eps 1 \
    --adv_alpha 0.2 \
    --adv_steps 5


CUDA_VISIBLE_DEVICES=0 python main_args.py \
    --task_type classification \
    --model_name abmil \
    --train_source test_task \
    --mode test \
    --seed 1 \
    --split /test_task/k=all.tsv \
    --task_config /test_task/config.yaml \
    --slide_merge_mode list \
    --slide_embeddings_dirs /features \
    --results_dir_root /debug_results \
    --modalities slide \
    --input_feature_dim 1024 \
    --embed_dim 512 \
    --bag_size None \
    --balanced \
    --base_learning_rate 1e-5 \
    --weight_decay 1e-5 \
    --optimizer_type AdamW \
    --scheduler_type cosine \
    --gradient_accumulation 1 \
    --num_epochs 300 \
    --batch_size 1 \
    --checkpoint_strategy best-val-loss \
    --early_stopping \
    --view_progress bar



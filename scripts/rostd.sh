for seed in 13 2023 2024 2025 2026
do
    CUDA_VISIBLE_DEVICES=3 python train.py \
        --model roberta-base \
        --output_dir ./output/rostd/$seed \
        --seed $seed \
        --dataset rostd \
        --log_file ./log/rostd/exp_$seed.txt \
        --lr 2e-5 \
        --epochs 5 \
        --batch_size 16

    CUDA_VISIBLE_DEVICES=3 python extract_full_features.py \
        --dataset rostd \
        --ood_datasets rostd_ood \
        --output_dir output/rostd/$seed \
        --model roberta-base \
        --pretrained_model output/rostd/$seed/model.pt

    CUDA_VISIBLE_DEVICES=3 python test.py \
        --dataset rostd \
        --ood_datasets rostd_ood \
        --input_dir output/rostd/$seed \
        --model roberta-base \
        --log_file ./log/rostd/exp_$seed.txt \
        --pretrained_model output/rostd/$seed/model.pt \
        --ood_method flats
done
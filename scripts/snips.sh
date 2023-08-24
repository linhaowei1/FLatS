for seed in 13 2023 2024 2025 2026
do
    CUDA_VISIBLE_DEVICES=4 python train.py \
        --model roberta-base \
        --output_dir ./output/snips/$seed \
        --seed $seed \
        --dataset snips \
        --log_file ./log/snips/exp_$seed.txt \
        --lr 2e-5 \
        --epochs 5 \
        --batch_size 16

    CUDA_VISIBLE_DEVICES=4 python extract_full_features.py \
        --dataset snips \
        --ood_datasets snips_ood \
        --output_dir output/snips/$seed \
        --model roberta-base \
        --pretrained_model output/snips/$seed/model.pt

    for baseline in 'base' 'maxlogit' 'energy' 'maha' 'knn' 'd2u' 'odin' 'lof' 'pout'
    do
        CUDA_VISIBLE_DEVICES=4 python test.py \
            --dataset snips \
            --ood_datasets snips_ood \
            --input_dir output/snips/$seed \
            --model roberta-base \
            --log_file ./log/snips/exp_$seed.txt \
            --pretrained_model output/snips/$seed/model.pt \
            --ood_method $baseline
    done
done
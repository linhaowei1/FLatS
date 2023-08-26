for seed in 13
do
    # CUDA_VISIBLE_DEVICES=2 python train.py \
    #     --model ../roberta-base \
    #     --output_dir ./output/clinc/$seed \
    #     --seed $seed \
    #     --dataset clinc \
    #     --log_file ./log/clinc/exp_$seed.txt \
    #     --lr 2e-5 \
    #     --epochs 5 \
    #     --batch_size 16

    # CUDA_VISIBLE_DEVICES=2 python extract_full_features.py \
    #     --dataset clinc \
    #     --ood_datasets clinc_ood \
    #     --output_dir output/clinc/$seed \
    #     --model ../roberta-base \
    #     --pretrained_model output/clinc/$seed/model.pt

    for baseline in 'knn'
    do
        CUDA_VISIBLE_DEVICES=2 python test.py \
            --dataset clinc \
            --ood_datasets clinc_ood \
            --input_dir output/clinc/$seed \
            --model ../roberta-base \
            --log_file ./log/clinc/ablation_exp_$seed.txt \
            --pretrained_model output/clinc/$seed/model.pt \
            --ood_method $baseline
    done
done
for seed in 13
do
    CUDA_VISIBLE_DEVICES=1 python train.py \
        --model roberta-base \
        --output_dir ./output/banking77/$seed \
        --seed $seed \
        --dataset banking77 \
        --log_file ./log/banking77/exp_$seed.txt \
        --lr 2e-5 \
        --epochs 5 \
        --batch_size 16

    CUDA_VISIBLE_DEVICES=1 python extract_full_features.py \
        --dataset banking77 \
        --ood_datasets banking77_ood \
        --output_dir output/banking77/$seed \
        --model roberta-base \
        --pretrained_model output/banking77/$seed/model.pt

    CUDA_VISIBLE_DEVICES=1 python test.py \
        --dataset banking77 \
        --ood_datasets banking77_ood \
        --input_dir output/banking77/$seed \
        --model roberta-base \
        --log_file ./log/banking77/exp_$seed.txt \
        --pretrained_model output/banking77/$seed/model.pt \
        --ood_method flats
done
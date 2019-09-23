#!/usr/bin/env bash
DATA_DIR=/data/embeddings
src=en
tgt=fr
seed=988
dev=4
#
#python main-trainer.py --src_lang $src --src_emb $DATA_DIR/wiki.$src.vec --tgt_lang $tgt --tgt_emb $DATA_DIR/wiki.$tgt.vec  --norm_embeddings center --cuda_device 0 --exp_id tune32-van --state 1
#python main-trainer.py --src_lang $src --src_emb $DATA_DIR/$src.emb.txt --tgt_lang $tgt --tgt_emb $DATA_DIR/$tgt.emb.txt  --norm_embeddings none --cuda_device $dev --exp_id tune32-van-wacky --state 1 --dataset wacky

#python main-trainer.py --src_lang $src --src_emb $DATA_DIR/wiki.$src.vec --tgt_lang $tgt --tgt_emb $DATA_DIR/wiki.$tgt.vec  --norm_embeddings none --cuda_device 3 --exp_id tune32-van --state 1 --mode 1 --src_pretrain tune32-van/$src-$tgt/none/best/seed_$seed\_dico_S2T\&T2S_stage_2_best_X.t7  --tgt_pretrain tune32-van/$src-$tgt/none/best/seed_$seed\_dico_S2T\&T2S_stage_2_best_Y.t7 --mode 1

random_list=$(python3 -c "import random; random.seed(0); print(' '.join([str(random.randint(0, 1000)) for _ in range(10)]))") # random seeds
echo $random_list
#random_list=(265 430 523 776 864)
for seed in ${random_list[@]}
do
echo $seed
### Wiki dataset
python main-trainer.py --src_lang $src --src_emb $DATA_DIR/wiki.$src.vec --tgt_lang $tgt --tgt_emb $DATA_DIR/wiki.$tgt.vec  --norm_embeddings none --cuda_device 3 --exp_id tune32-van --state 1 --mode 1 --src_pretrain tune32-van/$src-$tgt/none/best/seed_$seed\_dico_S2T\&T2S_stage_2_best_X.t7  --tgt_pretrain tune32-van/$src-$tgt/none/best/seed_$seed\_dico_S2T\&T2S_stage_2_best_Y.t7 --mode 1
done

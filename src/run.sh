#!/usr/bin/env bash
DATA_DIR=/data/embeddings
## en <-> zh
#python main-trainer.py --src_lang en --src_emb /data/embeddings/wiki.en.vec --tgt_lang zh --tgt_emb /data/embeddings/wiki.zh.vec  --norm_embeddings none --cuda_device 0 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang en --src_emb /data/embeddings/wiki.en.vec --tgt_lang zh --tgt_emb /data/embeddings/wiki.zh.vec  --norm_embeddings unit --cuda_device 1 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang en --src_emb /data/embeddings/wiki.en.vec --tgt_lang zh --tgt_emb /data/embeddings/wiki.zh.vec  --norm_embeddings center --cuda_device 2 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang en --src_emb /data/embeddings/wiki.en.vec --tgt_lang zh --tgt_emb /data/embeddings/wiki.zh.vec  --norm_embeddings unit_center_unit --cuda_device 3 --exp_id tune32-van --state 3
#
## zh <-> en
#python main-trainer.py --src_lang zh --src_emb /data/embeddings/wiki.zh.vec --tgt_lang en --tgt_emb /data/embeddings/wiki.en.vec  --norm_embeddings none --cuda_device 0 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang zh --src_emb /data/embeddings/wiki.zh.vec --tgt_lang en --tgt_emb /data/embeddings/wiki.en.vec  --norm_embeddings unit --cuda_device 1 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang zh --src_emb /data/embeddings/wiki.zh.vec --tgt_lang en --tgt_emb /data/embeddings/wiki.en.vec  --norm_embeddings center --cuda_device 0 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang zh --src_emb /data/embeddings/wiki.zh.vec --tgt_lang en --tgt_emb /data/embeddings/wiki.en.vec  --norm_embeddings unit_center_unit --cuda_device 3 --exp_id tune32-van --state 3
#
# en <-> es
#
#python main-trainer.py --src_lang en --src_emb /data/embeddings/wiki.en.vec --tgt_lang es --tgt_emb /data/embeddings/wiki.es.vec  --norm_embeddings none --cuda_device 0 --exp_id tune32-van --state 3
python main-trainer.py --src_lang en --src_emb $DATA_DIR/wiki.en.vec --tgt_lang es --tgt_emb $DATA_DIR/wiki.es.vec  --norm_embeddings none --cuda_device 1 --exp_id tune32-van --state 1
#python main-trainer.py --src_lang en --src_emb /data/embeddings/wiki.en.vec --tgt_lang es --tgt_emb /data/embeddings/wiki.es.vec  --norm_embeddings center --cuda_device 0 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang en --src_emb /data/embeddings/wiki.en.vec --tgt_lang es --tgt_emb /data/embeddings/wiki.es.vec  --norm_embeddings unit_center_unit --cuda_device 1 --exp_id tune32-van --state 3
#
## es <-> en
#python main-trainer.py --src_lang es --src_emb /data/embeddings/wiki.es.vec --tgt_lang en --tgt_emb /data/embeddings/wiki.en.vec  --norm_embeddings none --cuda_device 0 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang es --src_emb /data/embeddings/wiki.es.vec --tgt_lang en --tgt_emb /data/embeddings/wiki.en.vec  --norm_embeddings unit --cuda_device 1 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang es --src_emb $DATA_DIR/wiki.es.vec --tgt_lang en --tgt_emb $DATA_DIR/wiki.en.vec  --norm_embeddings none --cuda_device 0 --exp_id tune32-van --state 1
#python main-trainer.py --src_lang es --src_emb /data/embeddings/wiki.es.vec --tgt_lang en --tgt_emb /data/embeddings/wiki.en.vec  --norm_embeddings center --cuda_device 2 --exp_id tune32-van --state 3 --dico_build T2S
#python main-trainer.py --src_lang es --src_emb /data/embeddings/wiki.es.vec --tgt_lang en --tgt_emb /data/embeddings/wiki.en.vec  --norm_embeddings center --cuda_device 3 --exp_id tune32-van --state 3 --dico_build S2T|T2S
#python main-trainer.py --src_lang es --src_emb /data/embeddings/wiki.es.vec --tgt_lang en --tgt_emb /data/embeddings/wiki.en.vec  --norm_embeddings unit_center_unit --cuda_device 3 --exp_id tune32-van --state 3
#
## en <-> fr
#python main-trainer.py --src_lang en --src_emb /data/embeddings/wiki.en.vec --tgt_lang fr --tgt_emb /data/embeddings/wiki.fr.vec  --norm_embeddings none --cuda_device 0 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang en --src_emb /data/embeddings/wiki.en.vec --tgt_lang fr --tgt_emb /data/embeddings/wiki.fr.vec  --norm_embeddings unit --cuda_device 1 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang en --src_emb /data/embeddings/wiki.en.vec --tgt_lang fr --tgt_emb /data/embeddings/wiki.fr.vec  --norm_embeddings center --cuda_device 2 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang en --src_emb /data/embeddings/wiki.en.vec --tgt_lang fr --tgt_emb /data/embeddings/wiki.fr.vec  --norm_embeddings unit_center_unit --cuda_device 3 --exp_id tune32-van --state 3
#
## fr <-> en
#python main-trainer.py --src_lang fr --src_emb /data/embeddings/wiki.fr.vec --tgt_lang en --tgt_emb /data/embeddings/wiki.en.vec  --norm_embeddings none --cuda_device 0 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang fr --src_emb /data/embeddings/wiki.fr.vec --tgt_lang en --tgt_emb /data/embeddings/wiki.en.vec  --norm_embeddings unit --cuda_device 1 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang fr --src_emb /data/embeddings/wiki.fr.vec --tgt_lang en --tgt_emb /data/embeddings/wiki.en.vec  --norm_embeddings center --cuda_device 2 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang fr --src_emb /data/embeddings/wiki.fr.vec --tgt_lang en --tgt_emb /data/embeddings/wiki.en.vec  --norm_embeddings unit_center_unit --cuda_device 3 --exp_id tune32-van --state 3
## en <-> ru
#python main-trainer.py --src_lang en --src_emb /data/embeddings/wiki.en.vec --tgt_lang ru --tgt_emb /data/embeddings/wiki.ru.vec  --norm_embeddings none --cuda_device 0 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang en --src_emb /data/embeddings/wiki.en.vec --tgt_lang ru --tgt_emb /data/embeddings/wiki.ru.vec  --norm_embeddings unit --cuda_device 1 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang en --src_emb /data/embeddings/wiki.en.vec --tgt_lang ru --tgt_emb /data/embeddings/wiki.ru.vec  --norm_embeddings center --cuda_device 2 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang en --src_emb /data/embeddings/wiki.en.vec --tgt_lang ru --tgt_emb /data/embeddings/wiki.ru.vec  --norm_embeddings unit_center_unit --cuda_device 3 --exp_id tune32-van --state 3
#
## ru <-> en
#python main-trainer.py --src_lang ru --src_emb /data/embeddings/wiki.ru.vec --tgt_lang en --tgt_emb /data/embeddings/wiki.en.vec  --norm_embeddings none --cuda_device 0 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang ru --src_emb /data/embeddings/wiki.ru.vec --tgt_lang en --tgt_emb /data/embeddings/wiki.en.vec  --norm_embeddings unit --cuda_device 1 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang ru --src_emb /data/embeddings/wiki.ru.vec --tgt_lang en --tgt_emb /data/embeddings/wiki.en.vec  --norm_embeddings center --cuda_device 2 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang ru --src_emb /data/embeddings/wiki.ru.vec --tgt_lang en --tgt_emb /data/embeddings/wiki.en.vec  --norm_embeddings unit_center_unit --cuda_device 3 --exp_id tune32-van --state 3
#
## en <-> eo
#python main-trainer.py --src_lang en --src_emb /data/embeddings/wiki.en.vec --tgt_lang eo --tgt_emb /data/embeddings/wiki.eo.vec  --norm_embeddings none --cuda_device 0 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang en --src_emb /data/embeddings/wiki.en.vec --tgt_lang eo --tgt_emb /data/embeddings/wiki.eo.vec  --norm_embeddings unit --cuda_device 1 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang en --src_emb /data/embeddings/wiki.en.vec --tgt_lang eo --tgt_emb /data/embeddings/wiki.eo.vec  --norm_embeddings center --cuda_device 2 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang en --src_emb /data/embeddings/wiki.en.vec --tgt_lang eo --tgt_emb /data/embeddings/wiki.eo.vec  --norm_embeddings unit_center_unit --cuda_device 3 --exp_id tune32-van --state 3
#
#
## eo <-> en
#python main-trainer.py --src_lang eo --src_emb /data/embeddings/wiki.ru.vec --tgt_lang en --tgt_emb /data/embeddings/wiki.en.vec  --norm_embeddings none --cuda_device 0 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang eo --src_emb /data/embeddings/wiki.ru.vec --tgt_lang en --tgt_emb /data/embeddings/wiki.en.vec  --norm_embeddings unit --cuda_device 1 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang eo --src_emb /data/embeddings/wiki.ru.vec --tgt_lang en --tgt_emb /data/embeddings/wiki.en.vec  --norm_embeddings center --cuda_device 2 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang eo --src_emb /data/embeddings/wiki.ru.vec --tgt_lang en --tgt_emb /data/embeddings/wiki.en.vec  --norm_embeddings unit_center_unit --cuda_device 3 --exp_id tune32-van --state 3
#
## en<->it
#python main-trainer.py --src_lang en --src_emb /data/embeddings/en.emb.txt --tgt_lang it --tgt_emb /data/embeddings/it.emb.txt --norm_embeddings none --cuda_device 0 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang en --src_emb /data/embeddings/en.emb.txt --tgt_lang it --tgt_emb /data/embeddings/it.emb.txt --norm_embeddings unit --cuda_device 1 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang en --src_emb /data/embeddings/en.emb.txt --tgt_lang it --tgt_emb /data/embeddings/it.emb.txt --norm_embeddings center --cuda_device 2 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang en --src_emb /data/embeddings/en.emb.txt --tgt_lang it --tgt_emb /data/embeddings/it.emb.txt --norm_embeddings unit_center_unit --cuda_device 3 --exp_id tune32-van --state 3
#
#
## en<->de
#
#python main-trainer.py --src_lang en --src_emb /data/embeddings/en.emb.txt --tgt_lang de --tgt_emb /data/embeddings/de.emb.txt --cuda_device 0 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang en --src_emb /data/embeddings/en.emb.txt --tgt_lang de --tgt_emb /data/embeddings/de.emb.txt --norm_embeddings unit --cuda_device 1 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang en --src_emb /data/embeddings/en.emb.txt --tgt_lang de --tgt_emb /data/embeddings/de.emb.txt --norm_embeddings center --cuda_device 2 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang en --src_emb /data/embeddings/en.emb.txt --tgt_lang de --tgt_emb /data/embeddings/de.emb.txt --norm_embeddings unit_center_unit --cuda_device 3 --exp_id tune32-van --state 3
#
## en<->es
#
#python main-trainer.py --src_lang en --src_emb /data/embeddings/en.emb.txt --tgt_lang es --tgt_emb /data/embeddings/es.emb.txt --cuda_device 0 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang en --src_emb /data/embeddings/en.emb.txt --tgt_lang es --tgt_emb /data/embeddings/es.emb.txt --norm_embeddings unit --cuda_device 1 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang en --src_emb /data/embeddings/en.emb.txt --tgt_lang es --tgt_emb /data/embeddings/es.emb.txt --norm_embeddings center --cuda_device 2 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang en --src_emb /data/embeddings/en.emb.txt --tgt_lang es --tgt_emb /data/embeddings/es.emb.txt --norm_embeddings unit_center_unit --cuda_device 3 --exp_id tune32-van --state 3
#
## en<->fi
#
#python main-trainer.py --src_lang en --src_emb /data/embeddings/en.emb.txt --tgt_lang fi --tgt_emb /data/embeddings/fi.emb.txt --cuda_device 0 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang en --src_emb /data/embeddings/en.emb.txt --tgt_lang fi --tgt_emb /data/embeddings/fi.emb.txt --norm_embeddings unit --cuda_device 1 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang en --src_emb /data/embeddings/en.emb.txt --tgt_lang fi --tgt_emb /data/embeddings/fi.emb.txt --norm_embeddings center --cuda_device 2 --exp_id tune32-van --state 3
#python main-trainer.py --src_lang en --src_emb /data/embeddings/en.emb.txt --tgt_lang fi --tgt_emb /data/embeddings/fi.emb.txt --norm_embeddings unit_center_unit --cuda_device 3 --exp_id tune32-van --state 3

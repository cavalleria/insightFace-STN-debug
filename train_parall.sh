#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

# step_per_epoch = nums/len(ctx)/per_batch_size
# eg: step_per_epoch = 5272942/16/200=1287
# verbose=1287
# step1=12870 # 10 epoch
# step2=20592 # 16 epoch
# step3=25760 # 22 epoch
# #verbose=2574
# #step1=25740 # 10 epoch
# #step2=41184 # 16 epoch
# #step3=51480 # 20 epoch
# # pretrained=./models/r152-arcface-retina/model
# pretrained_epoch=0
#
# lr=0.1
# lr_steps=$step1,$step2,$step3
#
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15' python -u train_parall.py \
#                                                                        --dataset emore \
#                                                                        --network r152 \
#                                                                        --loss arcface \
#                                                                        --per-batch-size 128 \
#                                                                        --verbose $verbose \
#                                                                        --lr $lr \
#                                                                        --lr-steps $lr_steps \
#                                                                        # --pretrained $pretrained \
#                                                                        # --pretrained-epoch $pretrained_epoch

# step_per_epoch = nums/len(ctx)/per_batch_size
# eg: step_per_epoch = 5272942/16/200=1287
# verbose=10950
# verbose=27600
verbose=24600
step1=67380 # 3 epoch
step2=157220 # 7 epoch
step3=247060 # 11 epoch
# verbose=25
# step1=250 # 10 epoch
# step2=410 # 16 epoch
# step3=514 # 20 epoch
# pretrained=./models/r152-arcface-retina/model
# pretrained_epoch=0

lr=0.01
lr_steps=$step1,$step2,$step3

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15' python -u train_parall.py \
#                                                                        --dataset emore \
#                                                                        --loss arcface \
#                                                                        --per-batch-size  2 \
#                                                                        --verbose $verbose \
#                                                                        --lr $lr \
#                                                                        --lr-steps $lr_steps \
#                                                                        # --pretrained $pretrained \
#                                                                        # --pretrained-epoch $pretrained_epoch

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15' python -u train_parall.py \
                                                                       --dataset emore \
                                                                       --loss arcface \
                                                                       --per-batch-size  56 \
                                                                       --verbose $verbose \
                                                                       --lr $lr \
                                                                       --lr-steps $lr_steps \

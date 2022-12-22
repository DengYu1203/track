#!/bin/bash
rate=0.3 
bag_path_1=/data/deng/RadarScenes/bag/validation/sequence_1*.bag
bag_path_2=/data/deng/RadarScenes/bag/validation/sequence_2*.bag
bag_path_3=/data/deng/RadarScenes/bag/validation/sequence_3*.bag
bag_path_4=/data/deng/RadarScenes/bag/validation/sequence_4*.bag
bag_path_5=/data/deng/RadarScenes/bag/validation/sequence_5*.bag
bag_path_6=/data/deng/RadarScenes/bag/validation/sequence_6*.bag
bag_path_7=/data/deng/RadarScenes/bag/validation/sequence_7*.bag
bag_path_8=/data/deng/RadarScenes/bag/validation/sequence_8*.bag
bag_path_9=/data/deng/RadarScenes/bag/validation/sequence_9*.bag
bag_path_10=/data/deng/RadarScenes/bag/validation/sequence_5.bag
bag_path_11=/data/deng/RadarScenes/bag/validation/sequence_6.bag
bag_path_12=/data/deng/RadarScenes/bag/validation/sequence_155.bag
bag_path_13=/data/deng/RadarScenes/bag/validation/sequence_19.bag
bag_path=/data/deng/RadarScenes/bag/validation/*.bag

bag_train=/data/deng/RadarScenes/bag/train/*.bag
bag_path_train_1=/data/deng/RadarScenes/bag/train/sequence_1*.bag
bag_path_train_2=/data/deng/RadarScenes/bag/train/sequence_2*.bag
bag_path_train_3=/data/deng/RadarScenes/bag/train/sequence_3*.bag
bag_path_train_4=/data/deng/RadarScenes/bag/train/sequence_4*.bag
bag_path_train_5=/data/deng/RadarScenes/bag/train/sequence_5*.bag
bag_path_train_6=/data/deng/RadarScenes/bag/train/sequence_6*.bag
bag_path_train_7=/data/deng/RadarScenes/bag/train/sequence_7*.bag
bag_path_train_8=/data/deng/RadarScenes/bag/train/sequence_8*.bag
bag_path_train_9=/data/deng/RadarScenes/bag/train/sequence_9*.bag

bag_path_train_10=/data/deng/RadarScenes/bag/train/sequence_1.bag
bag_path_train_11=/data/deng/RadarScenes/bag/train/sequence_2.bag
bag_path_train_12=/data/deng/RadarScenes/bag/train/sequence_3.bag
bag_path_train_13=/data/deng/RadarScenes/bag/train/sequence_4.bag
bag_path_train_14=/data/deng/RadarScenes/bag/train/sequence_8.bag
bag_path_train_15=/data/deng/RadarScenes/bag/train/sequence_9.bag
bag_path_train_16=/data/deng/RadarScenes/bag/train/sequence_10.bag
bag_path_train_17=/data/deng/RadarScenes/bag/train/sequence_11.bag
bag_path_train_18=/data/deng/RadarScenes/bag/train/sequence_12.bag
bag_path_train_19=/data/deng/RadarScenes/bag/train/sequence_13.bag
bag_path_train_20=/data/deng/RadarScenes/bag/train/sequence_15.bag
bag_path_train_21=/data/deng/RadarScenes/bag/train/sequence_16.bag
bag_path_train_22=/data/deng/RadarScenes/bag/train/sequence_17.bag
bag_path_train_23=/data/deng/RadarScenes/bag/train/sequence_18.bag
bag_path_train_24=/data/deng/RadarScenes/bag/train/sequence_151.bag
bag_path_train_25=/data/deng/RadarScenes/bag/train/sequence_152.bag
bag_path_train_26=/data/deng/RadarScenes/bag/train/sequence_154.bag
bag_path_train_27=/data/deng/RadarScenes/bag/train/sequence_156.bag
bag_path_train_28=/data/deng/RadarScenes/bag/train/sequence_157.bag
bag_path_train_29=/data/deng/RadarScenes/bag/train/sequence_158.bag

# bag_path=($bag_path_2 $bag_path_3)
# for f in $bag_path_5 $bag_path_6 $bag_path_7 $bag_path_8 $bag_path_9 $bag_path_10 $bag_path_11 $bag_path_1
for f in $bag_path_2 $bag_path_3 $bag_path_4 $bag_path_5 $bag_path_6 $bag_path_7 $bag_path_8 $bag_path_9 $bag_path_10 $bag_path_11 $bag_path_1
do
  rosbag play $f --clock -r $rate -s 3
  # echo "$f"
done
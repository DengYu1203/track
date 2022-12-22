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

for f in $bag_path_13 $bag_path_12 
do
  rosbag play $f --clock -r $rate -s 3
done
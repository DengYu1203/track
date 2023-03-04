#!/bin/bash
rate=0.6
# 1(4), 3(7), 5(4), 10(19), 20(1), 23(11), 28(4), 40(6), 42(4), 43(10), 46(12), 50(10), 52(8), 56(12), 57(6), 58(9), 60(8), 65(15)
bag_path_1=/data2/nuScenes/bag_sweep_final/log1_*.bag
bag_path_2=/data2/nuScenes/bag_sweep_final/log3_*.bag
bag_path_3=/data2/nuScenes/bag_sweep_final/log5_*.bag
bag_path_4=/data2/nuScenes/bag_sweep_final/log10_*.bag
bag_path_5=/data2/nuScenes/bag_sweep_final/log20_*.bag
bag_path_6=/data2/nuScenes/bag_sweep_final/log23_*.bag
bag_path_7=/data2/nuScenes/bag_sweep_final/log28_*.bag
bag_path_8=/data2/nuScenes/bag_sweep_final/log40_*.bag
bag_path_9=/data2/nuScenes/bag_sweep_final/log42_*.bag
bag_path_10=/data2/nuScenes/bag_sweep_final/log43_*.bag
bag_path_11=/data2/nuScenes/bag_sweep_final/log46_*.bag
bag_path_12=/data2/nuScenes/bag_sweep_final/log50_*.bag
bag_path_13=/data2/nuScenes/bag_sweep_final/log52_*.bag
bag_path_14=/data2/nuScenes/bag_sweep_final/log56_*.bag
bag_path_15=/data2/nuScenes/bag_sweep_final/log57_*.bag
bag_path_16=/data2/nuScenes/bag_sweep_final/log58_*.bag
bag_path_17=/data2/nuScenes/bag_sweep_final/log60_*.bag
bag_path_18=/data2/nuScenes/bag_sweep_final/log65_*.bag
# bag_path=($bag_path_3 $bag_path_1)
# for f in $bag_path_1 $bag_path_2 $bag_path_3 $bag_path_4 $bag_path_5 $bag_path_6 $bag_path_7 $bag_path_8 $bag_path_9 $bag_path_10 $bag_path_11 $bag_path_12 $bag_path_13 $bag_path_14 $bag_path_15 $bag_path_16 $bag_path_17 $bag_path_18
for f in $bag_path_2 $bag_path_3 $bag_path_4 $bag_path_5 $bag_path_6 $bag_path_7 $bag_path_8 $bag_path_9 $bag_path_10 $bag_path_11 $bag_path_12 $bag_path_13 $bag_path_14 $bag_path_15 $bag_path_16 $bag_path_17 $bag_path_18
# for f in $bag_path_1
# for f in $bag_path_18
do
  rosbag play $f --clock -r $rate
done

# rosbag play $bag_path_3 $bag_path_1 --clock -r $rate
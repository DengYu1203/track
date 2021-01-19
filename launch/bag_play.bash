#!/bin/bash
rate=0.5
bag_path_1=/data2/nuScenes/bag_sweep_final/log46_1537295833898809_scene-0626.bag
bag_path_2=/data2/nuScenes/bag_sweep_final/log0_*.bag
bag_path_3=/data2/nuScenes/bag_sweep_final/log46_1537295813898777_scene-0625.bag
for f in $bag_path_2
do
  rosbag play $f --clock -r $rate
done

# rosbag play $bag_path_3 $bag_path_1 --clock -r $rate
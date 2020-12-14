#!/bin/bash
rate=0.5
bag_path_1=/data2/nuScenes/bag_sweep_final/log46_1537295833898809_scene-0626.bag
bag_path_2=/data2/nuScenes/bag_sweep_final/log3*.bag
for f in $bag_path_1
do
  rosbag play $f --clock -r $rate
done
#!/bin/bash
for f in /data2/nuScenes/bag_sweep_final/log6*.bag
do
  rosbag play $f --clock
done
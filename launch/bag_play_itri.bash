#!/bin/bash
rate=1

topic_list="/car_state \
            /radar_objs \
            /tf /tf_static \
            /velodyne_points \
            /pylon_cameras/gige_120_0/h264 \
            /gmsl_cameras/gmsl_60_315/h264 \
            /gmsl_cameras/gmsl_60_190/h264 \
            /gmsl_cameras/gmsl_120_225/h264"

itri_dir=/data2/itri/20200911-nctu
bag1=2020-09-11-17-15-15.bag
bag2=2020-09-11-17-20-30.bag
bag3=2020-09-11-17-22-10.bag
bag4=2020-09-11-17-26-31.bag
bag5=2020-09-11-17-29-19.bag
bag6=2020-09-11-17-31-33.bag
bag7=2020-09-11-17-37-12.bag
bag8=2020-09-11-17-42-38.bag
bag9=2020-09-11-17-47-35.bag
bag10=2020-09-11-17-51-40.bag
bag11=2020-09-11-17-55-25.bag
bag12=2020-09-11-18-00-58.bag

bag=$itri_dir/$bag7
for f in $bag
do
  echo rosbag play $f --clock -r $rate --topics $topic_list
  rosbag play $f --clock -r $rate --topics $topic_list
done
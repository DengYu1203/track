#!/bin/bash

# window 1
cmd1="bash -c 'cd ~/deng/catkin_deng;\
      source devel/setup.bash;\
      $SHELL'"
cmd2="bash -c 'source ~/deng/catkin_deng/devel/setup.bash;\
      cd ~/deng/catkin_deng/src/track;\
      $SHELL'"
cmd3="bash -c 'cd /data2/itri/20200911-nctu/;\
      $SHELL'"

# window 2
cmd4="bash -c 'cd '~/deng/catkin_deng';\
      source ~/deng/catkin_deng/devel/setup.bash;\
      cd src/track/launch; $SHELL'"
cmd5="bash -c 'cd /data2/nuScenes/bag_sweep_final; $SHELL'"

gnome-terminal --window --command="$cmd1" --tab --command="$cmd2" --tab --command="$cmd3" \
               --window --command="$cmd4" --tab --command="$cmd5"

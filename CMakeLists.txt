cmake_minimum_required(VERSION 2.8.3)
project(track)  

add_compile_options(-std=c++11)


find_package(catkin REQUIRED COMPONENTS
pcl_ros
pcl_conversions
roscpp
cv_bridge
conti_radar
itri_msgs
)
find_package( OpenCV REQUIRED )

find_package(PCL REQUIRED COMPONENTS)
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp pcl_ros
)


include_directories(
 include
 ${catkin_INCLUDE_DIRS}
 ${PCL_INCLUDE_DIRS}
 ${OpenCV_INCLUDE_DIRS}
)
link_directories(${PCL_LIBRARY_DIRS})



add_executable(kal
   src/modified_kf_track.cpp)

target_link_libraries(kal
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(radar_track
   src/radar_track.cpp
   src/Hungarian.cpp
   src/dbscan.cpp 
   src/cluster_visual.cpp
   src/dbtrack.cpp
   src/dbpda.cpp)

target_link_libraries(radar_track
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(radar_cluster_ann
   src/radar_cluster_ann.cpp
   src/Hungarian.cpp
   src/cluster_visual.cpp
   src/dbtrack.cpp)

target_link_libraries(radar_cluster_ann
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(radar_track_itri
   src/radar_track_itri.cpp
   src/Hungarian.cpp
   src/dbscan.cpp 
   src/cluster_visual.cpp
   src/dbpda.cpp)

target_link_libraries(radar_track_itri
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

# add_dependencies (combineSensor conti_radare_gencpp)



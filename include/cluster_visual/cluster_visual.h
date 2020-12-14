#ifndef CLUSTER_VISUAL_H_
#define CLUSTER_VISUAL_H_


#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/conditional_removal.h>

#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/don.h>

#include <pcl/kdtree/kdtree.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/common/projection_matrix.h>

#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>

#include <pcl/segmentation/extract_clusters.h>

#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/PolygonArray.h>
#include <jsk_rviz_plugins/PictogramArray.h>


#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <limits>
#include <cmath>
#include <chrono>
#include <Eigen/Dense>
#include <algorithm>

class Cluster_visual
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr pointcloud_colored_;
  pcl::PointXYZ min_point_;
  pcl::PointXYZ max_point_;
  pcl::PointXYZ average_point_;
  pcl::PointXYZ centroid_;
  double orientation_angle_;
  float length_, width_, height_;

  jsk_recognition_msgs::BoundingBox bounding_box_;
  geometry_msgs::PolygonStamped polygon_;
  std::vector<cv::Point2f> hull_;

  std::string label_;
  int id_;
  int r_, g_, b_;

  Eigen::Matrix3f eigen_vectors_;
  Eigen::Vector3f eigen_values_;

  bool valid_cluster_;

public:
  Cluster_visual();

  void SetCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr in_origin_cloud_ptr,
                 std_msgs::Header in_ros_header, bool in_estimate_pose);

  void SetColoredPointCloud(int intensity);

  pcl::PointCloud<pcl::PointXYZ>::Ptr GetCloud();
  pcl::PointCloud<pcl::PointXYZI>::Ptr GetColoredPointCloud();

  pcl::PointXYZ GetMinPoint();
  pcl::PointXYZ GetMaxPoint();
  pcl::PointXYZ GetAveragePoint();
  pcl::PointXYZ GetCentroid();

  jsk_recognition_msgs::BoundingBox GetBoundingBox();
  geometry_msgs::PolygonStamped GetPolygon();
  std::vector<cv::Point2f> GetHull();

  double GetOrientationAngle();

  float GetLength();
  float GetWidth();
  float GetHeight();
  float GetVolume();

  int GetId();
  std::string GetLabel();
  Eigen::Matrix3f GetEigenVectors();
  Eigen::Vector3f GetEigenValues();

};

#endif

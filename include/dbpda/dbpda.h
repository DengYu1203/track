#ifndef DBPDA_H
#define DBPDA_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <stdlib.h>
#include <string> 
#include <algorithm>
#include <queue>

// pcl for kdtree
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>

// opencv for tracking
#include <cv_bridge/cv_bridge.h>
#include <opencv-3.3.1-dev/opencv2/core.hpp>
#include <opencv-3.3.1-dev/opencv2/highgui.hpp>
#include <opencv-3.3.1-dev/opencv2/imgproc.hpp>
#include <opencv-3.3.1-dev/opencv2/calib3d.hpp>
#include <opencv-3.3.1-dev/opencv2/opencv.hpp>

using namespace std;

#ifndef cluster_struct
#define cluster_struct
enum class MOTION_STATE{
  move,
  stop,
  slow_down
};

typedef struct tracker_point{
  float x;
  float y;
  float z;
  float x_v;
  float y_v;
  float z_v;
  float rcs;
  double vel_ang;
  double vel;
  int id;
  int scan_id;
  int cluster_flag;
  bool vistited;
  MOTION_STATE motion;
}cluster_point,kf_tracker_point;

enum class FRAME_STATE{
    first,
    second,
    more
};
#endif

typedef struct dbpda_neighbor_info
{
  double search_radius;
  std::vector<int> pointIdxRadiusSearch;
  std::vector<float> pointRadiusSquaredDistance;
  int search_k;
  std::vector<int> pointIdxNKNSearch;
  std::vector<float> pointNKNSquaredDistance;
}dbpda_neighbor_info;

typedef struct dbpda_info
{
  double core_dist;
  double reachable_dist;
  std::vector<int> neighbors;
  dbpda_neighbor_info neighbor_info;
}dbpda_info;

typedef struct dbpda_parameter
{
    double eps;
    int Nmin;
}dbpda_para;

class dbpda
{
    private:
        std::vector< std::vector<cluster_point> > points_history;   // points (t, t-1, t-2), past points(including current scan)
        std::vector< cluster_point > points_t_current;                // points (t), current points
        std::vector< dbpda_info > cluster_queue;
        std::vector< int > cluster_order;                     // record the cluster result
        std::vector<int> center_list_info;
        std::vector<cluster_point> center, final_center;
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud;
        dbpda_para dbpda_param; // the parameter of the DBPDA Algorithm
        int history_frame_num;  // the frame numbers that decide to use
        double vel_scale;
        FRAME_STATE frame_state;
        dbpda_neighbor_info find_neighbors(cluster_point p);
        void expand_neighbor(std::vector< cluster_point > &process_data, std::vector<int> &temp_cluster, int core_index);
        double distance(cluster_point p1, cluster_point p2);
        std::vector<int> split_past(std::vector< cluster_point > process_data, std::vector< std::vector<int> > &final_cluster_order);
        void cluster_center(std::vector< std::vector<cluster_point> > cluster_list);
        void kalman_filter_init();
    public:
        dbpda(double eps=2.5, int Nmin=2);
        ~dbpda();
        std::vector< std::vector<cluster_point> > cluster(std::vector<cluster_point> data);
        std::vector<cluster_point> get_center(void);
        std::vector<cluster_point> get_history_points(void);
        int scan_num;
        double data_period;
        bool output_info;       // decide to print the dbscan output information or not

};

#endif
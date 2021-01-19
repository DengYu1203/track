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
#include <opencv-3.3.1-dev/opencv2/core/eigen.hpp>
#include <opencv-3.3.1-dev/opencv2/highgui.hpp>
// #include <opencv-3.3.1-dev/opencv2/imgproc.hpp>
// #include <opencv-3.3.1-dev/opencv2/calib3d.hpp>
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
    third,
    more
};
#endif

enum class CLUSTER_STATE{
  missing,    // once no matching
  tracking,   // tracking more than 2 frames
  unstable    // first frame tracker
};



typedef struct kalman_filter{
  cv::KalmanFilter kf;      // kalman filter of one cluster
  Eigen::VectorXd pred_state;   // predicted state
  int lose_frame;           // record the lost frames
  int track_frame;          // record the hit frames
  MOTION_STATE motion;      // the motion of the cluster
  CLUSTER_STATE cluster_state; // the stability of the cluster
  int cluster_id;
}cluster_filter;

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
        std::vector< dbpda_info > cluster_queue;                    // the queue of the current scan data info
        std::vector< int > cluster_order;                           // record the cluster result(contains current and past data)
        std::vector<cluster_point> final_center;                    // get the cluster center
        std::vector< std::vector< std::vector<cluster_point> > > history_points_with_cluster_order; // the vector ->  cluster order(time order(data))
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud;            // used for kd-tree
        dbpda_para dbpda_param; // the parameter of the DBPDA Algorithm
        int history_frame_num;  // the frame numbers that decide to use
        double vel_scale;
        FRAME_STATE frame_state;
        dbpda_neighbor_info find_neighbors(cluster_point p);
        void expand_neighbor(std::vector< cluster_point > &process_data, std::vector<int> &temp_cluster, int core_index);
        double distance(cluster_point p1, cluster_point p2);
        double mahalanobis_distance(Eigen::VectorXd v, cluster_filter cluster_kf);
        std::vector<int> split_past(std::vector< cluster_point > process_data, std::vector< std::vector<int> > &final_cluster_order);
        void cluster_center(std::vector< std::vector<cluster_point> > cluster_list);
        
        int stateDim = 4;   // kalman filter state : [x,y,vx,vy]
        int measureDim = 4; // [x,y]
        int controlDim = 0;
        int lose_max = 3;
        int cluster_count = 0;  // the counter to calculate the current occupied cluster id
        std::vector<cluster_filter> cluster_past;       // record the cluster filter result
        void kalman_filter_init(std::vector<cluster_point> cluster_list, int init_index);
        void kalman_filter_update_clusterPt(std::vector<cluster_point> cluster_update, cv::KalmanFilter &kf);
        void kalman_association(std::vector< cluster_point > process_data, std::vector< std::vector<int> > final_cluster_order);
        std::vector<cluster_point> time_encode();
    public:
        dbpda(double eps=2.5, int Nmin=2);
        ~dbpda();
        std::vector< std::vector<cluster_point> > cluster(std::vector<cluster_point> data);
        std::vector<cluster_point> get_center(void);
        std::vector<cluster_point> get_history_points(void);
        std::vector< std::vector< std::vector<cluster_point> > > get_history_points_with_cluster_order(void);
        void set_parameter(double eps, int Nmin, int frames_num);
        int scan_num;
        double data_period;
        bool output_info;       // decide to print the dbscan output information or not
        bool show_kf_info;
        bool use_kf;

};

#endif
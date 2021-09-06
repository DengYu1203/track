#ifndef DBTRACK_H
#define DBTRACK_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <stdlib.h>
#include <string> 
#include <algorithm>
#include <queue>
#include <time.h>   // for generate random number

#include "cluster_struct/cluster_struct.h"

// pcl for kdtree
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_representation.h>

#include <map>

using namespace std;

typedef struct dbtrack_neighbor_info
{
  double search_radius;
  std::vector<int> pointIdxRadiusSearch;
  std::vector<float> pointRadiusSquaredDistance;
  int search_k;
  std::vector<int> pointIdxNKNSearch;
  std::vector<float> pointNKNSquaredDistance;
}dbtrack_neighbor_info;

typedef struct dbtrack_info
{
  double core_dist;
  double reachable_dist;
  std::vector<int> neighbors;
  dbtrack_neighbor_info neighbor_info;
}dbtrack_info;

typedef struct dbtrack_parameter
{
  double eps;
  int Nmin;
  int time_threshold;
}dbtrack_para;

typedef struct regression_line
{
  /* 
   * Assume the motion euation:
   * x(t) = x1*t^2 + x2*t + x3
   * y(t) = y1*t^2 + y2*t + y3
   */
  double x1,x2,x3;
  double y1,y2,y3;
  double error_x1,error_x2,error_x3;
  double error_y1,error_y2,error_y3;
}regression_line;

typedef struct track_core{
  cluster_point point;
  Eigen::Vector2d vel;    // predicted vel
  std::vector<Eigen::Vector3d> hitsory; // (x,y,vel)

}track_core;

typedef struct rls_est
{
  Eigen::Vector2d vel;  // initialized with (0,0) or tracking history
  Eigen::Matrix2d P;    // initialized with random positive-define matrix
}rls_est;


class dbtrack
{
  private:
    std::vector< std::vector<cluster_point> > points_history;         // raw points (t, t-1, t-2), past points(including current scan)
    std::vector< std::vector<cluster_point> > cluster_with_history;   // cluster result with past points(including current scan) and the cluster composed of PAST data only
    std::vector< std::vector<cluster_point> > cluster_based_on_now;   // cluster result with past points(including current scan), but filter out the clsuter that only contains past
    std::vector< dbtrack_info > cluster_queue;                    // the neighbor info queue of the process data
    std::vector<cluster_point> final_center;                    // get the cluster center
    std::vector< std::vector< std::vector<cluster_point> > > history_points_with_cluster_order; // the vector ->  cluster order(time order(data))
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud;            // used for kd-tree
    dbtrack_para dbtrack_param; // the parameter of the CLUSTER Algorithm
    int history_frame_num;  // the frame numbers that decide to use
    double vel_scale;
    FRAME_STATE frame_state;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;             // k-d tree with x,y,vel
    dbtrack_neighbor_info find_neighbors(cluster_point p);
    void expand_neighbor(std::vector< cluster_point > &process_data, std::vector<int> &temp_cluster, int core_index);
    double distance(cluster_point p1, cluster_point p2);
    void split_past(std::vector< cluster_point > process_data, std::vector< std::vector<int> > &final_cluster_order);
    void split_past_new(std::vector< cluster_point > process_data, std::vector< std::vector<int> > &final_cluster_order);
    void cluster_center(std::vector< std::vector<cluster_point> > cluster_list);
    double vel_function(double vel, int frame_diff);
    /*
     * For cluster-tracking part
     * Use vector to store the cluster id
     * 
     * Layer 1: time layer(t-frame_num,...,t)
     * Layer 2: point layer(Nt(1),Nt(2),...,Nt(m))
     * Layer 3: tracking id list
     */
    std::map<int,std::vector<int>> tracking_id_history_map;  // record the cluster id and its hit numbers
    int tracking_id_max = 0;  // record the current tracking id max to generate the new id
    void cluster_track(std::vector< cluster_point > process_data, std::vector< std::vector<int> > final_cluster_order);
    void merge_cluster(std::map<int,std::vector<cluster_point*>> cluster_result_ptr);
    std::vector<int> final_cluster_idx;
    std::vector<std::vector<cluster_point>> current_final_cluster_vec;
    void motion_equation_optimizer(std::map<int,std::vector<cluster_point*>> cluster_result_pt);
    int voting_id(std::map<int,std::vector<cluster_point*>> &cluster_result_ptr, std::map<int,std::vector<cluster_point*>> cluster_id_map, int cluster_size);
    void tracking_id_adjust();
    std::vector<cluster_point> motion_model_center;
    /*
     * Output msg flag
     */
    bool output_cluster_track_msg = true;
    bool output_motion_eq_optimizer_msg = false;
    /*
     * Tracker Core points for DBSCAN 
     */
    bool tracker_core_flag = false;  // true: use the core points, false: not use it
    bool tracker_core_msg = true;
    int tracker_core_begin_index;

    std::map<int,track_core> tracker_cores; // record the tracking result as core points -> (cluster id,cluster points)
    /* 
     * RLS velocity estimation
     */
    bool use_RLS = true;
    bool RLS_msg_flag = false;
    std::map<int,rls_est> tracker_vel_map;
    void updateTrackerVelocity(std::vector<std::vector<cluster_point>> cluster_result, std::vector<int> tracker_idx);
    void RLS_vel(std::vector<cluster_point> cluster_group, rls_est &rls_v, Eigen::Vector2d ref_vel);
  public:
    dbtrack(double eps=2.5, int Nmin=2);
    ~dbtrack();
    // Modify the DBSCAN Algo
    std::vector< std::vector<cluster_point> > improve_cluster(std::vector<cluster_point> data);
    // Original DBSCAN + Merge/Split
    std::vector< std::vector<cluster_point> > cluster(std::vector<cluster_point> data);
    std::vector< std::vector<cluster_point> > cluster_with_past();
    std::vector< std::vector<cluster_point> > cluster_with_past_now();
    std::vector<cluster_point> get_center(void);
    std::vector< std::vector< std::vector<cluster_point> > > get_history_points_with_cluster_order(void);
    std::vector<int> cluster_tracking_result();
    void set_parameter(double eps, int Nmin, int frames_num, double dt_weight);
    void set_output_info(bool cluster_track_msg, bool motion_eq_optimizer_msg, bool rls_msg);
    int scan_num;
    double data_period;
    double dt_threshold_vel;  // use for the vel_function to decide the vel weight with scan difference
};

#endif
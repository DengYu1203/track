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

// DA for IOU score
#include "Hungarian/Hungarian.h"

// save/load file
#include <fstream>
#include <boost/filesystem.hpp>

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
  double eps, eps_min, eps_max;
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

typedef struct dynamic_param{
  std::vector<double> eps;
  int Nmin;
  double Vthresh;
}dynamic_param;

class dbtrack
{
  private:
    std::vector< std::vector<cluster_point> > points_history;         // raw points (t, t-1, t-2), past points(including current scan)
    std::vector< std::vector<cluster_point> > cluster_based_on_now;   // cluster result with past points(including current scan), but filter out the clsuter that only contains past
    std::vector< dbtrack_info > cluster_queue;                    // the neighbor info queue of the process data
    std::vector<cluster_point> final_center;                    // get the cluster center
    std::vector< std::vector< std::vector<cluster_point> > > history_points_with_cluster_order; // the vector ->  cluster order(time order(data))
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud;            // used for kd-tree
    dbtrack_para dbtrack_param; // the parameter of the CLUSTER Algorithm
    int history_frame_num;  // the frame numbers that decide to use
    double vel_scale;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;             // k-d tree with x,y,vel
    dbtrack_neighbor_info find_neighbors(cluster_point p, dbtrack_para *Optional_para=NULL);
    void expand_neighbor(std::vector< cluster_point > &process_data, std::vector<int> &temp_cluster, int core_index, std::vector<dbtrack_para> *Optional_para=NULL);
    double distance(cluster_point p1, cluster_point p2);
    void split_past(std::vector< cluster_point > process_data, std::vector< std::vector<int> > &final_cluster_order);
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
    std::vector<std::vector<cluster_point*>> cluster_track_vector_address;
    std::vector<int> ambiguous_cluster_vec;
    void motion_equation_optimizer(std::map<int,std::vector<cluster_point*>> cluster_result_pt);
    int voting_id(std::map<int,std::vector<cluster_point*>> &cluster_result_ptr, std::map<int,std::vector<cluster_point*>> cluster_id_map, int cluster_size);
    void tracking_id_adjust();
    // std::vector<cluster_point> motion_model_center;
    /*
     * Output msg flag
     */
    bool output_cluster_track_msg = true;
    bool output_motion_eq_optimizer_msg = false;
    /*
     * Tracker Core points for DBSCAN 
     */
    bool tracker_core_msg = true;

    std::map<int,track_core> tracker_cores; // record the tracking result as core points -> (cluster id,cluster points)
    /* 
     * RLS velocity estimation
     */
    bool use_RLS = false;
    bool RLS_msg_flag = false;
    std::map<int,rls_est> tracker_vel_map;
    void updateTrackerVelocity(std::vector<std::vector<cluster_point>> cluster_result, std::vector<int> tracker_idx);
    void RLS_vel(std::vector<cluster_point> cluster_group, rls_est &rls_v, Eigen::Vector2d ref_vel);
    // Modified DBSCAN
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> input_cloud_vec;   // record the point cloud as vector with size: accumulated frame numbers

    /*
     *  Learning DBSCAN Parameter
     */
    std::string parameter_path; // save/load the parameter path
    std::string para_dataload = "dbtrack_parameter.csv";  // parameter filename
    boost::filesystem::path dbscan_para_path;  // the complete parameter path
    bool parameter_training = true; // true: learning parameters, false: using parameters
    bool use_dbscan_training_para = true;  // true: use the training parameters, false use the default parameters
    void check_path(std::string path);
    void save_parameter(Eigen::MatrixXd parameter_matrix, std::string filepath, bool appendFlag=true);
    Eigen::MatrixXd load_parameter(std::string filepath);
    Eigen::MatrixXd dbscan_parameter_matrix;
    int radarInput_scanTime = -1;
    std::string training_scene_name;
    bool training_scene_flag = false; // true: append the radar input vector, false: rewrite new file
    void trainingInit();
    void trainingCluster(std::vector<cluster_point> data, bool skip);
    double trainingOptimize(std::vector< cluster_point > &process_data,
                          std::vector<dbtrack_para> &trainParameterVec,
                          std::vector< std::vector<cluster_point*> > &clusterVec,
                          std::map< int,std::vector<cluster_point*> > &trackMap);
    std::vector< std::vector<int> > trainingDBSCAN(std::vector< cluster_point > &process_data,
                                                   std::vector<dbtrack_para> &trainParameterVec);
    std::vector< std::map<std::string,std::vector<kf_tracker_point>> >  GTMap_vec;
    bool use_dynamic_eps = true;
    bool use_dynamic_Nmin = true;
    bool Nmin_train = false;
    bool use_dbscan_only = true;  // use cluster algo only without using vote ID
    int vel_slot_size = 6;
    dynamic_param dynamic_dbscan_result;
    typedef struct Nmin_train_{
      double moving_obj_percentage;
      int current_scan_points;
      int past_scan_points;
      std::vector<int> vel_slot; // record the input velocity distribution(from 0m/s to 30m/s, 5m/s for one slot)
    }Nmin_train_object;
    Nmin_train_object Nmin_training_record;
    // vel, r, dt, scan num
    Eigen::MatrixXd MLP_eps_v3(Eigen::MatrixXd input);
    int MLP_classfication_Nmin_v1(Eigen::MatrixXd input);
    bool use_dynamic_vel_threshold = true;
    double cluster_vel_threshold = 0.2;
    void MLP_vel_v1(Eigen::MatrixXd input);
    typedef struct Score{
      int frame;
      int object_num = 0;
      int good_cluster = 0;   // good cluster in gt
      int multi_cluster = 0;  // multiple clusters in one gt
      int under_cluster =0;    // cluster cover over than one gt
      int no_cluster = 0; // no cluster in gt
      double v_measure_score = 0;
      double homo = 0;
      double h_ck = 0;
      double h_c = 0;
      double comp = 0;
      double h_kc = 0;
      double h_k = 0;
    }clusterScore;
    typedef struct F1Score{
      double f1_score;
      double precision;
      double recall;
      int TP;
      int FP;
      int FN;
    }F1Score;
    std::string output_score_dir_name = "v34_dynamic_eps_Nmin_vel_threshold_parameter_without_feedback_dynamic";
    void f1_score_Vth_test(std::vector<std::vector<cluster_point>>GroundTruthCluster, std::vector<std::vector<cluster_point>>ResultCluster, double v_test);
    bool Vthreshold_test_flag = false;
    void outputScore(clusterScore result, double beta=1.0);
    void outputScore(F1Score f1_1, F1Score f1_2);
  public:
    dbtrack(double eps=1.5, int Nmin=2);
    ~dbtrack();
    // DBSCAN + Merge/Split
    std::vector< std::vector<cluster_point> > cluster(std::vector<cluster_point> data);
    std::vector< std::vector<cluster_point> > cluster_with_past_now();
    std::vector<cluster_point> get_center(void);
    std::vector< std::vector< std::vector<cluster_point> > > get_history_points_with_cluster_order(void);
    std::vector<int> cluster_tracking_result();
    std::vector<int> ambiguous_cluster();
    dynamic_param dynamic_viz(); // output the Dynamic eps, Nmin, V_threshold of dbscan
    void set_parameter(double eps, double eps_min, double eps_max, int Nmin, int frames_num, double dt_weight, double v_threshold);
    void set_dynamic(bool dynamic_eps, bool dynamic_Nmin, bool dynamic_vel_thrshold);
    void set_output_info(bool cluster_track_msg, bool motion_eq_optimizer_msg, bool rls_msg);
    void set_clustr_score_name(std::string score_file_name);
    void training_dbtrack_para(bool training_, std::string filepath="/home/user/deng/catkin_deng/src/track/src/dbscan_parameter"); // decide to train the parameter or not and give the filepath
    void get_training_name(std::string scene_name); // get bag name
    void get_gt_cluster(std::vector< std::vector<cluster_point> > gt_cluster);
    double cluster_score(std::vector<std::vector<cluster_point>>GroundTruthCluster, std::vector<std::vector<cluster_point>>ResultCluster, double beta=1.0);
    std::pair<double, double> f1_score(std::vector<std::vector<cluster_point>>GroundTruthCluster, std::vector<std::vector<cluster_point>>ResultCluster);
    void train_radarScenes(std::map<std::string,std::vector<kf_tracker_point>> GroundTruthMap, int scan_points);
    void Nmin_training_cluster(std::vector<std::vector<cluster_point>>GroundTruthCluster,std::vector<cluster_point> data);
    void reset();
    void update_tracker_association(std::vector<std::pair<int,int>> update_pair);
    int scan_num;
    double time_stamp;
    double data_period;
    double dt_threshold_vel;  // use for the vel_function to decide the vel weight with scan difference
};

#endif
#ifndef DBSCAN_H
#define DBSCAN_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace std;

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

// use the radar velocity to adjust the parameter(slow to fast)
typedef struct multi_param{
    double eps[4];  // distance threshold
    int MinPts[4];  // neighbor number threshold
}dbscan_param;

typedef struct analysis{
    double max;
    double ave;
    double min;
    double range;
}statistic;

enum class FRAME_STATE{
    first,
    second,
    more
};
class dbscan
{
    private:
        std::vector<cluster_point> points;          // current (t)
        std::vector<cluster_point> history_points;  // past (t-1)
        std::vector<cluster_point> oldest_points;   // past (t-2)
        std::vector<cluster_point> history_back;    // all past data
        std::vector<cluster_point> in_points;
        std::vector<cluster_point> out_points;
        std::vector<cluster_point> center, final_center;
        std::vector< Eigen::VectorXd > var_list, final_var_list;
        std::vector<int> center_list_info;
        std::vector< std::vector<cluster_point> > stage_one_cluster;
        std::vector<int> cluster_idx;
        dbscan_param param;
        statistic data_vel;
        double vel_scaling;
        int cluster_count;
        int state;
        FRAME_STATE frame_state;
        double scan_eps;
        bool use_RANSAC;
        bool use_vel_scaling;
        std::vector<int> find_neighbor(cluster_point pt, int vel_level);
        void expand_neighbor(std::vector<int> neighbor);
        int decide_vel_level(double vel);
        int decide_vel_level_test(double vel);
        std::vector<cluster_point> delist(std::vector< std::vector<cluster_point> > cluster_l);
        std::vector< std::vector<cluster_point> > stage_one_filter(std::vector< std::vector<cluster_point> > &cluster_list);
        void cluster_center(std::vector< std::vector<cluster_point> > cluster_list);
        void merge(std::vector< std::vector<cluster_point> > &cluster_list);
        void split(std::vector< std::vector<cluster_point> > &cluster_list);
        void remove(std::vector< std::vector<cluster_point> > &cluster_list);
        void analysis_data(void);
        void grid(double grid_size);
        
    public:
        dbscan();
        ~dbscan();
        std::vector< std::vector<cluster_point> > cluster_test(std::vector<cluster_point> data);
        std::vector< std::vector<cluster_point> > cluster_from_RANSAC(std::vector<cluster_point> inlier,std::vector<cluster_point> outlier);
        std::vector< std::vector<cluster_point> > stage_one_result(void);
        std::vector<cluster_point> get_center(void);
        std::vector<cluster_point> get_history_points(void);
        int cluster_num(void);
        int scan_num;           // get the current scan callback number
        float data_period;      // get the data period (s)
        bool output_info;       // decide to print the dbscan output information or not
};

#endif
#ifndef DBSCAN_H
#define DBSCAN_H

#include <iostream>
#include <vector>

using namespace std;

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
  int cluster_flag;
  bool vistited;
}cluster_point,kf_tracker_point;

// use the radar velocity to adjust the parameter(slow to fast)
typedef struct multi_param{
    double eps[4];  // distance threshold
    int MinPts[4];  // neighbor number threshold
}dbscan_param;

class dbscan
{
    private:
        std::vector<cluster_point> points;
        std::vector<cluster_point> in_points;
        std::vector<cluster_point> out_points;
        std::vector<cluster_point> center;
        std::vector< Eigen::VectorXd > var_list;
        std::vector<int> center_list_info;
        std::vector< std::vector<cluster_point> > stage_one_cluster;
        std::vector<int> cluster_idx;
        dbscan_param param;
        int cluster_count;
        int state;
        bool use_RANSAC;
        std::vector<int> find_neighbor(cluster_point pt, int vel_level);
        void expand_neighbor(std::vector<int> neighbor);
        int decide_vel_level(double vel);
        std::vector<cluster_point> delist(std::vector< std::vector<cluster_point> > cluster_l);
        std::vector< std::vector<cluster_point> > stage_one_filter(std::vector< std::vector<cluster_point> > &cluster_list);
        void cluster_center(std::vector< std::vector<cluster_point> > cluster_list);
        void merge(std::vector< std::vector<cluster_point> > &cluster_list);
        
    public:
        dbscan(std::vector<cluster_point> &data);
        ~dbscan();
        std::vector< std::vector<cluster_point> > cluster(void);
        std::vector< std::vector<cluster_point> > cluster_from_RANSAC(std::vector<cluster_point> inlier,std::vector<cluster_point> outlier);
        std::vector< std::vector<cluster_point> > cluster_from_RANSAC_out2in(std::vector<cluster_point> inlier,std::vector<cluster_point> outlier);
        std::vector< std::vector<cluster_point> > stage_one_result(void);
        std::vector<cluster_point> get_center(void);
        int cluster_num(void);
        bool output_info;       // decide to print the dbscan output information or not
};

#endif
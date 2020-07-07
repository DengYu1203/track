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
  double vel_ang;
  double vel;
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
    std::vector<cluster_point> center;
    std::vector<int> cluster_idx;
    dbscan_param param;
    int cluster_count;
    std::vector<int> find_neighbor(cluster_point pt, int vel_level);
    void expand_neighbor(std::vector<int> neighbor);
    int decide_vel_level(double vel);
    
public:
    dbscan(std::vector<cluster_point> &data);
    ~dbscan();
    std::vector< std::vector<cluster_point> > cluster(void);
    int cluster_num(void);
};

#endif
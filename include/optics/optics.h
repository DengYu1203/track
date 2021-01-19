#ifndef OPTICS_H
#define OPTICS_H

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

typedef struct optics_neighbor_info
{
  double search_radius;
  std::vector<int> pointIdxRadiusSearch;
  std::vector<float> pointRadiusSquaredDistance;
  int search_k;
  std::vector<int> pointIdxNKNSearch;
  std::vector<float> pointNKNSquaredDistance;
}optics_neighbor_info;

typedef struct optics_info
{
  double core_dist;
  double reachable_dist;
  std::vector<int> neighbors;
  optics_neighbor_info neighbor_info;
}optics_info;

typedef struct optics_parameter
{
  double eps;
  int Nmin;
}param;

struct queue_element{
  double reachable_dist;
  int queue_index;
  queue_element(double r,int index){reachable_dist=r;queue_index=index;}
  bool operator<(const queue_element &element) const{
    return reachable_dist < element.reachable_dist;
  }
};


class optics
{
  private:
    std::vector< std::vector<cluster_point> > points_history;   // points (t, t-1, t-2), past points(including current scan)
    std::vector< cluster_point > points_t_current;                // points (t), current points
    std::vector< optics_info > cluster_queue;
    std::vector< int > cluster_order;                     // record the cluster result
    std::vector<int> center_list_info;
    std::vector<cluster_point> center, final_center;
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud;
    param optics_param; // the parameter of the OPTICS Algorithm
    int history_frame_num;  // the frame numbers that decide to use
    double vel_scale;
    FRAME_STATE frame_state;
    void update(int core_index, std::priority_queue< queue_element > &seeds_queue, std::vector< cluster_point > &process_data);
    optics_neighbor_info find_neighbors(cluster_point p);
    double distance(cluster_point p1, cluster_point p2);
    std::vector<int> split_past(std::vector< cluster_point > process_data, std::vector< std::vector<int> > &final_cluster_order);
    void cluster_center(std::vector< std::vector<cluster_point> > cluster_list);

      
  public:
    optics(double eps=2.5, int Nmin=2);
    ~optics();
    std::vector< std::vector<cluster_point> > cluster(std::vector<cluster_point> data);
    std::vector<cluster_point> get_center(void);
    std::vector<cluster_point> get_history_points(void);
    int scan_num;
    double data_period;
    bool output_info;       // decide to print the dbscan output information or not

};

#endif
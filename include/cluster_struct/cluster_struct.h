#ifndef cluster_struct
#define cluster_struct
#include <vector>

enum class MOTION_STATE{
  move,
  stop,
  slow_down
};

enum class CLUSTER_STATE{
  missing,    // once no matching
  tracking,   // tracking more than 2 frames
  unstable    // first frame tracker
};

typedef struct tracker_point{
  float x;
  float y;
  float z;
  float x_v;
  float y_v;
  float z_v;
  float rcs;
  float r;              // the distance from sensor to measurement
  double vel_ang;       // the angle(deg) of the radar point w.r.t. map frame
  double vel;           // the velocity(m/s) of the radar point
  bool vel_dir;         // true: leave, false: close
  int id;               // callback msg index
  int scan_time;        // callback time stamp
  int cluster_id;       // -1: not clustered
  int tracking_id;      // record the tracking id(defalt:-1)
  std::vector<int> tracking_history;  // record the history tracking result
  int current_cluster_order;    // record the current cluster result order(not the tracking id)
  std::vector<int> cluster_state; // to know the current cluster is needed to be merge or not
  std::vector<int> noise_detect;  // the vector would be pushed 1 while this point is not clusterd
  bool visited;         // has been visited or not
  MOTION_STATE motion;
  bool operator==(const tracker_point &r){
    return x==r.x && y==r.y && x_v==r.x_v && y_v==r.y_v;
  }
}cluster_point,kf_tracker_point;


enum class FRAME_STATE{
  first,
  second,
  third,
  more
};
#endif
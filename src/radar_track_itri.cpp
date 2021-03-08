#include <iostream>
#include <stdlib.h>
#include <string> 
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <limits>
#include <utility>
#include <time.h>

#include <ros/ros.h>
#include <ros/package.h>
#include <rosgraph_msgs/Log.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/conversions.h>
#include "pcl_ros/point_cloud.h"
#include <pcl/impl/point_types.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/geometry.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h>
#include <pcl/ModelCoefficients.h>

#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseWithCovariance.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv-3.3.1-dev/opencv2/core.hpp>
#include <opencv-3.3.1-dev/opencv2/highgui.hpp>
#include <opencv-3.3.1-dev/opencv2/imgproc.hpp>
#include <opencv-3.3.1-dev/opencv2/calib3d.hpp>
#include <opencv-3.3.1-dev/opencv2/opencv.hpp>

#include "kf_tracker/featureDetection.h"
#include "kf_tracker/CKalmanFilter.h"
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Int32MultiArray.h>

#include "Hungarian/Hungarian.h"
 
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>


#include <tf/LinearMath/Matrix3x3.h>
#include <tf/LinearMath/Vector3.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>


#include <nav_msgs/Odometry.h>
#include <itri_msgs/Ars40xObjects.h>
#include <itri_msgs/CarState.h>

// to sync the subscriber
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// cluster lib
#include "dbscan/dbscan.h"
#include "dbpda/dbpda.h"

// visualize the cluster result
#include "cluster_visual/cluster_visual.h"
#include <geometry_msgs/PolygonStamped.h>

// get keyboard
#include <termios.h>

using namespace std;
using namespace cv;

#define trajectory_frame 10
#define tracking_stable_frames 10 // the most stable tracking frames

typedef message_filters::sync_policies::ApproximateTime<itri_msgs::Ars40xObjects, itri_msgs::CarState>radar_sync;

ros::Publisher pub_cluster_marker;  // pub cluster index
ros::Publisher pub_marker;          // pub tracking id
ros::Publisher pub_filter;          // pub filter points
ros::Publisher pub_in_filter;       // pub the inlier points
ros::Publisher pub_out_filter;      // pub the outlier points
ros::Publisher pub_cluster_center;  // pub cluster points
ros::Publisher pub_cluster_pointcloud;  // pub cluster point cloud
ros::Publisher vis_vel_comp;        // pub radar comp vel
ros::Publisher vis_tracking_vel;    // pub radar tracking vel
ros::Publisher vis_vel;             // pub radar vel
ros::Publisher pub_trajectory;      // pub the tracking trajectory
ros::Publisher pub_trajectory_smooth;  // pub the tracking trajectory using kf smoother
ros::Publisher pub_pt;              // pub the tracking history pts
ros::Publisher pub_pred;            // pub the predicted pts
ros::Publisher pub_annotation;      // pub the label annotaion to /map frame and set the duration to 0.48s
ros::Publisher pub_anno_cluster;    // pub the cluster from the annotation
ros::Publisher pub_cluster_hull;    // pub the cluster polygon result
ros::Publisher pub_radar_debug;     // pub the radar debug lines
ros::Publisher pub_debug_id_text;   // pub the radar invalid_id to debug the weird vel
ros::Publisher pub_tracker_cov;      // pub the tracker covariance

tf::StampedTransform echo_transform;    // tf between map and radar
tf::StampedTransform car_transform;    // tf between radar and car
tf::StampedTransform map_transform;    // tf between map and car
/*
 * ROS parameter
 */
bool DA_choose = false;             // true: hungarian, false: my DA
bool output_KFT_result = true;      // output the tracking information in KFT
bool output_obj_id_result = true;
bool output_radar_info = true;
bool output_cluster_info = true;    // output the cluster result in KFT
bool output_dbscan_info = true;     // output the DBSCAN cluster info
bool output_DA_pair = true;
bool output_exe_time = false;       // print the program exe time
bool output_label_info = false;     // print the annotation callback info
bool output_score_info = false;     // print the score_cluster info
bool output_ego_vel_info = true;
bool output_transform_info = true;  // print the tf::transform echo info
bool output_gt_pt_info = false;
bool output_score_mat_info = false;
bool write_out_score_file = false;
bool use_annotation_module = false; // true: repulish the global annotation and score the cluster performance
bool use_KFT_module = true;   // true: cluster and tracking, fasle: cluster only
bool show_stopObj_id = true;  // decide to show the tracker id at stop object or not
bool show_vel_marker = false;
int kft_id_num = 1;           // 1: only show the id tracked over 5 frames, 2: show the id tracked over 1 frames, 3: show all id
bool use_score_cluster = false; // true: score the cluster performance

#define KFT_INFO(msg)     if (output_KFT_result) {std::cout << msg;}
#define RADAR_INFO(msg)   if (output_radar_info) {std::cout << msg;}
#define DBSCAN_INFO(msg)  if (output_dbscan_info) {std::cout << msg;}
#define EXE_INFO(msg)     if (output_exe_time) {std::cout << msg;}
#define EGO_INFO(msg)     if (output_ego_vel_info) {std::cout << msg;}

#define MAP_FRAME "/map"
#define CAR_FRAME "/base_link"
#define RADAR_FRONT_FRAME "/radar"

int call_back_num = 0;  // record the callback number
std::list<int> invalid_state = {1,2,3,5,6,7,13,14}; // to check the state of the conti radar

ros::Time radar_stamp,label_stamp;
float vx = 0; // vehicle vel x
float vy = 0; // vehicle vel y

/*
 * To get the groudtruth from the 3D bounding box annotation
 */
typedef struct label_point
{
  string type;  // the label class type
  tf::Pose pose;

  // 2D bounding box vertex
  tf::Point front_left;
  tf::Point front_right;
  tf::Point back_left;
  tf::Point back_right;

  visualization_msgs::Marker marker;  // record the label's marker
  vector<cluster_point> cluster_list; // record the radar points inside the box
}label_point;

#define box_bias 1.25
vector< label_point> label_vec;
bool get_label = false;

/*
 * cluster and tracking center
 */
vector<kf_tracker_point> cens;
vector<kf_tracker_point> cluster_cens;

/*
 * For visualization
 */
visualization_msgs::MarkerArray m_s,l_s,cluster_s;
visualization_msgs::MarkerArray ClusterHulls; // show the radar cluster polygon marker
int max_size = 0, cluster_marker_max_size = 0, cluster_polygon_max_size = 0, trajcetory_max_size = 0, vel_marker_max_size = 0, tracking_vel_marker_max_size = 0, cov_marker_max_size = 0;
int debug_id_max_size = 0;
typedef struct marker_color
{
  std_msgs::ColorRGBA color;
  geometry_msgs::Vector3 scale;
}marker_color;
marker_color vel_color, vel_comp_color, itri_vel_color;
marker_color tra_color, itri_tra_color, history_pt_color, pred_pt_color;
marker_color cluster_polygon_color, cluster_circle_color;
void marker_color_init(){
  /*
   * the marker of the original velocity of the radar measurement
   */
  vel_color.scale.y = 0.8;
  vel_color.scale.z = 0.8;
  // blue
  vel_color.color.a = 0.5;
  vel_color.color.r = 0.0;
  vel_color.color.g = 0.5;
  vel_color.color.b = 1.0;

  /*
   * the marker that shows the tracking vel with the same color as ITRI
   */
  itri_vel_color.scale.y = 0.5;
  itri_vel_color.scale.z = 0.5;
  // purple
  itri_vel_color.color.a = 1.0;
  itri_vel_color.color.r = 1.0;
  itri_vel_color.color.g = 0;
  itri_vel_color.color.b = 1.0;
  

  /*
   * the marker of the compensated velocity of the radar measurement
   */
  vel_comp_color.scale.y = 0.5;
  vel_comp_color.scale.z = 0.5;

  vel_comp_color.color.a = 0.8;
  // yellow
  // vel_comp_color.color.r = 225.0f/255.0f;
  // vel_comp_color.color.g = 228.0f/255.0f;
  // vel_comp_color.color.b = 144.0f/255.0f;
  // salmon pink
  vel_comp_color.color.r = 255.0f/255.0f;
  vel_comp_color.color.g = 152.0f/255.0f;
  vel_comp_color.color.b = 144.0f/255.0f;

  /*
   * the marker of the tracjectory
   */
  tra_color.scale.x = 0.2f;
  tra_color.scale.y = 0.2f;
  tra_color.scale.z = 0.2f;
  // pink
  tra_color.color.a = 0.7;
  tra_color.color.r = 226.0f/255.0f;
  tra_color.color.g = 195.0f/255.0f;
  tra_color.color.b = 243.0f/255.0f;
  
  /*
   * the marker of the tracjectory that is the same as the ITRI
   */
  itri_tra_color.scale.x = 0.2f;
  itri_tra_color.scale.y = 0.2f;
  itri_tra_color.scale.z = 0.2f;
  // green
  itri_tra_color.color.a = 1.0;
  itri_tra_color.color.r = 0;
  itri_tra_color.color.g = 0.9;
  itri_tra_color.color.b = 0;

  /*
   * the marker of the history points
   */
  history_pt_color.scale.x = 0.2f;
  history_pt_color.scale.y = 0.2f;
  // red
  history_pt_color.color.r = 1.0f;
  history_pt_color.color.a = 1;

  /*
   * the marker of the predicted points
   */
  pred_pt_color.scale.x = 0.4f;
  pred_pt_color.scale.y = 0.4f;
  // 
  pred_pt_color.color.r = 155.0f/255.0f;
  pred_pt_color.color.g = 99.0f/255.0f;
  pred_pt_color.color.b = 227.0f/255.0f;
  pred_pt_color.color.a = 1;
  
  /*
   * the marker of the polygon line
   */
  cluster_polygon_color.scale.x = 0.1;
  cluster_polygon_color.scale.y = 0.1;
  cluster_polygon_color.scale.z = 0.1;
  // orange
  cluster_polygon_color.color.r = 1.0;
  cluster_polygon_color.color.g = 0.5;
  cluster_polygon_color.color.b = 0.0;
  cluster_polygon_color.color.a = 1.0;
  
  /*
   * the marker of the polygon circle
   */
  cluster_circle_color.scale.x = 0.1;
  cluster_circle_color.scale.y = 0.0;
  cluster_circle_color.scale.z = 0.0;
  // orange
  cluster_circle_color.color.r = 1.0;
  cluster_circle_color.color.g = 0.55;
  cluster_circle_color.color.b = 0.0;
  cluster_circle_color.color.a = 1.0;
}

/*
 * KFT setting
 */
float dt = 0.07f;     //0.07f=1/14Hz(radar)
float sigmaP = 0.01;  //0.01
float sigmaQ = 0.1;   //0.1
#define bias 4.0      // used for data association bias 5 6
#define mahalanobis_bias 5.5 // used for data association(mahalanobis dist)
int id_count = 0; //the counter to calculate the current occupied track_id
#define frame_lost 10   // 5 10

std::vector <int> id;
ros::Publisher objID_pub;
// KF init
int stateDim = 6; // [x,y,z,v_x,v_y,v_z]
int measDim = 3;  // [x,y,z//,v_x,v_y,v_z]
int ctrlDim = 0;  // control input 0(acceleration=0,constant v model)


bool firstFrame = true;
bool get_transformer = false;
tf::TransformListener *tf_listener;

#define motion_vel_threshold  0.80  // 2.88km/h

enum class TRACK_STATE{
  missing,    // once no matching
  tracking,   // tracking more than 2 frames
  unstable    // first frame tracker
};

typedef struct cluster_visual_unit{
  pcl::PointXYZ center;
  float x_radius;
  float y_radius;
}cluster_visual_unit;

// cluster type
string cluster_type = "dbscan";
double cluster_eps;
int cluster_Nmin;
int cluster_history_frames;
double cluster_dt_weight; // use for the vel_function in cluster lib
// dbscan init
dbscan dbscan_seg;

// DBPDA init
dbpda dbpda_seg;

// past radars
bool show_past_points = true;
std::vector< std::vector<cluster_point> > past_radars;
std::vector< std::vector< std::vector<cluster_point> > > past_radars_with_cluster;

// for score the cluster performance
typedef struct cluster_score{
  int frame;
  int object_num = 0;
  int good_cluster = 0;   // good cluster in gt
  int multi_cluster = 0;  // multiple clusters in one gt
  int over_cluster =0;    // cluster cover over than one gt
  int no_cluster = 0; // no cluster in gt
  double v_measure_score = 0;
  double homo = 0;
  double h_ck = 0;
  double h_c = 0;
  double comp = 0;
  double h_kc = 0;
  double h_k = 0;
}v_measure_score;
bool filename_change = false;  // to know if get the bag name yet
string score_file_name; // bag name
int score_frame = 0;


// type filter
typedef struct track{
  cv::KalmanFilter kf;
  geometry_msgs::Point pred_pose;
  geometry_msgs::Point pred_v;
  
  // update the new state
  cv::KalmanFilter kf_us;
  geometry_msgs::Point pred_pose_us;
  geometry_msgs::Point pred_v_us;
  
  vector<geometry_msgs::Point> history;
  vector<geometry_msgs::Point> history_smooth;
  vector<geometry_msgs::Point> future;

  int lose_frame;
  int track_frame;
  MOTION_STATE motion;
  TRACK_STATE tracking_state;
  int match_clus = 1000;
  int cluster_idx;
  int uuid ;
}track;

// tracking result vector
std::vector<track> filters;



// publish the tracking id
void marker_id(bool firstFrame,std::vector<int> obj_id,std::vector<int> marker_obj_id,std::vector<int> tracking_frame_obj_id){
  visualization_msgs::Marker marker;
  int k;
  for(k=0; k<cens.size(); k++){
    marker.header.frame_id = MAP_FRAME;
    marker.header.stamp = ros::Time();//to show every tag  (ros::Time::now()for latest tag)
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.id = k;
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;

    marker.scale.z = 1.0f;
    marker.color.r = 255.0f/255.0f;
    marker.color.g = 128.0f/255.0f;
    marker.color.b = 171.0f/255.0f;
    if(kft_id_num>1)
      marker.color.a = 1.0;
    else
      marker.color.a = 0;

    geometry_msgs::Pose pose;
    pose.position.x = cens[k].x;
    pose.position.y = cens[k].y;
    pose.position.z = cens[k].z+2.0f+k/20;
    
    //-----------first frame要先發佈tag 為initial 
    stringstream ss;
    if(firstFrame){
      ss << k;
    }
    else{
      ss << obj_id.at(k);
      if(marker_obj_id.at(k)==-1){    // track id first appeaar
        marker.color.r = 255.0f/255.0f;
        marker.color.g = 241.0f/255.0f;
        marker.color.b = 118.0f/255.0f;
        if(kft_id_num<3)
          marker.color.a = 0.0;
      }
      if(tracking_frame_obj_id.at(k)!=-1){  // track id appeaars more than tracking_stable_frames
        marker.color.r = 0.0f/255.0f;
        marker.color.g = 229.0f/255.0f;
        marker.color.b = 255.0f/255.0f;
        marker.color.a = 1;
      }

    }
    if(cens.at(k).motion == MOTION_STATE::stop && !show_stopObj_id)
      marker.color.a = 0;
    marker.text = ss.str();
    marker.pose = pose;
    m_s.markers.push_back(marker);
  }

  if (m_s.markers.size() > max_size)
    max_size = m_s.markers.size();

  for (int a = k; a < max_size; a++)
  {
    marker.id = a;
    marker.color.a = 0;
    marker.pose.position.x = 0;
    marker.pose.position.y = 0;
    marker.pose.position.z = 0;
    marker.scale.z = 0;
    m_s.markers.push_back(marker);
  }
  if(!firstFrame)
    pub_marker.publish(m_s);
}

// publish the cluster points with different color and id
void color_cluster(std::vector< std::vector<kf_tracker_point> > cluster_list, bool cluster_result = true){
  // republish radar points
  pcl::PointCloud<pcl::PointXYZI>::Ptr cluster_points(new pcl::PointCloud<pcl::PointXYZI>);
  sensor_msgs::PointCloud2::Ptr cluster_cloud(new sensor_msgs::PointCloud2);
  cluster_points->clear();
  // end republish
  visualization_msgs::Marker marker;
  cluster_s.markers.clear();

  for(int i=0;i<cluster_list.size();i++){
    srand(i+1);
    float color = 255 * rand() / (RAND_MAX + 1.0);
    // add cluster index marker
    marker.header.frame_id = MAP_FRAME;
    marker.header.stamp = radar_stamp;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.id = i;
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.scale.z = 1.5f;  // rgb(255,127,122)
    marker.color.r = 255.0 / 255.0;
    marker.color.g = 154.0 / 255.0;
    marker.color.b = 38.0 / 255.0;
    marker.color.a = 1;
    stringstream ss;
    ss << i;
    marker.text = ss.str();
    geometry_msgs::Pose pose;
    pose.position.x = cluster_list.at(i).at(0).x + 1.5f;
    pose.position.y = cluster_list.at(i).at(0).y + 1.5f;
    pose.position.z = cluster_list.at(i).at(0).z + 1.0f;
    marker.pose = pose;
    marker.lifetime = ros::Duration(dt-0.005);
    cluster_s.markers.push_back(marker);

    for(int j=0;j<cluster_list.at(i).size();j++){
      pcl::PointXYZI pt;
      pt.x = cluster_list.at(i).at(j).x;
      pt.y = cluster_list.at(i).at(j).y;
      pt.z = cluster_list.at(i).at(j).z;
      pt.intensity = color;
      cluster_points->push_back(pt);
    }
  }
  pcl::toROSMsg(*cluster_points,*cluster_cloud);
  cluster_cloud->header.frame_id = MAP_FRAME;
  cluster_cloud->header.stamp = radar_stamp;
  if(cluster_result)
    pub_cluster_pointcloud.publish(cluster_cloud);
  else{
    pub_anno_cluster.publish(cluster_cloud);
    return;
  }
  if (cluster_s.markers.size() > cluster_marker_max_size)
       cluster_marker_max_size = cluster_s.markers.size();

  for (int a = cluster_list.size(); a < cluster_marker_max_size; a++)
  {
      marker.id = a;
      marker.color.a = 0;
      marker.pose.position.x = 0;
      marker.pose.position.y = 0;
      marker.pose.position.z = 0;
      marker.scale.z = 0;
      cluster_s.markers.push_back(marker);
  }
  if(cluster_list.size()==0){
    for(int j=0;j<cluster_s.markers.size();j++){
      cluster_s.markers.at(j).id = 0;
      cluster_s.markers.at(j).color.a = 0;
      cluster_s.markers.at(j).pose.position.x = 0;
      cluster_s.markers.at(j).pose.position.y = 0;
      cluster_s.markers.at(j).pose.position.z = 0;
      cluster_s.markers.at(j).scale.z = 0;
      cluster_s.markers.at(j).action = visualization_msgs::Marker::DELETE;
    }
    // std::cout << "In cluster_list size = 0, the cluster_s size is:"<<cluster_s.markers.size()<<endl;
  }
  if(cluster_result)
    pub_cluster_marker.publish(cluster_s);
}

// publish the cluster points with polygon
void polygon_cluster_visual(std::vector< std::vector<kf_tracker_point> > cluster_list){
  std_msgs::Header header;
  header.frame_id = MAP_FRAME;
  header.stamp = radar_stamp;
  std::vector<geometry_msgs::PolygonStamped> polygon_vec;
  std::vector<cluster_visual_unit> single_point_cluster;
  std::vector<bool> record_moving_cluster_poly, record_moving_cluster_single;
  for(int i=0;i<cluster_list.size();i++){
    pcl::PointCloud <pcl::PointXYZ>::Ptr pointsCluster(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointXYZ pt;
    cluster_visual_unit cluster_unit;
    std::vector<float> temp_x_radius;
    std::vector<float> temp_y_radius;
    bool high_vel = false;
    /*
     * If the vector v = (a,b,c), then vector T = (c,c,-a-b) is the orthogonal vector of it.
     * However, it would fail if v = (-1,1,0).
     * We can test the norm of the T, if the ||T|| = 0 then we can let T be (-b-c,a,a).
     */
    Eigen::Vector3d ab_vec, bc_vec, T_vec, T2_vec;
    switch (cluster_list.at(i).size())
    {
      case 1:
        pt.x = cluster_list.at(i).at(0).x;
        pt.y = cluster_list.at(i).at(0).y;
        pt.z = cluster_list.at(i).at(0).z;
        if(cluster_list.at(i).at(0).vel > motion_vel_threshold){
          high_vel = true;
        }
        cluster_unit.center = pt;
        cluster_unit.x_radius = 1.0f;
        cluster_unit.y_radius = 1.0f;
        single_point_cluster.push_back(cluster_unit);
        record_moving_cluster_single.push_back(high_vel);
        break;
      case 2:
        ab_vec = Eigen::Vector3d( cluster_list.at(i).at(0).x-cluster_list.at(i).at(1).x,
                                  cluster_list.at(i).at(0).y-cluster_list.at(i).at(1).y,
                                  cluster_list.at(i).at(0).z-cluster_list.at(i).at(1).z);
        T_vec = Eigen::Vector3d(-ab_vec.y()-ab_vec.z(),ab_vec.x(),ab_vec.x());
        if(T_vec.norm() == 0)
          T_vec = Eigen::Vector3d(ab_vec.z(),ab_vec.z(),-ab_vec.x()-ab_vec.y());
        T_vec.normalize();
        pt.x = cluster_list.at(i).at(1).x;
        pt.y = cluster_list.at(i).at(1).y;
        pt.z = cluster_list.at(i).at(1).z;
        pointsCluster->points.push_back(pt);
        if(ab_vec.norm()<1){
          ab_vec.normalize();
          pt.x = cluster_list.at(i).at(1).x + ab_vec.x();
          pt.y = cluster_list.at(i).at(1).y + ab_vec.y();
          pt.z = cluster_list.at(i).at(1).z;
          pointsCluster->points.push_back(pt);
          pt.x = cluster_list.at(i).at(1).x - ab_vec.x();
          pt.y = cluster_list.at(i).at(1).y - ab_vec.y();
          pt.z = cluster_list.at(i).at(1).z;
          pointsCluster->points.push_back(pt);
        }
        else{
          pt.x = cluster_list.at(i).at(0).x;
          pt.y = cluster_list.at(i).at(0).y;
          pt.z = cluster_list.at(i).at(0).z;
          pointsCluster->points.push_back(pt);
        }
        
        for(int j=0;j<2;j++){
          pt.x = cluster_list.at(i).at(j).x + T_vec.x();
          pt.y = cluster_list.at(i).at(j).y + T_vec.y();
          pt.z = cluster_list.at(i).at(j).z;
          pointsCluster->points.push_back(pt);
          pt.x = cluster_list.at(i).at(j).x - T_vec.x();
          pt.y = cluster_list.at(i).at(j).y - T_vec.y();
          pt.z = cluster_list.at(i).at(j).z;
          if(cluster_list.at(i).at(j).vel > motion_vel_threshold){
            high_vel = true;
          }
          pointsCluster->points.push_back(pt);
        }
        break;
      case 3:
        ab_vec = Eigen::Vector3d( cluster_list.at(i).at(0).x-cluster_list.at(i).at(1).x,
                                  cluster_list.at(i).at(0).y-cluster_list.at(i).at(1).y,
                                  cluster_list.at(i).at(0).z-cluster_list.at(i).at(1).z);
        bc_vec = Eigen::Vector3d( cluster_list.at(i).at(2).x-cluster_list.at(i).at(1).x,
                                  cluster_list.at(i).at(2).y-cluster_list.at(i).at(1).y,
                                  cluster_list.at(i).at(2).z-cluster_list.at(i).at(1).z);
        T_vec = Eigen::Vector3d(-ab_vec.y()-ab_vec.z(),ab_vec.x(),ab_vec.x());
        if(T_vec.norm() == 0)
          T_vec = Eigen::Vector3d(ab_vec.z(),ab_vec.z(),-ab_vec.x()-ab_vec.y());
        T_vec.normalize();
        
        T2_vec = Eigen::Vector3d(-bc_vec.y()-bc_vec.z(),bc_vec.x(),bc_vec.x());
        if(T2_vec.norm() == 0)
          T2_vec = Eigen::Vector3d(bc_vec.z(),bc_vec.z(),-bc_vec.x()-bc_vec.y());
        T2_vec.normalize();

        pt.x = cluster_list.at(i).at(1).x;
        pt.y = cluster_list.at(i).at(1).y;
        pt.z = cluster_list.at(i).at(1).z;
        pointsCluster->points.push_back(pt);
        if(ab_vec.norm()<1){
          ab_vec.normalize();
          pt.x = cluster_list.at(i).at(1).x + ab_vec.x();
          pt.y = cluster_list.at(i).at(1).y + ab_vec.y();
          pt.z = cluster_list.at(i).at(1).z;
          pointsCluster->points.push_back(pt);
        }
        else{
          pt.x = cluster_list.at(i).at(0).x;
          pt.y = cluster_list.at(i).at(0).y;
          pt.z = cluster_list.at(i).at(0).z;
          pointsCluster->points.push_back(pt);
        }
        if(bc_vec.norm()<1){
          bc_vec.normalize();
          pt.x = cluster_list.at(i).at(1).x + bc_vec.x();
          pt.y = cluster_list.at(i).at(1).y + bc_vec.y();
          pt.z = cluster_list.at(i).at(1).z;
          pointsCluster->points.push_back(pt);
        }
        else{
          pt.x = cluster_list.at(i).at(2).x;
          pt.y = cluster_list.at(i).at(2).y;
          pt.z = cluster_list.at(i).at(2).z;
          pointsCluster->points.push_back(pt);
        }
        for(int j=0;j<2;j++){
          pt.x = cluster_list.at(i).at(j).x + T_vec.x();
          pt.y = cluster_list.at(i).at(j).y + T_vec.y();
          pt.z = cluster_list.at(i).at(j).z;
          pointsCluster->points.push_back(pt);
          pt.x = cluster_list.at(i).at(j).x - T_vec.x();
          pt.y = cluster_list.at(i).at(j).y - T_vec.y();
          pt.z = cluster_list.at(i).at(j).z;
          if(cluster_list.at(i).at(j).vel > motion_vel_threshold){
            high_vel = true;
          }
          pointsCluster->points.push_back(pt);
        }
        for(int j=1;j<3;j++){
          pt.x = cluster_list.at(i).at(j).x + T2_vec.x();
          pt.y = cluster_list.at(i).at(j).y + T2_vec.y();
          pt.z = cluster_list.at(i).at(j).z;
          pointsCluster->points.push_back(pt);
          pt.x = cluster_list.at(i).at(j).x - T2_vec.x();
          pt.y = cluster_list.at(i).at(j).y - T2_vec.y();
          pt.z = cluster_list.at(i).at(j).z;
          pointsCluster->points.push_back(pt);
        }
        break;
      default:
        for(int j=0;j<cluster_list.at(i).size();j++){
          pt.x = cluster_list.at(i).at(j).x;
          pt.y = cluster_list.at(i).at(j).y;
          pt.z = cluster_list.at(i).at(j).z;
          if(cluster_list.at(i).at(j).vel > motion_vel_threshold){
            high_vel = true;
          }
          pointsCluster->points.push_back(pt);
        }
        break;
    }
    if(pointsCluster->points.size()>0){
      Cluster_visual clusterObject;
      clusterObject.SetCloud(pointsCluster, header, true);
      polygon_vec.push_back(clusterObject.GetPolygon());
      record_moving_cluster_poly.push_back(high_vel);
    }
  }
  // visualization_msgs::MarkerArray ClusterHulls;
  ClusterHulls.markers.clear();
  for(int cluster_idx=0;cluster_idx<polygon_vec.size();cluster_idx++){
    visualization_msgs::Marker ObjHull;
    ObjHull.header = header;
    ObjHull.header.frame_id = MAP_FRAME;
    ObjHull.ns = "cluster_polygon";
    ObjHull.action = visualization_msgs::Marker::ADD;
    ObjHull.pose.orientation.w = 1.0;
    ObjHull.id = cluster_idx;
    ObjHull.type = visualization_msgs::Marker::LINE_STRIP;
    ObjHull.scale = cluster_polygon_color.scale;
    ObjHull.color = cluster_polygon_color.color;
    if(record_moving_cluster_poly.at(cluster_idx)){
      ObjHull.color.r = 250.0f/255.0f;
      ObjHull.color.g = 112.0f/255.0f;
      ObjHull.color.b = 188.0f/255.0f;
      ObjHull.scale.x = ObjHull.scale.y = ObjHull.scale.z = 0.2;
    }
    ObjHull.lifetime = ros::Duration(dt-0.01);

    geometry_msgs::Point markerPt;
    for(int j=0;j<polygon_vec.at(cluster_idx).polygon.points.size();j++){
      markerPt.x = polygon_vec.at(cluster_idx).polygon.points[j].x;
      markerPt.y = polygon_vec.at(cluster_idx).polygon.points[j].y;
      markerPt.z = polygon_vec.at(cluster_idx).polygon.points[j].z;
      ObjHull.points.push_back(markerPt);
    }
    ClusterHulls.markers.push_back(ObjHull);
  }
  for(int cluster_idx=0;cluster_idx< single_point_cluster.size();cluster_idx++){
    visualization_msgs::Marker ObjHull;
    ObjHull.header = header;
    ObjHull.header.frame_id = MAP_FRAME;
    ObjHull.ns = "cluster_polygon";
    ObjHull.action = visualization_msgs::Marker::ADD;
    ObjHull.pose.orientation.w = 1.0;
    ObjHull.id = polygon_vec.size() + cluster_idx;
    ObjHull.type = visualization_msgs::Marker::LINE_STRIP;
    ObjHull.scale = cluster_circle_color.scale;
    ObjHull.color = cluster_circle_color.color;
    if(record_moving_cluster_single.at(cluster_idx)){
      ObjHull.color.r = 250.0f/255.0f;
      ObjHull.color.g = 112.0f/255.0f;
      ObjHull.color.b = 188.0f/255.0f;
      ObjHull.scale.x = 0.2;
    }
    ObjHull.lifetime = ros::Duration(dt-0.01);

    for(int i=0;i<361;i++){
      geometry_msgs::Point markerPt;
      markerPt.x = single_point_cluster.at(cluster_idx).x_radius * sin(i*2*M_PI/360) + single_point_cluster.at(cluster_idx).center.x;
      markerPt.y = single_point_cluster.at(cluster_idx).y_radius * cos(i*2*M_PI/360) + single_point_cluster.at(cluster_idx).center.y;
      markerPt.z = single_point_cluster.at(cluster_idx).center.z;
      ObjHull.points.push_back(markerPt);
    }
    ClusterHulls.markers.push_back(ObjHull);
  }
  if(ClusterHulls.markers.size() > cluster_polygon_max_size)
    cluster_polygon_max_size = ClusterHulls.markers.size();
  for (int a = single_point_cluster.size() + polygon_vec.size(); a < cluster_polygon_max_size; a++)
  {
    visualization_msgs::Marker marker;
    marker.header = header;
    marker.ns = "cluster_polygon";
    marker.id = a;
    marker.color.a = 0;
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.pose.position.x = 0;
    marker.pose.position.y = 0;
    marker.pose.position.z = 0;
    marker.scale.x = 1;
    marker.lifetime = ros::Duration(dt-0.01);
    ClusterHulls.markers.push_back(marker);
  }
  pub_cluster_hull.publish(ClusterHulls);
}

// show the radar velocity arrow
void vel_marker(const itri_msgs::Ars40xObjectsConstPtr& msg){
	visualization_msgs::MarkerArray marker_array_comp;
	visualization_msgs::MarkerArray marker_array;
  visualization_msgs::MarkerArray debug_id_array;
  
  // check the invalid state of the points that vel > 0.5
  visualization_msgs::Marker id_marker;
  id_marker.header.frame_id = RADAR_FRONT_FRAME;
  id_marker.header.stamp = radar_stamp;
  id_marker.action = visualization_msgs::Marker::ADD;
  id_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  id_marker.pose.orientation.w = 1.0;
  id_marker.scale.z = 1.5f;
  id_marker.scale.x = 0.5f;
  // id_marker.color = vel_color.color;
  id_marker.color.a = 1;
  id_marker.color.r = 10.0f/255.0f;
  id_marker.color.g = 215.0f/255.0f;
  id_marker.color.b = 255.0f/255.0f;
  id_marker.lifetime = ros::Duration(dt-0.005);
  int id_marker_idx = 0;
  int count_vel_num = 0;
	for(int i=0; i<msg->objs.size(); i++){
    // std::list<int>::iterator invalid_state_it;
    // invalid_state_it = std::find(invalid_state.begin(),invalid_state.end(),msg->objs[i].id);
    // if(invalid_state_it!=invalid_state.end())
    //   continue;
		visualization_msgs::Marker marker;
		marker.header.frame_id = RADAR_FRONT_FRAME;
		marker.header.stamp = ros::Time();
		marker.id = count_vel_num++;
		marker.type = visualization_msgs::Marker::ARROW;
		marker.action = visualization_msgs::Marker::ADD;
		marker.pose.position.x = msg->objs[i].pose.position.x;
		marker.pose.position.y = msg->objs[i].pose.position.y;
		marker.pose.position.z = 0;
		float theta = atan2(msg->objs[i].velocity.linear.y+vy,
												msg->objs[i].velocity.linear.x+vx);
		tf2::Quaternion Q;
		Q.setRPY( 0, 0, theta );
		marker.pose.orientation = tf2::toMsg(Q);
    marker.scale = vel_comp_color.scale;
		marker.scale.x = sqrt(pow(msg->objs[i].velocity.linear.y+vy,2) + 
				 									pow(msg->objs[i].velocity.linear.x+vx,2)); //~lenght~//
    
    marker.color = vel_comp_color.color;
		marker_array_comp.markers.push_back(marker);
    // if(marker.scale.x > 0.5){
    id_marker.id = id_marker_idx++;
    id_marker.text = to_string((int)msg->objs[i].probability);
    id_marker.pose.position = marker.pose.position;
    id_marker.pose.position.z = 1.0f;
    debug_id_array.markers.push_back(id_marker);
    // }

		///////////////////////////////////////////////////////////////////
    tf::Vector3 msg_vel(msg->objs[i].velocity.linear.x+vx,
                        msg->objs[i].velocity.linear.y+vy,
                        0);
    tf::Vector3 msg_pos(msg->objs[i].pose.position.x,
                        msg->objs[i].pose.position.y,
                        0);
    msg_pos.normalize();
    tf::Vector3 project_radial_vel = msg_pos.dot(msg_vel) * msg_pos;
    theta = atan2(project_radial_vel.y(),project_radial_vel.x());

		Q.setRPY( 0, 0, theta );
		marker.pose.orientation = tf2::toMsg(Q);

		marker.scale = vel_color.scale;
    marker.scale.x = project_radial_vel.length();

		marker.color = vel_color.color;
		marker_array.markers.push_back(marker);
	}
  if(count_vel_num > vel_marker_max_size)
    vel_marker_max_size = count_vel_num;
  for(int a=count_vel_num;a<vel_marker_max_size;a++){
    visualization_msgs::Marker marker;
		marker.header.frame_id = RADAR_FRONT_FRAME;
		marker.header.stamp = ros::Time();
		marker.id = a;
		marker.type = visualization_msgs::Marker::ARROW;
		marker.action = visualization_msgs::Marker::DELETE;
		marker.pose.position.x = 0;
		marker.pose.position.y = 0;
		marker.pose.position.z = 1;
    marker.color.a = 0;
		marker_array_comp.markers.push_back(marker);
		marker_array.markers.push_back(marker);
  }
  if(debug_id_array.markers.size()>debug_id_max_size)
    debug_id_max_size = debug_id_array.markers.size();
  for(int a=debug_id_array.markers.size();a<debug_id_max_size;a++){
    id_marker.color.a = 0;
    id_marker.id = a;
    id_marker.action = visualization_msgs::Marker::DELETE;
    id_marker.pose.position.x = 0;
    id_marker.pose.position.y = 0;
    id_marker.pose.position.z = 0;
    id_marker.scale.z = 0;
    debug_id_array.markers.push_back(id_marker);
  }
	vis_vel_comp.publish( marker_array_comp );
	vis_vel.publish( marker_array );
  pub_debug_id_text.publish(debug_id_array);
}

geometry_msgs::PoseWithCovariance show_tracker_cov(track track_pt){
  geometry_msgs::PoseWithCovariance pose_pt;
  std::cout << "before get\n";
  pose_pt.pose.position = track_pt.history.at(track_pt.history.size()-1);
  std::cout << "after get\n";
  cv::Mat cov = track_pt.kf.measurementMatrix * track_pt.kf.errorCovPre * track_pt.kf.measurementMatrix.t() + track_pt.kf.measurementNoiseCov;
  // cv::Mat S = (Mat_<float>(3,3) << cov.at<float>(0,0),cov.at<float>(0,1),cov.at<float>(0,2),
  //                                  cov.at<float>(1,0),cov.at<float>(1,1),cov.at<float>(1,2),
  //                                  cov.at<float>(2,0),cov.at<float>(2,1),cov.at<float>(2,2));
  for(int i=0;i<3;i++){
    // std::cout << "Loop i = "<< i << "\n";
    for(int j=0;j<3;j++){
      // std::cout << "Loop j = " << j << "\n";
      pose_pt.covariance[6*i+j] = cov.at<float>(i,j);
      pose_pt.covariance[6*i+j+3] = pose_pt.covariance[6*(i+3)+j] = 0;
      if(i==j){
        pose_pt.covariance[6*(i+3)+(j+3)] = 1;
      }
      else{
        pose_pt.covariance[6*(i+3)+(j+3)] = 0;
      }
    }
  }
  return pose_pt;
}

visualization_msgs::Marker get_ellipse(track track_pt, int id_count){
  cv::Mat S = track_pt.kf.measurementMatrix * track_pt.kf.errorCovPre * track_pt.kf.measurementMatrix.t() + track_pt.kf.measurementNoiseCov;
  cv::Mat cov = (Mat_<double>(2,2) << S.at<float>(0,0),S.at<float>(0,1),
                                     S.at<float>(1,0),S.at<float>(1,1));
  cv::Point2f mean(track_pt.history.back().x,track_pt.history.back().y);
  double chisquare_val = 2.4477;  // 95% confidence
  //Get the eigenvalues and eigenvectors
	cv::Mat eigenvalues, eigenvectors;
	bool temp_flag = cv::eigen(cov, eigenvalues, eigenvectors);

	//Calculate the angle between the largest eigenvector and the x-axis
	double angle = atan2(eigenvectors.at<double>(0,1), eigenvectors.at<double>(0,0));

	//Shift the angle to the [0, 2pi] interval instead of [-pi, pi]
	if(angle < 0)
		angle += 6.28318530718;

	//Conver to degrees instead of radians
	angle = 180*angle/3.14159265359;

	//Calculate the size of the minor and major axes
	double halfmajoraxissize=chisquare_val*sqrt(eigenvalues.at<double>(0));
	double halfminoraxissize=chisquare_val*sqrt(eigenvalues.at<double>(1));
  // std::cout << "Cov: \n[" << S.at<float>(0,0)<< ", " << S.at<float>(0,1) << ", " << S.at<float>(0,2) << ",\n"
  //                       << S.at<float>(1,0)<< ", " << S.at<float>(1,1) << ", " << S.at<float>(1,2) << ",\n"
  //                       << S.at<float>(2,0)<< ", " << S.at<float>(2,1) << ", " << S.at<float>(2,2) << "]\n";
  // std::cout << "Ellipse-> major: " << halfmajoraxissize << ", minor: " << halfminoraxissize << std::endl << std::endl;
  visualization_msgs::Marker cov_marker;
  cov_marker.header.frame_id = MAP_FRAME;
  cov_marker.header.stamp = radar_stamp;
  cov_marker.id = id_count;
  cov_marker.type = visualization_msgs::Marker::CYLINDER;
  cov_marker.action = visualization_msgs::Marker::ADD;
  cov_marker.ns = "tracking cov";
  cov_marker.lifetime = ros::Duration(dt);
  cov_marker.scale.x = 1.2*halfmajoraxissize;
  cov_marker.scale.y = 1.2*halfminoraxissize;
  cov_marker.scale.z = 0.1;
  cov_marker.color.a = 0.6;
  cov_marker.color.r = 1.0f;
  cov_marker.color.g = 0;
  cov_marker.color.b = 1.0f;
  cov_marker.pose.position = track_pt.history.back();
  tf2::Quaternion Q;
  Q.setRPY( 0, 0, angle );
  cov_marker.pose.orientation = tf2::toMsg(Q);
  return cov_marker;
}

// show the tracking trajectory
void show_trajectory(){
  int k=0;
  visualization_msgs::MarkerArray tra_array, tra_array_smooth, point_array, pred_point_array, cov_array;
  tra_array.markers.clear();
  point_array.markers.clear();
  pred_point_array.markers.clear();
  cov_array.markers.clear();
  
  // vel marker init
  visualization_msgs::MarkerArray marker_tracking_vel_array;
  int count_vel_num = 0;
  int count_tra_num = 0;
  int count_cov_num = 0;

  for(k=0; k<filters.size(); k++){
    geometry_msgs::Point pred_vel;
    pred_vel.x = filters.at(k).pred_v.x;
    pred_vel.y = filters.at(k).pred_v.y;
    pred_vel.z = filters.at(k).pred_v.z;
    float velocity = sqrt(pred_vel.x*pred_vel.x + pred_vel.y*pred_vel.y + pred_vel.z*pred_vel.z);
    

    // if ( filters.at(k).tracking_state == TRACK_STATE::tracking || filters.at(k).tracking_state == TRACK_STATE::unstable ){
    if (filters.at(k).tracking_state == TRACK_STATE::tracking){
      visualization_msgs::Marker marker, P;
      marker.header.frame_id = MAP_FRAME;
      // marker.header.stamp = ros::Time();//to show every tag  (ros::Time::now()for latest tag)
      marker.header.stamp = radar_stamp;
      marker.action = visualization_msgs::Marker::ADD;
      marker.lifetime = ros::Duration(dt);
      marker.type = visualization_msgs::Marker::LINE_STRIP;
      marker.ns = "trajectory";
      marker.id = count_tra_num++;
      marker.pose.orientation.w = 1.0;
      marker.scale = itri_tra_color.scale;
      marker.color = itri_tra_color.color;

      // history pts marker
      P.header.frame_id = MAP_FRAME;
      P.header.stamp = radar_stamp;
      P.action = visualization_msgs::Marker::ADD;
      P.lifetime = ros::Duration(dt);
      P.type = visualization_msgs::Marker::POINTS;
      P.ns = "trajectory";
      P.id = k+1;
      P.scale = history_pt_color.scale;
      P.color = history_pt_color.color;

      visualization_msgs::Marker marker_smooth = marker;

      if (filters.at(k).history.size() < trajectory_frame){
        for (int i=0; i<filters.at(k).history.size(); i++){
            geometry_msgs::Point pt = filters.at(k).history.at(i);
            // pt.z = 0;
            marker.points.push_back(pt);
            marker_smooth.points.push_back(filters.at(k).history_smooth.at(i));
            P.points.push_back(pt);
        }
      }
      else{
        for (vector<geometry_msgs::Point>::const_reverse_iterator r_iter = filters.at(k).history.rbegin(); r_iter != filters.at(k).history.rbegin() + trajectory_frame; ++r_iter){
            geometry_msgs::Point pt = *r_iter;
            // pt.z = 0;
            marker.points.push_back(pt);
            P.points.push_back(pt);
        }
        for (vector<geometry_msgs::Point>::const_reverse_iterator r_iter = filters.at(k).history_smooth.rbegin(); r_iter != filters.at(k).history_smooth.rbegin() + trajectory_frame; ++r_iter){
            geometry_msgs::Point pt = *r_iter;
            marker_smooth.points.push_back(pt);
        }
      }
      // show the covariance
      if(filters.at(k).history.size()!=0){
        visualization_msgs::Marker cov_marker = get_ellipse(filters.at(k),count_cov_num++);
        cov_array.markers.push_back(cov_marker);
      }
      

      // predicted pose (wrong, its the predicted pose for current frame, not next)
      visualization_msgs::Marker Predict_p;
      Predict_p.header.frame_id = MAP_FRAME;
      Predict_p.header.stamp = radar_stamp;
      Predict_p.action = visualization_msgs::Marker::ADD;
      Predict_p.lifetime = ros::Duration(dt);
      Predict_p.type = visualization_msgs::Marker::POINTS;
      Predict_p.id = k+2;
      Predict_p.scale = pred_pt_color.scale;
      Predict_p.color = pred_pt_color.color;
      geometry_msgs::Point pred = filters.at(k).pred_pose;
      Predict_p.points.push_back(pred);

      if(filters.at(k).motion == MOTION_STATE::stop && !show_stopObj_id){
        marker.color.a = 0;
        marker_smooth.color.a = 0;
        P.color.a = 0;
        Predict_p.color.a = 0;
      }

      tra_array.markers.push_back(marker);
      tra_array_smooth.markers.push_back(marker_smooth);
      point_array.markers.push_back(P);
      pred_point_array.markers.push_back(Predict_p);

      // Show the tracking velocity
      if(filters.at(k).track_frame > 5){
        visualization_msgs::Marker vel_marker;
        vel_marker.header.frame_id = MAP_FRAME;
        vel_marker.header.stamp = radar_stamp;
        vel_marker.id = count_vel_num++;
        vel_marker.type = visualization_msgs::Marker::ARROW;
        vel_marker.action = visualization_msgs::Marker::ADD;
        vel_marker.ns = "tracking vel";
        vel_marker.lifetime = ros::Duration(dt);
        int hist_size = filters.at(k).history.size();
        if(hist_size!=0){
          vel_marker.pose.position.x = filters.at(k).history.at(hist_size-1).x;
          vel_marker.pose.position.y = filters.at(k).history.at(hist_size-1).y;
          vel_marker.pose.position.z = filters.at(k).history.at(hist_size-1).z;
        }
        else{
          vel_marker.pose.position.x = filters.at(k).pred_pose.x;
          vel_marker.pose.position.y = filters.at(k).pred_pose.y;
          vel_marker.pose.position.z = filters.at(k).pred_pose.z;
        }
        
        float theta = atan2(pred_vel.y,
                            pred_vel.x);
        tf2::Quaternion Q;
        Q.setRPY( 0, 0, theta );
        vel_marker.pose.orientation = tf2::toMsg(Q);
        vel_marker.scale = itri_vel_color.scale;
        vel_marker.scale.x = sqrt(pow(pred_vel.y,2) + 
                              pow(pred_vel.x,2)); //~lenght~//
        
        vel_marker.color = itri_vel_color.color;
        vel_marker.color.a = 1.0;
        marker_tracking_vel_array.markers.push_back(vel_marker);
      }
      
    }
    
  }
  if(count_tra_num > trajcetory_max_size)
    trajcetory_max_size = count_tra_num;
  for(int a = count_tra_num;a<trajcetory_max_size;a++){
    visualization_msgs::Marker marker;
    marker.header.frame_id = MAP_FRAME;
    marker.header.stamp = radar_stamp;
    marker.lifetime = ros::Duration(dt);
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.ns = "trajectory";
    marker.id = a;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.2f;
    marker.scale.y = 0.2f;
    marker.color.a = 0;
    tra_array.markers.push_back(marker);
    tra_array_smooth.markers.push_back(marker);
  }

  if(count_vel_num > tracking_vel_marker_max_size)
    tracking_vel_marker_max_size = count_vel_num;
  for(int a=count_vel_num;a<vel_marker_max_size;a++){
    visualization_msgs::Marker marker;
		marker.header.frame_id = MAP_FRAME;
		marker.header.stamp = radar_stamp;
    marker.lifetime = ros::Duration(dt);
		marker.id = a;
		marker.type = visualization_msgs::Marker::ARROW;
		marker.action = visualization_msgs::Marker::DELETE;
		marker.pose.position.x = 0;
		marker.pose.position.y = 0;
		marker.pose.position.z = 1;
    marker.color.a = 0;
		marker_tracking_vel_array.markers.push_back(marker);
  }

  if(count_cov_num > cov_marker_max_size){
    cov_marker_max_size = count_cov_num;
  }
  for(int a=count_cov_num;a<cov_marker_max_size;a++){
    visualization_msgs::Marker marker;
		marker.header.frame_id = MAP_FRAME;
		marker.header.stamp = radar_stamp;
    marker.lifetime = ros::Duration(dt);
		marker.id = a;
		marker.type = visualization_msgs::Marker::CYLINDER;
		marker.action = visualization_msgs::Marker::DELETE;
		marker.pose.position.x = 0;
		marker.pose.position.y = 0;
		marker.pose.position.z = 1;
    marker.color.a = 0;
		cov_array.markers.push_back(marker);
  }
  pub_trajectory.publish(tra_array);
  pub_trajectory_smooth.publish(tra_array_smooth);
  pub_pt.publish(point_array);
  pub_pred.publish(pred_point_array);
  vis_tracking_vel.publish(marker_tracking_vel_array);
  pub_tracker_cov.publish(cov_array);
}

// show the past radar points with colors
void show_past_radar_points_with_color(std::vector<bool> cluster_tracking_bool_vec){
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr history_filter_points_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
  sensor_msgs::PointCloud2::Ptr history_filter_cloud(new sensor_msgs::PointCloud2);
  for(int i=0;i<past_radars_with_cluster.size();i++){
    for(int j=0;j<past_radars_with_cluster.at(i).size();j++){
      for(int k=0;k<past_radars_with_cluster.at(i).at(j).size();k++){
        pcl::PointXYZRGB pt;
        pt.x = past_radars_with_cluster.at(i).at(j).at(k).x;
        pt.y = past_radars_with_cluster.at(i).at(j).at(k).y;
        pt.z = past_radars_with_cluster.at(i).at(j).at(k).z;
        
        if(cluster_tracking_bool_vec.at(i)){
          // green to dark red
          pt.r = 56+184/cluster_history_frames*(cluster_history_frames-1-j);
          pt.g = 239-192/cluster_history_frames*(cluster_history_frames-1-j);
          pt.b = 125-89/cluster_history_frames*(cluster_history_frames-1-j);
        }
        else{
          // dark green
          pt.r = 51;
          pt.g = 92;
          pt.b = 43;
        }
        history_filter_points_rgb->points.push_back(pt);
      }
    }
  }
  pcl::toROSMsg(*history_filter_points_rgb,*history_filter_cloud);
  if(get_transformer)
    history_filter_cloud->header.frame_id = MAP_FRAME;
  else
    history_filter_cloud->header.frame_id = RADAR_FRONT_FRAME;
  history_filter_cloud->header.stamp = radar_stamp;
  pub_filter.publish(history_filter_cloud);
}

// show the red line to debug the invalid state radar points
void radar_debug_visualization(void){
  visualization_msgs::MarkerArray radar_front_fov_arrary;
  visualization_msgs::Marker line1, line2, line3, line4;
  line1.header.frame_id = RADAR_FRONT_FRAME;
  line1.header.stamp = radar_stamp;
  // line1.lifetime = ros::Duration(dt);
  line1.type = visualization_msgs::Marker::LINE_STRIP;
  line1.ns = "radar_debug_range";
  line1.id = 0;
  line1.pose.orientation.w = 1.0;
  line1.scale.x = 0.1;
  line1.scale.y = 0.1;
  line1.scale.z = 0.1;
  line1.color.a = 1.0;
  line1.color.g = 93/255;
  line1.color.r = 1.0;
  line1.action = visualization_msgs::Marker::ADD;
  line4 = line3 = line2 = line1;
  line2.id = 1;
  line3.id = 2;
  line4.id = 3;
  geometry_msgs::Point origin,left_pt,right_pt;
  origin.x = origin.y = origin.z = left_pt.z = right_pt.z = 0;
  left_pt.x = 100*cos(-60*M_PI/180);
  left_pt.y = 100*sin(-60*M_PI/180);
  right_pt.x = 100*cos(60*M_PI/180);
  right_pt.y = 100*sin(60*M_PI/180);
  line1.points.push_back(origin);
  line1.points.push_back(left_pt);
  line2.points.push_back(origin);
  line2.points.push_back(right_pt);
  left_pt.x = 100*cos(-10*M_PI/180);
  left_pt.y = 100*sin(-10*M_PI/180);
  right_pt.x = 100*cos(10*M_PI/180);
  right_pt.y = 100*sin(10*M_PI/180);
  line3.points.push_back(origin);
  line3.points.push_back(left_pt);
  line4.points.push_back(origin);
  line4.points.push_back(right_pt);
  radar_front_fov_arrary.markers.push_back(line1);
  radar_front_fov_arrary.markers.push_back(line2);
  radar_front_fov_arrary.markers.push_back(line3);
  radar_front_fov_arrary.markers.push_back(line4);
  pub_radar_debug.publish(radar_front_fov_arrary);
}

// calculate euclidean distance of two points
double euclidean_distance(geometry_msgs::Point p1, geometry_msgs::Point p2)
{
  return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}

double mahalanobis_distance(geometry_msgs::Point p1, geometry_msgs::Point p2,cv::Mat cov){
  cv::Vec3f x(p1.x,p1.y,p1.z);    // measurementNoiseCov:3x3 matrix
  cv::Vec3f y(p2.x,p2.y,p2.z);
  
  cv::Mat S = (Mat_<float>(3,3) << cov.at<float>(0,0),cov.at<float>(0,1),cov.at<float>(0,2),
                                   cov.at<float>(1,0),cov.at<float>(1,1),cov.at<float>(1,2),
                                   cov.at<float>(2,0),cov.at<float>(2,1),cov.at<float>(2,2));
  
  double distance = cv::Mahalanobis(x,y,S);
  return (distance);
}

typedef struct DA_match
{
  double min;
  int index;
}DA_match;

vector<int> find_min(vector<vector<double>>distMat){
  std::vector< std::vector<double> > dist_mat = distMat;
  std::vector<DA_match> match_list;
  int filter_size = dist_mat.size();
  int radar_num = dist_mat[0].size();
  int match_array[filter_size]={0}; // 紀錄是否有超過一次被紀錄最小值
  vector<int> match_mat[filter_size];
  bool try_again = false;
  for(int j = 0;j<radar_num;j++){
    double min = 10000;             // avoid to be 1000(used below)
    int index = -1;
    for(int i = 0;i<filter_size;i++){
      if(min>dist_mat[i][j]){
        min = dist_mat[i][j];
        index = i;
      }
    }
    match_mat[index].push_back(j);
    DA_match da;
    da.min = min;
    da.index = index;
    match_list.push_back(da);
    if(min==1000)                 // min=1000 implies the column doesn't match(#filter < #radar)
      continue;
    else if(++match_array[index]>1)
      try_again = true;
  }
  if(try_again){
    for(int i =0;i<filter_size;i++){
      if(match_array[i]>1){
        vector<double> row;
        for(int j = 0;j<match_mat[i].size();j++){
          row.push_back(distMat[i][match_mat[i][j]]);
        }
        auto smallest = min_element(row.begin(),row.end());
        int index = distance(row.begin(),smallest);
        for(int j = 0;j<match_mat[i].size();j++){
          if(j!=index)
            dist_mat[i][match_mat[i][j]] = 1000;
          
        }
      }
    }
    // std::cout<<"\nDA distMat:\n";
    // for(int i =0 ;i<filter_size;i++){
    //   std::cout<<"Row "<<i<<":";
    //   for(int j=0;j<radar_num;j++){
    //     std::cout<<"\t"<<dist_mat[i][j];
    //   }
    //   std::cout<<endl;
    // }
    vector<int> assign = find_min(dist_mat);
    return assign;
  }
  else{
    vector<int> assign(filter_size,-1);
    for(int i =0;i<match_list.size();i++){
      if(match_list[i].min!=1000)
        assign[match_list[i].index] = i;
    }
    return assign;
  }
}

int new_track(kf_tracker_point &cen, int idx){
  track tk;
  cv::KalmanFilter ka;
  ka.init(stateDim,measDim,ctrlDim,CV_32F);
  ka.transitionMatrix = (Mat_<float>(6, 6) << 1,0,0,dt,0,0,
                                              0,1,0,0,dt,0,
                                              0,0,1,0,0,dt,
                                              0,0,0,1,0,0,
                                              0,0,0,0,1,0,
                                              0,0,0,0,0,1);
  cv::setIdentity(ka.measurementMatrix);
  cv::setIdentity(ka.processNoiseCov, Scalar::all(sigmaP));
  cv::setIdentity(ka.measurementNoiseCov, cv::Scalar::all(sigmaQ));
  cv::setIdentity(ka.errorCovPost,cv::Scalar::all(0.1));
  ka.statePost.at<float>(0)=cen.x;
  ka.statePost.at<float>(1)=cen.y;
  ka.statePost.at<float>(2)=cen.z;
  ka.statePost.at<float>(3)=cen.x_v;// initial v_x
  ka.statePost.at<float>(4)=cen.y_v;// initial v_y
  ka.statePost.at<float>(5)=cen.z_v;// initial v_z

  // errorCovPost seems not work well in radar case?
  // ka.errorCovPost = (Mat_<float>(6, 6) << 1,0,0,0,0,0,
  //                                         0,1,0,0,0,0,
  //                                         0,0,1,0,0,0,
  //                                         0,0,0,1000.0,0,0,
  //                                         0,0,0,0,1000.0,0,
  //                                         0,0,0,0,0,1000.0);
  //predict phase to generate statePre( state X(K|K-1) ), to correct to X(K|K) 
  geometry_msgs::Point pt, pt_v;
  cv::Mat pred;
  pred = ka.predict();
  pt.x = pred.at<float>(0);
  pt.y = pred.at<float>(1);
  pt.z = pred.at<float>(2);
  pt_v.x = pred.at<float>(3);
  pt_v.y = pred.at<float>(4);
  pt_v.z = pred.at<float>(5);
  tk.pred_pose = pt;
  tk.pred_v = pt_v;
  tk.pred_pose_us = pt;
  tk.pred_v_us = pt_v;


  tk.kf = ka;
  tk.kf_us = ka;
  tk.tracking_state = TRACK_STATE::tracking;
  if(cen.vel < motion_vel_threshold){
    tk.motion = MOTION_STATE::stop;
    cen.motion = MOTION_STATE::stop;
  }
  else{
    tk.motion = MOTION_STATE::move;
    cen.motion = MOTION_STATE::move;
  }
  tk.lose_frame = 0;
  tk.track_frame = 0;
  tk.cluster_idx = idx;

  tk.uuid = ++id_count;
  filters.push_back(tk);
  // std::cout<<"Done init newT at "<<id_count<<" is ("<<tk.pred_pose.x <<"," <<tk.pred_pose.y<<")"<<endl;
  return tk.uuid;
}

void KFT(void)
{
  // get the predict result from KF, and each filter is one points(from past on)
  for(int i=0 ;i<filters.size() ;i++){
    geometry_msgs::Point pt, pt_v;
    cv::Mat pred;
    cv::KalmanFilter k = filters.at(i).kf;
    pred = k.predict();
    pt.x = pred.at<float>(0);
    pt.y = pred.at<float>(1);
    pt.z = pred.at<float>(2);
    pt_v.x = pred.at<float>(3);
    pt_v.y = pred.at<float>(4);
    pt_v.z = pred.at<float>(5);
    filters.at(i).pred_pose = pt;
    filters.at(i).pred_v = pt_v;

    geometry_msgs::Point pt_us, pt_v_us;
    cv::Mat pred_us;
    cv::KalmanFilter k_us = filters.at(i).kf_us;
    pred_us = k_us.predict();
    pt_us.x = pred_us.at<float>(0);
    pt_us.y = pred_us.at<float>(1);
    pt_us.z = pred_us.at<float>(2);
    pt_v.x = pred_us.at<float>(3);
    pt_v.y = pred_us.at<float>(4);
    pt_v.z = pred_us.at<float>(5);
    filters.at(i).pred_pose_us = pt_us;
    filters.at(i).pred_v_us = pt_v;
  }

  // Get measurements
  // Extract the position of the clusters forom the multiArray. To check if the data
  // coming in, check the .z (every third) coordinate and that will be 0.0
  std::vector<geometry_msgs::Point> clusterCenters;//clusterCenters
  pcl::PointCloud<pcl::PointXYZ>::Ptr center_points(new pcl::PointCloud<pcl::PointXYZ>);
  sensor_msgs::PointCloud2::Ptr center_cloud(new sensor_msgs::PointCloud2);
  center_points->clear();
  int i=0;
  if(output_cluster_info)
    std::cout << "Now center radar points :"<<cens.size()<<endl;  // The total radar outlier measurements now!
  for(i; i<cens.size(); i++){
    geometry_msgs::Point pt;
    pt.x=cens[i].x;
    pt.y=cens[i].y;
    pt.z=cens[i].z;
    pcl::PointXYZ center_pt;
    center_pt.x = pt.x;
    center_pt.y = pt.y;
    center_pt.z = pt.z;
    center_points->push_back(center_pt);

    if(output_cluster_info){
      std::cout << i+1 << ": \t(" << pt.x << "," << pt.y << "," << pt.z << ")";
      std::cout << "\tvel: (" << cens[i].vel << cens[i].x_v << "," << cens[i].y_v << ")\n";
    }
    clusterCenters.push_back(pt);
  }
  pcl::toROSMsg(*center_points,*center_cloud);
  center_cloud->header.frame_id = MAP_FRAME;
  center_cloud->header.stamp = radar_stamp;
  pub_cluster_center.publish(center_cloud);

  std::vector<geometry_msgs::Point> KFpredictions;
  i=0;
  


  //construct dist matrix (mxn): m tracks, n clusters.
  std::vector<geometry_msgs::Point> copyOfClusterCenters(clusterCenters);
  std::vector<std::vector<double> > distMat;
  // std::cout<<"distMat:\n";
  // for(int i=0;i<cens.size();i++)
  //   std::cout<<"\t"<<i;
  // std::cout<<endl;
  int row_count = 0;
  for(std::vector<track>::const_iterator it = filters.begin (); it != filters.end (); ++it)
  {
    std::vector<double> distVec;
    for(int n=0;n<cens.size();n++)
    {
      /*  Decide the covariance matrix
          errorCovPost(P):posteriori error estimate covariance matrix
          errorCovPre(P'):priori error estimate covariance matrix
          measurementNoiseCov(R):measurement noise covariance matrix
          processNoiseCov(Q):process noise covariance matrix
      */
      cv::Mat S = (*it).kf.measurementMatrix * (*it).kf.errorCovPre * (*it).kf.measurementMatrix.t() + (*it).kf.measurementNoiseCov;
      cv::Mat S_us = (*it).kf_us.measurementMatrix * (*it).kf_us.errorCovPre * (*it).kf_us.measurementMatrix.t() + (*it).kf_us.measurementNoiseCov;
      double dis_1 = mahalanobis_distance((*it).pred_pose,copyOfClusterCenters[n],S.inv());
      double dis_2 = mahalanobis_distance((*it).pred_pose_us,copyOfClusterCenters[n],S_us.inv());
      // double dis_1 = mahalanobis_distance((*it).pred_pose,copyOfClusterCenters[n],(*it).kf.measurementNoiseCov);
      // double dis_2 = mahalanobis_distance((*it).pred_pose_us,copyOfClusterCenters[n],(*it).kf_us.measurementNoiseCov);
      // std::cout<<"Mahalanobis:"<<ma_dis1<<" ,"<<ma_dis2<<endl;
      if (dis_1<dis_2){
        distVec.push_back(dis_1);
      }
      else
      {
        distVec.push_back(dis_2);
      }
      
      // distVec.push_back(dis_1);
      // distVec.push_back(euclidean_distance((*it).pred_pose,copyOfClusterCenters[n]));
      // std::cout<<"\t"<<distVec.at(n);
    }
    distMat.push_back(distVec);

  }

  vector<int> assignment;
  if(DA_choose){
    //hungarian method to optimize(minimize) the dist matrix
    HungarianAlgorithm HungAlgo;
    double cost = HungAlgo.Solve(distMat, assignment);
    
    if(output_DA_pair){
      std::cout<<"HungAlgo assignment pair:\n";
      for (unsigned int x = 0; x < distMat.size(); x++)
        std::cout << x << "," << assignment[x] << "\t";
      std::cout << "\ncost: " << cost << std::endl; // HungAlgo computed cost
    }
  }
  else{
    // my DA method
    assignment = find_min(distMat);
    if(output_DA_pair){
      std::cout<<"My DA assignment pair:\n";
      for (unsigned int x = 0; x < distMat.size(); x++)
        std::cout << x << "," << assignment[x] << "\t";
      std::cout<<endl;
    }
  }

  std::vector<int> obj_id(cens.size(),-1);
  std::vector<int> tracking_frame_obj_id(cens.size(),-1);
  
  // record the cluster tracking result
  std::vector<bool> cluster_tracking_bool_vec(past_radars_with_cluster.size(),false);

  int k=0;
  for(k=0; k<filters.size(); k++){
    std::vector<double> dist_vec = distMat.at(k); //float
    geometry_msgs::Point pred_v = filters.at(k).pred_v;
    geometry_msgs::Point pred_pos = filters.at(k).pred_pose;
    geometry_msgs::Point pred_v_us = filters.at(k).pred_v_us;
    geometry_msgs::Point pred_pos_us = filters.at(k).pred_pose_us;
    geometry_msgs::Point v_mean;
    v_mean.x = 0;
    v_mean.y = 0;
    v_mean.z = 0;
    
    float delta_x = (pred_pos.x-filters.at(k).kf.statePost.at<float>(0));
    float delta_y = (pred_pos.y-filters.at(k).kf.statePost.at<float>(1));
    // float dist_thres1_ma = mahalanobis_distance(pred_v,v_mean,filters.at(k).kf.measurementNoiseCov);
    // float dist_thres2_ma = mahalanobis_distance(pred_v_us,v_mean,filters.at(k).kf_us.measurementNoiseCov);
    // cv::Mat S = filters.at(k).kf.measurementMatrix * filters.at(k).kf.errorCovPre * filters.at(k).kf.measurementMatrix.t() + filters.at(k).kf.measurementNoiseCov;
    // cv::Mat S_us = filters.at(k).kf_us.measurementMatrix * filters.at(k).kf_us.errorCovPre * filters.at(k).kf_us.measurementMatrix.t() + filters.at(k).kf_us.measurementNoiseCov;
    // float dist_thres1_ma = mahalanobis_distance(pred_v,v_mean,S.inv());
    // float dist_thres2_ma = mahalanobis_distance(pred_v_us,v_mean,S_us.inv());
    float dist_thres1 = sqrt(pred_v.x * pred_v.x * dt * dt + pred_v.y * pred_v.y * dt * dt);
    float dist_thres2 = sqrt(pred_v_us.x * pred_v_us.x * dt * dt + pred_v_us.y * pred_v_us.y * dt * dt);
    
    // float dist_thres;
    // if(dist_thres1<dist_thres2){
    //   dist_thres = dist_thres2;
    // }
    // else
    //   dist_thres = dist_thres1;
    float dist_thres = sqrt(delta_x * delta_x + delta_y * delta_y);
    if(output_KFT_result){
      std::cout<<"------------------------------------------------\n";
      // float dist_thres = sqrt(delta_x * delta_x + delta_y * delta_y);
      std::cout << "The dist_thres for " <<filters.at(k).uuid<<" is "<<dist_thres<<endl;
      std::cout<<"predict v:"<<pred_v.x<<","<<pred_v.y<<endl;
      std::cout<<"predict pos:"<<pred_pos.x<<","<<pred_pos.y<<endl;
      std::cout<<"predict v(us):"<<pred_v_us.x<<","<<pred_v_us.y<<endl;
      std::cout<<"predict pos(us):"<<pred_pos_us.x<<","<<pred_pos_us.y<<endl;
    }
    
    //-1 for non matched tracks
    if ( assignment[k] != -1 ){
      // if( dist_vec.at(assignment[k]) <=  dist_thres + 1.5 ){//bias as gating function to filter the impossible matched detection 
      Eigen::Vector3d dist_diff(cens.at(assignment[k]).x-pred_pos.x,cens.at(assignment[k]).y-pred_pos.y,0);
      if( dist_vec.at(assignment[k]) <=  mahalanobis_bias && (dist_diff.norm() <= 30*dt)){
        obj_id[assignment[k]] = filters.at(k).uuid;
        if(filters.at(k).track_frame > tracking_stable_frames)
          tracking_frame_obj_id[assignment[k]] = filters.at(k).uuid;
        filters.at(k).cluster_idx = assignment[k];
        filters.at(k).tracking_state = TRACK_STATE::tracking;
        // update the KF state
        filters.at(k).kf_us.statePost.at<float>(0)=cens.at(assignment[k]).x;
        filters.at(k).kf_us.statePost.at<float>(1)=cens.at(assignment[k]).y;
        filters.at(k).kf_us.statePost.at<float>(3)=cens.at(assignment[k]).x_v;
        filters.at(k).kf_us.statePost.at<float>(4)=cens.at(assignment[k]).y_v;
        // check the traker's motion state
        if(cens.at(assignment[k]).vel < motion_vel_threshold && filters.at(k).motion != MOTION_STATE::stop){
          filters.at(k).motion = MOTION_STATE::slow_down;
          cens.at(assignment[k]).motion = MOTION_STATE::slow_down;
        }
        else if(cens.at(assignment[k]).vel < motion_vel_threshold){
          filters.at(k).motion = MOTION_STATE::stop;
          cens.at(assignment[k]).motion = MOTION_STATE::stop;
        }
        else
        {
          filters.at(k).motion = MOTION_STATE::move;
          cens.at(assignment[k]).motion = MOTION_STATE::move;
          
        }
        
        geometry_msgs::Point tracking_pt_history;
        tracking_pt_history.x = cens.at(assignment[k]).x;
        tracking_pt_history.y = cens.at(assignment[k]).y;
        tracking_pt_history.z = cens.at(assignment[k]).z;
        filters.at(k).history.push_back(tracking_pt_history);

        geometry_msgs::Point tracking_pt_history_smooth;
        tracking_pt_history_smooth.x = filters.at(k).kf.statePre.at<float>(0);
        tracking_pt_history_smooth.y = filters.at(k).kf.statePre.at<float>(1);
        tracking_pt_history_smooth.z = filters.at(k).kf.statePre.at<float>(2);
        filters.at(k).history_smooth.push_back(tracking_pt_history_smooth);

        distMat[k]=std::vector<double>(cens.size(),10000.0); //float
        for(int row=0;row<distMat.size();row++)//set the column to a high number
        {
            distMat[row][assignment[k]]=10000.0;
        }  
      }
      else
      {
        filters.at(k).tracking_state = TRACK_STATE::missing;
      }
      if(filters.at(k).track_frame>10)
        cluster_tracking_bool_vec.at(assignment[k]) = true;
    }
    else
    {
      filters.at(k).tracking_state = TRACK_STATE::missing;
    }
    

    if(output_KFT_result){
      //get tracked or lost
      if (filters.at(k).tracking_state == TRACK_STATE::tracking){
          std::cout<<"The state of "<<k<<" filters is \033[1;32mtracking\033[0m,to cluster_idx "<< filters.at(k).cluster_idx <<" track: "<<filters.at(k).track_frame;
          std::cout<<" ("<<cens.at(assignment[k]).x<<","<<cens.at(assignment[k]).y<<")"<<endl;
      }
      else if (filters.at(k).tracking_state == TRACK_STATE::missing)
          std::cout<<"The state of "<<k<<" filters is \033[1;34mlost\033[0m,to cluster_idx "<< filters.at(k).cluster_idx <<" lost: "<<filters.at(k).lose_frame<<endl;
      else
          std::cout<<"\033[1;31mUndefined state for trackd "<<k<<"\033[0m"<<endl;
    }
  
  }

  if(output_obj_id_result){
    std::cout<<"\033[1;33mThe obj_id:\033[0m\n";
    for (i=0; i<cens.size(); i++){
      std::cout<<obj_id.at(i)<<" ";
      }
    std::cout<<endl;
  }
  vector<int> marker_obj_id = obj_id;
  //initiate new tracks for not-matched(obj_id = -1) cluster
  for (i=0; i<cens.size(); i++){
    if(obj_id.at(i) == -1){
      int track_uuid = new_track(cens.at(i),i);
      obj_id.at(i) = track_uuid;
    }
  }

  if(output_obj_id_result){
    std::cout<<"\033[1;33mThe obj_id after new track:\033[0m\n";
    for (i=0; i<cens.size(); i++){
      std::cout<<obj_id.at(i)<<" ";
      }
    std::cout<<endl;
  }
  //deal with lost tracks and let it remain const velocity -> update pred_pose to meas correct
  for(std::vector<track>::iterator pit = filters.begin (); pit != filters.end ();){
      if( (*pit).tracking_state == TRACK_STATE::missing  )//true for 0
          (*pit).lose_frame += 1;

      if( (*pit).tracking_state == TRACK_STATE::tracking  ){
          (*pit).track_frame += 1;
          (*pit).lose_frame = 0;
      }
      
      if ( (*pit).lose_frame == frame_lost )
          //remove track from filters
          pit = filters.erase(pit);
      else
          pit ++;
          
  }
  std::cout<<"\033[1;33mTracker number: "<<filters.size()<<" tracks.\033[0m\n"<<endl;


  //begin mark
  m_s.markers.clear();
  m_s.markers.shrink_to_fit();

  // marker_id(false,marker_obj_id);   // publish the "tracking" id
  marker_id(false,obj_id,marker_obj_id,tracking_frame_obj_id);       // publish all id
  show_trajectory();
  show_past_radar_points_with_color(cluster_tracking_bool_vec);
  
///////////////////////////////////////////////////estimate

  int num = filters.size();
  float meas[num][measDim];
  float meas_us[num][measDim];
  cv::Mat measMat[num];
  cv::Mat measMat_us[num];
  cv::Mat estimated[num];
  cv::Mat estimated_us[num];
  i = 0;
  for(std::vector<track>::iterator it = filters.begin(); it != filters.end(); ++it){
    if ( (*it).tracking_state == TRACK_STATE::tracking ){
        kf_tracker_point pt = cens[(*it).cluster_idx];
        meas[i][0] = pt.x;
        meas[i][1] = pt.y;
        meas[i][2] = pt.z;
        // meas[i][3] = pt.x_v;
        // meas[i][4] = pt.y_v;
        // meas[i][5] = pt.z_v;
        meas_us[i][0] = pt.x;
        meas_us[i][1] = pt.y;
        meas_us[i][2] = pt.z;
        // meas_us[i][3] = pt.x_v;
        // meas_us[i][4] = pt.y_v;
        // meas_us[i][5] = pt.z_v;
       
    }
    else if ( (*it).tracking_state == TRACK_STATE::missing ){
        meas[i][0] = (*it).pred_pose.x; //+ (*it).pred_v.x;
        meas[i][1] = (*it).pred_pose.y;//+ (*it).pred_v.y;
        meas[i][2] = (*it).pred_pose.z; //+ (*it).pred_v.z;
        // meas[i][3] = (*it).pred_v.x;
        // meas[i][4] = (*it).pred_v.y;
        // meas[i][5] = (*it).pred_v.z;
        meas_us[i][0] = (*it).pred_pose_us.x;
        meas_us[i][1] = (*it).pred_pose_us.y;
        meas_us[i][2] = (*it).pred_pose_us.z;
        // meas_us[i][3] = (*it).pred_v_us.x;
        // meas_us[i][4] = (*it).pred_v_us.y;
        // meas_us[i][5] = (*it).pred_v_us.z;
    }
    else
    {
        std::cout<<"Some tracks state not defined to tracking/lost."<<std::endl;
    }
    measMat[i]=cv::Mat(measDim,1,CV_32F,meas[i]);
    measMat_us[i]=cv::Mat(measDim,1,CV_32F,meas_us[i]);
    estimated[i] = (*it).kf.correct(measMat[i]);
    estimated_us[i] = (*it).kf_us.correct(measMat_us[i]);
    i++;
  }

  // cv::Mat measMat[num];
  // cv::Mat measMat_us[num];
  // for(int i=0;i<num;i++){
  //   measMat[i]=cv::Mat(measDim,1,CV_32F,meas[i]);
  //   measMat_us[i]=cv::Mat(measDim,1,CV_32F,meas_us[i]);
  // }

  // The update phase 
  
  // if (!(measMat[0].at<float>(0,0)==0.0f || measMat[0].at<float>(1,0)==0.0f))
  //     Mat estimated0 = KF[0].correct(measMat[0]);
  // cv::Mat estimated[num];
  // cv::Mat estimated_us[num];
  // i = 0;
  // for(std::vector<track>::iterator it = filters.begin(); it != filters.end(); ++it){
  //   estimated[i] = (*it).kf.correct(measMat[i]);
  //   estimated_us[i] = (*it).kf_us.correct(measMat_us[i]);
  //   // std::cout << "The corrected state of "<<i<<"th KF is: "<<estimated[i].at<float>(0)<<","<<estimated[i].at<float>(1)<<endl;
  //   i++;
  // }

  return;
}

void get_transform(ros::Time timestamp){
  // get the tf from radar to car
  if(firstFrame){
    tf_listener->waitForTransform(CAR_FRAME,RADAR_FRONT_FRAME,timestamp,ros::Duration(1));
    std::cout << std::fixed; std::cout.precision(3);
    try{
      tf_listener->lookupTransform(CAR_FRAME,RADAR_FRONT_FRAME,timestamp,car_transform);
      if(output_transform_info){
        std::cout << "\033[1;33mGet Transform:\033[0m" << std::endl;
        std::cout << "- At time: " << car_transform.stamp_.toSec() << std::endl;
        tf::Quaternion q = car_transform.getRotation();
        tf::Vector3 v = car_transform.getOrigin();
        std::cout << "- Translation: [" << v.getX() << ", " << v.getY() << ", " << v.getZ() << "]" << std::endl;
        std::cout << "- Rotation: in Quaternion: [" << q.getX() << ", " << q.getY() << ", " 
                  << q.getZ() << ", " << q.getW() << "]" << std::endl;
        std::cout<<"- Frame id: "<<car_transform.frame_id_<<", Child id: "<<car_transform.child_frame_id_<<endl;
      }
    }
    catch(tf::TransformException& ex)
    {
      // std::cout << "Failure at "<< ros::Time::now() << std::endl;
      std::cout << "Exception thrown:" << ex.what()<< std::endl;
      // std::cout << "The current list of frames is:" <<std::endl;
      // std::cout << tf_listener.allFramesAsString()<<std::endl;
      
    }
  }
  
  // get the tf from car to map
  tf_listener->waitForTransform(MAP_FRAME,CAR_FRAME,timestamp,ros::Duration(1));
  std::cout << std::fixed; std::cout.precision(3);
  try{
    tf_listener->lookupTransform(MAP_FRAME,CAR_FRAME,timestamp,map_transform);
    if(output_transform_info){
      std::cout << "\033[1;33mGet Transform:\033[0m" << std::endl;
      std::cout << "- At time: " << map_transform.stamp_.toSec() << std::endl;
      tf::Quaternion q = map_transform.getRotation();
      tf::Vector3 v = map_transform.getOrigin();
      std::cout << "- Translation: [" << v.getX() << ", " << v.getY() << ", " << v.getZ() << "]" << std::endl;
      std::cout << "- Rotation: in Quaternion: [" << q.getX() << ", " << q.getY() << ", " 
                << q.getZ() << ", " << q.getW() << "]" << std::endl;
      std::cout<<"- Frame id: "<<map_transform.frame_id_<<", Child id: "<<map_transform.child_frame_id_<<endl;
    }
  }
  catch(tf::TransformException& ex)
  {
    // std::cout << "Failure at "<< ros::Time::now() << std::endl;
    std::cout << "Exception thrown:" << ex.what()<< std::endl;
    // std::cout << "The current list of frames is:" <<std::endl;
    // std::cout << tf_listener.allFramesAsString()<<std::endl;
  }

  // test tf from radar to map
  // tf_listener->waitForTransform(MAP_FRAME,RADAR_FRONT_FRAME,timestamp,ros::Duration(1));
  // tf::StampedTransform temp;
  // try{
  //   tf_listener->lookupTransform(MAP_FRAME,RADAR_FRONT_FRAME,timestamp,temp);
  //   std::cout << "\033[1;33mGet Transform frome tf:\033[0m" << std::endl;
  //   std::cout << "- At time: " << temp.stamp_.toSec() << std::endl;
  //   tf::Quaternion q = temp.getRotation();
  //   tf::Vector3 v = temp.getOrigin();
  //   std::cout << "- Translation: [" << v.getX() << ", " << v.getY() << ", " << v.getZ() << "]" << std::endl;
  //   std::cout << "- Rotation: in Quaternion: [" << q.getX() << ", " << q.getY() << ", " 
  //             << q.getZ() << ", " << q.getW() << "]" << std::endl;
  //   std::cout<<"- Frame id: "<<temp.frame_id_<<", Child id: "<<temp.child_frame_id_<<endl;
  // }
  // catch(tf::TransformException& ex)
  // {
  //   std::cout << "Exception thrown:" << ex.what()<< std::endl;
  // }
  
  tf::Transform A(map_transform.getBasis(), map_transform.getOrigin());
  tf::Transform B(car_transform.getBasis(), car_transform.getOrigin());
  tf::Transform C = A*B;
  echo_transform.stamp_ = map_transform.stamp_;
  echo_transform.frame_id_ = map_transform.frame_id_;
  echo_transform.child_frame_id_ = car_transform.child_frame_id_;
  echo_transform.setOrigin(C.getOrigin());
  echo_transform.setBasis(C.getBasis());

  // test tf from 2 transform
  // std::cout << "\033[1;33mGet Transform frome 2 transform:\033[0m" << std::endl;
  // std::cout << "- At time: " << echo_transform.stamp_.toSec() << std::endl;
  // tf::Quaternion q = echo_transform.getRotation();
  // tf::Vector3 v = echo_transform.getOrigin();
  // std::cout << "- Translation: [" << v.getX() << ", " << v.getY() << ", " << v.getZ() << "]" << std::endl;
  // std::cout << "- Rotation: in Quaternion: [" << q.getX() << ", " << q.getY() << ", " 
  //           << q.getZ() << ", " << q.getW() << "]" << std::endl;
  // std::cout<<"- Frame id: "<<echo_transform.frame_id_<<", Child id: "<<echo_transform.child_frame_id_<<endl;
  return;
}

void first_frame_KFT(void){
  int current_id = cens.size();
  for(int i=0; i<current_id;i++){
      track tk;
      cv::KalmanFilter ka;
      ka.init(stateDim,measDim,ctrlDim,CV_32F);
      ka.transitionMatrix = (Mat_<float>(6, 6) << 1,0,0,dt,0,0,
                                                  0,1,0,0,dt,0,
                                                  0,0,1,0,0,dt,
                                                  0,0,0,1,0,0,
                                                  0,0,0,0,1,0,
                                                  0,0,0,0,0,1);
      cv::setIdentity(ka.measurementMatrix);
      cv::setIdentity(ka.processNoiseCov, Scalar::all(sigmaP));
      cv::setIdentity(ka.measurementNoiseCov, cv::Scalar(sigmaQ));
      cv::setIdentity(ka.errorCovPost,cv::Scalar::all(0.1));
      std::cout<<"( "<<cens.at(i).x<<","<<cens.at(i).y<<","<<cens.at(i).z<<")\n";
      ka.statePost.at<float>(0)=cens.at(i).x;
      ka.statePost.at<float>(1)=cens.at(i).y;
      ka.statePost.at<float>(2)=cens.at(i).z;
      ka.statePost.at<float>(3)=cens.at(i).x_v;// initial v_x
      ka.statePost.at<float>(4)=cens.at(i).y_v;// initial v_y
      ka.statePost.at<float>(5)=cens.at(i).z_v;// initial v_z

      tk.kf = ka;
      tk.kf_us = ka;  // ready to update the state
      tk.tracking_state = TRACK_STATE::tracking;
      if(cens.at(i).vel < motion_vel_threshold)
        tk.motion = MOTION_STATE::stop;
      else
        tk.motion = MOTION_STATE::move;
      tk.lose_frame = 0;
      tk.track_frame = 1;

      tk.uuid = i;
      id_count = i;
      filters.push_back(tk);
  }
  std::cout<<"Initiate "<<filters.size()<<" tracks."<<endl;
  std::cout << m_s.markers.size() <<endl;
  m_s.markers.clear();
  std::cout<< "\033[1;33mMarkerarray is empty(1):\033[0m"<<m_s.markers.empty()<< endl;
  std::vector<int> no_use_id;
  marker_id(firstFrame,no_use_id,no_use_id,no_use_id);
  firstFrame=false;
  return;
}

void callback(const itri_msgs::Ars40xObjectsConstPtr& msg, const itri_msgs::CarStateConstPtr& car_msg){
  int radar_point_size = msg->objs.size();
  int radar_id_count = 0;
  std::cout<<"\n\n\033[1;4;100m"<<++call_back_num<<" Call Back radar points:"<<radar_point_size<<"\033[0m"<<endl;
  if(radar_point_size==0) //radar points cannot be zero, it would cause segement fault
    return;
  radar_stamp = msg->header.stamp;
  std::cout << std::fixed; std::cout.precision(3);
  std::cout<<"Radar time:"<<radar_stamp.toSec()<<endl;

  
  get_transform(radar_stamp);

  // republish radar points
  pcl::PointCloud<pcl::PointXYZI>::Ptr in_filter_points(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr out_filter_points(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr history_filter_points(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr history_filter_points_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
  sensor_msgs::PointCloud2::Ptr in_filter_cloud(new sensor_msgs::PointCloud2);
  sensor_msgs::PointCloud2::Ptr out_filter_cloud(new sensor_msgs::PointCloud2);
  sensor_msgs::PointCloud2::Ptr history_filter_cloud(new sensor_msgs::PointCloud2);
  in_filter_points->clear();
  out_filter_points->clear();
  // end republish
  
  cens.clear();
  cluster_cens.clear();
  
  // get the ego velocity
  tf::StampedTransform twist_transform = car_transform;
  twist_transform.setOrigin(tf::Vector3(0,0,0));
  tf::Vector3 trans_twist_vel;
  trans_twist_vel.setValue(car_msg->twist.twist.linear.x, car_msg->twist.twist.linear.y, car_msg->twist.twist.linear.z);
  trans_twist_vel = twist_transform.inverse() * trans_twist_vel;
  vx = trans_twist_vel.x();
  vy = trans_twist_vel.y();
  EGO_INFO("\033[1;33mVehicle velocity:\n\033[0m");
  EGO_INFO("Time stamp:"<<car_msg->header.stamp.toSec()<<endl);
  EGO_INFO("vx: "<<vx<<" ,vy: "<<vy<<" ,vz: "<<car_msg->twist.twist.linear.z<<endl);
  
  if(show_vel_marker){
    vel_marker(msg);
  }
  cens.clear();
  cens.shrink_to_fit();
  // get the velocity transform matrix to map
  tf::StampedTransform vel_transform = echo_transform;
  vel_transform.setOrigin(tf::Vector3(0,0,0));

  RADAR_INFO(endl<<"Radar information\n");
  for (int i = 0;i<msg->objs.size();i++){
    tf::Point trans_pt;
    tf::Vector3 trans_vel;
    trans_pt.setValue(msg->objs[i].pose.position.x, msg->objs[i].pose.position.y, 0);
    trans_pt = echo_transform * trans_pt;
    // get compensated vel
    tf::Vector3 msg_vel(msg->objs[i].velocity.linear.x+vx,
                        msg->objs[i].velocity.linear.y+vy,
                        0);
    tf::Vector3 msg_pos(msg->objs[i].pose.position.x,
                        msg->objs[i].pose.position.y,
                        0);
    msg_pos.normalize();
    tf::Vector3 project_radial_vel = msg_pos.dot(msg_vel) * msg_pos;

    trans_vel.setValue(project_radial_vel.x(), project_radial_vel.y(), 0);
    trans_vel = vel_transform * trans_vel;
    float range = sqrt(pow(msg->objs[i].pose.position.x,2) + 
                        pow(msg->objs[i].pose.position.y,2));
    float velocity = sqrt(pow(project_radial_vel.y(),2) + 
                          pow(project_radial_vel.x(),2));
    double angle = atan2(project_radial_vel.y(),project_radial_vel.x())*180/M_PI;

    RADAR_INFO("----------------------------------------------\n");
    RADAR_INFO(i<<"\tposition:("<<msg->objs[i].pose.position.x<<","<<msg->objs[i].pose.position.y<<")\n");
    RADAR_INFO("\tposition to map:("<<trans_pt.x()<<","<<trans_pt.y()<<","<<trans_pt.z()<<")\n");
    RADAR_INFO("\tvel:"<<velocity<<"("<<project_radial_vel.x()<<","<<project_radial_vel.y()<<")\n\tdegree:"<<angle);
    RADAR_INFO("\n\tinvalid_state:"<<(int)msg->objs[i].id);
    RADAR_INFO("\n\tRCS value:"<<msg->objs[i].radar_cross_section);
    RADAR_INFO(endl);

    // std::list<int>::iterator invalid_state_it;
    // invalid_state_it = std::find(invalid_state.begin(),invalid_state.end(),msg->objs[i].id);
    // // remove the invalid state points
    // if(invalid_state_it!=invalid_state.end()){
    //   radar_point_size--;
    // }
    if(msg->objs[i].probability<80){
      radar_point_size--;
      RADAR_INFO("Low Probability radar point with prob = "<<(int)msg->objs[i].probability);
      pcl::PointXYZI pt;
      pt.x = msg->objs[i].pose.position.x;
      pt.y = msg->objs[i].pose.position.y;
      pt.z = 0;
      out_filter_points->push_back(pt);
    }
    else{
      kf_tracker_point c;
      c.x = trans_pt.x();
      c.y = trans_pt.y();
      c.z = trans_pt.z();
      c.x_v = trans_vel.x();
      c.y_v = trans_vel.y();
      c.z_v = 0;
      c.vel_ang = angle;
      c.vel = velocity;
      c.cluster_flag = -1;  // -1: not clustered
      c.vistited = false;
      c.id = radar_id_count++;
      c.scan_id = call_back_num;
      c.rcs = msg->objs[i].radar_cross_section;
      
      pcl::PointXYZI pt;
      pt.x = c.x;
      pt.y = c.y;
      pt.z = c.z;
      pt.intensity = msg->objs[i].radar_cross_section;
      in_filter_points->push_back(pt);
      pt.x = msg->objs[i].pose.position.x;
      pt.y = msg->objs[i].pose.position.y;
      pt.z = 0;
      out_filter_points->push_back(pt);
      cens.push_back(c);
    }
  }

  if(cens.size() == 0){
    ROS_WARN("No Valid Callback Radar points In This Frame!");
    return;
  }
  if(show_past_points && !use_KFT_module){
    past_radars.push_back(cens);
    if(past_radars.size()>cluster_history_frames){
      past_radars.erase(past_radars.begin());
    }
  }
  // republish filtered radar points
  pcl::toROSMsg(*in_filter_points,*in_filter_cloud);
  if(get_transformer)
    in_filter_cloud->header.frame_id = MAP_FRAME;
  else
    in_filter_cloud->header.frame_id = RADAR_FRONT_FRAME;
  in_filter_cloud->header.stamp = radar_stamp;
  // test radar transform coordinate
  pcl::toROSMsg(*out_filter_points,*out_filter_cloud);
  out_filter_cloud->header.frame_id = RADAR_FRONT_FRAME;
  out_filter_cloud->header.stamp = radar_stamp;
  pub_in_filter.publish(in_filter_cloud);
  pub_out_filter.publish(out_filter_cloud);
  
  std::vector<kf_tracker_point> multi_frame_pts;
  // cluster the points with DBSCAN-based method
  if(cluster_type=="dbscan"){
    double DBSCAN_START, DBSCAN_END;
    DBSCAN_START = clock();
    dbscan_seg.scan_num = call_back_num;
    dbscan_seg.data_period = dt;
    std::vector< std::vector<kf_tracker_point> > dbscan_cluster;
    dbscan_seg.output_info = output_dbscan_info;
    dbscan_cluster = dbscan_seg.cluster(cens);
    DBSCAN_END = clock();
    
    // check the DBSCAN exe performance
    EXE_INFO(endl << "\033[42mDBSCAN Execution time: " << (DBSCAN_END - DBSCAN_START) / CLOCKS_PER_SEC << "s\033[0m");
    EXE_INFO(std::endl);

    // cluster visualization
    color_cluster(dbscan_cluster);
    polygon_cluster_visual(dbscan_cluster);
    
    cens.clear();
    cens = dbscan_seg.get_center();
    
    // publish the history radar points
    multi_frame_pts = dbscan_seg.get_history_points();
  }
  // cluster the points with DBPDA method
  else{
    double DBPDA_START, DBPDA_END;
    DBPDA_START = clock();
    dbpda_seg.scan_num = call_back_num;
    dbpda_seg.data_period = dt;
    dbpda_seg.set_parameter(cluster_eps, cluster_Nmin, cluster_history_frames, cluster_dt_weight);
    std::vector< std::vector<kf_tracker_point> > dbpda_cluster;
    dbpda_seg.output_info = output_dbscan_info;
    dbpda_cluster = dbpda_seg.cluster(cens);
    dbpda_cluster.clear();
    dbpda_cluster = dbpda_seg.cluster_with_past();
    DBPDA_END = clock();
    
    // check the DBSCAN exe performance
    EXE_INFO(endl << "\033[42mDBPDA Execution time: " << (DBPDA_END - DBPDA_START) / CLOCKS_PER_SEC << "s\033[0m");
    EXE_INFO(std::endl);

    // cluster visualization
    color_cluster(dbpda_cluster);
    polygon_cluster_visual(dbpda_cluster);
    
    cens.clear();
    cens = dbpda_seg.get_center();
    
    // publish the history radar points
    multi_frame_pts = dbpda_seg.get_history_points();
    past_radars_with_cluster.clear();
    std::cout << "ready to get past\n";
    past_radars_with_cluster = dbpda_seg.get_history_points_with_cluster_order();
  }

  if(show_past_points && !use_KFT_module){
    for(int i=0;i<past_radars.size();i++){
      for(int j=0;j<past_radars.at(i).size();j++){
        pcl::PointXYZRGB pt;
        pt.x = past_radars.at(i).at(j).x;
        pt.y = past_radars.at(i).at(j).y;
        pt.z = past_radars.at(i).at(j).z;

        // green to dark red
        pt.r = 56+184/past_radars.size()*(past_radars.size()-1-i);
        pt.g = 239-192/past_radars.size()*(past_radars.size()-1-i);
        pt.b = 125-89/past_radars.size()*(past_radars.size()-1-i);

        history_filter_points_rgb->points.push_back(pt);
      }
    }
    pcl::toROSMsg(*history_filter_points_rgb,*history_filter_cloud);
  }
  else if(!use_KFT_module){
    for(int i=0;i<multi_frame_pts.size();i++){
      pcl::PointXYZI pt;
      pt.x = multi_frame_pts.at(i).x;
      pt.y = multi_frame_pts.at(i).y;
      pt.z = multi_frame_pts.at(i).z;
      pt.intensity = multi_frame_pts.at(i).rcs;
      history_filter_points->points.push_back(pt);
    }
    pcl::toROSMsg(*history_filter_points,*history_filter_cloud);
  }
  if(get_transformer)
    history_filter_cloud->header.frame_id = MAP_FRAME;
  else
    history_filter_cloud->header.frame_id = RADAR_FRONT_FRAME;
  history_filter_cloud->header.stamp = radar_stamp;
  pub_filter.publish(history_filter_cloud);
  
  if(cens.size()==0){
    ROS_WARN("No Valid Cluster Radar points In This Frame!");
    return;
  }

  if(use_KFT_module){
    if( firstFrame ){
      ROS_INFO("Initialize the KFT Tracking");
      first_frame_KFT();
      return;
    }
    // check the Tracking exe performance
    double START_TIME, END_TIME;
    START_TIME = clock();
    KFT();
    END_TIME = clock();
    EXE_INFO("\033[1;42mKFT Execution time: " << (END_TIME - START_TIME) / CLOCKS_PER_SEC << "s\033[0m");
    EXE_INFO(std::endl);
  }
  return;
}

void get_bag_info_callback(const rosgraph_msgs::LogConstPtr& log_msg){
  if(!use_score_cluster)
    return;
  if(log_msg->msg.find(".bag") != log_msg->msg.npos){
    filename_change = true;
    score_file_name = log_msg->msg;
    int bag_start = score_file_name.find("2020");
    int bag_end = score_file_name.find(".");
    score_file_name = score_file_name.substr(bag_start,bag_end-bag_start);
  }
  return;
}

int main(int argc, char** argv){
  ros::init(argc,argv,"radar_kf_track");
  ros::NodeHandle nh;
  marker_color_init();
  tf_listener = new tf::TransformListener();
  message_filters::Subscriber<itri_msgs::Ars40xObjects> sub_radar(nh,"/radar_objs",1);
  message_filters::Subscriber<itri_msgs::CarState> sub_car_state(nh,"/car_state",1);

  pub_cluster_marker = nh.advertise<visualization_msgs::MarkerArray>("cluster_index", 1);
  pub_marker = nh.advertise<visualization_msgs::MarkerArray>("tracking_id", 1);
  pub_filter = nh.advertise<sensor_msgs::PointCloud2>("radar_history",500);
  pub_in_filter = nh.advertise<sensor_msgs::PointCloud2>("transform_global_radar",200);
  pub_out_filter = nh.advertise<sensor_msgs::PointCloud2>("radar_points_with_radar_frame",200);
  pub_cluster_center = nh.advertise<sensor_msgs::PointCloud2>("cluster_center",500);
  pub_cluster_pointcloud = nh.advertise<sensor_msgs::PointCloud2>("cluster_radar_front_pointcloud",500);
  vis_vel_comp = nh.advertise<visualization_msgs::MarkerArray> ("radar_front_v_comp", 1);
  vis_tracking_vel = nh.advertise<visualization_msgs::MarkerArray> ("radar_tracking_v", 1);
  vis_vel = nh.advertise<visualization_msgs::MarkerArray> ("radar_front_v", 1);
  pub_trajectory = nh.advertise<visualization_msgs::MarkerArray> ("tracking_trajectory_path", 1);
  pub_trajectory_smooth = nh.advertise<visualization_msgs::MarkerArray> ("tracking_smooth_trajectory", 1);
  pub_pt = nh.advertise<visualization_msgs::MarkerArray> ("tracking_pt", 1);
  pub_pred = nh.advertise<visualization_msgs::MarkerArray> ("predict_pt", 1);
  pub_annotation = nh.advertise<visualization_msgs::MarkerArray> ("map_annotation", 1);
  pub_anno_cluster = nh.advertise<sensor_msgs::PointCloud2> ("annotation_pub", 300);
  pub_cluster_hull = nh.advertise<visualization_msgs::MarkerArray> ("cluster_hull", 1);
  pub_radar_debug = nh.advertise<visualization_msgs::MarkerArray> ("radar_debug_lines", 1);
  pub_debug_id_text = nh.advertise<visualization_msgs::MarkerArray> ("radar_debug_invalidID", 1);
  pub_tracker_cov = nh.advertise<visualization_msgs::MarkerArray> ("tracker_cov", 1);

  nh.param<bool>("output_KFT_result"    ,output_KFT_result    ,true);
  nh.param<bool>("output_obj_id_result" ,output_obj_id_result ,true);
  nh.param<bool>("output_radar_info"    ,output_radar_info    ,true);
  nh.param<bool>("output_cluster_info"  ,output_cluster_info  ,true);
  nh.param<bool>("output_dbscan_info"   ,output_dbscan_info   ,true);
  nh.param<bool>("output_DA_pair"       ,output_DA_pair       ,true);
  nh.param<bool>("output_exe_time"      ,output_exe_time      ,false);
  nh.param<bool>("output_label_info"    ,output_label_info    ,false);
  nh.param<bool>("output_score_info"    ,output_score_info    ,false);
  nh.param<bool>("output_ego_vel_info"  ,output_ego_vel_info  ,true);
  nh.param<bool>("output_transform_info",output_transform_info,true);
  nh.param<bool>("output_gt_pt_info"    ,output_gt_pt_info    ,true);
  nh.param<bool>("output_score_mat_info",output_score_mat_info,true);
  nh.param<bool>("write_out_score_file" ,write_out_score_file ,true);
  nh.param<bool>("DA_method"            ,DA_choose            ,false);
  nh.param<bool>("use_KFT_module"       ,use_KFT_module       ,true);
  nh.param<bool>("show_stopObj_id"      ,show_stopObj_id      ,true);
  nh.param<int> ("kft_id_num"           ,kft_id_num           ,1);
  nh.param<bool>("show_vel_marker"      ,show_vel_marker      ,false);
  nh.param<bool>("use_score_cluster"    ,use_score_cluster    ,false);
  nh.param<bool>("get_transformer"      ,get_transformer      ,false);
  nh.param<string>("cluster_type"       ,cluster_type         ,"dbscan");
  nh.param<double>("eps"                ,cluster_eps          ,2.5);
  nh.param<int>("Nmin"                  ,cluster_Nmin         ,2);
  nh.param<int>("history_frames"        ,cluster_history_frames  ,3);
  nh.param<double>("cluster_dt_weight"  ,cluster_dt_weight    ,0.0);
  
  ros::Subscriber bag_sub;
  // bag_sub = nh.subscribe("rosout",10,&get_bag_info_callback); // sub the bag name

  message_filters::Synchronizer<radar_sync>* radar_car_sync;
  radar_car_sync = new message_filters::Synchronizer<radar_sync>(radar_sync(3), sub_radar, sub_car_state);
  radar_car_sync->registerCallback(boost::bind(&callback, _1, _2));

  ROS_INFO("Setting:");
  ROS_INFO("Data association method : %s" , DA_choose ? "hungarian" : "my DA method(Greedy-like)");
  ROS_INFO_COND(output_radar_info     ,"Output Radar information");
  ROS_INFO_COND(output_cluster_info   ,"Output Cluster information");
  ROS_INFO_COND(output_dbscan_info    ,"Output DBSCAN information");
  ROS_INFO_COND(output_KFT_result     ,"Output KFT Result");
  ROS_INFO_COND(output_obj_id_result  ,"Output Obj-id Result");
  ROS_INFO_COND(output_DA_pair        ,"Output Data Association Pair");
  ROS_INFO_COND(output_exe_time       ,"Output Execution Time");
  ROS_INFO_COND(output_label_info     ,"Output Label information");
  ROS_INFO_COND(output_ego_vel_info   ,"Output Ego Velocity information");
  ROS_INFO_COND(output_transform_info ,"Output Transform information");
  ROS_INFO_COND(get_transformer       ,"Transform to Frame:Map");
  ROS_INFO_COND(show_vel_marker       ,"Publish Velocity marker");
  ROS_INFO_COND(use_KFT_module        ,"Use KFT module (Tracking part)");
  if(use_KFT_module){
    ROS_INFO("\tShow %d types marker id"          , kft_id_num);
    ROS_INFO("\t%s the stop object tracker"       ,show_stopObj_id ? "Show":"Mask");
  }
  ROS_INFO("Cluster Type : %s"        ,cluster_type.c_str());
  ROS_INFO("Cluster -> Eps:%f, Nmin:%d, Accumulate Frames:%d",cluster_eps,cluster_Nmin,cluster_history_frames);
  ROS_INFO("Cluster dt weight for vel function:%f",cluster_dt_weight);

  while(ros::ok()){
    ros::spinOnce();
    pub_marker.publish(m_s);
    radar_debug_visualization();
  }
  return 0;
}

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
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h>
#include <pcl/ModelCoefficients.h>

#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Twist.h>

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

#include <radar_scenes/RadarScenes.h>
#include <nav_msgs/Odometry.h>

// to sync the subscriber
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// cluster lib
#include "dbtrack/dbtrack.h"

// visualize the cluster result
#include "cluster_visual/cluster_visual.h"
#include <geometry_msgs/PolygonStamped.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>

#include <json/json.h>
#include <json/value.h>

#include <Eigen/Dense>

using namespace std;
using namespace cv;

#define trajectory_frame 10
#define tracking_stable_frames 10 // the most stable tracking frames

typedef message_filters::sync_policies::ApproximateTime<radar_scenes::RadarScenes,radar_scenes::RadarScenes,
														                            radar_scenes::RadarScenes,radar_scenes::RadarScenes> radar_sync;


ros::Publisher pub_cluster_marker;  // pub cluster index
ros::Publisher pub_marker;          // pub tracking id
ros::Publisher pub_filter;          // pub filter points
ros::Publisher pub_in_filter;       // pub the inlier points
ros::Publisher pub_cluster_center;  // pub cluster points
ros::Publisher pub_cluster_pointcloud;  // pub cluster point cloud
ros::Publisher pub_tracker_point;   // pub the tracker with pointcloud2
ros::Publisher pub_gt_cloud;        // pub the GT cluster result with pointcloud2
ros::Publisher vis_vel_rls;         // pub radar RLS vel
ros::Publisher vis_vel_comp;        // pub radar comp vel
ros::Publisher vis_tracking_vel;    // pub radar tracking vel
ros::Publisher pub_trajectory;      // pub the tracking trajectory
ros::Publisher pub_trajectory_smooth;  // pub the tracking trajectory using kf smoother
ros::Publisher pub_pt;              // pub the tracking history pts
ros::Publisher pub_pred;            // pub the predicted pts
ros::Publisher pub_anno_cluster;    // pub the cluster from the annotation
ros::Publisher pub_cluster_hull;    // pub the cluster polygon result
ros::Publisher pub_cluster_bbox;    // pub the cluster Bounding Box
ros::Publisher pub_tracker_cov;      // pub the tracker covariance


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
bool output_gt_pt_info = false;
bool output_score_mat_info = false;
bool write_out_score_file = false;
bool use_annotation_module = false; // true: repulish the global annotation and score the cluster performance
bool use_KFT_module = true;   // true: cluster and tracking, fasle: cluster only
bool show_stopObj_id = true;  // decide to show the tracker id at stop object or not
bool show_vel_marker = false;
int kft_id_num = 1;           // 1: only show the id tracked over 5 frames, 2: show the id tracked over 1 frames, 3: show all id
bool use_score_cluster = false; // true: score the cluster performance
bool KFT_detection_choose = true; // true: use the cluster_KFT(), false: use KFT()

#define KFT_INFO(msg)     if (output_KFT_result) {std::cout << msg;}
#define RADAR_INFO(msg)   if (output_radar_info) {std::cout << msg;}
#define DBSCAN_INFO(msg)  if (output_dbscan_info) {std::cout << msg;}
#define EXE_INFO(msg)     if (output_exe_time) {std::cout << msg;}

#define MAP_FRAME "/map"
#define CAR_FRAME "/car"

std::vector<tf::Transform> radar2car;
tf::StampedTransform car2map;

int call_back_num = 0;  // record the callback number

ros::Time radar_stamp,label_stamp;
float vx = 0; // vehicle vel x
float vy = 0; // vehicle vel y

#define box_bias_w 1.5
#define box_bias_l 2.0
bool get_label = false;

/*
 * cluster and tracking center
 */
std::vector< radar_scenes::RadarScenesConstPtr > call_back_msg_list;
vector<kf_tracker_point> cens;
std::vector<int> cluster_tracking_result_dbpda;

/*
 * For visualization
 */
visualization_msgs::MarkerArray m_s,l_s,cluster_s;
visualization_msgs::MarkerArray ClusterHulls; // show the radar cluster polygon marker
int max_size = 0, cluster_marker_max_size = 0, cluster_polygon_max_size = 0, trajcetory_max_size = 0, vel_marker_max_size = 0, tracking_vel_marker_max_size = 0, cov_marker_max_size = 0;
int anno_vel_max = 0, anno_tra_max = 0, anno_id_max = 0;
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
  tra_color.color.a = 1.0;
  tra_color.color.r = 226.0f/255.0f;
  tra_color.color.g = 195.0f/255.0f;
  tra_color.color.b = 243.0f/255.0f;
  
  /*
   * the marker of the tracjectory that is the same as the ITRI
   */
  itri_tra_color.scale.x = 0.5f;
  itri_tra_color.scale.y = 0.5f;
  itri_tra_color.scale.z = 0.2f;
  // green
  itri_tra_color.color.a = 0.8;
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
float dt = 0.8f;     //0.1f 0.08f=1/13Hz(radar)
float sigmaP = 0.01;  //0.01
float sigmaQ = 0.1;   //0.1
#define bias 4.0      // used for data association bias 5 6
#define mahalanobis_bias 5.5 // used for data association(mahalanobis dist)
int id_count = 0; //the counter to calculate the current occupied track_id
int anno_id_count = 0; //the counter to calculate the current occupied annotation track_id
#define frame_lost 10   // 5 10

std::vector <int> id;
ros::Publisher objID_pub;
// KF init
int stateDim = 6; // [x,y,z,v_x,v_y,v_z]
int measDim = 3;  // [x,y,z,v_x,v_y,v_z]
int ctrlDim = 0;  // control input 0(acceleration=0,constant v model)


bool firstFrame = true;
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
double cluster_eps;
int cluster_Nmin;
int cluster_history_frames;
double cluster_dt_weight; // use for the vel_function in cluster lib
bool viz_cluster_with_past = true;  // True : visualize the cluster result with past and now, false : with current only
bool cluster_track_msg = false;
bool motion_eq_optimizer_msg = false;
bool rls_msg = false;

// DBTRACK init
dbtrack dbtrack_seg;
bool dbtrack_para_train = false;
bool dbtrack_para_train_direct = false;

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
  int under_cluster =0;    // cluster cover over than one gt
  int no_cluster = 0; // no cluster in gt
  double v_measure_score = 0;
  double homo = 0;
  double h_ck = 0;
  double h_c = 0;
  double comp = 0;
  double h_kc = 0;
  double h_k = 0;
  // scene info
  int scene_num = 0;
  double ave_doppler_vel;

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

  vector<ros::Time> timestamp;

  int lose_frame;
  int track_frame;
  MOTION_STATE motion;
  TRACK_STATE tracking_state;
  int match_clus = 1000;
  int cluster_idx;
  int uuid ;
  int tracker_id;

  std::vector<cluster_point> cluster_pc;  // for annotation cluster
}track;

// tracking result vector
std::vector<track> filters;
std::vector<track> anno_filters;  // the annotation tracking filters
std::vector<int> anno_obj_id;    // record the current scan points' cluster tracking id
// std::vector< std::vector<cluster_point> > gt_cluster_vec;  // the vector contains the gt cluster result
std::vector<string> anno_token;
Json::Value anno_root;
int anno_root_idx = 0;

void transform_init(){
  tf::Transform trans2car;
  tf::Quaternion q;
  // trans2car.frame_id_ = "/car";
  // trans2car.child_frame_id_ = "/radar_1";
  trans2car.setOrigin(tf::Vector3(3.663,-0.873,0));
  q.setEuler(-1.48418552,0,0);
  trans2car.setRotation(q);
  radar2car.push_back(trans2car);

  // trans2car.child_frame_id_ = "/radar_2";
  trans2car.setOrigin(tf::Vector3(3.86,-0.7,0));
  q.setEuler(-0.436185662,0,0);
  trans2car.setRotation(q);
  radar2car.push_back(trans2car);

  // trans2car.child_frame_id_ = "/radar_3";
  trans2car.setOrigin(tf::Vector3(3.86,0.7,0));
  q.setEuler(0.436,0,0);
  trans2car.setRotation(q);
  radar2car.push_back(trans2car);

  // trans2car.child_frame_id_ = "/radar_4";
  trans2car.setOrigin(tf::Vector3(3.663,0.873,0));
  q.setEuler(1.484,0,0);
  trans2car.setRotation(q);
  radar2car.push_back(trans2car);
  
}

ros::Time fileStamp(string stamp_str){
    long int stamp=stod(stamp_str.c_str());
    ros::Time timestamp;
    long int stamp_temp = (stamp%1000000)*1000;
    while(stamp_temp/100000000<0){
        stamp_temp=stamp_temp*10;
    }
    timestamp.nsec=stamp_temp;
    timestamp.sec=stamp/1000000;
    return timestamp;
}

// publish the tracking id
void marker_id(bool firstFrame,std::vector<int> obj_id,std::vector<int> marker_obj_id,std::vector<int> tracking_frame_obj_id){
  visualization_msgs::Marker marker;
  pcl::PointCloud<pcl::PointXYZI>::Ptr tracker_points(new pcl::PointCloud<pcl::PointXYZI>);
  sensor_msgs::PointCloud2::Ptr tracker_cloud(new sensor_msgs::PointCloud2);
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
    if(marker.color.a!=0){
      pcl::PointXYZI pt;
      pt.x = cens[k].x;
      pt.y = cens[k].y;
      pt.z = cens[k].z;
      pt.intensity = std::atof(marker.text.c_str());
      tracker_points->push_back(pt);
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

  pcl::toROSMsg(*tracker_points,*tracker_cloud);
  tracker_cloud->header.frame_id = MAP_FRAME;
  tracker_cloud->header.stamp = radar_stamp;
  pub_tracker_point.publish(tracker_cloud);
}

// publish the cluster points with different color and id
void color_cluster(std::vector< std::vector<kf_tracker_point> > cluster_list, bool cluster_result = true, std::vector<int> cluster_tracking_result={}){
  // republish radar points
  pcl::PointCloud<pcl::PointXYZI>::Ptr cluster_points(new pcl::PointCloud<pcl::PointXYZI>);
  sensor_msgs::PointCloud2::Ptr cluster_cloud(new sensor_msgs::PointCloud2);
  cluster_points->clear();
  // end republish
  visualization_msgs::Marker marker;
  cluster_s.markers.clear();
  if((cens.size()!=cluster_tracking_result.size() || cens.size()!=cluster_list.size()) && cluster_tracking_result.size()!=0)
    ROS_WARN("Cens size: %d, cluster tracking vec size: %d, cluster list size: %d",(int)cens.size(),(int)cluster_tracking_result.size(),(int)cluster_list.size());
  for(int i=0;i<cluster_list.size();i++){
    srand(i+1);
    float color = 255 * rand() / (RAND_MAX + 1.0);
    // add cluster index marker
    marker.header.frame_id = MAP_FRAME;
    marker.header.stamp = radar_stamp;
    marker.action = visualization_msgs::Marker::ADD;
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.scale.z = 1.5f;  // rgb(255,127,122)
    marker.color.r = 255.0 / 255.0;
    marker.color.g = 154.0 / 255.0;
    marker.color.b = 38.0 / 255.0;
    marker.color.a = 1;
    stringstream ss;
    marker.id = i;
    if(cluster_tracking_result.size()==0){
      geometry_msgs::Pose pose;
      pose.position.x = cluster_list.at(i).at(0).x + 1.5f;
      pose.position.y = cluster_list.at(i).at(0).y + 1.5f;
      pose.position.z = cluster_list.at(i).at(0).z + 1.0f;
      pose.orientation.w = 1.0;
      marker.pose = pose;
      ss << i;
    }
    else{
      geometry_msgs::Pose pose;
      pose.position.x = cens.at(i).x;
      pose.position.y = cens.at(i).y;
      pose.position.z = cens.at(i).z + 2.0f;
      pose.orientation.w = 1.0;
      marker.pose = pose;
      marker.color.a = 0.8;
      if(cens.at(i).vel >= motion_vel_threshold){
        marker.color.r = 250.0f/255.0f;
        marker.color.g = 112.0f/255.0f;
        marker.color.b = 188.0f/255.0f;
        marker.color.a = 1.0;
        marker.scale.x = 0.2;
      }
      ss << cluster_tracking_result.at(i);
    }
    marker.text = ss.str();
    
    marker.lifetime = ros::Duration(dt-0.01);
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
void polygon_cluster_visual(std::vector< std::vector<kf_tracker_point> > clusterResult){
  std_msgs::Header header;
  header.frame_id = MAP_FRAME;
  header.stamp = radar_stamp;
  std::vector<geometry_msgs::PolygonStamped> polygon_vec;
  std::vector<cluster_visual_unit> single_point_cluster;
  std::vector<bool> record_moving_cluster_poly, record_moving_cluster_single;
  jsk_recognition_msgs::BoundingBoxArray bbox_array;
  bbox_array.header = header;
  double viz_hight = 2.5, viz_height_bias = 2.5;
  // Do not viz stationary cluster
  std::vector< std::vector<kf_tracker_point> > cluster_list = clusterResult;
  for(auto cluster_it=cluster_list.begin();cluster_it!=cluster_list.end();){
    Eigen::Vector2d vel(0,0);
    for(auto pt:*cluster_it){
      vel += Eigen::Vector2d(pt.x_v,pt.y_v);
    }
    if(vel.norm()<motion_vel_threshold){
      cluster_list.erase(cluster_it);
    }
    else{
      cluster_it++;
    }
  }
  for(int i=0;i<cluster_list.size();i++){
    pcl::PointCloud <pcl::PointXYZ>::Ptr pointsCluster(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointXYZ pt;
    cluster_visual_unit cluster_unit;
    std::vector<float> temp_x_radius;
    std::vector<float> temp_y_radius;
    double vel_motion = 0;
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
        pt.z = 0;
        pointsCluster->points.push_back(pt);
        if(ab_vec.norm()<1){
          ab_vec.normalize();
          pt.x = cluster_list.at(i).at(1).x + ab_vec.x();
          pt.y = cluster_list.at(i).at(1).y + ab_vec.y();
          pt.z = 0;
          pointsCluster->points.push_back(pt);
          pt.x = cluster_list.at(i).at(1).x - ab_vec.x();
          pt.y = cluster_list.at(i).at(1).y - ab_vec.y();
          pt.z = 0;
          pointsCluster->points.push_back(pt);
        }
        else{
          pt.x = cluster_list.at(i).at(0).x;
          pt.y = cluster_list.at(i).at(0).y;
          pt.z = 0;
          pointsCluster->points.push_back(pt);
        }
        for(int j=0;j<2;j++){
          pt.x = cluster_list.at(i).at(j).x + T_vec.x();
          pt.y = cluster_list.at(i).at(j).y + T_vec.y();
          pt.z = 0;
          pointsCluster->points.push_back(pt);
          pt.x = cluster_list.at(i).at(j).x - T_vec.x();
          pt.y = cluster_list.at(i).at(j).y - T_vec.y();
          pt.z = 0;
          vel_motion += cluster_list.at(i).at(j).vel;
          
          pointsCluster->points.push_back(pt);
        }
        if(vel_motion/cluster_list.at(i).size() > motion_vel_threshold){
          high_vel = true;
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
        pt.z = 0;
        pointsCluster->points.push_back(pt);
        if(ab_vec.norm()<1){
          ab_vec.normalize();
          pt.x = cluster_list.at(i).at(1).x + ab_vec.x();
          pt.y = cluster_list.at(i).at(1).y + ab_vec.y();
          pt.z = 0;
          pointsCluster->points.push_back(pt);
        }
        else{
          pt.x = cluster_list.at(i).at(0).x;
          pt.y = cluster_list.at(i).at(0).y;
          pt.z = 0;
          pointsCluster->points.push_back(pt);
        }
        if(bc_vec.norm()<1){
          bc_vec.normalize();
          pt.x = cluster_list.at(i).at(1).x + bc_vec.x();
          pt.y = cluster_list.at(i).at(1).y + bc_vec.y();
          pt.z = 0;
          pointsCluster->points.push_back(pt);
        }
        else{
          pt.x = cluster_list.at(i).at(2).x;
          pt.y = cluster_list.at(i).at(2).y;
          pt.z = 0;
          pointsCluster->points.push_back(pt);
        }
        for(int j=0;j<2;j++){
          pt.x = cluster_list.at(i).at(j).x + T_vec.x();
          pt.y = cluster_list.at(i).at(j).y + T_vec.y();
          pt.z = 0;
          pointsCluster->points.push_back(pt);
          pt.x = cluster_list.at(i).at(j).x - T_vec.x();
          pt.y = cluster_list.at(i).at(j).y - T_vec.y();
          pt.z = 0;
          pointsCluster->points.push_back(pt);
        }
        vel_motion = cluster_list.at(i).at(0).vel;
        for(int j=1;j<3;j++){
          pt.x = cluster_list.at(i).at(j).x + T2_vec.x();
          pt.y = cluster_list.at(i).at(j).y + T2_vec.y();
          pt.z = 0;
          pointsCluster->points.push_back(pt);
          pt.x = cluster_list.at(i).at(j).x - T2_vec.x();
          pt.y = cluster_list.at(i).at(j).y - T2_vec.y();
          pt.z = 0;
          pointsCluster->points.push_back(pt);
          vel_motion += cluster_list.at(i).at(j).vel;
        }
        if(vel_motion/cluster_list.at(i).size() > motion_vel_threshold){
          high_vel = true;
        }
        break;
      default:
        for(int j=0;j<cluster_list.at(i).size();j++){
          pt.x = cluster_list.at(i).at(j).x;
          pt.y = cluster_list.at(i).at(j).y;
          pt.z = 0;
          pointsCluster->points.push_back(pt);
          vel_motion += cluster_list.at(i).at(j).vel;
        }
        if(vel_motion/cluster_list.at(i).size() > motion_vel_threshold){
          high_vel = true;
        }
        break;
    }
    if(pointsCluster->points.size()>0){
      Cluster_visual clusterObject;
      clusterObject.SetCloud(pointsCluster, header, true);
      polygon_vec.push_back(clusterObject.GetPolygon());
      bbox_array.boxes.push_back(clusterObject.GetBoundingBox());
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
    // ObjHull.color.r = 255.0f/255.0f;
    // ObjHull.color.g = 0.0f/255.0f;
    // ObjHull.color.b = 0.0f/255.0f;
    if(record_moving_cluster_poly.at(cluster_idx)){
      // ObjHull.color.r = 0.0f/255.0f;
      // ObjHull.color.g = 255.0f/255.0f;
      // ObjHull.color.b = 0.0f/255.0f;
      // ObjHull.color.r = 250.0f/255.0f;
      // ObjHull.color.g = 112.0f/255.0f;
      // ObjHull.color.b = 188.0f/255.0f;
      ObjHull.scale.x = ObjHull.scale.y = ObjHull.scale.z = 0.2;
    }
    ObjHull.lifetime = ros::Duration(dt-0.01);

    geometry_msgs::Point markerPt;
    for(int j=0;j<polygon_vec.at(cluster_idx).polygon.points.size();j++){
      markerPt.x = polygon_vec.at(cluster_idx).polygon.points[j].x;
      markerPt.y = polygon_vec.at(cluster_idx).polygon.points[j].y;
      markerPt.z = polygon_vec.at(cluster_idx).polygon.points[j].z*viz_hight+viz_height_bias;
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
    // ObjHull.color.r = 255.0f/255.0f;
    // ObjHull.color.g = 0.0f/255.0f;
    // ObjHull.color.b = 0.0f/255.0f;
    if(record_moving_cluster_single.at(cluster_idx)){
      // ObjHull.color.r = 0.0f/255.0f;
      // ObjHull.color.g = 255.0f/255.0f;
      // ObjHull.color.b = 0.0f/255.0f;
      // ObjHull.color.r = 250.0f/255.0f;
      // ObjHull.color.g = 112.0f/255.0f;
      // ObjHull.color.b = 188.0f/255.0f;
      ObjHull.scale.x = 0.2;
    }
    ObjHull.lifetime = ros::Duration(dt-0.01);

    for(int i=0;i<361;i++){
      geometry_msgs::Point markerPt;
      markerPt.x = single_point_cluster.at(cluster_idx).x_radius * sin(i*2*M_PI/360) + single_point_cluster.at(cluster_idx).center.x;
      markerPt.y = single_point_cluster.at(cluster_idx).y_radius * cos(i*2*M_PI/360) + single_point_cluster.at(cluster_idx).center.y;
      markerPt.z = single_point_cluster.at(cluster_idx).center.z*viz_hight+viz_height_bias;
      ObjHull.points.push_back(markerPt);
    }
    // Set the cluster polygon color invisable if the cluster containts only one point
    ObjHull.color.a = 1.0;
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
  // pub_cluster_bbox.publish(bbox_array);
}

// show the radar velocity arrow
void vel_marker(std::vector< radar_scenes::RadarScenesConstPtr > call_back_msg_list){
  visualization_msgs::MarkerArray marker_array_comp;
  int count_vel_num = 0;
  for(int msg_index=0;msg_index<call_back_msg_list.size();msg_index++){
    for(int i=0; i<call_back_msg_list.at(msg_index)->points.size(); i++){
      
      visualization_msgs::Marker marker;
      marker.header.frame_id = call_back_msg_list.at(msg_index)->points[i].header.frame_id;
      // marker.header.stamp = ros::Time();
      marker.header.stamp = radar_stamp;
      marker.id = count_vel_num++;
      marker.type = visualization_msgs::Marker::ARROW;
      marker.action = visualization_msgs::Marker::ADD;
      marker.pose.position.x = call_back_msg_list.at(msg_index)->points[i].range*std::cos(call_back_msg_list.at(msg_index)->points[i].azimuth);
      marker.pose.position.y = call_back_msg_list.at(msg_index)->points[i].range*std::sin(call_back_msg_list.at(msg_index)->points[i].azimuth);
      marker.pose.position.z = 0;
      float theta = call_back_msg_list.at(msg_index)->points[i].azimuth;
      tf2::Quaternion Q;
      Q.setRPY( 0, 0, theta );
      marker.pose.orientation = tf2::toMsg(Q);
      marker.scale = vel_comp_color.scale;
      // vel length
      marker.scale.x = call_back_msg_list.at(msg_index)->points[i].vr_compensated;
      marker.color = vel_comp_color.color;
      marker_array_comp.markers.push_back(marker);
    }
  }

  if(count_vel_num > vel_marker_max_size)
    vel_marker_max_size = count_vel_num;
  for(int a=count_vel_num;a<vel_marker_max_size;a++){
    visualization_msgs::Marker marker;
		marker.header.frame_id = MAP_FRAME;
		marker.header.stamp = ros::Time();
		marker.id = a;
		marker.type = visualization_msgs::Marker::ARROW;
		marker.action = visualization_msgs::Marker::DELETE;
		marker.pose.position.x = 0;
		marker.pose.position.y = 0;
		marker.pose.position.z = 1;
    marker.color.a = 0;
		marker_array_comp.markers.push_back(marker);
  }

	vis_vel_comp.publish( marker_array_comp );
}

void rls_vel_marker(std::vector<kf_tracker_point> rls_points){
  visualization_msgs::MarkerArray rls_marker;
  int count_vel_num = 0;
  for(auto pt=rls_points.begin();pt!=rls_points.end();pt++){
    visualization_msgs::Marker marker;
    marker.header.frame_id = MAP_FRAME;
    marker.header.stamp = ros::Time();
    marker.id = count_vel_num++;
    marker.type = visualization_msgs::Marker::ARROW;
    marker.action = visualization_msgs::Marker::ADD;
    tf::Point map2car_pt(pt->x,pt->y,0);
    marker.pose.position.x = map2car_pt.x();
    marker.pose.position.y = map2car_pt.y();
    marker.pose.position.z = 5;
    double rls_vx = pt->x_v;
    double rls_vy = pt->y_v;
    float theta = atan2(rls_vy,
                        rls_vx);
    tf2::Quaternion Q;
    Q.setRPY( 0, 0, theta );
    marker.pose.orientation = tf2::toMsg(Q);
    marker.scale = vel_comp_color.scale;
    // vel length
    marker.scale.x = sqrt(pow(rls_vy,2) + 
                          pow(rls_vx,2));
    
    if(marker.scale.x<1){
      marker.color.a = 0;
    }
    marker.color = vel_color.color;
    marker.color.a = 0.9;
    // marker.color.r = 0;
    // marker.color.g = 1.0;
    // marker.color.b = 0;

    rls_marker.markers.push_back(marker);
    
  }

  if(count_vel_num > vel_marker_max_size)
    vel_marker_max_size = count_vel_num;
  for(int a=count_vel_num;a<vel_marker_max_size;a++){
    visualization_msgs::Marker marker;
		marker.header.frame_id = MAP_FRAME;
		marker.header.stamp = ros::Time();
		marker.id = a;
		marker.type = visualization_msgs::Marker::ARROW;
		marker.action = visualization_msgs::Marker::DELETE;
		marker.pose.position.x = 0;
		marker.pose.position.y = 0;
		marker.pose.position.z = 1;
    marker.color.a = 0;
		rls_marker.markers.push_back(marker);
  }
	vis_vel_rls.publish(rls_marker);
}

// show the tracking trajectory
void show_trajectory(){
  int k=0;
  visualization_msgs::MarkerArray tra_array, tra_array_smooth, point_array, pred_point_array, cov_array;
  tra_array.markers.clear();
  point_array.markers.clear();
  pred_point_array.markers.clear();

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
    // show the covariance
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
void show_past_radar_points_with_color(){
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr history_filter_points_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
  sensor_msgs::PointCloud2::Ptr history_filter_cloud(new sensor_msgs::PointCloud2);
  for(int i=0;i<past_radars_with_cluster.size();i++){
    for(int j=0;j<past_radars_with_cluster.at(i).size();j++){
      for(int k=0;k<past_radars_with_cluster.at(i).at(j).size();k++){
        pcl::PointXYZRGB pt;
        pt.x = past_radars_with_cluster.at(i).at(j).at(k).x;
        pt.y = past_radars_with_cluster.at(i).at(j).at(k).y;
        pt.z = past_radars_with_cluster.at(i).at(j).at(k).z;
        
        pcl::_PointXYZRGB color1, color2;

        color1.r = 145;
        color1.g = 255;
        color1.b = 124;

        color2.r = 1;
        color2.g = 15;
        color2.b = 10;
        
        // light orange to dark orange 1
        // pt.r = 255-201/cluster_history_frames*(cluster_history_frames-1-j);
        // pt.g = 170-134/cluster_history_frames*(cluster_history_frames-1-j);
        // pt.b = 50-39/cluster_history_frames*(cluster_history_frames-1-j);

        pt.r = color1.r-(color1.r-color2.r)/cluster_history_frames*(cluster_history_frames-1-j);
        pt.g = color1.g-(color1.g-color2.g)/cluster_history_frames*(cluster_history_frames-1-j);
        pt.b = color1.b-(color1.b-color2.b)/cluster_history_frames*(cluster_history_frames-1-j);
        
        history_filter_points_rgb->points.push_back(pt);
      }
    }
  }
  pcl::toROSMsg(*history_filter_points_rgb,*history_filter_cloud);
  history_filter_cloud->header.frame_id = MAP_FRAME;
  history_filter_cloud->header.stamp = radar_stamp;
  pub_filter.publish(history_filter_cloud);
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

int new_track(kf_tracker_point &cen, int idx, int direct_assign_tracker_id=-1){
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
  if(measDim<=3){
    ka.statePost.at<float>(3)=0;// initial v_x
    ka.statePost.at<float>(4)=0;// initial v_y
    ka.statePost.at<float>(5)=0;// initial v_z
  }
  else{
    ka.statePost.at<float>(3)=cen.x_v;// initial v_x
    ka.statePost.at<float>(4)=cen.y_v;// initial v_y
    ka.statePost.at<float>(5)=cen.z_v;// initial v_z
  }

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
  tk.track_frame = 1;
  tk.cluster_idx = idx;
  if(direct_assign_tracker_id!=-1)
    tk.uuid = direct_assign_tracker_id;
  else
    tk.uuid = ++id_count;
  filters.push_back(tk);
  // std::cout<<"Done init newT at "<<id_count<<" is ("<<tk.pred_pose.x <<"," <<tk.pred_pose.y<<")"<<endl;
  return tk.uuid;
}

void cluster_KFT(std::vector<int> cluster_tracking_result){
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

  i=0;

  std::vector<int> obj_id(cens.size(),-1);
  std::vector<int> tracking_frame_obj_id(cens.size(),-1);
  
  // record the cluster tracking result
  std::vector<bool> cluster_tracking_bool_vec(past_radars_with_cluster.size(),false);

  for(int k=0; k<filters.size(); k++){
    geometry_msgs::Point pred_v = filters.at(k).pred_v;
    geometry_msgs::Point pred_pos = filters.at(k).pred_pose;

    
    float delta_x = (pred_pos.x-filters.at(k).kf.statePost.at<float>(0));
    float delta_y = (pred_pos.y-filters.at(k).kf.statePost.at<float>(1));
    float dist_thres = sqrt(delta_x * delta_x + delta_y * delta_y);
    if(output_KFT_result){
      std::cout<<"------------------------------------------------\n";
      std::cout << "Tracker " << "\033[1;31m" << filters.at(k).uuid << "\033[0m\n";
      // std::cout << "The dist_thres for " <<filters.at(k).uuid<<" is "<<dist_thres<<endl;
      std::cout<<"predict v:"<<pred_v.x<<","<<pred_v.y<<endl;
      std::cout<<"predict pos:"<<pred_pos.x<<","<<pred_pos.y<<endl;
    }
    int cluster_id;
    bool new_tracker = true;
    for(int idx=0;idx<cluster_tracking_result.size();idx++){
      if(filters[k].uuid==cluster_tracking_result[idx]){
        new_tracker = false;
        cluster_id = idx;
        continue;
      }
    }

    //-1 for non matched tracks
    if (!new_tracker){
      Eigen::Vector3d dist_diff(cens.at(cluster_id).x-pred_pos.x,cens.at(cluster_id).y-pred_pos.y,0);
      obj_id[cluster_id] = filters.at(k).uuid;
      if(filters.at(k).track_frame > tracking_stable_frames)
        tracking_frame_obj_id[cluster_id] = filters.at(k).uuid;
      filters.at(k).cluster_idx = cluster_id;
      filters.at(k).tracking_state = TRACK_STATE::tracking;
      // check the traker's motion state
      if(cens.at(cluster_id).vel < motion_vel_threshold && filters.at(k).motion != MOTION_STATE::stop){
        filters.at(k).motion = MOTION_STATE::slow_down;
        cens.at(cluster_id).motion = MOTION_STATE::slow_down;
      }
      else if(cens.at(cluster_id).vel < motion_vel_threshold){
        filters.at(k).motion = MOTION_STATE::stop;
        cens.at(cluster_id).motion = MOTION_STATE::stop;
      }
      else
      {
        filters.at(k).motion = MOTION_STATE::move;
        cens.at(cluster_id).motion = MOTION_STATE::move;
      }
      
      geometry_msgs::Point tracking_pt_history;
      tracking_pt_history.x = cens.at(cluster_id).x;
      tracking_pt_history.y = cens.at(cluster_id).y;
      tracking_pt_history.z = cens.at(cluster_id).z;
      filters.at(k).history.push_back(tracking_pt_history);

      geometry_msgs::Point tracking_pt_history_smooth;
      tracking_pt_history_smooth.x = filters.at(k).kf.statePre.at<float>(0);
      tracking_pt_history_smooth.y = filters.at(k).kf.statePre.at<float>(1);
      tracking_pt_history_smooth.z = filters.at(k).kf.statePre.at<float>(2);
      filters.at(k).history_smooth.push_back(tracking_pt_history_smooth);
      
      if(filters.at(k).track_frame>10)
        cluster_tracking_bool_vec.at(cluster_id) = true;
    }
    else
    {
      filters.at(k).tracking_state = TRACK_STATE::missing;
    }
    

    if(output_KFT_result){
      //get tracked or lost
      if (filters.at(k).tracking_state == TRACK_STATE::tracking){
          std::cout<<"The state of "<<k<<" filters is \033[1;32mtracking\033[0m, to cluster_idx "<< filters.at(k).cluster_idx <<" track: "<<filters.at(k).track_frame;
          std::cout<<" ("<<cens.at(cluster_id).x<<","<<cens.at(cluster_id).y<<")"<<endl;
      }
      else if (filters.at(k).tracking_state == TRACK_STATE::missing)
          std::cout<<"The state of "<<k<<" filters is \033[1;34mlost\033[0m, to cluster_idx "<< filters.at(k).cluster_idx <<" lost: "<<filters.at(k).lose_frame<<endl;
      else
          std::cout<<"\033[1;31mUndefined state for tracker "<<k<<"\033[0m"<<endl;
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
      int track_uuid = new_track(cens.at(i),i,cluster_tracking_result.at(i));
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

  marker_id(false,obj_id,marker_obj_id,tracking_frame_obj_id);       // publish all id
  show_trajectory();
  show_past_radar_points_with_color();
  
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
        if(measDim>3){
          meas[i][3] = pt.x_v;
          meas[i][4] = pt.y_v;
          meas[i][5] = pt.z_v;
        }
    }
    else if ( (*it).tracking_state == TRACK_STATE::missing ){
        meas[i][0] = (*it).pred_pose.x; //+ (*it).pred_v.x;
        meas[i][1] = (*it).pred_pose.y;//+ (*it).pred_v.y;
        meas[i][2] = (*it).pred_pose.z; //+ (*it).pred_v.z;
        if(measDim>3){
          meas[i][3] = (*it).pred_v.x;
          meas[i][4] = (*it).pred_v.y;
          meas[i][5] = (*it).pred_v.z;
          }
    }
    else
    {
        std::cout<<"Some tracks state not defined to tracking/lost."<<std::endl;
    }
    measMat[i]=cv::Mat(measDim,1,CV_32F,meas[i]);
    estimated[i] = (*it).kf.correct(measMat[i]);
    i++;
  }

  return;
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
    // dist_thres = dist_thres1;
    float dist_thres = sqrt(delta_x * delta_x + delta_y * delta_y);
    if(output_KFT_result){
      std::cout<<"------------------------------------------------\n";
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
      if( dist_vec.at(assignment[k]) <=  mahalanobis_bias && (dist_diff.norm() <= 5)){
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
  show_past_radar_points_with_color();
  
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


void first_frame_KFT(std::vector<int> cluster_tracking_result={}){
  for(int i=0; i<cens.size();i++){
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
    // std::cout<<"("<<cens.at(i).x<<","<<cens.at(i).y<<","<<cens.at(i).z<<")\n";
    ka.statePost.at<float>(0)=cens.at(i).x;
    ka.statePost.at<float>(1)=cens.at(i).y;
    ka.statePost.at<float>(2)=cens.at(i).z;
    if(measDim<=3){
      ka.statePost.at<float>(3)=0;// initial v_x
      ka.statePost.at<float>(4)=0;// initial v_y
      ka.statePost.at<float>(5)=0;// initial v_z
    }
    else{
      ka.statePost.at<float>(3)=cens.at(i).x_v;// initial v_x
      ka.statePost.at<float>(4)=cens.at(i).y_v;// initial v_y
      ka.statePost.at<float>(5)=cens.at(i).z_v;// initial v_z
    }
    tk.kf = ka;
    tk.kf_us = ka;  // ready to update the state
    tk.tracking_state = TRACK_STATE::tracking;
    if(cens.at(i).vel < motion_vel_threshold)
      tk.motion = MOTION_STATE::stop;
    else
      tk.motion = MOTION_STATE::move;
    tk.lose_frame = 0;
    tk.track_frame = 1;
    if(cluster_tracking_result.size()!=0)
      tk.uuid = cluster_tracking_result.at(i);
    else
      tk.uuid = i;
    id_count++;
    filters.push_back(tk);
  }
  std::cout<<"Initiate "<<filters.size()<<" tracks."<<endl;
  std::cout << m_s.markers.size() <<endl;
  m_s.markers.clear();
  std::vector<int> no_use_id;
  marker_id(firstFrame,no_use_id,no_use_id,no_use_id);
  firstFrame=false;
  return;
}

void output_score(double beta, v_measure_score result){
  string dir_path = ros::package::getPath("track") + "/cluster_score/" + to_string(ros::WallTime::now().toBoost().date()) + "/";
  string csv_name_with_para;
  stringstream eps_s,Nmin_s;
  eps_s << cluster_eps;
  Nmin_s << cluster_Nmin;
  csv_name_with_para = "Eps-"+eps_s.str()+"_Nmin-"+Nmin_s.str()+"_";
  string csv_file_name = csv_name_with_para + score_file_name + ".csv";
  string output_path = dir_path + csv_file_name;
  if(! boost::filesystem::exists(dir_path)){
    boost::filesystem::create_directories(dir_path);   
  }
  if(filename_change){
    filename_change = false;
    score_frame = 1;
    result.frame = 1;
  }
  if(result.frame == 1){
    ofstream file;
    file.open(output_path, ios::out);
    file << "frame" << ",";
    file << "obj num" << ",";
    file << "correct-obj num" << ",";
    file << "over-obj num" << ",";
    file << "under-obj num" << ",";
    file << "no-obj num" << ",";
    file << "V measure score (beta=" << fixed << setprecision(2) << beta << ")" << ",";
    file << "Homogeneity" << ",";
    file << "H(C|K)" << ",";
    file << "H(C)" << ",";
    file << "Completeness" << ",";
    file << "H(K|C)" << ",";
    file << "H(K)" << ",";
    file << "Scene Num" << ",";
    file << "Ave vel" << endl;
    file.close();
  }
  ofstream file;
  file.open(output_path, ios::out|ios::app);
  file << setprecision(0) << result.frame << ",";
  file << setprecision(0) << result.object_num << ",";
  file << setprecision(0) << result.good_cluster << ",";
  file << setprecision(0) << result.multi_cluster << ",";
  file << setprecision(0) << result.under_cluster << ",";
  file << setprecision(0) << result.no_cluster << ",";
  file << fixed << setprecision(3);
  file << result.v_measure_score << ",";
  file << result.homo << ",";
  file << result.h_ck << ",";
  file << result.h_c << ",";
  file << result.comp << ",";
  file << result.h_kc << ",";
  file << result.h_k << ",";
  file << result.scene_num << ",";
  file << result.ave_doppler_vel << endl;
  file.close();
  return;
}

void score_cluster(std::vector< std::vector<kf_tracker_point> > ResultCluster){
  std::string LABEL_STR[12]={ "CAR",
                            "LARGE_VEHICLE",
                            "TRUCK",
                            "BUS",
                            "TRAIN",
                            "BICYCLE",
                            "TWO_WHEELER",
                            "PEDESTRIAN",
                            "GROUP_PEDESTRIAN",
                            "ANIMAL",
                            "DYNAMIC_OBJECT",
                            "STATIC_OBJECT"};
  std::vector< std::vector<kf_tracker_point> > GTCluster;
  std::map<std::string,std::vector<kf_tracker_point>> GT_Map;
  std::map<std::string,std::pair<int,int>> LabelMap;  // pair:track_id, label_id
  int track_id = 0;
  pcl::PointCloud<pcl::PointXYZI>::Ptr gt_points(new pcl::PointCloud<pcl::PointXYZI>);
  sensor_msgs::PointCloud2::Ptr gt_cloud(new sensor_msgs::PointCloud2);
  for(auto msg:call_back_msg_list){
    for(auto pt:msg->points){
      if(pt.track_id=="")
        continue;
      kf_tracker_point cluster_pt;
      cluster_pt.x = pt.pose_seq.position.x;
      cluster_pt.y = pt.pose_seq.position.y;
      cluster_pt.x_v = pt.vr_compensated;
      cluster_pt.y_v = 0;
      GT_Map[pt.track_id].push_back(cluster_pt);
      if(LabelMap.find(pt.track_id)==LabelMap.end())
        LabelMap[pt.track_id] = std::pair<int,int>(track_id++,pt.label_id);
      pcl::PointXYZI pcl_pt;
      pcl_pt.x = pt.pose_seq.position.x;
      pcl_pt.y = pt.pose_seq.position.y;
      pcl_pt.z = 0;
      pcl_pt.intensity = LabelMap[pt.track_id].first;
      gt_points->points.push_back(pcl_pt);
    }
  }
  for(auto vec:GT_Map){
    GTCluster.push_back(vec.second);
    std::cout<<"Track ID: "<<vec.first
             <<", Label: "<<LABEL_STR[LabelMap[vec.first].second]
             <<", size: "<<vec.second.size()<<std::endl;
    
  }
  pcl::toROSMsg(*gt_points,*gt_cloud);
  gt_cloud->header.frame_id = MAP_FRAME;
  gt_cloud->header.stamp = radar_stamp;
  pub_gt_cloud.publish(gt_cloud);

  dbtrack_seg.cluster_score(GTCluster,ResultCluster);
  dbtrack_seg.f1_score(GTCluster,ResultCluster);

}

void transform_radar_msg_to_cluster_data( const radar_scenes::RadarScenesConstPtr& msg, int &radar_id_count,
                                          pcl::PointCloud<pcl::PointXYZI>::Ptr &pub_filter_points, std::vector<kf_tracker_point> &cluster_data, int sensor_id){
  // get the velocity transform matrix to map
  tf::Transform radar2map;
  radar2map = car2map*radar2car.at(sensor_id);
  radar2map.setOrigin(tf::Vector3(0,0,0));
  for(int i = 0;i<msg->points.size();i++){
    tf::Vector3 velocity_vec;
    velocity_vec.setValue(msg->points[i].vr_compensated*std::cos(msg->points[i].azimuth),
                          msg->points[i].vr_compensated*std::sin(msg->points[i].azimuth),
                          0);
    velocity_vec = radar2map*velocity_vec;
    float velocity = msg->points[i].vr_compensated;
    double vel_angle = msg->points[i].azimuth;
    double angle = msg->points[i].azimuth;


    RADAR_INFO("----------------------------------------------\n");
    RADAR_INFO(i<<"\tposition:("<<msg->points[i].pose_cc.position.x<<","<<msg->points[i].pose_cc.position.y<<")\n");
    RADAR_INFO("\tposition to map:("<<msg->points[i].pose_seq.position.x<<","<<msg->points[i].pose_seq.position.y<<","<<msg->points[i].pose_seq.position.z<<")\n");
    RADAR_INFO("\tvel:"<<velocity<<std::endl);
    RADAR_INFO(endl);

    kf_tracker_point c;
    c.x = msg->points[i].pose_seq.position.x;
    c.y = msg->points[i].pose_seq.position.y;
    c.z = msg->points[i].pose_seq.position.z;
    c.r = msg->points[i].range;
    c.x_v = velocity_vec.x();
    c.y_v = velocity_vec.y();
    c.z_v = 0;
    c.vel_ang = angle;
    c.vel = std::fabs(velocity);
    c.vel_dir = (velocity>=0 ? true : false);
    c.cluster_id = -1;  // -1: not clustered
    c.tracking_id = -1; // -1: not be tracked
    c.visited = false;
    c.id = radar_id_count++;
    c.scan_time = call_back_num;
    c.rcs = msg->points[i].rcs;
    
    pcl::PointXYZI pt;
    pt.x = c.x;
    pt.y = c.y;
    pt.z = c.z;
    pt.intensity = msg->points[i].rcs;
    pub_filter_points->push_back(pt);
    cluster_data.push_back(c);
    
  }
}

void cluster_state(){
  // cluster the points with DBSCAN-based method
  if(firstFrame){
    dbtrack_seg.set_parameter(cluster_eps, cluster_Nmin, cluster_history_frames, cluster_dt_weight);
    dbtrack_seg.set_output_info(cluster_track_msg,motion_eq_optimizer_msg,rls_msg);
    dbtrack_seg.training_dbtrack_para(false);
  }
  double CLUSTER_START, CLUSTER_END;
  CLUSTER_START = clock();
  dbtrack_seg.scan_num = call_back_num;
  dbtrack_seg.data_period = dt;
  std::vector< std::vector<kf_tracker_point> > dbtrack_cluster, dbtrack_cluster_current;
  dbtrack_cluster_current = dbtrack_seg.cluster(cens);    // the current scan cluster result
  std::vector<int> cluster_tracking_result = dbtrack_seg.cluster_tracking_result(); // record the cluster-tracking id
  dbtrack_cluster.clear();
  dbtrack_cluster = dbtrack_seg.cluster_with_past_now();  // the cluster result that consist the past data
  CLUSTER_END = clock();
  
  // check the DBSCAN exe performance
  EXE_INFO(endl << "\033[42mDBTRACK Execution time: " << (CLUSTER_END - CLUSTER_START) / CLOCKS_PER_SEC << "s\033[0m");
  EXE_INFO(std::endl);

  // score the cluster performance
  // if(use_score_cluster && get_label && (fabs(label_vec.at(0).marker.header.stamp.toSec()-radar_stamp.toSec()) <= 0.08))
  score_cluster(dbtrack_cluster_current);
  
  cens.clear();
  cens = dbtrack_seg.get_center();        // the cluster center based on the current points
  rls_vel_marker(cens);
  
  // cluster visualization
  if(!viz_cluster_with_past){
    // For current cluster only
    color_cluster(dbtrack_cluster_current,true,cluster_tracking_result);
    polygon_cluster_visual(dbtrack_cluster_current);
  }
  else{
    // Cluster contains past and current
    color_cluster(dbtrack_cluster,true,cluster_tracking_result);
    polygon_cluster_visual(dbtrack_cluster);
  }

  // publish the history radar points
  past_radars_with_cluster.clear();
  past_radars_with_cluster = dbtrack_seg.get_history_points_with_cluster_order();
  cluster_tracking_result_dbpda.clear();
  cluster_tracking_result_dbpda.shrink_to_fit();
  cluster_tracking_result_dbpda = cluster_tracking_result;
}

void viz_overlay_radar(){
  past_radars.push_back(cens);
  if(past_radars.size()>cluster_history_frames){
    past_radars.erase(past_radars.begin());
  }
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr history_filter_points_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
  sensor_msgs::PointCloud2::Ptr history_filter_cloud(new sensor_msgs::PointCloud2);
  pcl::_PointXYZRGB color1, color2;
  color1.r = 145;
  color1.g = 255;
  color1.b = 124;

  color2.r = 1;
  color2.g = 15;
  color2.b = 10;

  for(int i=0;i<past_radars.size();i++){
    for(int j=0;j<past_radars.at(i).size();j++){
      pcl::PointXYZRGB pt;
      pt.x = past_radars.at(i).at(j).x;
      pt.y = past_radars.at(i).at(j).y;
      pt.z = past_radars.at(i).at(j).z;
      
      // light green to dark green
      pt.r = color1.r-(color1.r-color2.r)/past_radars.size()*(past_radars.size()-1-i);
      pt.g = color1.g-(color1.g-color2.g)/past_radars.size()*(past_radars.size()-1-i);
      pt.b = color1.b-(color1.b-color2.b)/past_radars.size()*(past_radars.size()-1-i);

      history_filter_points_rgb->points.push_back(pt);
    }
  }
  pcl::toROSMsg(*history_filter_points_rgb,*history_filter_cloud);
  history_filter_cloud->header.frame_id = MAP_FRAME;
  history_filter_cloud->header.stamp = radar_stamp;
  pub_filter.publish(history_filter_cloud);
}

void radar_process(){
  vel_marker(call_back_msg_list);
  tf_listener->waitForTransform(MAP_FRAME,CAR_FRAME,call_back_msg_list.at(0)->header.stamp,ros::Duration(1));
  try{
    tf_listener->lookupTransform(MAP_FRAME,CAR_FRAME,call_back_msg_list.at(0)->header.stamp,car2map);
  }
  catch(tf::TransformException& ex)
  {
    std::cout << "Exception thrown:" << ex.what()<< std::endl;
  }

  int radar_id_count = 0;
  // republish radar points
  pcl::PointCloud<pcl::PointXYZI>::Ptr in_filter_points(new pcl::PointCloud<pcl::PointXYZI>);
  sensor_msgs::PointCloud2::Ptr in_filter_cloud(new sensor_msgs::PointCloud2);
  in_filter_points->clear();
  // end republish init
  
  cens.clear();
  cens.shrink_to_fit();
  int sensor_id = 0;
  for(auto msg:call_back_msg_list){
    transform_radar_msg_to_cluster_data(msg,radar_id_count,in_filter_points,cens,sensor_id++);
  }
  if(cens.size() == 0){
    ROS_WARN("No Valid Callback Radar points In This Frame!");
    return;
  }

  // show the past radar points
  if(show_past_points && !use_KFT_module){
    viz_overlay_radar();
  }
  
  // republish filtered radar points
  pcl::toROSMsg(*in_filter_points,*in_filter_cloud);
  in_filter_cloud->header.frame_id = MAP_FRAME;
  in_filter_cloud->header.stamp = radar_stamp;
  pub_in_filter.publish(in_filter_cloud);
  
  // Cluster the radar points
  cluster_state();

  if(cens.size()==0){
    ROS_WARN("No Valid Cluster Radar points In This Frame!");
    return;
  }
  if(use_KFT_module){
    if( firstFrame ){
      ROS_INFO("Initialize the KFT Tracking");
      first_frame_KFT(cluster_tracking_result_dbpda);
      return;
    }
    // check the Tracking exe performance
    double START_TIME, END_TIME;
    START_TIME = clock();
    if(cluster_history_frames==1 || !KFT_detection_choose)
      KFT();
    else
      cluster_KFT(cluster_tracking_result_dbpda);
    END_TIME = clock();
    EXE_INFO("\033[1;42mKFT Execution time: " << (END_TIME - START_TIME) / CLOCKS_PER_SEC << "s\033[0m");
    EXE_INFO(std::endl);
  }
  return;
}

void single_radar_callback(const radar_scenes::RadarScenesConstPtr& msg){
  if(msg->points.size()==0){
    radar_stamp = msg->header.stamp;
    return;
  }
  std::cout << std::fixed; std::cout.precision(3);
  std::cout<<"\n\n\033[1;4;100m"<<++call_back_num<<" Call Back radar points:"<<msg->points.size()<<", Frame:"<<msg->header.frame_id.c_str()<<", Stamp: "<<msg->header.stamp.toSec()<<"\033[0m"<<endl;
  if(!firstFrame){
    dt = msg->header.stamp.toSec() - radar_stamp.toSec();
    std::cout<<"dt: "<<dt<<std::endl;
  }
  radar_stamp = msg->header.stamp;
  call_back_msg_list.clear();
  call_back_msg_list.push_back(msg);

  // double CALLBACK_START, CALLBACK_END;
  // CALLBACK_START = clock();
  radar_process();
  // CALLBACK_END = clock();
  // EXE_INFO("\n");
  // EXE_INFO("\033[1;42mCallback Execution time: " << (CALLBACK_END - CALLBACK_START) / CLOCKS_PER_SEC << "s\033[0m");
}

void radar_callback(const radar_scenes::RadarScenesConstPtr& msg_1,
                    const radar_scenes::RadarScenesConstPtr& msg_2,
                    const radar_scenes::RadarScenesConstPtr& msg_3,
                    const radar_scenes::RadarScenesConstPtr& msg_4){
  int radar_point_size = msg_1->points.size() + msg_2->points.size() + msg_3->points.size() + msg_4->points.size();
  std::cout<<"\n\n\033[1;4;100m"<<++call_back_num<<" Call Back radar points:"<<radar_point_size
           <<"("<<msg_1->points.size()<<"+"<<msg_2->points.size()<<"+"<<msg_3->points.size()<<"+"<<msg_4->points.size()<<")\033[0m"<<endl;
  if(radar_point_size==0) //radar points cannot be zero, it would cause segement fault
    return;
  // double CALLBACK_START, CALLBACK_END;
  // CALLBACK_START = clock();
  if(!firstFrame){
    dt = msg_1->header.stamp.toSec() - radar_stamp.toSec();
    std::cout<<"dt: "<<dt<<std::endl;
  }
  radar_stamp = msg_1->header.stamp;
  std::cout << std::fixed; std::cout.precision(3);
  std::cout<<"Radar time:"<<msg_1->header.stamp.toSec()<<" / "<<msg_2->header.stamp.toSec()<<" / "<<msg_3->header.stamp.toSec()<<" / "<<msg_4->header.stamp.toSec()<<endl;
  
  call_back_msg_list.clear();
  call_back_msg_list.push_back(msg_1);
  call_back_msg_list.push_back(msg_2);
  call_back_msg_list.push_back(msg_3);
  call_back_msg_list.push_back(msg_4);

  radar_process();
  // CALLBACK_END = clock();
  // EXE_INFO("\033[1;42mCallback Execution time: " << (CALLBACK_END - CALLBACK_START) / CLOCKS_PER_SEC << "s\033[0m");

}

void get_bag_info_callback(const rosgraph_msgs::LogConstPtr& log_msg){
  if(log_msg->msg.find(".bag") != log_msg->msg.npos){
    filename_change = true;
    score_file_name = log_msg->msg;
    int bag_start = score_file_name.find("s");
    int bag_end = score_file_name.find(".");
    score_file_name = score_file_name.substr(bag_start,bag_end-bag_start);
    dbtrack_seg.get_training_name(score_file_name);
  }
  return;
}

int main(int argc, char** argv){
  ros::init(argc,argv,"radar_kf_track_radarscenes");
  ros::NodeHandle nh;
  marker_color_init();
  transform_init();
  tf_listener = new tf::TransformListener();
  message_filters::Subscriber<radar_scenes::RadarScenes> sub_1(nh,"/radar_1",1);   // sub the radar points(outlier)
  message_filters::Subscriber<radar_scenes::RadarScenes> sub_2(nh,"/radar_2",1);   // sub the radar points(outlier)
  message_filters::Subscriber<radar_scenes::RadarScenes> sub_3(nh,"/radar_3",1);   // sub the radar points(outlier)
  message_filters::Subscriber<radar_scenes::RadarScenes> sub_4(nh,"/radar_4",1);   // sub the radar points(outlier)
  // ros::Subscriber sub_r1 = nh.subscribe("radar_1",1,&single_radar_callback);
  // ros::Subscriber sub_r2 = nh.subscribe("radar_2",1,&single_radar_callback);
  // ros::Subscriber sub_r3 = nh.subscribe("radar_3",1,&single_radar_callback);
  // ros::Subscriber sub_r4 = nh.subscribe("radar_4",1,&single_radar_callback);

  pub_cluster_marker = nh.advertise<visualization_msgs::MarkerArray>("cluster_index", 1);
  pub_marker = nh.advertise<visualization_msgs::MarkerArray>("tracking_id", 1);
  pub_filter = nh.advertise<sensor_msgs::PointCloud2>("radar_history",500);
  pub_in_filter = nh.advertise<sensor_msgs::PointCloud2>("inlier_radar",200);
  pub_cluster_center = nh.advertise<sensor_msgs::PointCloud2>("cluster_center",500);
  pub_cluster_pointcloud = nh.advertise<sensor_msgs::PointCloud2>("cluster_radar_front_pointcloud",500);
  pub_tracker_point = nh.advertise<sensor_msgs::PointCloud2>("tracker_point",500);
  pub_gt_cloud = nh.advertise<sensor_msgs::PointCloud2>("radarScenes_gt_pc",10);
  vis_vel_rls = nh.advertise<visualization_msgs::MarkerArray> ("radar_rls_v", 1);
  vis_vel_comp = nh.advertise<visualization_msgs::MarkerArray> ("radar_front_v_comp", 1);
  vis_tracking_vel = nh.advertise<visualization_msgs::MarkerArray> ("radar_tracking_v", 1);
  pub_trajectory = nh.advertise<visualization_msgs::MarkerArray> ("tracking_trajectory_path", 1);
  pub_trajectory_smooth = nh.advertise<visualization_msgs::MarkerArray> ("tracking_smooth_trajectory", 1);
  pub_pt = nh.advertise<visualization_msgs::MarkerArray> ("tracking_pt", 1);
  pub_pred = nh.advertise<visualization_msgs::MarkerArray> ("predict_pt", 1);
  pub_anno_cluster = nh.advertise<sensor_msgs::PointCloud2> ("annotation_pub", 300);
  pub_cluster_hull = nh.advertise<visualization_msgs::MarkerArray> ("cluster_hull", 1);
  pub_cluster_bbox = nh.advertise<jsk_recognition_msgs::BoundingBoxArray> ("cluster_bbox", 1);
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
  nh.param<bool>("output_gt_pt_info"    ,output_gt_pt_info    ,true);
  nh.param<bool>("output_score_mat_info",output_score_mat_info,true);
  nh.param<bool>("write_out_score_file" ,write_out_score_file ,true);
  nh.param<bool>("DA_method"            ,DA_choose            ,false);
  nh.param<bool>("use_KFT_module"       ,use_KFT_module       ,true);
  nh.param<bool>("KFT_detection_choose" ,KFT_detection_choose ,true);
  nh.param<bool>("show_stopObj_id"      ,show_stopObj_id      ,true);
  nh.param<int> ("kft_id_num"           ,kft_id_num           ,1);
  nh.param<bool>("show_vel_marker"      ,show_vel_marker      ,false);
  nh.param<bool>("use_score_cluster"    ,use_score_cluster    ,false);
  // cluster state parameters
  nh.param<bool>("dbtrack_para_train"   ,dbtrack_para_train   ,false);
  nh.param<bool>("dbtrack_para_train_direct"   ,dbtrack_para_train_direct   ,false);
  nh.param<double>("eps"                ,cluster_eps          ,2.5);
  nh.param<int>("Nmin"                  ,cluster_Nmin         ,2);
  nh.param<int>("history_frames"        ,cluster_history_frames  ,3);
  nh.param<double>("cluster_dt_weight"  ,cluster_dt_weight    ,0.0);
  nh.param<bool>("viz_cluster_with_past",viz_cluster_with_past,true);
  nh.param<bool>("cluster_track_msg"    ,cluster_track_msg    ,false);
  nh.param<bool>("motion_eq_optimizer_msg",motion_eq_optimizer_msg,false);
  nh.param<bool>("rls_msg"              ,rls_msg              ,false);
  
  ros::Subscriber bag_sub;
  bag_sub = nh.subscribe("rosout",1,&get_bag_info_callback); // sub the bag name
  message_filters::Synchronizer<radar_sync>* all_radar_sync_;
  all_radar_sync_ = new message_filters::Synchronizer<radar_sync>(radar_sync(10), sub_1, sub_2, sub_3, sub_4);
  all_radar_sync_->registerCallback(boost::bind(&radar_callback, _1, _2, _3, _4));
  

  ROS_INFO("Setting:");
  ROS_INFO("Data association method : %s" , DA_choose ? "hungarian" : "my DA method(Greedy-like)");
  ROS_INFO_COND(output_radar_info     ,"Output Radar information");
  ROS_INFO_COND(output_cluster_info   ,"Output Cluster information");
  ROS_INFO_COND(output_dbscan_info    ,"Output DBSCAN information");
  ROS_INFO_COND(output_obj_id_result  ,"Output Obj-id Result");
  ROS_INFO_COND(output_DA_pair        ,"Output Data Association Pair");
  ROS_INFO_COND(output_exe_time       ,"Output Execution Time");
  ROS_INFO_COND(output_label_info     ,"Output Label information");
  ROS_INFO_COND(show_vel_marker       ,"Publish Velocity marker");
  // kalman filter info
  if(use_KFT_module){
    ROS_INFO_COND(use_KFT_module        ,"Use KFT module (Tracking part)");
    ROS_INFO("Show %d types marker id"          , kft_id_num);
    ROS_INFO("%s the stop object tracker"       ,show_stopObj_id ? "Show":"Mask");
    ROS_INFO_COND(output_KFT_result     ,"Output KFT Result");
  }
  // cluster info
  ROS_INFO("Cluster -> Eps:%f, Nmin:%d, Accumulate Frames:%d",cluster_eps,cluster_Nmin,cluster_history_frames);
  ROS_INFO("Cluster dt weight for vel function:%f",cluster_dt_weight);
  ROS_INFO("Visualize the cluster with %s" ,viz_cluster_with_past ? "past and now scans":"now scan only");
  // cluster score info
  if(use_score_cluster){
    ROS_INFO("\tBox bias -> w: %f, l: %f", box_bias_w, box_bias_l);
    ROS_INFO("\tOutput score info : %s"       , output_score_info ? "True" : "False");
    ROS_INFO("\tOutput score mat info : %s"   , output_score_mat_info ? "True" : "False");
    ROS_INFO("\tOutput GT inside points info : %s"   , output_gt_pt_info ? "True" : "False");
    ROS_INFO("\tWrite out the csv files : %s" , write_out_score_file ? "True" : "False");
  }

  while(ros::ok()){
    ros::spinOnce();
    // train DBSCAN parameter directly
    if(dbtrack_para_train_direct){
      dbtrack_seg.data_period = dt;
      dbtrack_seg.set_parameter(cluster_eps, cluster_Nmin, cluster_history_frames, cluster_dt_weight);
      dbtrack_seg.set_output_info(cluster_track_msg,motion_eq_optimizer_msg,rls_msg);
      dbtrack_seg.training_dbtrack_para(dbtrack_para_train_direct);
      // visualize the training result
      std::vector< std::vector<kf_tracker_point> > dbtrack_cluster = dbtrack_seg.cluster_with_past_now();
      std::vector<int> cluster_tracking_result = dbtrack_seg.cluster_tracking_result();
      past_radars_with_cluster.clear();
      past_radars_with_cluster = dbtrack_seg.get_history_points_with_cluster_order();
      color_cluster(dbtrack_cluster,true,cluster_tracking_result);
      polygon_cluster_visual(dbtrack_cluster);
      show_past_radar_points_with_color();
      dbtrack_para_train_direct = false;
    }
    pub_marker.publish(m_s);
  }
  return 0;
}

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
// #include <pcl/io/pcd_io.h>
#include "pcl_ros/point_cloud.h"
#include <pcl/impl/point_types.hpp>
// #include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/geometry.h>
// #include <pcl/filters/extract_indices.h>
// #include <pcl/filters/passthrough.h>
// #include <pcl/filters/voxel_grid.h>
// #include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
// #include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h>
#include <pcl/ModelCoefficients.h>
// #include <pcl/filters/crop_box.h>

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


#include <conti_radar/Measurement.h>
#include <nav_msgs/Odometry.h>

// to sync the subscriber
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// cluster lib
#include "dbscan/dbscan.h"
#include "optics/optics.h"
#include "dbpda/dbpda.h"

// visualize the cluster result
#include "cluster_visual/cluster_visual.h"
#include <geometry_msgs/PolygonStamped.h>

// get keyboard
#include <termios.h>

using namespace std;
using namespace cv;

#define iteration 20 //plan segmentation
#define trajectory_frame 15

typedef message_filters::sync_policies::ApproximateTime<conti_radar::Measurement,
														                            nav_msgs::Odometry> NoCloudSyncPolicy;
typedef message_filters::sync_policies::ApproximateTime<conti_radar::Measurement,conti_radar::Measurement,
														                            nav_msgs::Odometry> ego_in_out_sync;

typedef message_filters::sync_policies::ApproximateTime<conti_radar::Measurement,conti_radar::Measurement,
														                            conti_radar::Measurement,conti_radar::Measurement,
                                                        conti_radar::Measurement> radar_sync;

typedef pcl::PointXYZ PointT;

ros::Publisher pub_cluster_marker;  // pub cluster index
ros::Publisher pub_marker;          // pub tracking id
ros::Publisher pub_filter;          // pub filter points
ros::Publisher pub_in_filter;       // pub the inlier points
ros::Publisher pub_out_filter;      // pub the outlier points
ros::Publisher pub_cluster_center;  // pub cluster points
ros::Publisher pub_cluster_pointcloud;  // pub cluster point cloud
ros::Publisher vis_vel_comp;        // pub radar comp vel
ros::Publisher vis_vel;             // pub radar vel
ros::Publisher pub_trajectory;      // pub the tracking trajectory
ros::Publisher pub_pt;              // pub the tracking history pts
ros::Publisher pub_pred;            // pub the predicted pts
ros::Publisher pub_annotation;      // pub the label annotaion to /map frame and set the duration to 0.48s
ros::Publisher pub_anno_cluster;    // pub the cluster from the annotation
ros::Publisher pub_cluster_hull;    // pub the cluster polygon result
ros::Publisher pub_radar_debug;     // pub the radar debug lines
ros::Publisher pub_debug_id_text;   // pub the radar invalid_id to debug the weird vel

std::vector<string> frame_list{ "nuscenes_radar_front",
                                "nuscenes_radar_front_right",
                                "nuscenes_radar_front_left",
                                "nuscenes_radar_back_right",
                                "nuscenes_radar_back_left"};
tf::StampedTransform echo_transform;    // tf between map and radar_front
tf::StampedTransform echo_transform_fr; // tf between map and radar_front_right
tf::StampedTransform echo_transform_fl; // tf between map and radar_front_left
tf::StampedTransform echo_transform_br; // tf between map and radar_back_right
tf::StampedTransform echo_transform_bl; // tf between map and radar_back_left
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
bool use_5_radar = false;       // true: use 5 radar points, false: use follow callback
bool use_ego_callback = false;  // true: use the ego motion in/outlier pts, false: use the original pts
bool use_score_cluster = false; // true: score the cluster performance

#define KFT_INFO(msg)     if (output_KFT_result) {std::cout << msg;}
#define RADAR_INFO(msg)   if (output_radar_info) {std::cout << msg;}
#define DBSCAN_INFO(msg)  if (output_dbscan_info) {std::cout << msg;}
#define EXE_INFO(msg)     if (output_exe_time) {std::cout << msg;}
#define EGO_INFO(msg)     if (output_ego_vel_info) {std::cout << msg;}

#define MAP_FRAME "/map"
#define RADAR_FRONT_FRAME "/nuscenes_radar_front"

int call_back_num = 0;  // record the callback number
std::list<int> invalid_state = {1,2,3,5,6,7,13,14}; // to check the state of the conti radar
std::vector<std::string> score_skip_ns = {"movable_object"}; // to skip score the cluster performance in some gt_marker

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
int max_size = 0, cluster_marker_max_size = 0, cluster_polygon_max_size = 0, trajcetory_max_size = 0, vel_marker_max_size = 0;
int debug_id_max_size = 0;
typedef struct marker_color
{
  std_msgs::ColorRGBA color;
  geometry_msgs::Vector3 scale;
}marker_color;
marker_color vel_color, vel_comp_color;
marker_color tra_color, history_pt_color, pred_pt_color;
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
float dt = 0.08f;     //0.1f 0.08f=1/13Hz(radar)
float sigmaP = 0.01;  //0.01
float sigmaQ = 0.1;   //0.1
#define bias 4.0      // used for data association bias 5 6
#define mahalanobis_bias 5.5 // used for data association(mahalanobis dist)
int id_count = 0; //the counter to calculate the current occupied track_id
#define frame_lost 10   // 5 10

std::vector <int> id;
ros::Publisher objID_pub;
// KF init
int stateDim = 6;// [x,y,v_x,v_y]//,w,h]
int measDim = 3;// [z_x,z_y//,z_w,z_h]
int ctrlDim = 0;// control input 0(acceleration=0,constant v model)


bool firstFrame = true;
bool get_transformer = false;
tf::TransformListener *tf_listener;

float motion_vel_threshold = 0.10;

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
// dbscan init
dbscan dbscan_seg;

// OPTICS init
optics optics_seg(2.5,3);

// DBPDA init
dbpda dbpda_seg;

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
string csv_name;        // to output the specific csv name
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
    // marker.header.frame_id="/nuscenes_radar_front";
    marker.header.frame_id = MAP_FRAME;
    marker.header.stamp = ros::Time();//to show every tag  (ros::Time::now()for latest tag)
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.id = k;
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;

    marker.scale.z = 1.0f;  // rgb(255,127,122)
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
      if(tracking_frame_obj_id.at(k)!=-1){  // track id appeaars more than 5
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
    // float color = 255 * 2 / (RAND_MAX + 1.0);
    // add cluster index marker
    // marker.header.frame_id="/nuscenes_radar_front";
    marker.header.frame_id = MAP_FRAME;
    marker.header.stamp = radar_stamp;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.id = i;
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.scale.z = 1.5f;  // rgb(255,127,122)
    // marker.color.r = color * 3.34448160535f;
    // marker.color.g = color * 1.70357751278f;
    // marker.color.b = color * 8.77192982456f;
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
  // cluster_cloud->header.frame_id = "/nuscenes_radar_front";
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
  for(int i=0;i<cluster_list.size();i++){
    pcl::PointCloud <pcl::PointXYZ>::Ptr pointsCluster(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointXYZ pt;
    cluster_visual_unit cluster_unit;
    std::vector<float> temp_x_radius;
    std::vector<float> temp_y_radius;
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
        cluster_unit.center = pt;
        cluster_unit.x_radius = 1.0f;
        cluster_unit.y_radius = 1.0f;
        single_point_cluster.push_back(cluster_unit);
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
        for(int j=1;j<3;j++){
          pt.x = cluster_list.at(i).at(j).x + T2_vec.x();
          pt.y = cluster_list.at(i).at(j).y + T2_vec.y();
          pt.z = 0;
          pointsCluster->points.push_back(pt);
          pt.x = cluster_list.at(i).at(j).x - T2_vec.x();
          pt.y = cluster_list.at(i).at(j).y - T2_vec.y();
          pt.z = 0;
          pointsCluster->points.push_back(pt);
        }
        break;
      default:
        for(int j=0;j<cluster_list.at(i).size();j++){
          pt.x = cluster_list.at(i).at(j).x;
          pt.y = cluster_list.at(i).at(j).y;
          pt.z = 0;
          pointsCluster->points.push_back(pt);
        }
        break;
    }
    if(pointsCluster->points.size()>0){
      Cluster_visual clusterObject;
      clusterObject.SetCloud(pointsCluster, header, true);
      polygon_vec.push_back(clusterObject.GetPolygon());
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
void vel_marker(const conti_radar::MeasurementConstPtr& input){
	visualization_msgs::MarkerArray marker_array_comp;
	visualization_msgs::MarkerArray marker_array;
  visualization_msgs::MarkerArray debug_id_array;
  visualization_msgs::Marker id_marker;
  id_marker.header.frame_id = RADAR_FRONT_FRAME;
  id_marker.header.stamp = ros::Time();
  id_marker.action = visualization_msgs::Marker::ADD;
  id_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  id_marker.pose.orientation.w = 1.0;
  id_marker.scale.z = 0.8f;
  id_marker.color = vel_color.color;
  id_marker.color.a = 1;

	for(int i=0; i<input->points.size(); i++){
		visualization_msgs::Marker marker;
		marker.header.frame_id = RADAR_FRONT_FRAME;
		marker.header.stamp = ros::Time();
		marker.id = i;
		marker.type = visualization_msgs::Marker::ARROW;
		marker.action = visualization_msgs::Marker::ADD;
		marker.pose.position.x = input->points[i].longitude_dist;
		marker.pose.position.y = input->points[i].lateral_dist;
		marker.pose.position.z = 0;
		float theta = atan2(input->points[i].lateral_vel_comp,
												input->points[i].longitude_vel_comp);
		tf2::Quaternion Q;
		Q.setRPY( 0, 0, theta );
		marker.pose.orientation = tf2::toMsg(Q);

    marker.scale = vel_comp_color.scale;
		marker.scale.x = sqrt(pow(input->points[i].lateral_vel_comp,2) + 
				 									pow(input->points[i].longitude_vel_comp,2)); //~lenght~//

		marker.color = vel_comp_color.color;
		marker_array_comp.markers.push_back(marker);
    if(marker.scale.x > 0.2){
      id_marker.id = i;
      id_marker.text = to_string(input->points[i].invalid_state);
      id_marker.pose.position = marker.pose.position;
      debug_id_array.markers.push_back(id_marker);
    }

		///////////////////////////////////////////////////////////////////
    tf::Vector3 msg_vel(input->points[i].longitude_vel,
                        input->points[i].lateral_vel,
                        0);
    tf::Vector3 msg_pos(input->points[i].longitude_dist,
                        input->points[i].lateral_dist,
                        0);
    msg_pos.normalize();
    tf::Vector3 project_radial_vel = msg_pos.dot(msg_vel) * msg_pos;
    theta = atan2(project_radial_vel.y(),project_radial_vel.x());
		// theta = atan2(input->points[i].lateral_vel,
		// 										input->points[i].longitude_vel);
		Q.setRPY( 0, 0, theta );
		marker.pose.orientation = tf2::toMsg(Q);

		// marker.scale.x = sqrt(pow(input->points[i].lateral_vel,2) + 
		// 		 									pow(input->points[i].longitude_vel,2)); //~lenght~//
		marker.scale = vel_color.scale;
    marker.scale.x = project_radial_vel.length();

		marker.color = vel_color.color;
		marker_array.markers.push_back(marker);
	}
  if(input->points.size() > vel_marker_max_size)
    vel_marker_max_size = input->points.size();
  for(int a=input->points.size();a<vel_marker_max_size;a++){
    visualization_msgs::Marker marker;
		marker.header.frame_id = RADAR_FRONT_FRAME;
		marker.header.stamp = ros::Time();
		marker.id = a;
		marker.type = visualization_msgs::Marker::ARROW;
		marker.action = visualization_msgs::Marker::ADD;
		marker.pose.position.x = 0;
		marker.pose.position.y = 0;
		marker.pose.position.z = 0;
    marker.color.a = 0;
		marker_array_comp.markers.push_back(marker);
		marker_array.markers.push_back(marker);
  }
  if(debug_id_array.markers.size()>debug_id_max_size)
    debug_id_max_size = debug_id_array.markers.size();
  for(int a=debug_id_array.markers.size();a<debug_id_max_size;a++){
    id_marker.color.a = 0;
    debug_id_array.markers.push_back(id_marker);
  }
	vis_vel_comp.publish( marker_array_comp );
	vis_vel.publish( marker_array );
  pub_debug_id_text.publish(debug_id_array);
}

// show the radar velocity arrow (inlier & outlier)
void vel_marker(const conti_radar::MeasurementConstPtr& in_msg, const conti_radar::MeasurementConstPtr& out_msg){
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
  id_marker.scale.z = 2.3f;
  id_marker.scale.x = 0.5f;
  // id_marker.color = vel_color.color;
  id_marker.color.a = 1;
  id_marker.color.r = 10.0f/255.0f;
  id_marker.color.g = 215.0f/255.0f;
  id_marker.color.b = 255.0f/255.0f;
  id_marker.lifetime = ros::Duration(dt-0.005);
  int id_marker_idx = 0;
  int count_vel_num = 0;
	for(int i=0; i<in_msg->points.size(); i++){
    std::list<int>::iterator invalid_state_it;
    invalid_state_it = std::find(invalid_state.begin(),invalid_state.end(),in_msg->points[i].invalid_state);
    if(invalid_state_it!=invalid_state.end())
      continue;
		visualization_msgs::Marker marker;
		marker.header.frame_id = RADAR_FRONT_FRAME;
		marker.header.stamp = ros::Time();
		marker.id = count_vel_num++;
		marker.type = visualization_msgs::Marker::ARROW;
		marker.action = visualization_msgs::Marker::ADD;
		marker.pose.position.x = in_msg->points[i].longitude_dist;
		marker.pose.position.y = in_msg->points[i].lateral_dist;
		marker.pose.position.z = 0;
		float theta = atan2(in_msg->points[i].lateral_vel_comp,
												in_msg->points[i].longitude_vel_comp);
		tf2::Quaternion Q;
		Q.setRPY( 0, 0, theta );
		marker.pose.orientation = tf2::toMsg(Q);
    marker.scale = vel_comp_color.scale;
		marker.scale.x = sqrt(pow(in_msg->points[i].lateral_vel_comp,2) + 
				 									pow(in_msg->points[i].longitude_vel_comp,2)); //~lenght~//
    
    marker.color = vel_comp_color.color;
		marker_array_comp.markers.push_back(marker);
    if(marker.scale.x > 0.5){
      id_marker.id = id_marker_idx++;
      id_marker.text = to_string(in_msg->points[i].invalid_state);
      id_marker.pose.position = marker.pose.position;
      id_marker.pose.position.z = 1.0f;
      debug_id_array.markers.push_back(id_marker);
    }

		///////////////////////////////////////////////////////////////////
    tf::Vector3 msg_vel(in_msg->points[i].longitude_vel,
                        in_msg->points[i].lateral_vel,
                        0);
    tf::Vector3 msg_pos(in_msg->points[i].longitude_dist,
                        in_msg->points[i].lateral_dist,
                        0);
    msg_pos.normalize();
    tf::Vector3 project_radial_vel = msg_pos.dot(msg_vel) * msg_pos;
    theta = atan2(project_radial_vel.y(),project_radial_vel.x());
		// theta = atan2(in_msg->points[i].lateral_vel,
		// 										in_msg->points[i].longitude_vel);
		Q.setRPY( 0, 0, theta );
		marker.pose.orientation = tf2::toMsg(Q);

		// marker.scale.x = sqrt(pow(in_msg->points[i].lateral_vel,2) + 
		// 		 									pow(in_msg->points[i].longitude_vel,2)); //~lenght~//
		marker.scale = vel_color.scale;
    marker.scale.x = project_radial_vel.length();

		marker.color = vel_color.color;
		marker_array.markers.push_back(marker);
	}
  for(int i=0; i<out_msg->points.size(); i++){
    std::list<int>::iterator invalid_state_it;
    invalid_state_it = std::find(invalid_state.begin(),invalid_state.end(),out_msg->points[i].invalid_state);
    if(invalid_state_it!=invalid_state.end())
      continue;
		visualization_msgs::Marker marker;
		marker.header.frame_id = RADAR_FRONT_FRAME;
		marker.header.stamp = ros::Time();
		marker.id = count_vel_num++;
		marker.type = visualization_msgs::Marker::ARROW;
		marker.action = visualization_msgs::Marker::ADD;
		marker.pose.position.x = out_msg->points[i].longitude_dist;
		marker.pose.position.y = out_msg->points[i].lateral_dist;
		marker.pose.position.z = 0;
		float theta = atan2(out_msg->points[i].lateral_vel_comp,
												out_msg->points[i].longitude_vel_comp);
		tf2::Quaternion Q;
		Q.setRPY( 0, 0, theta );
		marker.pose.orientation = tf2::toMsg(Q);

    marker.scale = vel_comp_color.scale;
		marker.scale.x = sqrt(pow(out_msg->points[i].lateral_vel_comp,2) + 
				 									pow(out_msg->points[i].longitude_vel_comp,2)); //~lenght~//

		marker.color = vel_comp_color.color;
		marker_array_comp.markers.push_back(marker);
    if(marker.scale.x > 0.5){
      id_marker.id = id_marker_idx++;
      id_marker.text = to_string(out_msg->points[i].invalid_state);
      id_marker.pose.position = marker.pose.position;
      id_marker.pose.position.z = 1.0f;
      debug_id_array.markers.push_back(id_marker);
    }
		///////////////////////////////////////////////////////////////////
    tf::Vector3 msg_vel(out_msg->points[i].longitude_vel,
                        out_msg->points[i].lateral_vel,
                        0);
    tf::Vector3 msg_pos(out_msg->points[i].longitude_dist,
                        out_msg->points[i].lateral_dist,
                        0);
    msg_pos.normalize();
    tf::Vector3 project_radial_vel = msg_pos.dot(msg_vel) * msg_pos;
    theta = atan2(project_radial_vel.y(),project_radial_vel.x());
		// theta = atan2(out_msg->points[i].lateral_vel,
		// 										out_msg->points[i].longitude_vel);
		Q.setRPY( 0, 0, theta );
		marker.pose.orientation = tf2::toMsg(Q);

		// marker.scale.x = sqrt(pow(out_msg->points[i].lateral_vel,2) + 
		// 		 									pow(out_msg->points[i].longitude_vel,2)); //~lenght~//
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

// show the tracking trajectory
void show_trajectory(){
  int k=0;
  visualization_msgs::MarkerArray tra_array, point_array, pred_point_array;
  tra_array.markers.clear();
  point_array.markers.clear();
  pred_point_array.markers.clear();
  for(k=0; k<filters.size(); k++){
    geometry_msgs::Point pred;
    pred.x = filters.at(k).pred_v.x;
    pred.y = filters.at(k).pred_v.y;
    pred.z = filters.at(k).pred_v.z;
    float velocity = sqrt(pred.x*pred.x + pred.y*pred.y + pred.z*pred.z);

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
      marker.id = k;
      marker.pose.orientation.w = 1.0;
      marker.scale = tra_color.scale;
      marker.color = tra_color.color;

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

      if (filters.at(k).history.size() < trajectory_frame){
        for (int i=0; i<filters.at(k).history.size(); i++){
            geometry_msgs::Point pt = filters.at(k).history.at(i);
            marker.points.push_back(pt);
            P.points.push_back(pt);
        }
      }
      else{
        for (vector<geometry_msgs::Point>::const_reverse_iterator r_iter = filters.at(k).history.rbegin(); r_iter != filters.at(k).history.rbegin() + trajectory_frame; ++r_iter){
            geometry_msgs::Point pt = *r_iter;
            marker.points.push_back(pt);
            P.points.push_back(pt);
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
        P.color.a = 0;
        Predict_p.color.a = 0;
      }

      tra_array.markers.push_back(marker);
      point_array.markers.push_back(P);
      pred_point_array.markers.push_back(Predict_p);
    }
  }
  if(tra_array.markers.size() > trajcetory_max_size)
    trajcetory_max_size = tra_array.markers.size();
  for(int a = tra_array.markers.size();a<trajcetory_max_size;a++){
    visualization_msgs::Marker marker;
    marker.header.frame_id = MAP_FRAME;
    marker.header.stamp = radar_stamp;
    marker.lifetime = ros::Duration(dt);
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.ns = "trajectory";
    marker.id = a;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.2f;
    marker.color.a = 0;
    tra_array.markers.push_back(marker);
  }

  pub_trajectory.publish(tra_array);
  pub_pt.publish(point_array);
  pub_pred.publish(pred_point_array);
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
  double distance = cv::Mahalanobis(x,y,cov);
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
    cv::setIdentity(ka.measurementNoiseCov, cv::Scalar(sigmaQ));
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
  // std::vector<cv::Mat> pred; // get the predict result from KF, and each filter is one points(from past on)
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
      // std::cout<<"Row "<<row_count++<<":";
      std::vector<double> distVec;
      for(int n=0;n<cens.size();n++)
      {
        // double dis_1 = euclidean_distance((*it).pred_pose,copyOfClusterCenters[n]);
        // double dis_2 = euclidean_distance((*it).pred_pose_us,copyOfClusterCenters[n]);
        
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
        
        // distVec.push_back(euclidean_distance((*it).pred_pose,copyOfClusterCenters[n]));
        // std::cout<<"\t"<<distVec.at(n);
      }
      // std::cout<<endl;
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
    
    float dist_thres;
    if(dist_thres1<dist_thres2){
      dist_thres = dist_thres2;
    }
    else
      dist_thres = dist_thres1;
    
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
      if( dist_vec.at(assignment[k]) <=  mahalanobis_bias ){
        obj_id[assignment[k]] = filters.at(k).uuid;
        if(filters.at(k).track_frame>10)
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
        
        // cens.at(assignment[k]).
        geometry_msgs::Point tracking_pt_history;
        tracking_pt_history.x = cens.at(assignment[k]).x;
        tracking_pt_history.y = cens.at(assignment[k]).y;
        tracking_pt_history.z = cens.at(assignment[k]).z;
        filters.at(k).history.push_back(tracking_pt_history);

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
  // m_s.markers.resize(cens.size());
  m_s.markers.clear();
  m_s.markers.shrink_to_fit();

  // marker_id(false,marker_obj_id);   // publish the "tracking" id
  marker_id(false,obj_id,marker_obj_id,tracking_frame_obj_id);       // publish all id
  show_trajectory();
  
///////////////////////////////////////////////////estimate

  int num = filters.size();
  float meas[num][3];
  float meas_us[num][3];
  i = 0;
  for(std::vector<track>::const_iterator it = filters.begin(); it != filters.end(); ++it){
    if ( (*it).tracking_state == TRACK_STATE::tracking ){
        kf_tracker_point pt = cens[(*it).cluster_idx];
        meas[i][0] = pt.x;
        meas[i][1] = pt.y;
        meas[i][2] = pt.z;
        meas_us[i][0] = pt.x;
        meas_us[i][1] = pt.y;
        meas_us[i][2] = pt.z;
    }
    else if ( (*it).tracking_state == TRACK_STATE::missing ){
        meas[i][0] = (*it).pred_pose.x; //+ (*it).pred_v.x;
        meas[i][1] = (*it).pred_pose.y;//+ (*it).pred_v.y;
        meas[i][2] = (*it).pred_pose.z; //+ (*it).pred_v.z;
        meas_us[i][0] = (*it).pred_pose_us.x;
        meas_us[i][1] = (*it).pred_pose_us.y;
        meas_us[i][2] = (*it).pred_pose_us.z;
    }
    else
    {
        std::cout<<"Some tracks state not defined to tracking/lost."<<std::endl;
    }
    
    i++;
  }

  // std::cout<<"mesurement record."<<std::endl;
  cv::Mat measMat[num];
  cv::Mat measMat_us[num];
  for(int i=0;i<num;i++){
    measMat[i]=cv::Mat(3,1,CV_32F,meas[i]);
    measMat_us[i]=cv::Mat(3,1,CV_32F,meas_us[i]);
  }

  // The update phase 
  
  // if (!(measMat[0].at<float>(0,0)==0.0f || measMat[0].at<float>(1,0)==0.0f))
  //     Mat estimated0 = KF[0].correct(measMat[0]);
  cv::Mat estimated[num];
  cv::Mat estimated_us[num];
  i = 0;
  for(std::vector<track>::iterator it = filters.begin(); it != filters.end(); ++it){
    estimated[i] = (*it).kf.correct(measMat[i]);
    estimated_us[i] = (*it).kf_us.correct(measMat_us[i]);
    // std::cout << "The corrected state of "<<i<<"th KF is: "<<estimated[i].at<float>(0)<<","<<estimated[i].at<float>(1)<<endl;
    i++;
  }

  return;
}

void get_transform(ros::Time timestamp, int frame_index=0){
  string chlild_frame = frame_list.at(frame_index);
  tf_listener->waitForTransform(chlild_frame,MAP_FRAME,timestamp,ros::Duration(1));
  std::cout << std::fixed; std::cout.precision(3);
  tf::StampedTransform temp_transform;
  try{
    tf_listener->lookupTransform(MAP_FRAME,chlild_frame,timestamp,temp_transform);
    if(output_transform_info){
      std::cout << "\033[1;33mGet Transform:\033[0m" << std::endl;
      std::cout << "- At time: " << temp_transform.stamp_.toSec() << std::endl;
      tf::Quaternion q = temp_transform.getRotation();
      tf::Vector3 v = temp_transform.getOrigin();
      std::cout << "- Translation: [" << v.getX() << ", " << v.getY() << ", " << v.getZ() << "]" << std::endl;
      std::cout << "- Rotation: in Quaternion: [" << q.getX() << ", " << q.getY() << ", " 
                << q.getZ() << ", " << q.getW() << "]" << std::endl;
      std::cout<<"- Frame id: "<<temp_transform.frame_id_<<", Child id: "<<temp_transform.child_frame_id_<<endl;
    }
  }
  catch(tf::TransformException& ex)
  {
    // std::cout << "Failure at "<< ros::Time::now() << std::endl;
    std::cout << "Exception thrown:" << ex.what()<< std::endl;
    // std::cout << "The current list of frames is:" <<std::endl;
    // std::cout << tf_listener.allFramesAsString()<<std::endl;
    
  }
  switch (frame_index)
  {
    case 0:
      echo_transform = temp_transform;
      break;
    case 1:
      echo_transform_fr = temp_transform;
      break;
    case 2:
      echo_transform_fl = temp_transform;
      break;
    case 3:
      echo_transform_br = temp_transform;
      break;
    case 4:
      echo_transform_br = temp_transform;
      break;
    default:
      break;
  }
  
  return;
}

void first_frame_KFT(void){
  int current_id = cens.size();
  float sigmaP=0.01;//0.01
  float sigmaQ=0.1;//0.1
  //initialize new tracks(function)
  //state = [x,y,z,vx,vy,vz]
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

void output_score(double beta, v_measure_score result){
  string dir_path = ros::package::getPath("track") + "/cluster_score/" + to_string(ros::WallTime::now().toBoost().date()) + "/";
  string csv_file_name = csv_name + score_file_name + ".csv";
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
    file << "good-obj num" << ",";
    file << "multi-obj num" << ",";
    file << "no-obj num" << ",";
    file << "V measure score (beta=" << fixed << setprecision(2) << beta << ")" << ",";
    file << "Homogeneity" << ",";
    file << "H(C|K)" << ",";
    file << "H(C)" << ",";
    file << "Completeness" << ",";
    file << "H(K|C)" << ",";
    file << "H(K)" << endl;
    file.close();
  }
  ofstream file;
  file.open(output_path, ios::out|ios::app);
  file << setprecision(0) << result.frame << ",";
  file << setprecision(0) << result.object_num << ",";
  file << setprecision(0) << result.good_cluster << ",";
  file << setprecision(0) << result.multi_cluster << ",";
  file << setprecision(0) << result.no_cluster << ",";
  file << fixed << setprecision(3);
  file << result.v_measure_score << ",";
  file << result.homo << ",";
  file << result.h_ck << ",";
  file << result.h_c << ",";
  file << result.comp << ",";
  file << result.h_kc << ",";
  file << result.h_k << endl;
  file.close();
  return;
}

void score_cluster(std::vector<kf_tracker_point> score_cens, std::vector< std::vector<kf_tracker_point> > dbscan_cluster, double beta=1.0){
  get_label = false;
  std::vector< std::vector<kf_tracker_point> > gt_cluster;
  std::cout << "\033[1;33mScore cluster performance\033[0m" << endl;
  std::cout << "Timestamp: " << label_vec.at(0).marker.header.stamp.toSec() << endl;
  std::cout << "Reading bag: " << score_file_name << endl;
  std::cout << "Label size: " << label_vec.size() << endl;
  // std::cout << "Radar size: " << score_cens.size() << endl;
  int total_n = 0;
  for(int label_idx=0;label_idx<label_vec.size();label_idx++){
    std::vector<kf_tracker_point> label_list;
    int n = 0;
    label_point label_pt = label_vec.at(label_idx);
    bool skip = false;
    for(int ns_idx=0;ns_idx<score_skip_ns.size();ns_idx++){
      std::size_t pos = label_pt.marker.ns.find(score_skip_ns.at(ns_idx));
      if(pos != label_pt.marker.ns.npos){
        skip = true;
        break;
      }
    }
    if(skip)
      continue;
    tf::Vector3 v1,v2;
    v1 = label_pt.front_left - label_pt.back_left;
    v2 = label_pt.back_right - label_pt.back_left;
    double ang1 = atan2(v1.y(), v1.x());
    double ang2 = atan2(v2.y(), v2.x());
    if(output_score_info){
      std::cout << "----------------------------------------------------\n";
      std::cout << "Box " << label_idx << "-th:\n";
      std::cout << "Corner:" << endl;
      std::cout << "\t"
           << label_pt.front_left.getX() << ", "
           << label_pt.front_left.getY() << ", "
           << label_pt.front_left.getZ() << endl;
      std::cout << "\t"
           << label_pt.front_right.getX() << ", "
           << label_pt.front_right.getY() << ", "
           << label_pt.front_right.getZ() << endl;
      std::cout << "\t"
           << label_pt.back_left.getX() << ", "
           << label_pt.back_left.getY() << ", "
           << label_pt.back_left.getZ() << endl;
      std::cout << "\t"
           << label_pt.back_right.getX() << ", "
           << label_pt.back_right.getY() << ", "
           << label_pt.back_right.getZ() << endl;
      std::cout << "v1: (" << v1.x() << ", " << v1.y() << ", " << v1.z() << ")\t";
      std::cout << "ang:" << ang1 << endl;
      std::cout << "v2: (" << v2.x() << ", " << v2.y() << ", " << v2.z() << ")\t";
      std::cout << "ang:" << ang2 << endl;
    }
    for(int i=0;i<score_cens.size();i++){
      if(score_cens.at(i).cluster_flag != -1)
        continue;
      kf_tracker_point score_pt = score_cens.at(i);
      tf::Vector3 v3;
      v3 = tf::Point(score_pt.x,score_pt.y,0) - label_pt.back_left;
      double ang3 = atan2(v3.y(), v3.x()) ;
      if(output_score_info){
        std::cout << "\t==========================\n";
        std::cout << "\tposition: (" << score_pt.x << ", " << score_pt.y << ", " << score_pt.z << ")\n";
        std::cout << "\tv: (" << v3.x() << ", " << v3.y() << ", " << v3.z() << ")\t";
        std::cout << "\tang:" << ang3 << endl;
      }
      if(!(0 <= v1.normalized().dot(v3.normalized()) && v1.normalized().dot(v3.normalized()) <=1 && 0<= v2.normalized().dot(v3.normalized()) && 0<= v2.normalized().dot(v3.normalized()) <=1))
        continue;
      if((v3.dot(v1) / v1.dot(v1) * v1).length() <= v1.length() && (v3.dot(v2) / v2.dot(v2) * v2).length() <= v2.length()){
        if(output_score_info){
          std::cout << "\t\033[1;32mInside the box\033[0m" << endl;
        }
        score_cens.at(i).cluster_flag = label_idx;
        label_list.push_back(score_cens.at(i));
        n++;
        total_n++;
      }
    }
    if(n != 0)
      gt_cluster.push_back(label_list);
  }
  Eigen::MatrixXd v_measure_mat(gt_cluster.size(),dbscan_cluster.size()+1); // +1 for the noise cluster from dbscan
  Eigen::MatrixXd cluster_list_mat(1,dbscan_cluster.size()+1); // check the cluster performance
  v_measure_mat.setZero();
  cluster_list_mat.setZero();
  color_cluster(gt_cluster,false);
  // std::cout << "Ground truth cluster:\n";
  for(int idx=0;idx<gt_cluster.size();idx++){
    std::vector<kf_tracker_point> temp_gt = gt_cluster.at(idx);
    if(output_gt_pt_info){
      std::cout << "-------------------------------\n";
      std::cout << "cluster index: " << idx << endl;
    }
    for(int i=0;i<temp_gt.size();i++){
      bool find_cluster =false;
      if(output_gt_pt_info){
        std::cout << "\t" << i << endl;
        std::cout << "\tPosition: (" <<temp_gt.at(i).x<<","<<temp_gt.at(i).y<<")"<<endl;
        std::cout << "\tVelocity: (" << temp_gt.at(i).x_v << "," << temp_gt.at(i).y_v << ")" << endl;
        std::cout << "\tCluster id: ";
      }
      for(int cluster_idx=0;cluster_idx<dbscan_cluster.size()+1;cluster_idx++){
        cluster_list_mat(0,cluster_idx) = cluster_idx;
        if(cluster_idx == dbscan_cluster.size()){
          if(!find_cluster){
            v_measure_mat(idx,cluster_idx) += 1;
            if(output_gt_pt_info){
              std::cout << "noise pt" << endl;
            }
          }
          continue;
        }
        std::vector<kf_tracker_point> temp_dbcluster = dbscan_cluster.at(cluster_idx);
        for(int j=0;j<temp_dbcluster.size();j++){
          tf::Vector3 distance = tf::Vector3(temp_gt.at(i).x,temp_gt.at(i).y,0) - tf::Vector3(temp_dbcluster.at(j).x,temp_dbcluster.at(j).y,0);
          if(distance.length() <= 0.001){
            v_measure_mat(idx,cluster_idx) += 1;
            find_cluster = true;
            if(output_gt_pt_info){
              std::cout << cluster_idx << endl;
            }
          }
        }
      }
      if(output_gt_pt_info){
        std::cout << endl;
      }
    }
  }
  int n = v_measure_mat.sum();
  
  // v-measure score calculation
  Eigen::RowVectorXd row_sum = v_measure_mat.rowwise().sum();
  Eigen::RowVectorXd col_sum = v_measure_mat.colwise().sum();
  // for(int i=0;i<dbscan_cluster.size();i++){
  //   if(col_sum(i)==0)
  //     continue;
  //   else
  //     col_sum(i) = dbscan_cluster.at(i).size();
  // }
  std::cout << "Ground true (row): " << gt_cluster.size() << " ,DBSCAN cluster (col):" << dbscan_cluster.size() << endl;
  // std::cout << "n = " << total_n << endl;
  std::cout << "v_measure_mat sum = " << n;
  std::cout << endl;
  if(output_score_mat_info){
    Eigen::MatrixXd cluster_score_output_mat(gt_cluster.size()+1,dbscan_cluster.size()+1); // check the cluster performance
    cluster_score_output_mat << cluster_list_mat, v_measure_mat;
    // std::cout << setprecision(0) << cluster_list_mat << endl;
    // std::cout << setprecision(0) << v_measure_mat << endl;
    std::cout << "V-measurement matrix:\n";
    std::cout << setprecision(0) << cluster_score_output_mat << endl;
    
    std::cout << "Original Col sum: " << col_sum << endl;
  }
  v_measure_score result;
  result.frame = ++score_frame;
  result.object_num = gt_cluster.size();
  std::cout << setprecision(0);
  // std::cout << "Row sum: " << row_sum << endl;
  // std::cout << "After Col sum: " << col_sum << endl;
  for(int i=0;i<v_measure_mat.rows();i++){
    bool check_cluster_state = false;
    for(int j=0;j<v_measure_mat.cols();j++){
      if(v_measure_mat(i,j) == 0 || col_sum(j) == 0)
        continue;
      else
        result.h_ck -= (v_measure_mat(i,j) / n) * log(v_measure_mat(i,j)/col_sum(j));
      if(!check_cluster_state){
        check_cluster_state = true;
        if(v_measure_mat(i,j) < row_sum(i))
          result.multi_cluster ++;
        else if(j == v_measure_mat.cols()-1)
          result.no_cluster ++;
        else if(v_measure_mat(i,j) < col_sum(j))
          result.over_cluster ++;
        else
          result.good_cluster ++;
      }
    }
    if(row_sum(i) == 0)
      continue;
    else
      result.h_c -= (row_sum(i)/n) * log(row_sum(i)/n);
  }
  if(result.h_ck == 0)
    result.homo = 1;
  else
    result.homo = 1 - (result.h_ck / result.h_c);
  for(int j=0;j<v_measure_mat.cols();j++){
    for(int i=0;i<v_measure_mat.rows();i++){
      if(v_measure_mat(i,j) == 0 || row_sum(i) == 0)
        continue;
      else
        result.h_kc -= (v_measure_mat(i,j) / n) * log(v_measure_mat(i,j)/row_sum(i));
    }
    if(col_sum(j) == 0)
      continue;
    else
      result.h_k -= (col_sum(j)/n) * log(col_sum(j)/n);

  }
  if(result.h_kc == 0)
    result.comp = 1;
  else
    result.comp = 1 - (result.h_kc / result.h_k);
  result.v_measure_score = (1+beta)*result.homo*result.comp / (beta*result.homo + result.comp);
  std::cout << setprecision(3);
  std::cout << "Homogeneity:\t" << result.homo << "\t-> ";
  std::cout << "H(C|k) = " << result.h_ck << ", H(C) = " << result.h_c << endl;
  std::cout << "Completeness:\t" << result.comp << "\t-> ";
  std::cout << "H(K|C) = " << result.h_kc << ", H(K) = " << result.h_k << endl;
  std::cout << "\033[1;46mV-measure score: " << result.v_measure_score << "\033[0m" << endl;
  std::cout << "Total object:" << result.object_num << endl;
  std::cout << "Good object: " << result.good_cluster << endl;
  std::cout << "Multi object: " << result.multi_cluster << endl;
  std::cout << "Over Cluster object: " << result.over_cluster << endl;
  std::cout << "Bad object: " << result.no_cluster << endl;
  if(write_out_score_file && (result.object_num != 0))
    output_score(beta,result);

}

void callback(const conti_radar::MeasurementConstPtr& msg,const nav_msgs::OdometryConstPtr& odom){
  int radar_point_size = msg->points.size();
  int radar_id_count = 0;
  std::cout<<"\n\n\033[1;4;100m"<<++call_back_num<<" Call Back radar points:"<<radar_point_size<<"\033[0m"<<endl;
  if(radar_point_size==0) //radar points cannot be zero, it would cause segement fault
    return;
  radar_stamp = msg->header.stamp;
  std::cout << std::fixed; std::cout.precision(3);
  std::cout<<"Radar time:"<<radar_stamp.toSec()<<endl;
  if(show_vel_marker)
    vel_marker(msg);
  if(get_transformer)
    get_transform(radar_stamp);
  else
    echo_transform.setIdentity();
  // republish radar points
  pcl::PointCloud<pcl::PointXYZI>::Ptr filter_points(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr history_filter_points(new pcl::PointCloud<pcl::PointXYZI>);
  sensor_msgs::PointCloud2::Ptr filter_cloud(new sensor_msgs::PointCloud2);
  sensor_msgs::PointCloud2::Ptr history_filter_cloud(new sensor_msgs::PointCloud2);
  filter_points->clear();
  // end republish
  
  cens.clear();
  cluster_cens.clear();
  // get the ego velocity
  vx = odom->twist.twist.linear.x;
  vy = odom->twist.twist.linear.y;
  if(output_ego_vel_info){
    std::cout<<"\033[1;33mVehicle velocity:\n\033[0m";
    std::cout<<"Time stamp:"<<odom->header.stamp.toSec()<<endl;
    std::cout<<"vx: "<<vx<<" ,vy: "<<vy<<endl;
  }
  if(output_radar_info)
    std::cout<<endl<<"Inlier Radar information\n";
  tf::StampedTransform vel_transform = echo_transform;
  vel_transform.setOrigin(tf::Vector3(0,0,0));
  for (int i = 0;i<msg->points.size();i++){
    tf::Point trans_pt;
    tf::Vector3 trans_vel;
    trans_pt.setValue(msg->points[i].longitude_dist, msg->points[i].lateral_dist, 0);
    trans_pt = echo_transform * trans_pt;
    trans_vel.setValue(msg->points[i].longitude_vel_comp, msg->points[i].lateral_vel_comp, 0);
    trans_vel = vel_transform * trans_vel;
    float range = sqrt(pow(msg->points[i].longitude_dist,2) + 
                        pow(msg->points[i].lateral_dist,2));
    float velocity = sqrt(pow(msg->points[i].lateral_vel_comp,2) + 
                          pow(msg->points[i].longitude_vel_comp,2));
    double angle = atan2(msg->points[i].lateral_vel_comp,msg->points[i].longitude_vel_comp)*180/M_PI;

    // float velocity = sqrt(pow(msg->points[i].lateral_vel+vy,2) + 
    //                       pow(msg->points[i].longitude_vel+vx,2));
    // double angle = atan2(msg->points[i].lateral_vel+vy,msg->points[i].longitude_vel+vx)*180/M_PI;
    

    if(output_radar_info){
      std::cout<<"----------------------------------------------\n";
      std::cout<<i<<"\tposition:("<<msg->points[i].longitude_dist<<","<<msg->points[i].lateral_dist<<")\n";
      std::cout<<"\tposition to map:("<<trans_pt.x()<<","<<trans_pt.y()<<")\n";
      std::cout<<"\tvel:"<<velocity<<"("<<msg->points[i].longitude_vel_comp<<","<<msg->points[i].lateral_vel_comp<<")\n\tdegree:"<<angle;
      // std::cout<<"\tvel:"<<velocity<<"("<<msg->points[i].longitude_vel+vx<<","<<msg->points[i].lateral_vel+vy<<")\n\tdegree:"<<angle;
      std::cout<<"\n\tinvalid_state:"<<msg->points[i].invalid_state;
      std::cout<<"\n\tRCS value:"<<msg->points[i].rcs;
      std::cout<<endl;
    }
    std::list<int>::iterator invalid_state_it;
    invalid_state_it = std::find(invalid_state.begin(),invalid_state.end(),msg->points[i].invalid_state);
    if(range>100 || invalid_state_it!=invalid_state.end()){               //set radar threshold (distance)
        radar_point_size--;
        // if(output_radar_info)
        //   std::cout<<endl;
    }
    else{
      kf_tracker_point c;
      c.x = trans_pt.x();
      c.y = trans_pt.y();
      c.z = trans_pt.z();
      // c.x_v = msg->points[i].longitude_vel_comp+vx;
      // c.y_v = msg->points[i].lateral_vel_comp+vy;
      // c.x_v = msg->points[i].longitude_vel+vx;
      // c.y_v = msg->points[i].lateral_vel+vy;
      c.x_v = trans_vel.x();
      c.y_v = trans_vel.y();
      c.z_v = 0;
      c.vel_ang = angle;
      c.vel = velocity;
      c.cluster_flag = -1;  // -1: not clustered
      c.vistited = false;
      c.id = radar_id_count++;
      c.scan_id = call_back_num;
      c.rcs = msg->points[i].rcs;
      
      pcl::PointXYZI pt;
      pt.x = c.x;
      pt.y = c.y;
      pt.z = c.z;
      pt.intensity = msg->points[i].rcs;
      filter_points->push_back(pt);
      // if(output_radar_info)
      //   std::cout<<"\n\tposition:("<<c.x<<","<<c.y<<","<<c.z<<")\n";
      cens.push_back(c);
    }
  }
  pcl::toROSMsg(*filter_points,*filter_cloud);
  if(!get_transformer)
    filter_cloud->header.frame_id = "/nuscenes_radar_front";
  else
    filter_cloud->header.frame_id = "/map";
  filter_cloud->header.stamp = radar_stamp;
  pub_in_filter.publish(filter_cloud);

  // cluster the points with DBSCAN-based method
  double DBSCAN_START, DBSCAN_END;
  DBSCAN_START = clock();
  dbscan_seg.scan_num = call_back_num;
  dbscan_seg.data_period = dt;
  std::vector< std::vector<kf_tracker_point> > dbscan_cluster;
  dbscan_seg.output_info = output_dbscan_info;
  dbscan_cluster = dbscan_seg.cluster(cens);
  DBSCAN_END = clock();
  if(output_exe_time){
    std::cout << endl << "\033[1;42mDBSCAN Execution time: " << (DBSCAN_END - DBSCAN_START) / CLOCKS_PER_SEC << "s\033[0m";
    std::cout << endl;
  }
  color_cluster(dbscan_cluster);
  polygon_cluster_visual(dbscan_cluster);
  if(use_score_cluster && get_label && (fabs(label_vec.at(0).marker.header.stamp.toSec()-radar_stamp.toSec()) <= 0.08))
    score_cluster(cens, dbscan_cluster);
  cens.clear();
  cens = dbscan_seg.get_center();
  // publish the history radar points
  std::vector<kf_tracker_point> multi_frame_pts = dbscan_seg.get_history_points();
  for(int i=0;i<multi_frame_pts.size();i++){
    pcl::PointXYZI pt;
    pt.x = multi_frame_pts.at(i).x;
    pt.y = multi_frame_pts.at(i).y;
    pt.z = multi_frame_pts.at(i).z;
    pt.intensity = multi_frame_pts.at(i).rcs;
    history_filter_points->points.push_back(pt);
  }
  pcl::toROSMsg(*history_filter_points,*history_filter_cloud);
  if(get_transformer)
    history_filter_cloud->header.frame_id = "/map";
  else
    history_filter_cloud->header.frame_id = "/nuscenes_radar_front";
  history_filter_cloud->header.stamp = radar_stamp;
  pub_filter.publish(history_filter_cloud);

  if(cens.size()==0)
    return;
  

  if(use_KFT_module){
    if( firstFrame ){
      first_frame_KFT();
      return; // first initialization down 
    }
    double START_TIME, END_TIME;
    START_TIME = clock();
    KFT();
    END_TIME = clock();
    if(output_exe_time){
      std::cout << "\033[1;42mKFT Execution time: " << (END_TIME - START_TIME) / CLOCKS_PER_SEC << "s\033[0m";
      std::cout << endl;
    }
  }
  return;

}

void callback_ego(const conti_radar::MeasurementConstPtr& in_msg,const conti_radar::MeasurementConstPtr& out_msg,const nav_msgs::OdometryConstPtr& odom){
  int radar_point_size = in_msg->points.size() + out_msg->points.size();
  int radar_id_count = 0;
  std::cout<<"\n\n\033[1;4;100m"<<++call_back_num<<" Call Back radar points:"<<radar_point_size<<"\033[0m"<<endl;
  if(radar_point_size==0) //radar points cannot be zero, it would cause segement fault
    return;
  radar_stamp = in_msg->header.stamp;
  std::cout << std::fixed; std::cout.precision(3);
  std::cout<<"Radar time:"<<radar_stamp.toSec()<<endl;

  if(show_vel_marker){
    vel_marker(in_msg,out_msg);
  }
  if(get_transformer)
    get_transform(radar_stamp);
  else
    echo_transform.setIdentity();

  // republish radar points
  pcl::PointCloud<pcl::PointXYZI>::Ptr in_filter_points(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr out_filter_points(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr history_filter_points(new pcl::PointCloud<pcl::PointXYZI>);
  sensor_msgs::PointCloud2::Ptr in_filter_cloud(new sensor_msgs::PointCloud2);
  sensor_msgs::PointCloud2::Ptr out_filter_cloud(new sensor_msgs::PointCloud2);
  sensor_msgs::PointCloud2::Ptr history_filter_cloud(new sensor_msgs::PointCloud2);
  in_filter_points->clear();
  out_filter_points->clear();
  // end republish
  
  cens.clear();
  cluster_cens.clear();
  
  // get the ego velocity
  vx = odom->twist.twist.linear.x;
  vy = odom->twist.twist.linear.y;
  EGO_INFO("\033[1;33mVehicle velocity:\n\033[0m");
  EGO_INFO("Time stamp:"<<odom->header.stamp.toSec()<<endl);
  EGO_INFO("vx: "<<vx<<" ,vy: "<<vy<<endl);

  RADAR_INFO(endl<<"Inlier Radar information\n");
  
  // put the inlier/outlier radar data into vector
  std::vector<kf_tracker_point> in_cens,out_cens;
  
  // get the velocity transform matrix to map
  tf::StampedTransform vel_transform = echo_transform;
  vel_transform.setOrigin(tf::Vector3(0,0,0));
  
  for (int i = 0;i<in_msg->points.size();i++){
    tf::Point trans_pt;
    tf::Vector3 trans_vel;
    trans_pt.setValue(in_msg->points[i].longitude_dist, in_msg->points[i].lateral_dist, 0);
    trans_pt = echo_transform * trans_pt;
    trans_vel.setValue(in_msg->points[i].longitude_vel_comp, in_msg->points[i].lateral_vel_comp, 0);
    trans_vel = vel_transform * trans_vel;
    float range = sqrt(pow(in_msg->points[i].longitude_dist,2) + 
                        pow(in_msg->points[i].lateral_dist,2));
    float velocity = sqrt(pow(in_msg->points[i].lateral_vel_comp,2) + 
                          pow(in_msg->points[i].longitude_vel_comp,2));
    double angle = atan2(in_msg->points[i].lateral_vel_comp,in_msg->points[i].longitude_vel_comp)*180/M_PI;

    RADAR_INFO("----------------------------------------------\n");
    RADAR_INFO(i<<"\tposition:("<<in_msg->points[i].longitude_dist<<","<<in_msg->points[i].lateral_dist<<")\n");
    RADAR_INFO("\tposition to map:("<<trans_pt.x()<<","<<trans_pt.y()<<","<<trans_pt.z()<<")\n");
    RADAR_INFO("\tvel:"<<velocity<<"("<<in_msg->points[i].longitude_vel_comp<<","<<in_msg->points[i].lateral_vel_comp<<")\n\tdegree:"<<angle);
    RADAR_INFO("\n\tinvalid_state:"<<in_msg->points[i].invalid_state);
    RADAR_INFO("\n\tRCS value:"<<in_msg->points[i].rcs);
    RADAR_INFO(endl);

    std::list<int>::iterator invalid_state_it;
    invalid_state_it = std::find(invalid_state.begin(),invalid_state.end(),in_msg->points[i].invalid_state);
    // remove the invalid state points
    if(invalid_state_it!=invalid_state.end()){
        radar_point_size--;
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
      c.rcs = in_msg->points[i].rcs;
      
      pcl::PointXYZI pt;
      pt.x = c.x;
      pt.y = c.y;
      pt.z = c.z;
      pt.intensity = in_msg->points[i].rcs;
      in_filter_points->push_back(pt);
      in_cens.push_back(c);
  }
  }
  RADAR_INFO("\nOutlier Radar information\n");
  for (int i = 0;i<out_msg->points.size();i++){
    tf::Point trans_pt;
    tf::Vector3 trans_vel;
    trans_pt.setValue(out_msg->points[i].longitude_dist, out_msg->points[i].lateral_dist, 0);
    trans_pt = echo_transform * trans_pt;
    trans_vel.setValue(out_msg->points[i].longitude_vel_comp, out_msg->points[i].lateral_vel_comp, 0);
    trans_vel = vel_transform * trans_vel;
    float range = sqrt(pow(out_msg->points[i].longitude_dist,2) + 
                        pow(out_msg->points[i].lateral_dist,2));
    float velocity = sqrt(pow(out_msg->points[i].lateral_vel_comp,2) + 
                          pow(out_msg->points[i].longitude_vel_comp,2));
    double angle = atan2(out_msg->points[i].lateral_vel_comp,out_msg->points[i].longitude_vel_comp)*180/M_PI;

    RADAR_INFO("----------------------------------------------\n");
    RADAR_INFO(i<<"\tposition:("<<out_msg->points[i].longitude_dist<<","<<out_msg->points[i].lateral_dist<<")\n");
    RADAR_INFO("\tposition to map:("<<trans_pt.x()<<","<<trans_pt.y()<<","<<trans_pt.z()<<")\n");
    RADAR_INFO("\tvel:"<<velocity<<"("<<out_msg->points[i].longitude_vel_comp<<","<<out_msg->points[i].lateral_vel_comp<<")\n\tdegree:"<<angle);
    RADAR_INFO("\n\tinvalid_state:"<<out_msg->points[i].invalid_state);
    RADAR_INFO("\n\tRCS value:"<<out_msg->points[i].rcs);
    RADAR_INFO(endl);

    std::list<int>::iterator invalid_state_it;
    invalid_state_it = std::find(invalid_state.begin(),invalid_state.end(),out_msg->points[i].invalid_state);
    // remove the invalid state points
    if(invalid_state_it!=invalid_state.end()){
      radar_point_size--;
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
      c.rcs = out_msg->points[i].rcs;
      c.scan_id = call_back_num;
      
      pcl::PointXYZI pt;
      pt.x = c.x;
      pt.y = c.y;
      pt.z = c.z;
      pt.intensity = out_msg->points[i].rcs;
      out_filter_points->push_back(pt);
      out_cens.push_back(c);
    }
  }
  cens.clear();
  cens.shrink_to_fit();
  cens = in_cens;
  cens.insert(cens.end(),out_cens.begin(),out_cens.end());
  if(cens.size() == 0){
    ROS_WARN("No Valid Callback Radar points In This Frame!");
    return;
  }
  
  // republish filtered radar points
  pcl::toROSMsg(*in_filter_points,*in_filter_cloud);
  if(get_transformer)
    in_filter_cloud->header.frame_id = MAP_FRAME;
  else
    in_filter_cloud->header.frame_id = RADAR_FRONT_FRAME;
  in_filter_cloud->header.stamp = radar_stamp;
  pcl::toROSMsg(*out_filter_points,*out_filter_cloud);
  if(get_transformer)
    out_filter_cloud->header.frame_id = MAP_FRAME;
  else
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
    
    // score the cluster performance
    if(use_score_cluster && get_label && (fabs(label_vec.at(0).marker.header.stamp.toSec()-radar_stamp.toSec()) <= 0.08))
      score_cluster(cens, dbscan_cluster);
    
    cens.clear();
    cens = dbscan_seg.get_center();
    
    // publish the history radar points
    multi_frame_pts = dbscan_seg.get_history_points();
  }
  else if(cluster_type=="optics"){
    double OPTICS_START, OPTICS_END;
    OPTICS_START = clock();
    optics_seg.scan_num = call_back_num;
    optics_seg.data_period = dt;
    std::vector< std::vector<kf_tracker_point> > optics_cluster;
    optics_seg.output_info = output_dbscan_info;
    optics_cluster = optics_seg.cluster(cens);
    OPTICS_END = clock();
    
    // check the DBSCAN exe performance
    EXE_INFO(endl << "\033[42mOPTICS Execution time: " << (OPTICS_END - OPTICS_START) / CLOCKS_PER_SEC << "s\033[0m");
    EXE_INFO(std::endl);

    // cluster visualization
    color_cluster(optics_cluster);
    polygon_cluster_visual(optics_cluster);
    
    // score the cluster performance
    if(use_score_cluster && get_label && (fabs(label_vec.at(0).marker.header.stamp.toSec()-radar_stamp.toSec()) <= 0.08))
      score_cluster(cens, optics_cluster);
    
    cens.clear();
    cens = optics_seg.get_center();
    
    // publish the history radar points
    multi_frame_pts = optics_seg.get_history_points();
  }
  else{
    double DBPDA_START, DBPDA_END;
    DBPDA_START = clock();
    dbpda_seg.scan_num = call_back_num;
    dbpda_seg.data_period = dt;
    std::vector< std::vector<kf_tracker_point> > optics_cluster;
    dbpda_seg.output_info = output_dbscan_info;
    optics_cluster = dbpda_seg.cluster(cens);
    DBPDA_END = clock();
    
    // check the DBSCAN exe performance
    EXE_INFO(endl << "\033[42mDBPDA Execution time: " << (DBPDA_END - DBPDA_START) / CLOCKS_PER_SEC << "s\033[0m");
    EXE_INFO(std::endl);

    // cluster visualization
    color_cluster(optics_cluster);
    polygon_cluster_visual(optics_cluster);
    
    // score the cluster performance
    if(use_score_cluster && get_label && (fabs(label_vec.at(0).marker.header.stamp.toSec()-radar_stamp.toSec()) <= 0.08))
      score_cluster(cens, optics_cluster);
    
    cens.clear();
    cens = dbpda_seg.get_center();
    
    // publish the history radar points
    multi_frame_pts = dbpda_seg.get_history_points();
  }


  for(int i=0;i<multi_frame_pts.size();i++){
    pcl::PointXYZI pt;
    pt.x = multi_frame_pts.at(i).x;
    pt.y = multi_frame_pts.at(i).y;
    pt.z = multi_frame_pts.at(i).z;
    pt.intensity = multi_frame_pts.at(i).rcs;
    history_filter_points->points.push_back(pt);
  }
  pcl::toROSMsg(*history_filter_points,*history_filter_cloud);
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

void radar_callback(const conti_radar::MeasurementConstPtr& f_msg,
                    const conti_radar::MeasurementConstPtr& fr_msg,
                    const conti_radar::MeasurementConstPtr& fl_msg,
                    const conti_radar::MeasurementConstPtr& br_msg,
                    const conti_radar::MeasurementConstPtr& bl_msg){
  int radar_point_size = f_msg->points.size() + fr_msg->points.size() +
                         fl_msg->points.size() + br_msg->points.size() +
                         bl_msg->points.size();
  int radar_id_count = 0;
  std::cout<<"\n\n\033[1;4;100m"<<++call_back_num<<" Call Back radar points:"<<radar_point_size<<"\033[0m"<<endl;
  if(radar_point_size==0) //radar points cannot be zero, it would cause segement fault
    return;
  radar_stamp = f_msg->header.stamp;
  std::cout << std::fixed; std::cout.precision(3);
  std::cout<<"Radar time:"<<radar_stamp.toSec()<<endl;
  
  get_transform(radar_stamp,0);
  // republish radar points
  pcl::PointCloud<pcl::PointXYZI>::Ptr filter_points(new pcl::PointCloud<pcl::PointXYZI>);
  sensor_msgs::PointCloud2::Ptr filter_cloud(new sensor_msgs::PointCloud2);
  pcl::PointCloud<pcl::PointXYZI>::Ptr history_filter_points(new pcl::PointCloud<pcl::PointXYZI>);
  sensor_msgs::PointCloud2::Ptr history_filter_cloud(new sensor_msgs::PointCloud2);
  // end republish
  cens.clear();
  cluster_cens.clear();
  std::vector<kf_tracker_point> in_cens;
  
  tf::StampedTransform vel_transform = echo_transform;
  vel_transform.setOrigin(tf::Vector3(0,0,0));
  for (int i = 0;i<f_msg->points.size();i++){
    tf::Point trans_pt;
    tf::Vector3 trans_vel;
    trans_pt.setValue(f_msg->points[i].longitude_dist, f_msg->points[i].lateral_dist, 0);
    trans_pt = echo_transform * trans_pt;
    trans_vel.setValue(f_msg->points[i].longitude_vel_comp, f_msg->points[i].lateral_vel_comp, 0);
    trans_vel = vel_transform * trans_vel;
    float range = sqrt(pow(f_msg->points[i].longitude_dist,2) + 
                        pow(f_msg->points[i].lateral_dist,2));
    float velocity = sqrt(pow(f_msg->points[i].lateral_vel_comp,2) + 
                          pow(f_msg->points[i].longitude_vel_comp,2));
    double angle = atan2(f_msg->points[i].lateral_vel_comp,f_msg->points[i].longitude_vel_comp)*180/M_PI;

    if(output_radar_info){
      std::cout<<"----------------------------------------------\n";
      std::cout<<i<<"\tposition:("<<f_msg->points[i].longitude_dist<<","<<f_msg->points[i].lateral_dist<<")\n";
      std::cout<<"\tposition to map:("<<trans_pt.x()<<","<<trans_pt.y()<<","<<trans_pt.z()<<")\n";
      std::cout<<"\tvel:"<<velocity<<"("<<f_msg->points[i].longitude_vel_comp<<","<<f_msg->points[i].lateral_vel_comp<<")\n\tdegree:"<<angle;
      std::cout<<"\n\tinvalid_state:"<<f_msg->points[i].invalid_state;
      std::cout<<"\n\tRCS value:"<<f_msg->points[i].rcs;
      std::cout<<endl;
    }
    if(range>100){               //set radar threshold (distance)
        radar_point_size--;
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
        c.rcs = f_msg->points[i].rcs;
        
        pcl::PointXYZI pt;
        pt.x = c.x;
        pt.y = c.y;
        pt.z = c.z;
        pt.intensity = f_msg->points[i].rcs;
        filter_points->push_back(pt);
        in_cens.push_back(c);
    }
  }
  
  get_transform(fr_msg->header.stamp,1);
  vel_transform = echo_transform_fr;
  vel_transform.setOrigin(tf::Vector3(0,0,0));
  for (int i = 0;i<fr_msg->points.size();i++){
    tf::Point trans_pt;
    tf::Vector3 trans_vel;
    trans_pt.setValue(fr_msg->points[i].longitude_dist, fr_msg->points[i].lateral_dist, 0);
    trans_pt = echo_transform_fr * trans_pt;
    trans_vel.setValue(fr_msg->points[i].longitude_vel_comp, fr_msg->points[i].lateral_vel_comp, 0);
    trans_vel = vel_transform * trans_vel;
    float range = sqrt(pow(fr_msg->points[i].longitude_dist,2) + 
                        pow(fr_msg->points[i].lateral_dist,2));
    float velocity = sqrt(pow(fr_msg->points[i].lateral_vel_comp,2) + 
                          pow(fr_msg->points[i].longitude_vel_comp,2));
    double angle = atan2(fr_msg->points[i].lateral_vel_comp,fr_msg->points[i].longitude_vel_comp)*180/M_PI;

    if(output_radar_info){
      std::cout<<"----------------------------------------------\n";
      std::cout<<i<<"\tposition:("<<fr_msg->points[i].longitude_dist<<","<<fr_msg->points[i].lateral_dist<<")\n";
      std::cout<<"\tposition to map:("<<trans_pt.x()<<","<<trans_pt.y()<<","<<trans_pt.z()<<")\n";
      std::cout<<"\tvel:"<<velocity<<"("<<fr_msg->points[i].longitude_vel_comp<<","<<fr_msg->points[i].lateral_vel_comp<<")\n\tdegree:"<<angle;
      std::cout<<"\n\tinvalid_state:"<<fr_msg->points[i].invalid_state;
      std::cout<<"\n\tRCS value:"<<fr_msg->points[i].rcs;
      std::cout<<endl;
    }
    if(range>100){               //set radar threshold (distance)
        radar_point_size--;
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
        c.rcs = fr_msg->points[i].rcs;
        
        pcl::PointXYZI pt;
        pt.x = c.x;
        pt.y = c.y;
        pt.z = c.z;
        pt.intensity = fr_msg->points[i].rcs;
        filter_points->push_back(pt);
        in_cens.push_back(c);
    }
  }
  
  get_transform(fl_msg->header.stamp,2);
  vel_transform = echo_transform_fl;
  vel_transform.setOrigin(tf::Vector3(0,0,0));
  for (int i = 0;i<fl_msg->points.size();i++){
    tf::Point trans_pt;
    tf::Vector3 trans_vel;
    trans_pt.setValue(fl_msg->points[i].longitude_dist, fl_msg->points[i].lateral_dist, 0);
    trans_pt = echo_transform_fl * trans_pt;
    trans_vel.setValue(fl_msg->points[i].longitude_vel_comp, fl_msg->points[i].lateral_vel_comp, 0);
    trans_vel = vel_transform * trans_vel;
    float range = sqrt(pow(fl_msg->points[i].longitude_dist,2) + 
                        pow(fl_msg->points[i].lateral_dist,2));
    float velocity = sqrt(pow(fl_msg->points[i].lateral_vel_comp,2) + 
                          pow(fl_msg->points[i].longitude_vel_comp,2));
    double angle = atan2(fl_msg->points[i].lateral_vel_comp,fl_msg->points[i].longitude_vel_comp)*180/M_PI;

    if(output_radar_info){
      std::cout<<"----------------------------------------------\n";
      std::cout<<i<<"\tposition:("<<fl_msg->points[i].longitude_dist<<","<<fl_msg->points[i].lateral_dist<<")\n";
      std::cout<<"\tposition to map:("<<trans_pt.x()<<","<<trans_pt.y()<<","<<trans_pt.z()<<")\n";
      std::cout<<"\tvel:"<<velocity<<"("<<fl_msg->points[i].longitude_vel_comp<<","<<fl_msg->points[i].lateral_vel_comp<<")\n\tdegree:"<<angle;
      std::cout<<"\n\tinvalid_state:"<<fl_msg->points[i].invalid_state;
      std::cout<<"\n\tRCS value:"<<fl_msg->points[i].rcs;
      std::cout<<endl;
    }
    if(range>100){               //set radar threshold (distance)
        radar_point_size--;
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
        c.rcs = fl_msg->points[i].rcs;
        
        pcl::PointXYZI pt;
        pt.x = c.x;
        pt.y = c.y;
        pt.z = c.z;
        pt.intensity = fl_msg->points[i].rcs;
        filter_points->push_back(pt);
        in_cens.push_back(c);
    }
  }
  
  get_transform(br_msg->header.stamp,3);
  vel_transform = echo_transform_br;
  vel_transform.setOrigin(tf::Vector3(0,0,0));
  for (int i = 0;i<br_msg->points.size();i++){
    tf::Point trans_pt;
    tf::Vector3 trans_vel;
    trans_pt.setValue(br_msg->points[i].longitude_dist, br_msg->points[i].lateral_dist, 0);
    trans_pt = echo_transform_br * trans_pt;
    trans_vel.setValue(br_msg->points[i].longitude_vel_comp, br_msg->points[i].lateral_vel_comp, 0);
    trans_vel = vel_transform * trans_vel;
    float range = sqrt(pow(br_msg->points[i].longitude_dist,2) + 
                        pow(br_msg->points[i].lateral_dist,2));
    float velocity = sqrt(pow(br_msg->points[i].lateral_vel_comp,2) + 
                          pow(br_msg->points[i].longitude_vel_comp,2));
    double angle = atan2(br_msg->points[i].lateral_vel_comp,br_msg->points[i].longitude_vel_comp)*180/M_PI;

    if(output_radar_info){
      std::cout<<"----------------------------------------------\n";
      std::cout<<i<<"\tposition:("<<br_msg->points[i].longitude_dist<<","<<br_msg->points[i].lateral_dist<<")\n";
      std::cout<<"\tposition to map:("<<trans_pt.x()<<","<<trans_pt.y()<<","<<trans_pt.z()<<")\n";
      std::cout<<"\tvel:"<<velocity<<"("<<br_msg->points[i].longitude_vel_comp<<","<<br_msg->points[i].lateral_vel_comp<<")\n\tdegree:"<<angle;
      std::cout<<"\n\tinvalid_state:"<<br_msg->points[i].invalid_state;
      std::cout<<"\n\tRCS value:"<<br_msg->points[i].rcs;
      std::cout<<endl;
    }
    if(range>100){               //set radar threshold (distance)
        radar_point_size--;
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
        c.rcs = br_msg->points[i].rcs;
        
        pcl::PointXYZI pt;
        pt.x = c.x;
        pt.y = c.y;
        pt.z = c.z;
        pt.intensity = br_msg->points[i].rcs;
        filter_points->push_back(pt);
        in_cens.push_back(c);
    }
  }
  
  get_transform(bl_msg->header.stamp,4);
  vel_transform = echo_transform_bl;
  vel_transform.setOrigin(tf::Vector3(0,0,0));
  for (int i = 0;i<bl_msg->points.size();i++){
    tf::Point trans_pt;
    tf::Vector3 trans_vel;
    trans_pt.setValue(bl_msg->points[i].longitude_dist, bl_msg->points[i].lateral_dist, 0);
    trans_pt = echo_transform_bl * trans_pt;
    trans_vel.setValue(bl_msg->points[i].longitude_vel_comp, bl_msg->points[i].lateral_vel_comp, 0);
    trans_vel = vel_transform * trans_vel;
    float range = sqrt(pow(bl_msg->points[i].longitude_dist,2) + 
                        pow(bl_msg->points[i].lateral_dist,2));
    float velocity = sqrt(pow(bl_msg->points[i].lateral_vel_comp,2) + 
                          pow(bl_msg->points[i].longitude_vel_comp,2));
    double angle = atan2(bl_msg->points[i].lateral_vel_comp,bl_msg->points[i].longitude_vel_comp)*180/M_PI;

    if(output_radar_info){
      std::cout<<"----------------------------------------------\n";
      std::cout<<i<<"\tposition:("<<bl_msg->points[i].longitude_dist<<","<<bl_msg->points[i].lateral_dist<<")\n";
      std::cout<<"\tposition to map:("<<trans_pt.x()<<","<<trans_pt.y()<<","<<trans_pt.z()<<")\n";
      std::cout<<"\tvel:"<<velocity<<"("<<bl_msg->points[i].longitude_vel_comp<<","<<bl_msg->points[i].lateral_vel_comp<<")\n\tdegree:"<<angle;
      std::cout<<"\n\tinvalid_state:"<<bl_msg->points[i].invalid_state;
      std::cout<<"\n\tRCS value:"<<bl_msg->points[i].rcs;
      std::cout<<endl;
    }
    if(range>100){               //set radar threshold (distance)
        radar_point_size--;
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
        c.rcs = bl_msg->points[i].rcs;
        
        pcl::PointXYZI pt;
        pt.x = c.x;
        pt.y = c.y;
        pt.z = c.z;
        pt.intensity = bl_msg->points[i].rcs;
        filter_points->push_back(pt);
        in_cens.push_back(c);
    }
  }
  
  cens.clear();
  cens.shrink_to_fit();
  cens = in_cens;
  pcl::toROSMsg(*filter_points,*filter_cloud);
  filter_cloud->header.frame_id = "/map";
  filter_cloud->header.stamp = radar_stamp;
  pub_in_filter.publish(filter_cloud);

  double DBSCAN_START, DBSCAN_END;
  DBSCAN_START = clock();
  dbscan_seg.scan_num = call_back_num;
  dbscan_seg.data_period = dt;
  std::vector< std::vector<kf_tracker_point> > dbscan_cluster;
  dbscan_seg.output_info = output_dbscan_info;
  dbscan_cluster = dbscan_seg.cluster(cens);
  DBSCAN_END = clock();
  
  // score the cluster performance
  if(output_exe_time){
    std::cout << endl << "\033[42mDBSCAN Execution time: " << (DBSCAN_END - DBSCAN_START) / CLOCKS_PER_SEC << "s\033[0m";
    std::cout << endl;
  }
  color_cluster(dbscan_cluster);
  polygon_cluster_visual(dbscan_cluster);
  if(use_score_cluster && get_label && (fabs(label_vec.at(0).marker.header.stamp.toSec()-radar_stamp.toSec()) <= 0.08))
    score_cluster(cens, dbscan_cluster);
  cens.clear();
  cens = dbscan_seg.get_center();
  // publish the history radar points
  std::vector<kf_tracker_point> multi_frame_pts = dbscan_seg.get_history_points();
  for(int i=0;i<multi_frame_pts.size();i++){
    pcl::PointXYZI pt;
    pt.x = multi_frame_pts.at(i).x;
    pt.y = multi_frame_pts.at(i).y;
    pt.z = multi_frame_pts.at(i).z;
    pt.intensity = multi_frame_pts.at(i).rcs;
    history_filter_points->points.push_back(pt);
  }
  pcl::toROSMsg(*history_filter_points,*history_filter_cloud);
  history_filter_cloud->header.frame_id = "/map";
  history_filter_cloud->header.stamp = radar_stamp;
  pub_filter.publish(history_filter_cloud);

  if(use_KFT_module){
    if( firstFrame ){
      first_frame_KFT();
      return; // first initialization down 
    }
    double START_TIME, END_TIME;
    START_TIME = clock();
    KFT();
    END_TIME = clock();
    if(output_exe_time){
      std::cout << "\033[1;42mKFT Execution time: " << (END_TIME - START_TIME) / CLOCKS_PER_SEC << "s\033[0m";
      std::cout << endl;
    }
  }
  return;
}

void annotation(const visualization_msgs::MarkerArrayConstPtr& label){
  if(!use_score_cluster || label->markers.size() == 0)
    return;
  ros::Time label_stamp = label->markers.at(0).header.stamp;
  std::cout << std::fixed; std::cout.precision(3);
  if(output_label_info){
    std::cout << endl << "In annotation callback:" << endl;
    std::cout << "Time stamp:" <<label_stamp.toSec() << endl << endl;
  }
  tf::StampedTransform label_transform;
  string TARGET_FRAME = MAP_FRAME;
  string SOURCE_FRAME = "lidar_label";
  tf_listener->waitForTransform(TARGET_FRAME,SOURCE_FRAME,label_stamp,ros::Duration(1));
  try{
    tf_listener->lookupTransform(TARGET_FRAME,SOURCE_FRAME,label_stamp,label_transform);
    if(output_label_info){
      std::cout << "At time " << label_transform.stamp_.toSec() << std::endl;
      tf::Quaternion q = label_transform.getRotation();
      tf::Vector3 v = label_transform.getOrigin();
      std::cout << "- Translation: [" << v.getX() << ", " << v.getY() << ", " << v.getZ() << "]" << std::endl;
      std::cout << "- Rotation: in Quaternion [" << q.getX() << ", " << q.getY() << ", " 
                << q.getZ() << ", " << q.getW() << "]" << std::endl;
      std::cout<<"Frame id:"<<label_transform.frame_id_<<", Child id:"<<label_transform.child_frame_id_<<endl;
    }
  }
  catch(tf::TransformException& ex)
  {
    std::cout << "Exception thrown:" << ex.what()<< std::endl;
  }
  visualization_msgs::MarkerArray repub;
  repub.markers.clear();
  repub.markers.shrink_to_fit();
  label_vec.clear();
  label_vec.shrink_to_fit();
  for(int i=0;i<label->markers.size();i++){
    label_point label_pt;
    label_pt.type = label->markers.at(i).ns;
    label_pt.marker = label->markers.at(i);

    tf::poseMsgToTF(label->markers.at(i).pose,label_pt.pose);
    label_pt.pose = label_transform * label_pt.pose;
    tf::poseTFToMsg(label_pt.pose,label_pt.marker.pose);

    label_pt.front_left = tf::Point(-label->markers.at(i).scale.x/2 * box_bias,
                                    label->markers.at(i).scale.y/2 * box_bias,
                                    label->markers.at(i).scale.z/2);
    label_pt.front_right = tf::Point(label->markers.at(i).scale.x/2 * box_bias,
                                    label->markers.at(i).scale.y/2 * box_bias,
                                    label->markers.at(i).scale.z/2);
    label_pt.back_left = tf::Point(-label->markers.at(i).scale.x/2 * box_bias,
                                    -label->markers.at(i).scale.y/2 * box_bias,
                                    label->markers.at(i).scale.z/2);
    label_pt.back_right = tf::Point(label->markers.at(i).scale.x/2 * box_bias,
                                    -label->markers.at(i).scale.y/2 * box_bias,
                                    label->markers.at(i).scale.z/2);
    label_pt.front_left = label_pt.pose * label_pt.front_left;
    label_pt.front_right = label_pt.pose * label_pt.front_right;
    label_pt.back_left = label_pt.pose * label_pt.back_left;
    label_pt.back_right = label_pt.pose * label_pt.back_right;

    // tf::Quaternion label_q;
    // tf::quaternionMsgToTF(label->markers.at(i).pose.orientation,label_q);
    // label_pt.front_left = (label_transform * tf::Pose(label_q,tf::Vector3(label_pt.front_left.x(),label_pt.front_left.y(),label_pt.front_left.z()))).getOrigin();
    // label_pt.front_left.setZ(0);
    // label_pt.front_right = (label_transform * tf::Pose(label_q,tf::Vector3(label_pt.front_right.x(),label_pt.front_right.y(),label_pt.front_right.z()))).getOrigin();
    // label_pt.front_right.setZ(0);
    // label_pt.back_left = (label_transform * tf::Pose(label_q,tf::Vector3(label_pt.back_left.x(),label_pt.back_left.y(),label_pt.back_left.z()))).getOrigin();
    // label_pt.back_left.setZ(0);
    // label_pt.back_right = (label_transform * tf::Pose(label_q,tf::Vector3(label_pt.back_right.x(),label_pt.back_right.y(),label_pt.back_right.z()))).getOrigin();
    // label_pt.back_right.setZ(0);

    // label_pt.front_left = label_transform * label_pt.front_left;
    label_pt.front_left.setZ(0);
    // label_pt.front_right = label_transform * label_pt.front_right;
    label_pt.front_right.setZ(0);
    // label_pt.back_left = label_transform * label_pt.back_left;
    label_pt.back_left.setZ(0);
    // label_pt.back_right = label_transform * label_pt.back_right;
    label_pt.back_right.setZ(0);

    // if(output_label_info){
    //   std::cout << "--------------------------------------\n";
    //   std::cout << "The " << i << "-th marker pose:\n";
    //   std::cout << label_pt.type.c_str() << endl;
    //   std::cout << "- Orientation: ["  << label_pt.pose.getRotation().w() << ", "
    //                               << label_pt.pose.getRotation().x() << ", "
    //                               << label_pt.pose.getRotation().y() << ", "
    //                               << label_pt.pose.getRotation().z() << "]\n";
    //   std::cout << "- Position: [" << label_pt.pose.getOrigin().x() << ","
    //                           << label_pt.pose.getOrigin().y() << ","
    //                           << label_pt.pose.getOrigin().z() << "]\n";
    //   std::cout << "- Scale: [" << label_pt.marker.scale.x << ", "
    //                        << label_pt.marker.scale.y << ", "
    //                        << label_pt.marker.scale.z << "]\n";
    //   std::cout << "- Vertex: \n";
    //   std::cout << "\t"
    //        << label_pt.front_left.getX() << ", "
    //        << label_pt.front_left.getY() << ", "
    //        << label_pt.front_left.getZ() << endl;
    //   std::cout << "\t"
    //        << label_pt.front_right.getX() << ", "
    //        << label_pt.front_right.getY() << ", "
    //        << label_pt.front_right.getZ() << endl;
    //   std::cout << "\t"
    //        << label_pt.back_left.getX() << ", "
    //        << label_pt.back_left.getY() << ", "
    //        << label_pt.back_left.getZ() << endl;
    //   std::cout << "\t"
    //        << label_pt.back_right.getX() << ", "
    //        << label_pt.back_right.getY() << ", "
    //        << label_pt.back_right.getZ() << endl;
    // }
    label_pt.marker.header.frame_id = TARGET_FRAME;
    label_pt.marker.lifetime = ros::Duration(0.1);
    label_pt.marker.scale.x = label_pt.marker.scale.x;
    label_pt.marker.scale.y = label_pt.marker.scale.y;
    label_vec.push_back(label_pt);
    repub.markers.push_back(label_pt.marker);
  }
  get_label = true;
  // repub_global.markers.clear();
  // repub_global = repub;
  // repub_global.markers.shrink_to_fit();
  pub_annotation.publish(repub);
  return;
}

void get_bag_info_callback(const rosgraph_msgs::LogConstPtr& log_msg){
  if(!use_score_cluster)
    return;
  if(log_msg->msg.find(".bag") != log_msg->msg.npos){
    filename_change = true;
    score_file_name = log_msg->msg;
    int bag_start = score_file_name.find("log");
    int bag_end = score_file_name.find(".");
    score_file_name = score_file_name.substr(bag_start,bag_end-bag_start);
  }
  return;
}

void get_keyboard_refresh(){
  static struct termios oldt, newt;
  tcgetattr( STDIN_FILENO, &oldt);           // save old settings
  newt = oldt;
  newt.c_lflag &= ~(ICANON);                 // disable buffering      
  tcsetattr( STDIN_FILENO, TCSANOW, &newt);  // apply new settings
  int c = getchar();  // read character (non-blocking)
  tcsetattr( STDIN_FILENO, TCSANOW, &oldt);  // restore old settings
  if(c=='r'){
    ROS_WARN("Refresh the Program!");
    firstFrame = true;
    call_back_num = 0;
    cens.clear();
    cluster_cens.clear();
    max_size = 0, cluster_marker_max_size = 0, cluster_polygon_max_size = 0, trajcetory_max_size = 0, vel_marker_max_size = 0;
    debug_id_max_size = 0;
    id_count = 0;
    filename_change = false;
    score_frame = 0;
    filters.clear();
    filters.shrink_to_fit();
    dbscan_seg = dbscan();
  }
}

int main(int argc, char** argv){
  ros::init(argc,argv,"radar_kf_track");
  ros::NodeHandle nh;
  marker_color_init();
  tf_listener = new tf::TransformListener();
  message_filters::Subscriber<conti_radar::Measurement> sub_out(nh,"/radar_front_outlier",1);   // sub the radar points(outlier)
  message_filters::Subscriber<conti_radar::Measurement> sub_in(nh,"/radar_front_inlier",1);     // sub the radar points(inlier)
  message_filters::Subscriber<conti_radar::Measurement> sub(nh,"/radar_front",1);             // sub the radar points(total)
  message_filters::Subscriber<nav_msgs::Odometry> sub_vel(nh,"/vel",1);                       // sub the ego velocity
  // sub the 5 radar points
  message_filters::Subscriber<conti_radar::Measurement> sub_f (nh,"/radar_front",1);
  message_filters::Subscriber<conti_radar::Measurement> sub_fr(nh,"/radar_front_right",1);
  message_filters::Subscriber<conti_radar::Measurement> sub_fl(nh,"/radar_front_left",1);
  message_filters::Subscriber<conti_radar::Measurement> sub_br(nh,"/radar_back_right",1);
  message_filters::Subscriber<conti_radar::Measurement> sub_bl(nh,"/radar_back_left",1);

  pub_cluster_marker = nh.advertise<visualization_msgs::MarkerArray>("cluster_index", 1);
  pub_marker = nh.advertise<visualization_msgs::MarkerArray>("tracking_id", 1);
  pub_filter = nh.advertise<sensor_msgs::PointCloud2>("radar_history",500);
  pub_in_filter = nh.advertise<sensor_msgs::PointCloud2>("inlier_radar",200);
  pub_out_filter = nh.advertise<sensor_msgs::PointCloud2>("outlier_radar",200);
  pub_cluster_center = nh.advertise<sensor_msgs::PointCloud2>("cluster_center",500);
  pub_cluster_pointcloud = nh.advertise<sensor_msgs::PointCloud2>("cluster_radar_front_pointcloud",500);
  vis_vel_comp = nh.advertise<visualization_msgs::MarkerArray> ("radar_front_v_comp", 1);
  vis_vel = nh.advertise<visualization_msgs::MarkerArray> ("radar_front_v", 1);
  pub_trajectory = nh.advertise<visualization_msgs::MarkerArray> ("tracking_trajectory_path", 1);
  pub_pt = nh.advertise<visualization_msgs::MarkerArray> ("tracking_pt", 1);
  pub_pred = nh.advertise<visualization_msgs::MarkerArray> ("predict_pt", 1);
  pub_annotation = nh.advertise<visualization_msgs::MarkerArray> ("map_annotation", 1);
  pub_anno_cluster = nh.advertise<sensor_msgs::PointCloud2> ("annotation_pub", 300);
  pub_cluster_hull = nh.advertise<visualization_msgs::MarkerArray> ("cluster_hull", 1);
  pub_radar_debug = nh.advertise<visualization_msgs::MarkerArray> ("radar_debug_lines", 1);
  pub_debug_id_text = nh.advertise<visualization_msgs::MarkerArray> ("radar_debug_invalidID", 1);

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
  nh.param<bool>("use_5_radar"          ,use_5_radar          ,false);
  nh.param<bool>("use_ego_callback"     ,use_ego_callback     ,false);
  nh.param<bool>("use_score_cluster"    ,use_score_cluster    ,false);
  nh.param<string>("csv_name"           ,csv_name             ,"");
  nh.param<bool>("get_transformer"      ,get_transformer      ,false);
  nh.param<string>("cluster_type"       ,cluster_type         ,"dbscan");
  
  ros::Subscriber marker_sub;
  marker_sub = nh.subscribe("lidar_label", 10, &annotation); // sub the nuscenes annotation
  ros::Subscriber bag_sub;
  bag_sub = nh.subscribe("rosout",10,&get_bag_info_callback); // sub the bag name
  if(use_5_radar){
    message_filters::Synchronizer<radar_sync>* all_radar_sync_;
    all_radar_sync_ = new message_filters::Synchronizer<radar_sync>(radar_sync(6), sub_f, sub_fr, sub_fl, sub_br, sub_bl);
    all_radar_sync_->registerCallback(boost::bind(&radar_callback, _1, _2, _3, _4, _5));
  }
  else{
    if(use_ego_callback){
      message_filters::Synchronizer<ego_in_out_sync>* use_ego_motion_sync_;
      use_ego_motion_sync_ = new message_filters::Synchronizer<ego_in_out_sync>(ego_in_out_sync(4), sub_in, sub_out, sub_vel);
      use_ego_motion_sync_->registerCallback(boost::bind(&callback_ego, _1, _2, _3));
    }
    else{
      message_filters::Synchronizer<NoCloudSyncPolicy>* no_cloud_sync_;
      no_cloud_sync_ = new message_filters::Synchronizer<NoCloudSyncPolicy>(NoCloudSyncPolicy(3), sub, sub_vel);
      no_cloud_sync_->registerCallback(boost::bind(&callback, _1, _2));
    }
  }
  
  

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
  ROS_INFO("Cluster Type : %s"        ,cluster_type.c_str());
  if(use_KFT_module){
    ROS_INFO("\tShow %d types marker id"          , kft_id_num);
    ROS_INFO("\t%s the stop object tracker"       ,show_stopObj_id ? "Show":"Mask");
  }
  if(use_score_cluster){
    ROS_INFO("Use score-cluster and ego-motion in/outliner callback function");
    ROS_INFO("\tBox bias: %f", box_bias);
    ROS_INFO("\tOutput score info : %s"       , output_score_info ? "True" : "False");
    ROS_INFO("\tOutput score mat info : %s"   , output_score_mat_info ? "True" : "False");
    ROS_INFO("\tOutput GT inside points info : %s"   , output_gt_pt_info ? "True" : "False");
    ROS_INFO("\tWrite out the csv files : %s" , write_out_score_file ? "True" : "False");
  }
  else{
    ROS_INFO("use %s callback function"          , use_ego_callback ? "ego-motion in/outliner" : "original");
  }

  while(ros::ok()){
    ros::spinOnce();
    pub_marker.publish(m_s);
    radar_debug_visualization();
    // get_keyboard_refresh();
  }
  return 0;
}

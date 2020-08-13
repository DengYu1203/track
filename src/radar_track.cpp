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

#include <dbscan/dbscan.h>

using namespace std;
using namespace cv;

#define iteration 20 //plan segmentation
#define trajectory_frame 15

typedef message_filters::sync_policies::ApproximateTime<conti_radar::Measurement,
														                            nav_msgs::Odometry> NoCloudSyncPolicy;
typedef message_filters::sync_policies::ApproximateTime<conti_radar::Measurement,conti_radar::Measurement,
														                            nav_msgs::Odometry> ego_in_out_sync;

typedef pcl::PointXYZ PointT;

ros::Publisher pub_cluster_marker;  // pub cluster index
ros::Publisher pub_marker;          // pub tracking id
ros::Publisher pub_filter;          // pub filter points
ros::Publisher pub_in_filter;       // pub the inlier points
ros::Publisher pub_out_filter;      // pub the outlier points
ros::Publisher pub_cluster_center;         // pub cluster points
ros::Publisher pub_cluster_pointcloud;  // pub cluster point cloud
ros::Publisher vis_vel_comp;        // pub radar comp vel
ros::Publisher vis_vel;             // pub radar vel
ros::Publisher pub_trajectory;      // pub the tracking trajectory
ros::Publisher pub_pt;              // pub the tracking history pts
ros::Publisher pub_pred;            // pub the predicted pts
ros::Publisher pub_annotation;


tf::StampedTransform echo_transform;
bool lidar_tf_check = false;
bool radar_front_tf_check = false;
bool DA_choose = false;  // true: hungarian, false: my DA
bool output_KFT_result = true;
bool output_obj_id_result = true;
bool output_radar_info = true;
bool output_cluster_info = true; // output the cluster result
bool output_dbscan_info = true;
bool output_DA_pair = true;
bool output_exe_time = false; // print the program exe time
bool output_label_info = false; // print the annotation callback info
bool use_annotation_module = false; // true: repulish the global annotation and score the cluster performance
bool use_KFT_module = true; // true: cluster and tracking, fasle: cluster only
bool show_vel_marker = false;
int kft_id_num = 1; // 1: only show the id tracked over 5 frames, 2: show the id tracked over 1 frames, 3: show all id
bool use_ego_callback = false;  // true: use the ego motion in/outlier pts, false: use the original pts
bool use_score_cluster = false; // true: score the cluster performance

int call_back_num = 0;

#define cluster_distance 2.5 // euclidean distance 0.8,1.5,2.5,3 for car length
#define cluster_vel_angle 2  // velocity angle threshold 2
#define cluster_vel 1        // velocity threshold 1
bool filter_cluster_flag=true;       // decide cluster the radar points or not


ros::Time radar_stamp,label_stamp;
float vx = 0; // vehicle vel x
float vy = 0; // vehicle vel y

// label annotation
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


#define box_bias 0.3
vector< label_point> label_vec;
bool get_label = false;

vector<kf_tracker_point> cens;
int cens_index = 0;
vector<kf_tracker_point> cluster_cens;
visualization_msgs::MarkerArray m_s,l_s,cluster_s;
int max_size = 0, cluster_marker_max_size = 0;

float dt = 0.08f;     //0.1f 0.08f=1/13Hz(radar)
float sigmaP = 0.01;  //0.01
float sigmaQ = 0.1;   //0.1
#define bias 4.0      // used for data association bias 5 6
#define mahalanobis_bias 1.0 // used for data association(mahalanobis dist)
int id_count = 0; //the counter to calculate the current occupied track_id

////////////////////////kalman/////////////////////////
#define frame_lost 10   // 5 10

std::vector <string> tracked;
std::vector <int> count,id;
std::vector<geometry_msgs::Point> pred_velocity;
int un_assigned;

ros::Publisher objID_pub;
// KF init
int stateDim = 6;// [x,y,v_x,v_y]//,w,h]
int measDim = 3;// [z_x,z_y//,z_w,z_h]
int ctrlDim = 0;// control input 0(acceleration=0,constant v model)


bool firstFrame = true;
bool get_transformer = false;
tf::TransformListener *tf_listener;

enum class MOTION_STATE{
  move,
  stop,
  slow_down
};

enum class TRACK_STATE{
  missing,
  tracking,
  unstable
};

// for score the cluster performance
typedef struct cluster_score{
  int frame;
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
  vector<geometry_msgs::Point> future;

  int lose_frame;
  int track_frame;
  MOTION_STATE motion;
  TRACK_STATE tracking_state;
  int match_clus = 1000;
  int cluster_idx;
  int uuid ;
}track;

std::vector<track> filters;

// publish the tracking id
void marker_id(bool firstFrame,std::vector<int> obj_id,std::vector<int> marker_obj_id,std::vector<int> tracking_frame_obj_id){
  visualization_msgs::Marker marker;
    int k;
    for(k=0; k<cens.size(); k++){
        // marker.header.frame_id="/nuscenes_radar_front";
        marker.header.frame_id="/map";
        marker.header.stamp = ros::Time();//to show every tag  (ros::Time::now()for latest tag)
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.id = k;
        marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;

        marker.scale.z = 2.0f;  // rgb(255,127,122)
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
        marker.text = "ID:"+ss.str();
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
void color_cluster(std::vector< std::vector<kf_tracker_point> > cluster_list){
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
    // marker.header.frame_id="/nuscenes_radar_front";
    marker.header.frame_id="/map";
    marker.header.stamp = ros::Time();
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.id = i;
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.scale.z = 3.0f;  // rgb(255,127,122)
    marker.color.r = color * 3.34448160535f;
    marker.color.g = color * 1.70357751278f;
    marker.color.b = color * 8.77192982456f;
    marker.color.a = 1;
    stringstream ss;
    ss << i;
    marker.text = ss.str();
    geometry_msgs::Pose pose;
    pose.position.x = cluster_list.at(i).at(0).x;
    pose.position.y = cluster_list.at(i).at(0).y;
    pose.position.z = cluster_list.at(i).at(0).z+1.0f;
    marker.pose = pose;
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
  cluster_cloud->header.frame_id = "/map";
  cluster_cloud->header.stamp = radar_stamp;
  pub_cluster_pointcloud.publish(cluster_cloud);
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
    // cout << "In cluster_list size = 0, the cluster_s size is:"<<cluster_s.markers.size()<<endl;
  }
  pub_cluster_marker.publish(cluster_s);
}

// show the radar velocity arrow
void vel_marker(const conti_radar::MeasurementConstPtr& input){
	visualization_msgs::MarkerArray marker_array_comp;
	visualization_msgs::MarkerArray marker_array;

	for(int i=0; i<input->points.size(); i++){
		visualization_msgs::Marker marker;
		marker.header.frame_id = "/nuscenes_radar_front";
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

		marker.scale.x = sqrt(pow(input->points[i].lateral_vel_comp,2) + 
				 									pow(input->points[i].longitude_vel_comp,2)); //~lenght~//
		marker.scale.y = 0.8;
		marker.scale.z = 0.8;

		marker.color.a = 1.0;
		// marker.color.r = 231.0f/255.0f;
		// marker.color.g = 232.0f/255.0f;
		// marker.color.b = 135.0f/255.0f;
		marker.color.r = 225.0f/255.0f;
		marker.color.g = 228.0f/255.0f;
		marker.color.b = 144.0f/255.0f;

		marker_array_comp.markers.push_back(marker);
		///////////////////////////////////////////////////////////////////

		theta = atan2(input->points[i].lateral_vel,
												input->points[i].longitude_vel);
		Q.setRPY( 0, 0, theta );
		marker.pose.orientation = tf2::toMsg(Q);

		marker.scale.x = sqrt(pow(input->points[i].lateral_vel,2) + 
				 									pow(input->points[i].longitude_vel,2)); //~lenght~//
		marker.scale.y = 0.8;
		marker.scale.z = 0.8;

		marker.color.a = 1.0;
		marker.color.r = 0.0;
		marker.color.g = 1.0;
		marker.color.b = 1.0;

		marker_array.markers.push_back(marker);
	}
	vis_vel_comp.publish( marker_array_comp );
	vis_vel.publish( marker_array );
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

    // if (velocity >= moving && velocity < invalid_v && !(filters.at(k).state.compare("tracking"))){  
    if ( filters.at(k).tracking_state == TRACK_STATE::tracking || filters.at(k).tracking_state == TRACK_STATE::unstable ){ 
      visualization_msgs::Marker marker, P;
      marker.header.frame_id = "/map";
      marker.header.stamp = ros::Time();//to show every tag  (ros::Time::now()for latest tag)
      marker.action = visualization_msgs::Marker::ADD;
      marker.lifetime = ros::Duration(dt);
      marker.type = visualization_msgs::Marker::LINE_STRIP;
      marker.ns = "trajectory";
      marker.id = k;
      marker.pose.orientation.w = 1.0;
      marker.scale.x = 0.2f;
      marker.color.a = 0.7;
      marker.color.r = 226.0f/255.0f;
      marker.color.g = 195.0f/255.0f;
      marker.color.b = 243.0f/255.0f;


      // history pts marker
      P.header.frame_id = "/map";
      P.header.stamp = ros::Time();
      P.action = visualization_msgs::Marker::ADD;
      P.lifetime = ros::Duration(dt);
      P.type = visualization_msgs::Marker::POINTS;
      P.ns = "trajectory";
      P.id = k+1;
      P.scale.x = 0.2f;
      P.scale.y = 0.2f;
      P.color.r = 1.0f;
      P.color.a = 1;

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
      Predict_p.header.frame_id = "/map";
      Predict_p.header.stamp = ros::Time();
      Predict_p.action = visualization_msgs::Marker::ADD;
      Predict_p.lifetime = ros::Duration(dt);
      Predict_p.type = visualization_msgs::Marker::POINTS;
      Predict_p.id = k+2;
      Predict_p.scale.x = 0.4f;
      Predict_p.scale.y = 0.4f;
      Predict_p.color.r = (155.0f/255.0f);
      Predict_p.color.g = ( 99.0f/255.0f);
      Predict_p.color.b = (227.0f/255.0f);
      Predict_p.color.a = 1;
      geometry_msgs::Point pred = filters.at(k).pred_pose;
      Predict_p.points.push_back(pred);

    
      tra_array.markers.push_back(marker);
      point_array.markers.push_back(P);
      pred_point_array.markers.push_back(Predict_p);
    }
  }

  pub_trajectory.publish(tra_array);
  pub_pt.publish(point_array);
  pub_pred.publish(pred_point_array);
}

// calculate euclidean distance of two points
double euclidean_distance(geometry_msgs::Point p1, geometry_msgs::Point p2)
{
  return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}

double mahalanobis_distance(geometry_msgs::Point p1, geometry_msgs::Point p2,cv::Mat cov){
  // cv::Vec6f x(p1.x,p1.y,p1.z,0,0,0);
  // cv::Vec6f y(p2.x,p2.y,p2.z,0,0,0);
  cv::Vec3f x(p1.x,p1.y,p1.z);    // measurementNoiseCov:3x3 matrix
  cv::Vec3f y(p2.x,p2.y,p2.z);
  // cv::Mat x = cv::Mat(measDim, 1, CV_32F, {p1.x,p1.y,p1.z});
  // cv::Mat y = cv::Mat(measDim, 1, CV_32F, {p2.x,p2.y,p2.z});
  
  double distance = cv::Mahalanobis(x,y,cov);
  return (distance);
}

// int find_matching(std::vector<float> dist_vec){
//   float now_min = std::numeric_limits<float>::max();
//   int cluster_idx;
//   for (int i=0; i<dist_vec.size(); i++){
//     if(dist_vec.at(i)<now_min){
//       now_min = dist_vec.at(i);
//       cluster_idx = i;
//     }
//   }
//   // cout<<"minIndex="<<cluster_idx<<endl;
//   return cluster_idx;
// }

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
    // cout<<"\nDA distMat:\n";
    // for(int i =0 ;i<filter_size;i++){
    //   cout<<"Row "<<i<<":";
    //   for(int j=0;j<radar_num;j++){
    //     cout<<"\t"<<dist_mat[i][j];
    //   }
    //   cout<<endl;
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

int new_track(kf_tracker_point cen, int idx){
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
    if(cen.vel < 0.1)
      tk.motion = MOTION_STATE::stop;
    else
      tk.motion = MOTION_STATE::move;
    tk.lose_frame = 0;
    tk.track_frame = 0;
    tk.cluster_idx = idx;

    tk.uuid = ++id_count;
    filters.push_back(tk);
    // cout<<"Done init newT at "<<id_count<<" is ("<<tk.pred_pose.x <<"," <<tk.pred_pose.y<<")"<<endl;
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
      cout << "Now center radar points :"<<cens.size()<<endl;  // The total radar outlier measurements now!
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
        cout << i+1 << ": \t(" << pt.x << "," << pt.y << "," << pt.z << ")";
        cout << "\tvel: (" << cens[i].vel << cens[i].x_v << "," << cens[i].y_v << ")\n";
      }
      clusterCenters.push_back(pt);
    }
    pcl::toROSMsg(*center_points,*center_cloud);
    center_cloud->header.frame_id = "/map";
    center_cloud->header.stamp = radar_stamp;
    pub_cluster_center.publish(center_cloud);

    std::vector<geometry_msgs::Point> KFpredictions;
    i=0;
    


    //construct dist matrix (mxn): m tracks, n clusters.
    std::vector<geometry_msgs::Point> copyOfClusterCenters(clusterCenters);
    std::vector<std::vector<double> > distMat;
    // cout<<"distMat:\n";
    // for(int i=0;i<cens.size();i++)
    //   cout<<"\t"<<i;
    // cout<<endl;
    int row_count = 0;
    for(std::vector<track>::const_iterator it = filters.begin (); it != filters.end (); ++it)
    {
        // cout<<"Row "<<row_count++<<":";
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
          // cv::Mat S = (*it).kf.measurementMatrix * (*it).kf.errorCovPre * (*it).kf.measurementMatrix.t() + (*it).kf.measurementNoiseCov;
          // cv::Mat S_us = (*it).kf_us.measurementMatrix * (*it).kf_us.errorCovPre * (*it).kf_us.measurementMatrix.t() + (*it).kf_us.measurementNoiseCov;
          // double dis_1 = mahalanobis_distance((*it).pred_pose,copyOfClusterCenters[n],S.inv());
          // double dis_2 = mahalanobis_distance((*it).pred_pose_us,copyOfClusterCenters[n],S_us.inv());
          double dis_1 = mahalanobis_distance((*it).pred_pose,copyOfClusterCenters[n],(*it).kf.measurementNoiseCov);
          double dis_2 = mahalanobis_distance((*it).pred_pose_us,copyOfClusterCenters[n],(*it).kf_us.measurementNoiseCov);
          // cout<<"Mahalanobis:"<<ma_dis1<<" ,"<<ma_dis2<<endl;
          if (dis_1<dis_2){
            distVec.push_back(dis_1);
          }
          else
          {
            distVec.push_back(dis_2);
          }
          
          // distVec.push_back(euclidean_distance((*it).pred_pose,copyOfClusterCenters[n]));
          // cout<<"\t"<<distVec.at(n);
        }
        // cout<<endl;
        distMat.push_back(distVec);

    }

    vector<int> assignment;
    if(DA_choose){
      //hungarian method to optimize(minimize) the dist matrix
      HungarianAlgorithm HungAlgo;
      double cost = HungAlgo.Solve(distMat, assignment);
      
      if(output_DA_pair){
        cout<<"HungAlgo assignment pair:\n";
        for (unsigned int x = 0; x < distMat.size(); x++)
          std::cout << x << "," << assignment[x] << "\t";
        std::cout << "\ncost: " << cost << std::endl; // HungAlgo computed cost
      }
    }
    else{
      // my DA method
      assignment = find_min(distMat);
      if(output_DA_pair){
        cout<<"My DA assignment pair:\n";
        for (unsigned int x = 0; x < distMat.size(); x++)
          std::cout << x << "," << assignment[x] << "\t";
        cout<<endl;
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
      
      // float delta_x = (pred_pos.x-filters.at(k).kf.statePost.at<float>(0));
      // float delta_y = (pred_pos.y-filters.at(k).kf.statePost.at<float>(1));
      float dist_thres1_ma = mahalanobis_distance(pred_v,v_mean,filters.at(k).kf.measurementNoiseCov);
      float dist_thres2_ma = mahalanobis_distance(pred_v_us,v_mean,filters.at(k).kf_us.measurementNoiseCov);
      // cv::Mat S = filters.at(k).kf.measurementMatrix * filters.at(k).kf.errorCovPre * filters.at(k).kf.measurementMatrix.t() + filters.at(k).kf.measurementNoiseCov;
      // cv::Mat S_us = filters.at(k).kf_us.measurementMatrix * filters.at(k).kf_us.errorCovPre * filters.at(k).kf_us.measurementMatrix.t() + filters.at(k).kf_us.measurementNoiseCov;
      // float dist_thres1_ma = mahalanobis_distance(pred_v,v_mean,S.inv());
      // float dist_thres2_ma = mahalanobis_distance(pred_v_us,v_mean,S_us.inv());
      float dist_thres1 = sqrt(pred_v.x * pred_v.x * dt * dt + pred_v.y * pred_v.y * dt * dt); //float
      float dist_thres2 = sqrt(pred_v_us.x * pred_v_us.x * dt * dt + pred_v_us.y * pred_v_us.y * dt * dt); //float
      
      float dist_thres;
      if(dist_thres1<dist_thres2){
        dist_thres = dist_thres2;
      }
      else
        dist_thres = dist_thres1;
      
      if(output_KFT_result){
        cout<<"------------------------------------------------\n";
        // float dist_thres = sqrt(delta_x * delta_x + delta_y * delta_y);
        cout << "The dist_thres for " <<filters.at(k).uuid<<" is "<<dist_thres<<endl;
        cout<<"predict v:"<<pred_v.x<<","<<pred_v.y<<endl;
        cout<<"predict pos:"<<pred_pos.x<<","<<pred_pos.y<<endl;
        cout<<"predict v(us):"<<pred_v_us.x<<","<<pred_v_us.y<<endl;
        cout<<"predict pos(us):"<<pred_pos_us.x<<","<<pred_pos_us.y<<endl;
      }
      
      //-1 for non matched tracks
      if ( assignment[k] != -1 ){
        // if( dist_vec.at(assignment[k]) <=  dist_thres + mahalanobis_bias ){//bias as gating function to filter the impossible matched detection 
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
            cout<<"The state of "<<k<<" filters is \033[1;32mtracking\033[0m,to cluster_idx "<< filters.at(k).cluster_idx <<" track: "<<filters.at(k).track_frame;
            cout<<" ("<<cens.at(assignment[k]).x<<","<<cens.at(assignment[k]).y<<")"<<endl;
        }
        else if (filters.at(k).tracking_state == TRACK_STATE::missing)
            cout<<"The state of "<<k<<" filters is \033[1;34mlost\033[0m,to cluster_idx "<< filters.at(k).cluster_idx <<" lost: "<<filters.at(k).lose_frame<<endl;
        else
            cout<<"\033[1;31mUndefined state for trackd "<<k<<"\033[0m"<<endl;
      }
    
    }

    if(output_obj_id_result){
      cout<<"\033[1;33mThe obj_id:\033[0m\n";
      for (i=0; i<cens.size(); i++){
        cout<<obj_id.at(i)<<" ";
        }
      cout<<endl;
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
      cout<<"\033[1;33mThe obj_id after new track:\033[0m\n";
      for (i=0; i<cens.size(); i++){
        cout<<obj_id.at(i)<<" ";
        }
      cout<<endl;
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
    cout<<"\033[1;33mTracker number: "<<filters.size()<<" tracks.\033[0m\n"<<endl;


    //begin mark
    m_s.markers.resize(cens.size());
    m_s.markers.clear();

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
        // cout << "The corrected state of "<<i<<"th KF is: "<<estimated[i].at<float>(0)<<","<<estimated[i].at<float>(1)<<endl;
        i++;
    }

  return;
}

void cluster(void){
  int cluster_num = 0;
  for(int i=0;i<cens.size();i++){
    geometry_msgs::Point p1;
    p1.x = cens.at(i).x;
    p1.y = cens.at(i).y;
    p1.z = cens.at(i).z;
    for (int j=i;j<cens.size();j++){
      geometry_msgs::Point p2;
      p2.x = cens.at(j).x;
      p2.y = cens.at(j).y;
      p2.z = cens.at(j).z;
      double distance = euclidean_distance(p1,p2);
      if(distance<cluster_distance){
        // Using velocity angle as threshold
        bool ang_threshold = (abs(cens.at(i).vel_ang-cens.at(j).vel_ang)<cluster_vel_angle);
        // if(abs(cens.at(i).vel_ang-cens.at(j).vel_ang)<cluster_vel_angle){
        //   if(cens.at(i).cluster_flag==-1){
        //     cens.at(i).cluster_flag = cluster_num++;
        //   }
        //   cens.at(j).cluster_flag = cens.at(i).cluster_flag;
        // }
        // Using velocity as threshold
        bool vel_threshold = (abs(cens.at(i).vel-cens.at(j).vel)<cluster_vel);
        // if(abs(cens.at(i).vel-cens.at(j).vel)<cluster_vel){
        //   if(cens.at(i).cluster_flag==-1){
        //     cens.at(i).cluster_flag = cluster_num++;
        //   }
        //   cens.at(j).cluster_flag = cens.at(i).cluster_flag;
        // }

        // Using both as threshold
        if(ang_threshold||vel_threshold){
          if(cens.at(i).cluster_flag==-1){
            cens.at(i).cluster_flag = cluster_num++;
          }
          cens.at(j).cluster_flag = cens.at(i).cluster_flag;
        }
      }
    }
  }
  // republish cluster radar points
  pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_points(new pcl::PointCloud<pcl::PointXYZ>);
  sensor_msgs::PointCloud2::Ptr cluster_cloud(new sensor_msgs::PointCloud2);
  cluster_points->clear();
  // end republish
  if(output_cluster_info)
    cout<<"\nIn cluter:\n";
  for(int i=0;i<cluster_num;i++){
    kf_tracker_point c;
    c.x = 0;
    c.y = 0;
    c.z = 0;
    c.x_v = 0;
    c.y_v = 0;
    c.z_v = 0;
    int index=0;
    if(output_cluster_info){
      cout<<"==============================\n";
      cout<<"cluster "<<i<<":\n";
    }
    for(int j=0;j<cens.size();j++){
      if(cens.at(j).cluster_flag==i){
        index++;
        c.x += cens.at(j).x;
        c.y += cens.at(j).y;
        c.z += cens.at(j).z;
        c.x_v += cens.at(j).x_v;
        c.y_v += cens.at(j).y_v;
        c.z_v += cens.at(j).z_v;
        if(output_cluster_info){
          cout<<"position sum:"<<"("<<c.x<<","<<c.y<<")"<<endl;
          cout<<"vel sum:"<<"("<<c.x_v<<","<<c.y_v<<")"<<endl;
          cout<<"index="<<index<<endl;
        }
      }

    }

    c.x /= index;
    c.y /= index;
    c.z /= index;
    c.x_v /= index;
    c.y_v /= index;
    c.z_v /= index;
    cluster_cens.push_back(c);

    pcl::PointXYZ pt;
    pt.x = c.x;
    pt.y = c.y;
    pt.z = c.z;
    cluster_points->push_back(pt);
  }
  if(output_cluster_info)
    cout<<endl;
  pcl::toROSMsg(*cluster_points,*cluster_cloud);
  cluster_cloud->header.frame_id = "/nuscenes_radar_front";
  cluster_cloud->header.stamp = radar_stamp;
  pub_cluster_center.publish(cluster_cloud);

}

void get_transform(ros::Time timestamp){
  // ros::Rate rate(1);
  tf_listener->waitForTransform("nuscenes_radar_front","map",timestamp,ros::Duration(1));
  std::cout << std::fixed; std::cout.precision(3);
  try{
    
    tf_listener->lookupTransform("map","nuscenes_radar_front",timestamp,echo_transform);
    std::cout << "At time " << echo_transform.stamp_.toSec() << std::endl;
    // double yaw, pitch, roll;
    // echo_transform.getBasis().getRPY(roll, pitch, yaw);
    tf::Quaternion q = echo_transform.getRotation();
    tf::Vector3 v = echo_transform.getOrigin();
    std::cout << "- Translation: [" << v.getX() << ", " << v.getY() << ", " << v.getZ() << "]" << std::endl;
    std::cout << "- Rotation: in Quaternion [" << q.getX() << ", " << q.getY() << ", " 
              << q.getZ() << ", " << q.getW() << "]" << std::endl;
    cout<<"Frame id:"<<echo_transform.frame_id_<<", Child id:"<<echo_transform.child_frame_id_<<endl;
  }
  catch(tf::TransformException& ex)
  {
    // std::cout << "Failure at "<< ros::Time::now() << std::endl;
    std::cout << "Exception thrown:" << ex.what()<< std::endl;
    // std::cout << "The current list of frames is:" <<std::endl;
    // std::cout << tf_listener.allFramesAsString()<<std::endl;
    
  }
  // rate.sleep();
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
      cout<<"( "<<cens.at(i).x<<","<<cens.at(i).y<<","<<cens.at(i).z<<")\n";
      ka.statePost.at<float>(0)=cens.at(i).x;
      ka.statePost.at<float>(1)=cens.at(i).y;
      ka.statePost.at<float>(2)=cens.at(i).z;
      ka.statePost.at<float>(3)=cens.at(i).x_v;// initial v_x
      ka.statePost.at<float>(4)=cens.at(i).y_v;// initial v_y
      ka.statePost.at<float>(5)=cens.at(i).z_v;// initial v_z

      tk.kf = ka;
      tk.kf_us = ka;  // ready to update the state
      tk.tracking_state = TRACK_STATE::tracking;
      if(cens.at(i).vel <= 0.1)
        tk.motion = MOTION_STATE::stop;
      else
        tk.motion = MOTION_STATE::move;
      tk.lose_frame = 0;
      tk.track_frame = 1;

      tk.uuid = i;
      id_count = i;
      filters.push_back(tk);
  }
  cout<<"Initiate "<<filters.size()<<" tracks."<<endl;
  cout << m_s.markers.size() <<endl;
  m_s.markers.clear();
  cout<< "\033[1;33mMarkerarray is empty(1):\033[0m"<<m_s.markers.empty()<< endl;
  std::vector<int> no_use_id;
  marker_id(firstFrame,no_use_id,no_use_id,no_use_id);
  firstFrame=false;
  return;
}

void output_score(double beta, v_measure_score result){
  string dir_path = ros::package::getPath("track") + "/cluster_score/" + to_string(ros::Time::now().toBoost().date()) + "/";
  string csv_file_name = score_file_name + ".csv";
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

void score_cluster(std::vector< std::vector<kf_tracker_point> > dbscan_cluster, double beta=1.0){
  get_label = false;
  std::vector< std::vector<kf_tracker_point> > gt_cluster;
  std::vector<kf_tracker_point> score_cens = cens;
  cout << endl << "\033[1;33mScore cluster performance\033[0m" << endl;
  for(int label_idx=0;label_idx<label_vec.size();label_idx++){
    std::vector<kf_tracker_point> label_list;
    int n = 0;
    label_point label_pt = label_vec.at(label_idx);
    tf::Vector3 v1,v2;
    v1 = label_pt.front_left - label_pt.back_left;
    v2 = label_pt.back_right - label_pt.back_left;
    double ang1 = v1.angle(tf::Vector3(1,0,0));
    double ang2 = v2.angle(tf::Vector3(1,0,0));
    // make sure that the ang1 > ang2
    if(ang1 < ang2){
      std::swap(ang1,ang2);
      // std::swap(v1,v2);
    }
    for(int i=0;i<score_cens.size();i++){
      if(score_cens.at(i).cluster_flag != -1)
        continue;
      kf_tracker_point score_pt = score_cens.at(i);
      tf::Vector3 v3;
      v3 = tf::Point(score_pt.x,score_pt.y,score_pt.z) - label_pt.back_left;
      double ang3 = v3.angle(tf::Vector3(1,0,0));
      if(!(ang2<=ang3 && ang3<=ang1))
        continue;
      if((v3.dot(v1) / v1.dot(v1) * v1).length() <= v1.length() && (v3.dot(v2) / v2.dot(v2) * v2).length() <= v2.length()){
        score_cens.at(i).cluster_flag = label_idx;
        label_list.push_back(score_cens.at(i));
        n++;
      }
    }
    if(n != 0)
      gt_cluster.push_back(label_list);
  }
  cout << "Timestamp: " << label_vec.at(0).marker.header.stamp.toSec() << endl;
  // cout << "V-measurement matrix:\n";
  cout << "Ground true (row): " << gt_cluster.size() << " ,DBSCAN cluster (col):" << dbscan_cluster.size() << endl;
  Eigen::MatrixXd v_measure_mat(gt_cluster.size(),dbscan_cluster.size());
  v_measure_mat.setZero();
  // cout << "Ground truth cluster:\n";
  for(int idx=0;idx<gt_cluster.size();idx++){
    std::vector<kf_tracker_point> temp_gt = gt_cluster.at(idx);
    // cout << "-------------------------------\n";
    //   cout << "cluster index: " << idx << endl;
    for(int i=0;i<temp_gt.size();i++){
      // cout << "\tPosition: (" <<temp_gt.at(i).x<<","<<temp_gt.at(i).y<<")"<<endl;
      for(int cluster_idx=0;cluster_idx<dbscan_cluster.size();cluster_idx++){
        std::vector<kf_tracker_point> temp_dbcluster = dbscan_cluster.at(cluster_idx);
        for(int j=0;j<temp_dbcluster.size();j++){
          tf::Vector3 distance = tf::Vector3(temp_gt.at(i).x,temp_gt.at(i).y,0) - tf::Vector3(temp_dbcluster.at(j).x,temp_dbcluster.at(j).y,0);
          if(distance.length() <= 0.001){
            v_measure_mat(idx,cluster_idx) += 1;
          }
        }
      }
    }
  }
  int n = v_measure_mat.sum();
  // cout << setprecision(0) << v_measure_mat << endl;
  cout << "n = " << n << endl;

  // v-measure score calculation
  Eigen::RowVectorXd row_sum = v_measure_mat.rowwise().sum();
  Eigen::RowVectorXd col_sum = v_measure_mat.colwise().sum();
  v_measure_score result;
  result.frame = ++score_frame;
  // double result.h_ck = 0, result.h_c = 0, result.h_kc = 0, result.h_k = 0;
  // double result.homo = 0, result.comp = 0;
  cout << setprecision(0);
  // cout << "Row sum: " << row_sum << endl;
  // cout << "Col sum: " << col_sum << endl;
  for(int i=0;i<v_measure_mat.rows();i++){
    for(int j=0;j<v_measure_mat.cols();j++){
      if(v_measure_mat(i,j) == 0)
        continue;
      if(col_sum(j) == 0)
        result.h_ck -= (v_measure_mat(i,j) / n) * log(v_measure_mat(i,j)/0.00001);
      else
        result.h_ck -= (v_measure_mat(i,j) / n) * log(v_measure_mat(i,j)/col_sum(j));
    }
    if(row_sum(i) == 0)
      continue;
    else
      result.h_c -= (row_sum(i)/n) * log(row_sum(i)/n);
  }
  result.homo = 1 - (result.h_ck / result.h_c);
  for(int j=0;j<v_measure_mat.cols();j++){
    for(int i=0;i<v_measure_mat.rows();i++){
      if(v_measure_mat(i,j) == 0)
        continue;
      if(row_sum(i) == 0)
        result.h_kc -= (v_measure_mat(i,j) / n) * log(v_measure_mat(i,j)/0.00001);
      else
        result.h_kc -= (v_measure_mat(i,j) / n) * log(v_measure_mat(i,j)/row_sum(i));
    }
    if(col_sum(j) == 0)
      continue;
    else
      result.h_k -= (col_sum(j)/n) * log(col_sum(j)/n);

  }
  result.comp = 1 - (result.h_kc / result.h_k);
  result.v_measure_score = (1+beta)*result.homo*result.comp / (beta*result.homo + result.comp);
  cout << setprecision(3);
  cout << "Homogeneity:\t" << result.homo << "\t-> ";
  cout << "H(C|k) = " << result.h_ck << ", H(C) = " << result.h_c << endl;
  cout << "Completeness:\t" << result.comp << "\t-> ";
  cout << "H(K|C) = " << result.h_kc << ", H(K) = " << result.h_k << endl;
  cout << "\033[1;46mV-measure score: " << result.v_measure_score << "\033[0m" << endl;
  output_score(beta,result);

}

void callback(const conti_radar::MeasurementConstPtr& msg,const nav_msgs::OdometryConstPtr& odom){
    int radar_point_size = msg->points.size();
    cout<<"\033[1;33m\n\n"<<++call_back_num<<" Call Back radar points:"<<radar_point_size<<"\033[0m"<<endl;
    if(radar_point_size==0) //radar points cannot be zero, it would cause segement fault
      return;
    radar_stamp = msg->header.stamp;

    if(show_vel_marker)
      vel_marker(msg);
    if(get_transformer)
      get_transform(radar_stamp);
    else
      echo_transform.setIdentity();
    // republish radar points
    pcl::PointCloud<pcl::PointXYZI>::Ptr filter_points(new pcl::PointCloud<pcl::PointXYZI>);
    sensor_msgs::PointCloud2::Ptr filter_cloud(new sensor_msgs::PointCloud2);
    filter_points->clear();
    // end republish
    
    std::cout << std::fixed; std::cout.precision(3);
    cout<<"Radar time:"<<radar_stamp.toSec()<<endl;
    cens.clear();
    cluster_cens.clear();
    // get the ego velocity
    vx = odom->twist.twist.linear.x;
    vy = odom->twist.twist.linear.y;
    cout<<"\033[1;33mVehicle velocity:\n\033[0m";
    cout<<"Time stamp:"<<odom->header.stamp.toSec()<<endl;
    cout<<"vx: "<<vx<<" ,vy: "<<vy<<endl<<endl;
    if(output_radar_info)
      cout<<"Radar information\n";
    for (int i = 0;i<msg->points.size();i++){
        tf::Point trans_pt;
        trans_pt.setValue(msg->points[i].longitude_dist, msg->points[i].lateral_dist, 0);
        trans_pt = echo_transform * trans_pt;
        float range = sqrt(pow(msg->points[i].longitude_dist,2) + 
                           pow(msg->points[i].lateral_dist,2));
        float velocity = sqrt(pow(msg->points[i].lateral_vel_comp,2) + 
                              pow(msg->points[i].longitude_vel_comp,2));
        double angle = atan2(msg->points[i].lateral_vel_comp,msg->points[i].longitude_vel_comp)*180/M_PI;

        // float velocity = sqrt(pow(msg->points[i].lateral_vel+vy,2) + 
        //                       pow(msg->points[i].longitude_vel+vx,2));
        // double angle = atan2(msg->points[i].lateral_vel+vy,msg->points[i].longitude_vel+vx)*180/M_PI;
        

        if(output_radar_info){
          cout<<"----------------------------------------------\n";
          cout<<i<<"\tposition:("<<msg->points[i].longitude_dist<<","<<msg->points[i].lateral_dist<<")\n";
          cout<<"\tposition to map:("<<trans_pt.x()<<","<<trans_pt.y()<<")\n";
          cout<<"\tvel:"<<velocity<<"("<<msg->points[i].longitude_vel_comp<<","<<msg->points[i].lateral_vel_comp<<")\n\tdegree:"<<angle;
          // cout<<"\tvel:"<<velocity<<"("<<msg->points[i].longitude_vel+vx<<","<<msg->points[i].lateral_vel+vy<<")\n\tdegree:"<<angle;
          cout<<"\n\tinvalid_state:"<<msg->points[i].invalid_state;
          cout<<"\n\tRCS value:"<<msg->points[i].rcs;
          cout<<endl;
        }
        // if((range>100||(msg->points[i].invalid_state==7))){               //set radar threshold (distance and invalid state)
        if(range>100){               //set radar threshold (distance)
            radar_point_size--;
            // if(output_radar_info)
            //   cout<<endl;
        }
        else{
            kf_tracker_point c;
            c.x = trans_pt.x();
            c.y = trans_pt.y();
            c.z = 0;
            // c.x_v = msg->points[i].longitude_vel_comp+vx;
            // c.y_v = msg->points[i].lateral_vel_comp+vy;
            // c.x_v = msg->points[i].longitude_vel+vx;
            // c.y_v = msg->points[i].lateral_vel+vy;
            c.x_v = msg->points[i].longitude_vel_comp;
            c.y_v = msg->points[i].lateral_vel_comp;
            c.z_v = 0;
            c.vel_ang = angle;
            c.vel = velocity;
            c.cluster_flag = -1;  // -1: not clustered
            c.vistited = false;
            c.rcs = msg->points[i].rcs;
            
            pcl::PointXYZI pt;
            pt.x = c.x;
            pt.y = c.y;
            pt.z = c.z;
            pt.intensity = msg->points[i].rcs;
            filter_points->push_back(pt);
            // if(output_radar_info)
            //   cout<<"\n\tposition:("<<c.x<<","<<c.y<<","<<c.z<<")\n";
            cens.push_back(c);
        }
        
       

    }
    pcl::toROSMsg(*filter_points,*filter_cloud);
    if(!get_transformer)
      filter_cloud->header.frame_id = "/nuscenes_radar_front";
    else
      filter_cloud->header.frame_id = "/map";
    filter_cloud->header.stamp = radar_stamp;
    pub_filter.publish(filter_cloud);
    // cluster the radar points
    if (filter_cluster_flag){
      // cluster the points with DBSCAN-based method
      double DBSCAN_START, DBSCAN_END;
      DBSCAN_START = clock();
      dbscan dbscan(cens);
      std::vector< std::vector<kf_tracker_point> > dbscan_cluster, stage_one_cluster;
      dbscan.output_info = output_dbscan_info;
      dbscan_cluster = dbscan.cluster();
      DBSCAN_END = clock();
      if(output_exe_time){
        cout << endl << "\033[1;42mDBSCAN Execution time: " << (DBSCAN_END - DBSCAN_START) / CLOCKS_PER_SEC << "s\033[0m";
        cout << endl;
      }
      color_cluster(dbscan_cluster);

      // cluster the points with original method
      if(use_KFT_module){
        // cluster();
        cout<<"\nAfter cluster:\n";
        cens.clear();
        // cens = cluster_cens;
        cens = dbscan.get_center();
      }
      if(cens.size()==0)
        return;
    }

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
        cout << "\033[1;42mKFT Execution time: " << (END_TIME - START_TIME) / CLOCKS_PER_SEC << "s\033[0m";
        cout << endl;
      }
    }
    return;

}

void callback_ego(const conti_radar::MeasurementConstPtr& in_msg,const conti_radar::MeasurementConstPtr& out_msg,const nav_msgs::OdometryConstPtr& odom){
  int radar_point_size = in_msg->points.size() + out_msg->points.size();
  int radar_id_count = 0;
  cout<<"\033[1;33m\n\n"<<++call_back_num<<" Call Back radar points:"<<radar_point_size<<"\033[0m"<<endl;
  if(radar_point_size==0) //radar points cannot be zero, it would cause segement fault
    return;
  radar_stamp = in_msg->header.stamp;

  if(show_vel_marker)
    vel_marker(out_msg);
  if(get_transformer)
    get_transform(radar_stamp);
  else
    echo_transform.setIdentity();
  // republish radar points
  pcl::PointCloud<pcl::PointXYZI>::Ptr in_filter_points(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr out_filter_points(new pcl::PointCloud<pcl::PointXYZI>);
  sensor_msgs::PointCloud2::Ptr in_filter_cloud(new sensor_msgs::PointCloud2);
  sensor_msgs::PointCloud2::Ptr out_filter_cloud(new sensor_msgs::PointCloud2);
  in_filter_points->clear();
  out_filter_points->clear();
  // end republish
  
  std::cout << std::fixed; std::cout.precision(3);
  cout<<"Radar time:"<<radar_stamp.toSec()<<endl;
  cens.clear();
  cluster_cens.clear();
  
  // get the ego velocity
  vx = odom->twist.twist.linear.x;
  vy = odom->twist.twist.linear.y;
  cout<<"\033[1;33mVehicle velocity:\n\033[0m";
  cout<<"Time stamp:"<<odom->header.stamp.toSec()<<endl;
  cout<<"vx: "<<vx<<" ,vy: "<<vy<<endl<<endl;
  if(output_radar_info)
    cout<<"Inlier Radar information\n";
  std::vector<kf_tracker_point> in_cens,out_cens;
  for (int i = 0;i<in_msg->points.size();i++){
    tf::Point trans_pt;
    trans_pt.setValue(in_msg->points[i].longitude_dist, in_msg->points[i].lateral_dist, 0);
    trans_pt = echo_transform * trans_pt;
    float range = sqrt(pow(in_msg->points[i].longitude_dist,2) + 
                        pow(in_msg->points[i].lateral_dist,2));
    float velocity = sqrt(pow(in_msg->points[i].lateral_vel_comp,2) + 
                          pow(in_msg->points[i].longitude_vel_comp,2));
    double angle = atan2(in_msg->points[i].lateral_vel_comp,in_msg->points[i].longitude_vel_comp)*180/M_PI;

    // float velocity = sqrt(pow(in_msg->points[i].lateral_vel+vy,2) + 
    //                       pow(in_msg->points[i].longitude_vel+vx,2));
    // double angle = atan2(in_msg->points[i].lateral_vel+vy,in_msg->points[i].longitude_vel+vx)*180/M_PI;

    if(output_radar_info){
      cout<<"----------------------------------------------\n";
      cout<<i<<"\tposition:("<<in_msg->points[i].longitude_dist<<","<<in_msg->points[i].lateral_dist<<")\n";
      cout<<"\tposition to map:("<<trans_pt.x()<<","<<trans_pt.y()<<")\n";
      cout<<"\tvel:"<<velocity<<"("<<in_msg->points[i].longitude_vel_comp<<","<<in_msg->points[i].lateral_vel_comp<<")\n\tdegree:"<<angle;
      // cout<<"\tvel:"<<velocity<<"("<<in_msg->points[i].longitude_vel+vx<<","<<in_msg->points[i].lateral_vel+vy<<")\n\tdegree:"<<angle;
      cout<<"\n\tinvalid_state:"<<in_msg->points[i].invalid_state;
      cout<<"\n\tRCS value:"<<in_msg->points[i].rcs;
      cout<<endl;
    }
    if(range>100){               //set radar threshold (distance)
        radar_point_size--;
    }
    else{
        kf_tracker_point c;
        c.x = trans_pt.x();
        c.y = trans_pt.y();
        c.z = 0;
        // c.x_v = in_msg->points[i].longitude_vel_comp+vx;
        // c.y_v = in_msg->points[i].lateral_vel_comp+vy;
        // c.x_v = in_msg->points[i].longitude_vel+vx;
        // c.y_v = in_msg->points[i].lateral_vel+vy;
        c.x_v = in_msg->points[i].longitude_vel_comp;
        c.y_v = in_msg->points[i].lateral_vel_comp;
        c.z_v = 0;
        c.vel_ang = angle;
        c.vel = velocity;
        c.cluster_flag = -1;  // -1: not clustered
        c.vistited = false;
        c.id = radar_id_count++;
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
  if(output_radar_info)
    cout<<"\nOutlier Radar information\n";
  for (int i = 0;i<out_msg->points.size();i++){
    tf::Point trans_pt;
    trans_pt.setValue(out_msg->points[i].longitude_dist, out_msg->points[i].lateral_dist, 0);
    trans_pt = echo_transform * trans_pt;
    float range = sqrt(pow(out_msg->points[i].longitude_dist,2) + 
                        pow(out_msg->points[i].lateral_dist,2));
    float velocity = sqrt(pow(out_msg->points[i].lateral_vel_comp,2) + 
                          pow(out_msg->points[i].longitude_vel_comp,2));
    double angle = atan2(out_msg->points[i].lateral_vel_comp,out_msg->points[i].longitude_vel_comp)*180/M_PI;

    // float velocity = sqrt(pow(out_msg->points[i].lateral_vel+vy,2) + 
    //                       pow(out_msg->points[i].longitude_vel+vx,2));
    // double angle = atan2(out_msg->points[i].lateral_vel+vy,out_msg->points[i].longitude_vel+vx)*180/M_PI;
    
    if(output_radar_info){
      cout<<"----------------------------------------------\n";
      cout<<i<<"\tposition:("<<out_msg->points[i].longitude_dist<<","<<out_msg->points[i].lateral_dist<<")\n";
      cout<<"\tposition to map:("<<trans_pt.x()<<","<<trans_pt.y()<<")\n";
      cout<<"\tvel:"<<velocity<<"("<<out_msg->points[i].longitude_vel_comp<<","<<out_msg->points[i].lateral_vel_comp<<")\n\tdegree:"<<angle;
      // cout<<"\tvel:"<<velocity<<"("<<out_msg->points[i].longitude_vel+vx<<","<<out_msg->points[i].lateral_vel+vy<<")\n\tdegree:"<<angle;
      cout<<"\n\tinvalid_state:"<<out_msg->points[i].invalid_state;
      cout<<"\n\tRCS value:"<<out_msg->points[i].rcs;
      cout<<endl;
    }
    if(range>100){               //set radar threshold (distance)
      radar_point_size--;
    }
    else{
      kf_tracker_point c;
      c.x = trans_pt.x();
      c.y = trans_pt.y();
      c.z = 0;
      // c.x_v = out_msg->points[i].longitude_vel_comp+vx;
      // c.y_v = out_msg->points[i].lateral_vel_comp+vy;
      // c.x_v = out_msg->points[i].longitude_vel+vx;
      // c.y_v = out_msg->points[i].lateral_vel+vy;
      c.x_v = out_msg->points[i].longitude_vel_comp;
      c.y_v = out_msg->points[i].lateral_vel_comp;
      c.z_v = 0;
      c.vel_ang = angle;
      c.vel = velocity;
      c.cluster_flag = -1;  // -1: not clustered
      c.vistited = false;
      c.id = radar_id_count++;
      c.rcs = out_msg->points[i].rcs;
      
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
  cens = in_cens;
  cens.insert(cens.end(),out_cens.begin(),out_cens.end());
  pcl::toROSMsg(*in_filter_points,*in_filter_cloud);
  if(get_transformer)
    in_filter_cloud->header.frame_id = "/map";
  else
    in_filter_cloud->header.frame_id = "/nuscenes_radar_front";
  in_filter_cloud->header.stamp = radar_stamp;
  pcl::toROSMsg(*out_filter_points,*out_filter_cloud);
  if(get_transformer)
    out_filter_cloud->header.frame_id = "/map";
  else
    out_filter_cloud->header.frame_id = "/nuscenes_radar_front";
  out_filter_cloud->header.stamp = radar_stamp;
  pub_in_filter.publish(in_filter_cloud);
  pub_out_filter.publish(out_filter_cloud);
  // cluster the radar points
  if (filter_cluster_flag){
    // cluster the points with DBSCAN-based method
    double DBSCAN_START, DBSCAN_END;
    DBSCAN_START = clock();
    dbscan dbscan(cens);
    std::vector< std::vector<kf_tracker_point> > dbscan_cluster, stage_one_cluster;
    dbscan.output_info = output_dbscan_info;
    dbscan_cluster = dbscan.cluster_from_RANSAC(in_cens,out_cens);
    // dbscan_cluster = dbscan.cluster_from_RANSAC_out2in(in_cens,out_cens);
    DBSCAN_END = clock();
    
    // score the cluster performance
    if(output_exe_time){
      cout << endl << "\033[1;42mDBSCAN Execution time: " << (DBSCAN_END - DBSCAN_START) / CLOCKS_PER_SEC << "s\033[0m";
      cout << endl;
    }
    color_cluster(dbscan_cluster);
    if(use_score_cluster && get_label && (label_vec.at(0).marker.header.stamp.toSec()-radar_stamp.toSec() <= 0.08))
      score_cluster(dbscan_cluster);
    cens.clear();
    cens = dbscan.get_center();
    if(cens.size()==0)
      return;
  }

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
      cout << "\033[1;42mKFT Execution time: " << (END_TIME - START_TIME) / CLOCKS_PER_SEC << "s\033[0m";
      cout << endl;
    }
  }
  return;
}

void annotation(const visualization_msgs::MarkerArrayConstPtr& label){
  ros::Time label_stamp = label->markers.at(0).header.stamp;
  std::cout << std::fixed; std::cout.precision(3);
  if(output_label_info){
    cout << endl << "In annotation callback:" << endl;
    cout << "Time stamp:" <<label_stamp.toSec() << endl << endl;
  }
  tf::StampedTransform label_transform;
  string TARGET_FRAME = "map";
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
      cout<<"Frame id:"<<label_transform.frame_id_<<", Child id:"<<label_transform.child_frame_id_<<endl;
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

    label_pt.front_left = tf::Point(label->markers.at(i).pose.position.x - label->markers.at(i).scale.x/2 - box_bias,
                                    label->markers.at(i).pose.position.y + label->markers.at(i).scale.y/2 + box_bias,
                                    label->markers.at(i).pose.position.z - label->markers.at(i).scale.z/2);
    label_pt.front_right = tf::Point(label->markers.at(i).pose.position.x + label->markers.at(i).scale.x/2 + box_bias,
                                    label->markers.at(i).pose.position.y + label->markers.at(i).scale.y/2 + box_bias,
                                    label->markers.at(i).pose.position.z - label->markers.at(i).scale.z/2);
    label_pt.back_left = tf::Point(label->markers.at(i).pose.position.x - label->markers.at(i).scale.x/2 - box_bias,
                                    label->markers.at(i).pose.position.y - label->markers.at(i).scale.y/2 -box_bias,
                                    label->markers.at(i).pose.position.z - label->markers.at(i).scale.z/2);
    label_pt.back_right = tf::Point(label->markers.at(i).pose.position.x + label->markers.at(i).scale.x/2 + box_bias,
                                    label->markers.at(i).pose.position.y - label->markers.at(i).scale.y/2 -box_bias,
                                    label->markers.at(i).pose.position.z - label->markers.at(i).scale.z/2);
    label_pt.front_left = label_transform * label_pt.front_left;
    label_pt.front_left.setZ(0);
    label_pt.front_right = label_transform * label_pt.front_right;
    label_pt.front_right.setZ(0);
    label_pt.back_left = label_transform * label_pt.back_left;
    label_pt.back_left.setZ(0);
    label_pt.back_right = label_transform * label_pt.back_right;
    label_pt.back_right.setZ(0);

    // if(output_label_info){
    //   cout << "--------------------------------------\n";
    //   cout << "The " << i << "-th marker pose:\n";
    //   cout << label_pt.type.c_str() << endl;
    //   cout << "- Orientation: ["  << label_pt.pose.getRotation().w() << ", "
    //                               << label_pt.pose.getRotation().x() << ", "
    //                               << label_pt.pose.getRotation().y() << ", "
    //                               << label_pt.pose.getRotation().z() << "]\n";
    //   cout << "- Position: [" << label_pt.pose.getOrigin().x() << ","
    //                           << label_pt.pose.getOrigin().y() << ","
    //                           << label_pt.pose.getOrigin().z() << "]\n";
    //   cout << "- Scale: [" << label_pt.marker.scale.x << ", "
    //                        << label_pt.marker.scale.y << ", "
    //                        << label_pt.marker.scale.z << "]\n";
    //   cout << "- Vertex: \n";
    //   cout << "\t"
    //        << label_pt.front_left.getX() << ", "
    //        << label_pt.front_left.getY() << ", "
    //        << label_pt.front_left.getZ() << endl;
    //   cout << "\t"
    //        << label_pt.front_right.getX() << ", "
    //        << label_pt.front_right.getY() << ", "
    //        << label_pt.front_right.getZ() << endl;
    //   cout << "\t"
    //        << label_pt.back_left.getX() << ", "
    //        << label_pt.back_left.getY() << ", "
    //        << label_pt.back_left.getZ() << endl;
    //   cout << "\t"
    //        << label_pt.back_right.getX() << ", "
    //        << label_pt.back_right.getY() << ", "
    //        << label_pt.back_right.getZ() << endl;
    // }
    label_pt.marker.header.frame_id = TARGET_FRAME;
    label_pt.marker.lifetime = ros::Duration(0.48);
    // label_pt.marker.lifetime = ros::Duration();
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
  if(log_msg->msg.find(".bag") != log_msg->msg.npos){
    filename_change = true;
    score_file_name = log_msg->msg;
    int bag_start = score_file_name.find("log");
    int bag_end = score_file_name.find(".");
    score_file_name = score_file_name.substr(bag_start,bag_end-bag_start);
  }
  return;
}

int main(int argc, char** argv){
  ros::init(argc,argv,"radar_kf_track");
  ros::NodeHandle nh;

  tf_listener = new tf::TransformListener();
  message_filters::Subscriber<conti_radar::Measurement> sub_out(nh,"/radar_front_outlier",1);   // sub the radar points(outlier)
  message_filters::Subscriber<conti_radar::Measurement> sub_in(nh,"/radar_front_inlier",1);     // sub the radar points(inlier)
  message_filters::Subscriber<conti_radar::Measurement> sub(nh,"/radar_front",1);             // sub the radar points(total)
  message_filters::Subscriber<nav_msgs::Odometry> sub_vel(nh,"/vel",1);                       // sub the ego velocity
  
  pub_cluster_marker = nh.advertise<visualization_msgs::MarkerArray>("cluster_index", 1);
  pub_marker = nh.advertise<visualization_msgs::MarkerArray>("tracking_id", 1);
  pub_filter = nh.advertise<sensor_msgs::PointCloud2>("filter_radar_front",500);
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

  nh.param<bool>("output_KFT_result"    ,output_KFT_result    ,true);
  nh.param<bool>("output_obj_id_result" ,output_obj_id_result ,true);
  nh.param<bool>("output_radar_info"    ,output_radar_info    ,true);
  nh.param<bool>("output_cluster_info"  ,output_cluster_info  ,true);
  nh.param<bool>("output_dbscan_info"   ,output_dbscan_info   ,true);
  nh.param<bool>("output_DA_pair"       ,output_DA_pair       ,true);
  nh.param<bool>("output_exe_time"      ,output_exe_time      ,false);
  nh.param<bool>("output_label_info"    ,output_label_info    ,false);
  nh.param<bool>("DA_method"            ,DA_choose            ,false);
  nh.param<bool>("use_KFT_module"       ,use_KFT_module       ,true);
  nh.param<int> ("kft_id_num"           ,kft_id_num           ,1);
  nh.param<bool>("show_vel_marker"      ,show_vel_marker      ,false);
  nh.param<bool>("use_ego_callback"     ,use_ego_callback     ,false);
  nh.param<bool>("use_score_cluster"    ,use_score_cluster    ,false);
  nh.param<bool>("get_transformer"      ,get_transformer      ,false);
  
  ros::Subscriber marker_sub;
  marker_sub = nh.subscribe("lidar_label", 10, &annotation); // sub the nuscenes annotation
  ros::Subscriber bag_sub;
  bag_sub = nh.subscribe("rosout",10,&get_bag_info_callback); // sub the bag name
  
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
  
  

  ROS_INFO("Parameter:");
  ROS_INFO("Data association method : %s" , DA_choose ? "hungarian" : "my DA method(Gerrdy-like)");
  ROS_INFO("output radar info : %s"       , output_radar_info ? "True" : "False");
  ROS_INFO("output cluster info : %s"     , output_cluster_info ? "True" : "False");
  ROS_INFO("output dbscan info : %s"      , output_dbscan_info ? "True" : "False");
  ROS_INFO("output KFT result : %s"       , output_KFT_result ? "True" : "False");
  ROS_INFO("output obj id result : %s"    , output_obj_id_result ? "True" : "False");
  ROS_INFO("output DA pair : %s"          , output_DA_pair ? "True" : "False");
  ROS_INFO("output execution time : %s"   , output_exe_time ? "True" : "False");
  ROS_INFO("output label info : %s"       , output_label_info ? "True" : "False");
  ROS_INFO("get transform : %s"           , get_transformer ? "true" : "false");
  ROS_INFO("publish velocity marker : %s" , show_vel_marker ? "True" : "False");
  ROS_INFO("use KFT module (Tracking part) : %s"          , use_KFT_module ? "True" : "False");
  if(use_KFT_module){
    ROS_INFO("Show %d types marker id"          , kft_id_num);
  }
  if(use_score_cluster){
    ROS_INFO("use score-cluster callback function");
  }
  else{
    ROS_INFO("use %s callback function"          , use_ego_callback ? "ego-motion in/outliner" : "original");
  }
  // ros::Rate r(10);
  while(ros::ok()){
    ros::spinOnce();
    pub_marker.publish(m_s);
    // pub_annotation.publish(repub);
    // r.sleep();
  }

  return 0;
}

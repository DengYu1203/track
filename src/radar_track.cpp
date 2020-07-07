#include <iostream>
#include <stdlib.h>
#include <string> 
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <limits>
#include <utility>

#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include "pcl_ros/point_cloud.h"
#include <pcl/impl/point_types.hpp>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/geometry.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/crop_box.h>

#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Twist.h>

#include <image_transport/image_transport.h>
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


#include <conti_radar/Measurement.h>
#include <nav_msgs/Odometry.h>

// to sync the subscriber
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <dbscan/dbscan.h>

using namespace std;
using namespace cv;

#define iteration 20 //plan segmentation #

typedef message_filters::sync_policies::ApproximateTime<conti_radar::Measurement,
														                            nav_msgs::Odometry> NoCloudSyncPolicy;

typedef pcl::PointXYZ PointT;

ros::Publisher pub_marker;  // pub tracking id
ros::Publisher pub_filter;  // pub filter points
ros::Publisher pub_cluster; // pub cluster points
ros::Publisher pub_cluster_pointcloud;  // pub cluster point cloud


tf::Transform scan_transform ;
tf::Transform transform_lidar,transform_radar_front;
tf::StampedTransform echo_transform;
bool lidar_tf_check = false;
bool radar_front_tf_check = false;
bool DA_choose = false;  // true: hungarian, false: my DA
bool output_KFT_result = true;
bool output_obj_id_result = true;
bool output_radar_info = true;
bool output_DA_pair = true;

// typedef struct kf_tracker_point{
//   float x;
//   float y;
//   float z;
//   float x_v;
//   float y_v;
//   float z_v;
//   double vel_ang;
//   double vel;
//   int cluster_flag;
//   // ros::Time stamp;
// }kf_tracker_point;

ros::Time radar_stamp,label_stamp;
float vx = 0; // vehicle vel x
float vy = 0; // vehicle vel y
typedef struct label_point
{
  tf::Vector3 position;
  tf::Quaternion orientation;
  tf::Vector3 scale;
  tf::Transform transform;

}label_point;


vector<kf_tracker_point> cens;
int cens_index = 0;
vector<kf_tracker_point> cluster_cens;
visualization_msgs::MarkerArray m_s,l_s;
int max_size = 0;

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

// type filter
typedef struct track{
  cv::KalmanFilter kf;
  geometry_msgs::Point pred_pose;
  geometry_msgs::Point pred_v;
  
  // update the new state
  cv::KalmanFilter kf_us;
  geometry_msgs::Point pred_pose_us;
  geometry_msgs::Point pred_v_us;
  
  int lose_frame;
  int track_frame;
  string state;
  int match_clus = 1000;
  int cluster_idx;
  int uuid ;
}track;

std::vector<track> filters;

void marker_id(bool firstFrame,std::vector<int> obj_id,std::vector<int> marker_obj_id,std::vector<int> tracking_frame_obj_id){
  visualization_msgs::Marker marker;
    int k;
    for(k=0; k<cens.size(); k++){
        marker.header.frame_id="/nuscenes_radar_front";
        marker.header.stamp = ros::Time();//to show every tag  (ros::Time::now()for latest tag)
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.id = k;
        marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;

        marker.scale.z = 3.0f;  // rgb(255,127,122)
        marker.color.r = 255.0f/255.0f;
        marker.color.g = 128.0f/255.0f;
        marker.color.b = 171.0f/255.0f;
        marker.color.a = 1;

        geometry_msgs::Pose pose;
        pose.position.x = cens[k].x;
        pose.position.y = cens[k].y;
        pose.position.z = cens[k].z+1.0f+k/20;
        
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
            marker.color.a = 0.8;
          }
          if(tracking_frame_obj_id.at(k)!=-1){  // track id appeaars more than 5
            marker.color.r = 0.0f/255.0f;
            marker.color.g = 229.0f/255.0f;
            marker.color.b = 255.0f/255.0f;
          }

        }
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

void color_cluster(std::vector< std::vector<kf_tracker_point> > cluster_list){
  // republish radar points
  pcl::PointCloud<pcl::PointXYZI>::Ptr cluster_points(new pcl::PointCloud<pcl::PointXYZI>);
  sensor_msgs::PointCloud2::Ptr cluster_cloud(new sensor_msgs::PointCloud2);
  cluster_points->clear();
  // end republish
  for(int i=0;i<cluster_list.size();i++){
    for(int j=0;j<cluster_list.at(i).size();j++){
      pcl::PointXYZI pt;
      pt.x = cluster_list.at(i).at(j).x;
      pt.y = cluster_list.at(i).at(j).y;
      pt.z = cluster_list.at(i).at(j).z;
      pt.intensity = i*200;
      cluster_points->push_back(pt);
    }
  }
  pcl::toROSMsg(*cluster_points,*cluster_cloud);
  cluster_cloud->header.frame_id = "/nuscenes_radar_front";
  cluster_cloud->header.stamp = radar_stamp;
  pub_cluster_pointcloud.publish(cluster_cloud);
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
  
  double distance = cv::Mahalanobis(x,y,cov);
  return distance;
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
    tk.state = "tracking";
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
   
    int i=0;
    cout << "Now center radar points :"<<cens.size()<<endl;  // The total radar outlier measurements now!
    for(i; i<cens.size(); i++){
      geometry_msgs::Point pt;
      pt.x=cens[i].x;
      pt.y=cens[i].y;
      pt.z=cens[i].z;
      cout <<i+1<<": \t("<<pt.x<<","<<pt.y<<","<<pt.z<<")";
      cout<<"\tvel: ("<<cens[i].x_v<<","<<cens[i].y_v<<")\n";
      clusterCenters.push_back(pt);
    }

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
          filters.at(k).state = "tracking";  
          // update the KF state
          filters.at(k).kf_us.statePost.at<float>(0)=cens.at(assignment[k]).x;
          filters.at(k).kf_us.statePost.at<float>(1)=cens.at(assignment[k]).y;
          filters.at(k).kf_us.statePost.at<float>(3)=cens.at(assignment[k]).x_v;
          filters.at(k).kf_us.statePost.at<float>(4)=cens.at(assignment[k]).y_v;
          

          distMat[k]=std::vector<double>(cens.size(),10000.0); //float
          for(int row=0;row<distMat.size();row++)//set the column to a high number
          {
              distMat[row][assignment[k]]=10000.0;
          }  
        }
        else
        {
          filters.at(k).state= "lost";
        } 

      }
      else
      {
        filters.at(k).state= "lost";
      }
      
  
      if(output_KFT_result){
        //get tracked or lost
        if (filters.at(k).state== "tracking"){
            cout<<"The state of "<<k<<" filters is \033[1;32m"<<filters.at(k).state<<"\033[0m,to cluster_idx "<< filters.at(k).cluster_idx <<" track: "<<filters.at(k).track_frame;
            cout<<" ("<<cens.at(assignment[k]).x<<","<<cens.at(assignment[k]).y<<")"<<endl;
        }
        else if (filters.at(k).state== "lost")
            cout<<"The state of "<<k<<" filters is \033[1;34m"<<filters.at(k).state<<"\033[0m,to cluster_idx "<< filters.at(k).cluster_idx <<" lost: "<<filters.at(k).lose_frame<<endl;
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
        if( !(*pit).state.compare("lost")  )//true for 0
            (*pit).lose_frame += 1;

        if( !(*pit).state.compare("tracking")  ){//true for 0
            (*pit).track_frame += 1;
            (*pit).lose_frame = 0;
        }
        
        if ( (*pit).lose_frame == frame_lost )
            //remove track from filters
            pit = filters.erase(pit);
        else
            pit ++;
            
    }
    cout<<"We now have "<<filters.size()<<" tracks."<<endl;


    //begin mark
    m_s.markers.resize(cens.size());
    m_s.markers.clear();

    // marker_id(false,marker_obj_id);   // publish the "tracking" id
    marker_id(false,obj_id,marker_obj_id,tracking_frame_obj_id);       // publish all id
    
///////////////////////////////////////////////////estimate
 
    int num = filters.size();
    float meas[num][3];
    float meas_us[num][3];
    i = 0;
    for(std::vector<track>::const_iterator it = filters.begin(); it != filters.end(); ++it){
        if ( (*it).state == "tracking" ){
            kf_tracker_point pt = cens[(*it).cluster_idx];
            meas[i][0] = pt.x;
            meas[i][1] = pt.y;
            meas[i][2] = pt.z;
            meas_us[i][0] = pt.x;
            meas_us[i][1] = pt.y;
            meas_us[i][2] = pt.z;
        }
        else if ( (*it).state == "lost" ){
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



int call_back_num = 0;

#define cluster_distance 2.5 // euclidean distance 0.8,1.5,2.5,3 for car length
#define cluster_vel_angle 2  // velocity angle threshold 2
#define cluster_vel 1        // velocity threshold 1
bool filter_cluster_flag=true;       // decide cluster the radar points or not
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
    cout<<"==============================\n";
    cout<<"cluster "<<i<<":\n";
    for(int j=0;j<cens.size();j++){
      if(cens.at(j).cluster_flag==i){
        index++;
        c.x += cens.at(j).x;
        c.y += cens.at(j).y;
        c.z += cens.at(j).z;
        c.x_v += cens.at(j).x_v;
        c.y_v += cens.at(j).y_v;
        c.z_v += cens.at(j).z_v;
        cout<<"position sum:"<<"("<<c.x<<","<<c.y<<")"<<endl;
        cout<<"vel sum:"<<"("<<c.x_v<<","<<c.y_v<<")"<<endl;
        cout<<"index="<<index<<endl;
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
  cout<<endl;
  pcl::toROSMsg(*cluster_points,*cluster_cloud);
  cluster_cloud->header.frame_id = "/nuscenes_radar_front";
  cluster_cloud->header.stamp = radar_stamp;
  pub_cluster.publish(cluster_cloud);

}
void callback(const conti_radar::MeasurementConstPtr& msg,const nav_msgs::OdometryConstPtr& odom){
    
    // if(radar_front_tf_check){
    int radar_point_size = msg->points.size();
    cout<<"\033[1;33m\n\n"<<++call_back_num<<" Call Back radar points:"<<radar_point_size<<"\033[0m"<<endl;
    if(radar_point_size==0) //radar points cannot be zero, it would cause segement fault
      return;
    radar_stamp = msg->header.stamp;

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

    for (int i = 0;i<msg->points.size();i++){

        float velocity = sqrt(pow(msg->points[i].lateral_vel_comp,2) + 
                              pow(msg->points[i].longitude_vel_comp,2));
        float range = sqrt(pow(msg->points[i].longitude_dist,2) + 
                           pow(msg->points[i].lateral_dist,2));
        double angle = atan2(msg->points[i].lateral_vel_comp+vy,msg->points[i].longitude_vel_comp+vx)*180/M_PI;
        if(output_radar_info){
          cout<<"----------------------------------------------\n";
          cout<<i<<"\tvel:"<<velocity<<"("<<msg->points[i].longitude_vel_comp<<","<<msg->points[i].lateral_vel_comp<<")\n\tdegree:"<<angle;
          cout<<"\n\tinvalid_state:"<<msg->points[i].invalid_state;
        }
        if((range>100||(msg->points[i].invalid_state==7))){               //set radar threshold (distance and invalid state)
        // if(range>100){               //set radar threshold (distance)
            radar_point_size--;
            cout<<endl;
        }
        else{
            kf_tracker_point c;
            c.x = msg->points[i].longitude_dist;
            c.y = msg->points[i].lateral_dist;
            c.z = 0;
            // c.x_v = msg->points[i].longitude_vel_comp+vx;
            // c.y_v = msg->points[i].lateral_vel_comp+vy;
            c.x_v = msg->points[i].longitude_vel_comp;
            c.y_v = msg->points[i].lateral_vel_comp;
            c.z_v = 0;
            c.vel_ang = angle;
            c.vel = velocity;
            c.cluster_flag = -1;  // -1: not clustered
            c.vistited = false;
            
            pcl::PointXYZI pt;
            pt.x = c.x;
            pt.y = c.y;
            pt.z = c.z;
            pt.intensity = msg->points[i].rcs;
            filter_points->push_back(pt);
            if(output_radar_info)
              cout<<"\n\tposition:("<<c.x<<","<<c.y<<","<<c.z<<")\n";
            cens.push_back(c);
        }
        
       

    }
    pcl::toROSMsg(*filter_points,*filter_cloud);
    filter_cloud->header.frame_id = "/nuscenes_radar_front";
    filter_cloud->header.stamp = radar_stamp;
    pub_filter.publish(filter_cloud);
    // cluster the radar points
    if (filter_cluster_flag){
      cluster();
      dbscan dbscan(cens);
      std::vector< std::vector<kf_tracker_point> > dbscan_cluster;
      dbscan_cluster = dbscan.cluster();
      color_cluster(dbscan_cluster);
      cout<<"\nAfter cluster:\n";
      cens.clear();
      cens = cluster_cens;
      if(cens.size()==0)
        return;
    }


    if( firstFrame ){
      int current_id = cens.size();
      float sigmaP=0.01;//0.01
      float sigmaQ=0.1;//0.1

      //initialize new tracks(function)
      //state = [x,y,z,vx,vy,vz]
      for(int i=0; i<current_id;i++){
          //try ka.init 
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
          tk.state = "tracking";
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
      return; // first initialization down 
    }
    KFT();
    return;
    // }

}

// void tf_listen(const tf::tfMessage& msg){
  // if(!(lidar_tf_check && radar_front_tf_check)){
  //   for(int i=0;i<msg.transforms.size();i++){
  //   // cout<<msg.transforms.at(i).child_frame_id<<endl;
  //   if(((msg.transforms.at(i).child_frame_id)=="nuscenes_lidar") && ((msg.transforms.at(i).header.frame_id)=="car")){
  //     tf::transformMsgToTF(msg.transforms.at(i).transform,transform_lidar);
  //     lidar_tf_check = true;
  //     cout<<"Lidar transform:"<<transform_lidar.getOrigin().getX() <<endl;
  //     }
  //   else if(((msg.transforms.at(i).child_frame_id)=="nuscenes_radar_front") && ((msg.transforms.at(i).header.frame_id)=="car")){
  //     tf::transformMsgToTF(msg.transforms.at(i).transform,transform_radar_front);
  //     radar_front_tf_check = true;
  //     }
    
  //   }
  // }
  // tf::StampedTransform echo_transform;
  // tf::Transformer tf_listener;
  // tf_listener.lookupTransform("car","radar_front",ros::Time(),echo_transform);
  // return;

// }
void get_transform(){
  ros::Rate rate(1);
  tf::TransformListener tf_listener;
  tf_listener.waitForTransform("nuscenes_radar_front","car",ros::Time(),ros::Duration(1.0));
  std::cout << std::fixed; std::cout.precision(3);
  try{
    
    tf_listener.lookupTransform("car","nuscenes_radar_front",ros::Time(),echo_transform);
    std::cout << "At time " << echo_transform.stamp_.toSec() << std::endl;
    double yaw, pitch, roll;
    echo_transform.getBasis().getRPY(roll, pitch, yaw);
    tf::Quaternion q = echo_transform.getRotation();
    tf::Vector3 v = echo_transform.getOrigin();
    std::cout << "- Translation: [" << v.getX() << ", " << v.getY() << ", " << v.getZ() << "]" << std::endl;
    std::cout << "- Rotation: in Quaternion [" << q.getX() << ", " << q.getY() << ", " 
              << q.getZ() << ", " << q.getW() << "]" << std::endl;
    cout<<"Frame id:"<<echo_transform.frame_id_<<", Child id:"<<echo_transform.child_frame_id_<<endl;
    get_transformer = true;
  }
  catch(tf::TransformException& ex)
  {
    // std::cout << "Failure at "<< ros::Time::now() << std::endl;
    // std::cout << "Exception thrown:" << ex.what()<< std::endl;
    // std::cout << "The current list of frames is:" <<std::endl;
    // std::cout << tf_listener.allFramesAsString()<<std::endl;
    
  }
  rate.sleep();
  return;
}


int main(int argc, char** argv){
  ros::init(argc,argv,"radar_kf_track");
  ros::NodeHandle nh;
  // ros::Subscriber tf_sub;
  // tf_sub = nh.subscribe("tf",1000,&tf_listen);
  // sub_vel = nh.subscribe("vel",10,&vel_callback);
  // sub = nh.subscribe("radar_front_outlier",1000,&callback); //subscribe the outlier radar point
  // sub = nh.subscribe("radar_front_inlier",1000,&callback); //subscribe the inlier radar point(bad to track)

  // message_filters::Subscriber<conti_radar::Measurement> sub(nh,"/radar_front_outlier",1);  // sub the radar points(outlier)
  message_filters::Subscriber<conti_radar::Measurement> sub(nh,"/radar_front",1);             // sub the radar points(total)
  message_filters::Subscriber<nav_msgs::Odometry> sub_vel(nh,"/vel",1);                       // sub the ego velocity
  message_filters::Synchronizer<NoCloudSyncPolicy>* no_cloud_sync_;
  no_cloud_sync_ = new message_filters::Synchronizer<NoCloudSyncPolicy>(NoCloudSyncPolicy(3), sub, sub_vel);
  no_cloud_sync_->registerCallback(boost::bind(&callback, _1, _2));
  
  pub_marker = nh.advertise<visualization_msgs::MarkerArray>("marker", 1);
  pub_filter = nh.advertise<sensor_msgs::PointCloud2>("filter_radar_front",500);
  pub_cluster = nh.advertise<sensor_msgs::PointCloud2>("cluster_radar_front",500);
  pub_cluster_pointcloud = nh.advertise<sensor_msgs::PointCloud2>("cluster_radar_front_pointcloud",500);

  nh.param<bool>("output_KFT_result"    ,output_KFT_result    ,true);
  nh.param<bool>("output_obj_id_result" ,output_obj_id_result ,true);
  nh.param<bool>("output_radar_info"    ,output_radar_info    ,true);
  nh.param<bool>("output_DA_pair"       ,output_DA_pair       ,true);
  nh.param<bool>("DA_method"            ,DA_choose            ,false);

  

  ROS_INFO("Parameter:");
  ROS_INFO("Data association method : %s" , DA_choose ? "hungarian" : "my DA method(Gerrdy-like)");
  ROS_INFO("output radar info : %s"       , output_radar_info ? "True" : "False");
  ROS_INFO("Output KFT result : %s"       , output_KFT_result ? "True" : "False");
  ROS_INFO("output obj id result : %s"    , output_obj_id_result ? "True" : "False");
  ROS_INFO("output DA pair : %s"          , output_DA_pair ? "True" : "False");
  

  // ros::Rate r(10);
  while(ros::ok()){
    ros::spinOnce();
    pub_marker.publish(m_s);
    if(!get_transformer){
      get_transform();
    }
    // r.sleep();
  }

  return 0;
}

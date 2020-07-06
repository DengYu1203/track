#include <iostream>
#include <stdlib.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <pcl/conversions.h>
#include <pcl/filters/passthrough.h>
#include <tf/transform_broadcaster.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include "pcl_ros/point_cloud.h"
#include <pcl/impl/point_types.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/icp.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/filters/voxel_grid.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Point.h>
#include <iostream>
#include <tf/LinearMath/Matrix3x3.h>
#include <tf/LinearMath/Vector3.h>
//#include <localization/gps_transformer.h>	
#include <string> 
#include <fstream>
#include <sstream>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>


#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>


// #include "darknet_ros_msgs/BoundingBoxes.h"
// #include "darknet_ros_msgs/BoundingBox.h"


#include <opencv-3.3.1-dev/opencv2/core.hpp>
#include <opencv-3.3.1-dev/opencv2/highgui.hpp>
#include <opencv-3.3.1-dev/opencv2/imgproc.hpp>
#include <opencv-3.3.1-dev/opencv2/calib3d.hpp>
#include <opencv-3.3.1-dev/opencv2/opencv.hpp>


#include <algorithm>
#include <iterator>
#include "kf_tracker/featureDetection.h"
#include "kf_tracker/CKalmanFilter.h"
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Int32MultiArray.h>

#include <sensor_msgs/PointCloud2.h>
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
 
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <limits>
#include <utility>

#include <tf/LinearMath/Matrix3x3.h>
#include <tf/LinearMath/Vector3.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>


#include <conti_radar/Measurement.h>

using namespace std;
using namespace cv;
#define iteration 20 //plan segmentation #


typedef pcl::PointXYZ PointT;
ros::Subscriber sub;
ros::Publisher pub_marker;

tf::Transform scan_transform ;
tf::Transform transform_lidar,transform_radar_front;
bool lidar_tf_check = false;
bool radar_front_tf_check = false;
sensor_msgs::PointCloud2  output;


// pcl::PointCloud<PointT>::Ptr cloud_pcl(new pcl::PointCloud<PointT>);
// pcl::PointCloud<PointT>::Ptr cloud_pcl_whole(new pcl::PointCloud<PointT>);
// pcl::PointCloud<PointT>::Ptr cloud_f(new pcl::PointCloud<PointT>);
// pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
typedef struct kf_tracker_point{
  float x;
  float y;
  float z;
  float x_v;
  float y_v;
  float z_v;
  // ros::Time stamp;
}kf_tracker_point;
ros::Time radar_stamp,label_stamp;
typedef struct label_point
{
  tf::Vector3 position;
  tf::Quaternion orientation;
  tf::Vector3 scale;
  tf::Transform transform;

}label_point;
vector<label_point> label_pt[45];
int label_index =0;
vector<kf_tracker_point> cens[45];
int cens_index = 0;
vector<kf_tracker_point> cluster_cens;
vector<geometry_msgs::Point> pre_cens;
visualization_msgs::MarkerArray m_s,l_s;
int max_size = 0;
// int radar_point_size =0;
float dt = 0.5f;
float sigmaP=0.01;//0.01
float sigmaQ=0.1;//0.1
#define bias 5.0 
int id_count = 0; //the counter to calculate the current occupied track_id

////////////////////////kalman/////////////////////////
#define cluster_num 40 
#define frame_lost 5

std::vector <string> tracked;
std::vector <int> count,id;
std::vector<geometry_msgs::Point> pred_velocity;
int un_assigned;
std::vector <float> kf_pre_cens;


ros::Publisher objID_pub;
// KF init
int stateDim=6;// [x,y,v_x,v_y]//,w,h]
int measDim=3;// [z_x,z_y//,z_w,z_h]
int ctrlDim=0;// control input 0(acceleration=0,constant v model)
// std::vector<pcl::PointCloud<PointT>::Ptr> cluster_vec;

std::vector<cv::KalmanFilter> KF;//(stateDim,measDim,ctrlDim,CV_32F);

cv::Mat state(stateDim,1,CV_32F);
cv::Mat_<float> measurement(3,1);//x.y.z pose as measurement

                        // measurement.setTo(Scalar(0));
bool firstFrame=true;

// //////type filter
typedef struct track{
  cv::KalmanFilter kf;
  geometry_msgs::Point pred_pose;
  geometry_msgs::Point pred_v;
  int lose_frame;
  string state;
  int match_clus = 1000;
  int cluster_idx;
  int uuid ;
}track;

std::vector<track> filters;

void marker_id(bool firstFrame,std::vector<int> obj_id){
  visualization_msgs::Marker marker;
    int k;
    for(k=0; k<cluster_cens.size(); k++){
        marker.header.frame_id="/nuscenes_radar_front";
        marker.header.stamp = ros::Time();//to show every tag  (ros::Time::now()for latest tag)
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.id = k;
        marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;

        marker.scale.z = 3.0f;
        marker.color.r = 255.0f/255.0f;
        marker.color.g = 127.0f/255.0f;
        marker.color.b = 122.0f/255.0f;
        marker.color.a = 1;

        geometry_msgs::Pose pose;
        pose.position.x = cluster_cens[k].x;
        pose.position.y = cluster_cens[k].y;
        pose.position.z = cluster_cens[k].z+2.0f;
        
        //-----------first frame要先發佈tag 為initial 
        stringstream ss;
        if(firstFrame){
          ss << k;
        }
        else{
          ss << obj_id.at(k);
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

// calculate euclidean distance of two points
double euclidean_distance(geometry_msgs::Point p1, geometry_msgs::Point p2)
{
  return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}


// first_index -> track_idx, second_index->cluster_idx
// Find the min of distmat, and return indexpair(i,j)
// std::pair<int,int> findIndexOfMin(std::vector<std::vector<float> > distMat)
// {
//     cout<<"findIndexOfMin CALLED\n";
//     std::pair<int,int>minIndex;
//     float minEl=std::numeric_limits<float>::max();
//     cout<<"minEl="<<minEl<<"\n";

//     //第i row為KF[i]與第j群的距離 ,但這樣挑會有順序性：0->1->.....49 KF[0]有選擇優先權,可以設threshold  但要如何設？（若單純用距離又與bag播放速度有關 速度資訊？）
//     for (int i=0; i<distMat.size();i++)
//         for(int j=0;j<distMat.at(0).size();j++)
//         {
//             if( distMat[i][j]<minEl)
//             {
//                 minEl=distMat[i][j];
//                 minIndex=std::make_pair(i,j);

//             }

//         }
//     // cout<<"minIndex="<<minIndex.first<<","<<minIndex.second<<"\n";
//     return minIndex;
// }


int find_matching(std::vector<float> dist_vec){
  float now_min = std::numeric_limits<float>::max();
  int cluster_idx;
  for (int i=0; i<dist_vec.size(); i++){
    if(dist_vec.at(i)<now_min){
      now_min = dist_vec.at(i);
      cluster_idx = i;
    }
  }
  // cout<<"minIndex="<<cluster_idx<<endl;
  return cluster_idx;
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
    // ka.statePost.at<float>(3)=0;// initial v_x
    // ka.statePost.at<float>(4)=0;// initial v_y
    // ka.statePost.at<float>(5)=0;// initial v_z
    ka.statePost.at<float>(3)=cen.x_v;// initial v_x
    ka.statePost.at<float>(4)=cen.y_v;// initial v_y
    ka.statePost.at<float>(5)=cen.z_v;// initial v_z
  
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


    tk.kf = ka;
    tk.state = "tracking";
    tk.lose_frame = 0;
    tk.cluster_idx = idx;

    // uuid_t uu;
    // uuid_generate(uu);
    // for (int k = 0 ; k<16; k++){
    //   uu[i]
    // }
    tk.uuid = ++id_count;
    filters.push_back(tk);
    // cout<<"Done init newT at "<<id_count<<" is ("<<tk.pred_pose.x <<"," <<tk.pred_pose.y<<")"<<endl;
    
    
    return tk.uuid;
}



// void initiateKF( void ){
//   float dvx=0.01f;
//   float dvy=0.01f;
//   float dx=1.0f;
//   float dy=1.0f;
//   float dt=0.1f;//time interval btw state transition(10hz ros spin)
//   float sigmaP=0.01;//0.01
//   float sigmaQ=0.1;//0.1
//   for(int i=0;i<cluster_num;i++){
//       // KF[i].transitionMatrix = (Mat_<float>(4, 4) << dx,0,1,0,   0,dy,0,1,  0,0,dvx,0,  0,0,0,dvy);
//       //KF[i].transitionMatrix = (Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
//       KF[i].transitionMatrix = (Mat_<float>(4, 4) << 1,0,dt,0,   0,1,0,dt,  0,0,1,0,  0,0,0,1);
//       cv::setIdentity(KF[i].measurementMatrix);
//       setIdentity(KF[i].processNoiseCov, Scalar::all(sigmaP));
//       cv::setIdentity(KF[i].measurementNoiseCov, cv::Scalar(sigmaQ));
//   }
//   return;
// }


// pcl::PointCloud<PointT>::Ptr crop(pcl::PointCloud<PointT>::Ptr cloud_clusters){
//   Eigen::Vector4f box_min,box_max;
//   pcl::PointCloud<PointT>::Ptr cluster_box(new pcl::PointCloud<PointT> );
//   box_min << -30,-30,-30,1;
//   box_max << 30,30,30,1;//choose 60x60x60 cube (mask)
//   pcl::CropBox<PointT> in_box;

//   in_box.setInputCloud(cloud_clusters);
//   in_box.setMin(box_min);
//   in_box.setMax(box_max);
//   in_box.filter(*cluster_box);

//   // sensor_msgs::PointCloud2 cluster_cloud;
//   // pcl::toROSMsg(*cluster_box, cluster_cloud);
//   // cluster_cloud.header.frame_id = "map";
//   // cluster_pub.publish(cluster_cloud);
//   return cluster_box;
// }



void KFT(void)
{
    // std::vector<cv::Mat> pred;
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
   
    int i=0;
    cout << "Now cen is:"<<endl;
    for(i; i<cluster_cens.size(); i++){
      geometry_msgs::Point pt;
      pt.x=cluster_cens[i].x;
      pt.y=cluster_cens[i].y;
      pt.z=cluster_cens[i].z;
      // cout <<"("<<pt.x<<","<<pt.y<<","<<pt.z<<")"<<endl;
      clusterCenters.push_back(pt);
    }

    std::vector<geometry_msgs::Point> KFpredictions;
    i=0;
    
    // cout<<"--------------------\nThe prediction is:"<<endl;
    for (auto it=filters.begin();it!=filters.end();it++)
    {
        geometry_msgs::Point pt;
        pt.x = (*it).pred_pose.x;
        pt.y = (*it).pred_pose.y;
        pt.z = (*it).pred_pose.z;
        // cout << "("<<pt.x<<","<<pt.y<<","<<pt.z<<")"<<endl;

    }


    //construct dist matrix (mxn): m tracks, n clusters.
    std::vector<geometry_msgs::Point> copyOfClusterCenters(clusterCenters);
    std::vector<std::vector<float> > distMat;

    for(std::vector<track>::const_iterator it = filters.begin (); it != filters.end (); ++it)
    {
        std::vector<float> distVec;
        for(int n=0;n<cluster_cens.size();n++)
        {
            distVec.push_back(euclidean_distance((*it).pred_pose,copyOfClusterCenters[n]));
        }

        distMat.push_back(distVec);

    }
  
    // DEBUG: print the distMat
    // for ( const auto &row : distMat )
    // {
    //     for ( const auto &s : row ) std::cout << s << ' ';
    //     // std::cout << std::endl;
    // }


    //establish link for current existing tracks
    std::vector<int> obj_id(cluster_cens.size(),-1); //record track_uuid for every cluster, -1 for not matched
  
    
    int k=0;
    int cluster_idx = -1;
    std::pair<int,int> minIndex;
    //every track to find the min value

    for(k=0; k<filters.size(); k++){
      std::vector<float> dist_vec = distMat.at(k);

      cluster_idx = find_matching(dist_vec);

      if( dist_vec[cluster_idx] <=  bias ){//但回傳最小 表從某個回傳開始都會loss track 
        obj_id[cluster_idx] = filters.at(k).uuid;
        filters.at(k).cluster_idx = cluster_idx;
        filters.at(k).state = "tracking";  

        distMat[k]=std::vector<float>(cluster_cens.size(),10000.0);
        for(int row=0;row<distMat.size();row++)//set the column to a high number
        {
            distMat[row][cluster_idx]=10000.0;
        }  
      }
      else
      {
        filters.at(k).state= "lost";
      }
      // cout<<"The state of \033[1;34m"<<k<<" \033[0mfilters is "<<filters.at(k).state<<" ,to cluster_idx "<< filters.at(k).cluster_idx <<endl;
    }  
    // cout<<"\033[1;33mThe obj_id:\033[0m\n";
    // for (i=0; i<cens.size(); i++){
    //   cout<<obj_id.at(i)<<" ";
    //   }
    // cout<<endl;

    //initiate new tracks for not-matched cluster
    for (i=0; i<cluster_cens.size(); i++){
      if(obj_id.at(i) == -1){
        int track_uuid = new_track(cluster_cens.at(i),i);
        obj_id.at(i) = track_uuid;
      }
    }

    // cout<<"We now have "<<filters.size()<<"tracks."<<endl;

    // cout<<"\033[1;33mThe obj_id after new track:<<\033[0m\n";
    // for (i=0; i<cens.size(); i++){
    //   cout<<obj_id.at(i)<<" ";
    //   }
    // cout<<endl;

  

    //deal with lost tracks and let it remain const velocity -> update pred_pose to meas correct
    for(std::vector<track>::iterator pit = filters.begin (); pit != filters.end ();){
        if( !(*pit).state.compare("lost")  )//true for 0
            (*pit).lose_frame += 1;

            if ( (*pit).lose_frame == frame_lost )
                //remove track from filters
                pit = filters.erase(pit);
            else
                pit ++;
            
    }



    //begin mark
    // cout << m_s.markers.size() <<endl;
    m_s.markers.resize(cluster_cens.size());
    m_s.markers.clear();
    // cout<< m_s.markers.size()<< endl;
    marker_id(false,obj_id);
    
///////////////////////////////////////////////////estimate
 
    int num = filters.size();
    float meas[num][3];
    i = 0;
    for(std::vector<track>::const_iterator it = filters.begin(); it != filters.end(); ++it){
        if ( (*it).state == "tracking" ){
            kf_tracker_point pt = cluster_cens[(*it).cluster_idx];
            meas[i][0] = pt.x;
            meas[i][1] = pt.y;
            meas[i][2] = pt.z;
        }
        else if ( (*it).state == "lost" ){
            meas[i][0] = (*it).pred_pose.x; //+ (*it).pred_v.x;
            meas[i][1] = (*it).pred_pose.y;//+ (*it).pred_v.y;
            meas[i][2] = (*it).pred_pose.z; //+ (*it).pred_v.z;
        }
        else
        {
            std::cout<<"Some tracks state not defined to tracking/lost."<<std::endl;
        }
        
        i++;
    }

    // std::cout<<"mesurement record."<<std::endl;
    cv::Mat measMat[num];
    for(int i=0;i<num;i++){
        measMat[i]=cv::Mat(3,1,CV_32F,meas[i]);
    }

    // The update phase 
    
    // if (!(measMat[0].at<float>(0,0)==0.0f || measMat[0].at<float>(1,0)==0.0f))
    //     Mat estimated0 = KF[0].correct(measMat[0]);
    cv::Mat estimated[num];
    i = 0;
    for(std::vector<track>::iterator it = filters.begin(); it != filters.end(); ++it){
        estimated[i] = (*it).kf.correct(measMat[i]); 
        // cout << "The corrected state of "<<i<<"th KF is: "<<estimated[i].at<float>(0)<<","<<estimated[i].at<float>(1)<<endl;
        i++;
    }

  return;
}





void callback(const conti_radar::MeasurementPtr& msg){
    
  // sensor_msgs::PointCloud2 out;
  // out = msg;
  // out.header.frame_id="/map";
  // pub_get.publish(out);
  // cout<<"I get the car."<<endl;

  // pcl::fromROSMsg(out,*cloud_pcl_whole);
//   pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
//   pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
//   // Create the segmentation object
//   pcl::SACSegmentation<PointT> seg;
//   // Optional
//   seg.setOptimizeCoefficients (true);
//   // Mandatory
//   seg.setModelType (pcl::SACMODEL_PLANE);
//   seg.setMethodType (pcl::SAC_RANSAC);
//   seg.setMaxIterations (100);
//   seg.setDistanceThreshold (0.02);//0.01 for bag1

//   //Chose ROI to process
//   // cloud_pcl = crop(cloud_pcl_whole);
//   cloud_pcl = cloud_pcl_whole;

//   cout<<"Ready to segmentation."<<endl;
//   pcl::ExtractIndices<PointT> extract;
  
//   int i = 0;
//   for(i=0;i<iteration;i++){
//   // Segment the largest planar component from the remaining cloud
//     seg.setInputCloud (cloud_pcl);
//     seg.segment (*inliers, *coefficients);
//     if (inliers->indices.size () == 0)
//     {
//       std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
//       break;
//     }

//     // Extract the inliers
//     extract.setInputCloud (cloud_pcl);
//     extract.setIndices (inliers);

//     extract.setNegative (true);
//     extract.filter (*cloud_f);
//     cloud_pcl.swap (cloud_f);
//     cout<<i<<endl;
//   }


//   pcl::toROSMsg(*cloud_pcl,output);
//   output.header.frame_id="/map";
//   pub.publish(output);
// ////////////////////no downsampling
//   pcl::VoxelGrid<PointT> sor;
//   sor.setInputCloud (cloud_pcl);
//   sor.setLeafSize (0.25f, 0.25f, 0.25f); //0.25 for bag1; 0.1 for 3
//   sor.filter (*cloud_filtered);

//   sensor_msgs::PointCloud2 cloud_filtered_sensor;
//   pcl::toROSMsg(*cloud_filtered,cloud_filtered_sensor);
//   cloud_filtered_sensor.header.frame_id = "/map";
//   pub_voxel.publish(cloud_filtered_sensor);



//   //cluster
//   pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI>);
//   tree->setInputCloud (cloud_filtered);

//   std::vector<pcl::PointIndices> cluster_indices;
//   pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;

//   ec.setClusterTolerance (0.45); //0.5->cars merged to one cluster
//   ec.setMinClusterSize (25); //30 for bag1
//   ec.setMaxClusterSize (400); //300 for bag1
//   ec.setSearchMethod (tree);
//   ec.setInputCloud (cloud_filtered);
//   ec.extract (cluster_indices);
//   std::cout << "Cluster size: " << cluster_indices.size() << std::endl;

//   int j = 50;

//   pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_clusters (new pcl::PointCloud<pcl::PointXYZI>);
  
//   cens.clear();
//   cluster_vec.clear();

  // for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it) {
    
  //   // extract clusters and save as a single point cloud
  //   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
  //   pcl::fromROSMsg(msg,*cloud_cluster);
  //   for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit) {
  //     cloud_filtered->points[*pit].intensity = j;
  //     cloud_cluster->points.push_back (cloud_filtered->points[*pit]); //*
  //   }
  //   cloud_cluster->width = cloud_cluster->points.size ();
  //   cloud_cluster->height = 1;
  //   cloud_cluster->is_dense = true;
    
  //   // compute cluster centroid and publish
  //   cloud_cluster = compute_c(cloud_cluster,j);
  //   cluster_vec.push_back(cloud_cluster);

  //   *cloud_clusters += *cloud_cluster;
  //   j+=2;

  // }
  // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
  // pcl::fromROSMsg(msg,*cloud_cluster);
  int radar_point_size = msg->points.size();
  radar_stamp = msg->header.stamp;
  cout<<"Radar time:"<<radar_stamp.nsec<<endl;
  cens[cens_index].clear();
  for (int i = 0;i<msg->points.size();i++){
    kf_tracker_point c;
    float velocity = sqrt(pow(msg->points[i].lateral_vel_comp,2) + 
				 									pow(msg->points[i].longitude_vel_comp,2));
    cout<<"radar v:"<<velocity<<endl;
    if(velocity<0){
      radar_point_size--;
    }
    else{
      c.x = msg->points[i].longitude_dist;
      c.y = msg->points[i].lateral_dist;
      c.z = 0;
      c.x_v = msg->points[i].longitude_vel_comp;
      c.y_v = 0;
      // c.y_v = msg->points[i].lateral_vel_comp;
      c.z_v = 0;
      cens[cens_index].push_back(c);
    }
    
  }
  cens_index++;

  // if( firstFrame ){

  //   int current_id = radar_point_size;
  //   // float dt = 0.1f; 
  //   float sigmaP=0.01;//0.01
  //   float sigmaQ=0.1;//0.1

  //   //initialize new tracks(function)
  //   //state = [x,y,z,vx,vy,vz]
  //   for(int i=0; i<current_id;i++){
  //    //try ka.init 
  //       track tk;
  //       cv::KalmanFilter ka;
  //       ka.init(stateDim,measDim,ctrlDim,CV_32F);
  //       ka.transitionMatrix = (Mat_<float>(6, 6) << 1,0,0,dt,0,0,
  //                                                   0,1,0,0,dt,0,
  //                                                   0,0,1,0,0,dt,
  //                                                   0,0,0,1,0,0,
  //                                                   0,0,0,0,1,0,
  //                                                   0,0,0,0,0,1);
  //       cv::setIdentity(ka.measurementMatrix);
  //       cv::setIdentity(ka.processNoiseCov, Scalar::all(sigmaP));
  //       cv::setIdentity(ka.measurementNoiseCov, cv::Scalar(sigmaQ));
  //       cout<<"( "<<cens.at(i).x<<","<<cens.at(i).y<<","<<cens.at(i).z<<")\n";
  //       ka.statePost.at<float>(0)=cens.at(i).x;
  //       ka.statePost.at<float>(1)=cens.at(i).y;
  //       ka.statePost.at<float>(2)=cens.at(i).z;
  //       ka.statePost.at<float>(3)=cens.at(i).x_v;// initial v_x
  //       ka.statePost.at<float>(4)=cens.at(i).y_v;// initial v_y
  //       ka.statePost.at<float>(5)=cens.at(i).z_v;// initial v_z
  //         // ka.statePost.at<float>(5)=0;// initial v_z
  //       tk.kf = ka;
  //       tk.state = "tracking";
  //       tk.lose_frame = 0;

  //       // uuid_t uu;
  //       // uuid_generate(uu);
  //       tk.uuid = i;
  //       id_count = i;
  //       filters.push_back(tk);
  //   }

  //   cout<<"Initiate "<<filters.size()<<" tracks."<<endl;

  //   cout << m_s.markers.size() <<endl;
  //   m_s.markers.clear();
  //   cout<< "\033[1;33mMarkerarray is empty(1):\033[0m"<<m_s.markers.empty()<< endl;
  //   std::vector<int> no_use_id;
  //   marker_id(firstFrame,no_use_id);
          
  //   firstFrame=false;
  //   return;//////////////////////////first initialization down 
  // }
  // KFT();
  return;

}
void cluster_radar(label_point point,int r_index){
  //cluster_cens.clear();
  kf_tracker_point cluster_cen_pt;
  cluster_cen_pt.x=0;
  cluster_cen_pt.y=0;
  cluster_cen_pt.z=0;
  cluster_cen_pt.x_v=0;
  cluster_cen_pt.y_v=0;
  cluster_cen_pt.z_v=0;
  int index=0;
  cout<<"Marker:"<<point.scale.getX()<<" "<<point.scale.getY()<<" "<<point.scale.getZ()<<endl;
  for (int i=0;i<cens[r_index].size();i++){
    float x = cens[r_index].at(i).x;
    float y = cens[r_index].at(i).y;
    float z = cens[r_index].at(i).z;
    // cout<<"Radar point "<<i<<":"<<x<<" "<<y<<" "<<z<<endl;
    tf::Vector3 vec_in = tf::Vector3(x,y,z);
    // cout<<"Radar point "<<i<<":"<<vec_in.getX()<<" "<<vec_in.getY()<<" "<<vec_in.getZ()<<endl;
    tf::Vector3 vec_out;
    tf::Transform A;
    A = point.transform.inverse()*transform_lidar.inverse()*transform_radar_front;  //transform_lidar.inverse()*
    vec_out = A*(vec_in);
    
    if(fabs(vec_out.getX())<=(point.scale.getX()+0.05) && fabs(vec_out.getY())<=(point.scale.getY()+0.05)){
      index++;
      cluster_cen_pt.x+=cens[r_index].at(i).x;
      cluster_cen_pt.y+=cens[r_index].at(i).y;
      // cluster_cen_pt.z+=cens.at(i).z;
      cluster_cen_pt.x_v+=cens[r_index].at(i).x_v;
      cluster_cen_pt.y_v+=cens[r_index].at(i).y_v;
      // cluster_cen_pt.z_v+=cens.at(i).z_v;
    }
    // cout<<"In cluster "<<i<<": "<<vec_out.getX()<<" "<<vec_out.getY()<<" "<<vec_out.getZ()<<endl;
  }
  cout<<"Cluster pt:"<<index<<endl;
  cluster_cen_pt.x=cluster_cen_pt.x/index;
  cluster_cen_pt.y=cluster_cen_pt.y/index;
  cluster_cen_pt.z=0;
  cluster_cen_pt.x_v=cluster_cen_pt.x_v/index;
  // cluster_cen_pt.y_v=cluster_cen_pt.y_v/index;
  cluster_cen_pt.y_v=0;
  cluster_cen_pt.z_v=0;
  if(index!=0)
    cluster_cens.push_back(cluster_cen_pt);
  return;
}

void marker_label(const visualization_msgs::MarkerArray& msg){
  // label_pt.clear();
  cluster_cens.clear();
  label_stamp = msg.markers.at(0).header.stamp;
  cout<<"label time:"<<label_stamp.nsec<<endl;
  for(int i=0; i<msg.markers.size();i++)
  {
    label_point point;
    point.position = tf::Vector3(msg.markers.at(i).pose.position.x,msg.markers.at(i).pose.position.y,msg.markers.at(i).pose.position.z);
    point.orientation = tf::Quaternion(msg.markers.at(i).pose.orientation.x,msg.markers.at(i).pose.orientation.y,msg.markers.at(i).pose.orientation.z,msg.markers.at(i).pose.orientation.w);
    point.scale = tf::Vector3(msg.markers.at(i).scale.x,msg.markers.at(i).scale.y,msg.markers.at(i).scale.z);
    point.transform.setOrigin(point.position);
    point.transform.setRotation(point.orientation);
    cout<<i<<endl;
    label_pt[label_index].push_back(point);
    // if(lidar_tf_check && radar_front_tf_check){
    //   cout<<"Time stamp diff:"<<fabs(label_stamp.nsec-radar_stamp.nsec)<<endl;
    //   if(fabs(label_stamp.nsec-radar_stamp.nsec)<100000000)
    //     cluster_radar(point);
    // }
    // if(label_index++<=cens_index){
    //   cluster_radar(label_pt[label_index-1],label_index-1);
    // }
    // else{
    //   label_pt[label_index].push_back(point);
    // }
    // label_pt.push_back(point);
    // cout<< i <<":\n";
    // cout<<"Marker position "<<point.position.getX()<<" "<<point.position.getY()<<" "<<point.position.getZ()<<endl;
    // cout<<"Marker orientation "<<point.orientation.getX()<<" "<<point.orientation.getY()<<" "<<point.orientation.getZ()<<" "<<point.orientation.getW()<<endl;
  }
    if(label_index++<=cens_index){
      for (int i=0;i<label_pt[label_index-1].size();i++)
        cluster_radar(label_pt[label_index-1].at(i),label_index-1);
    }
    else if(cens_index!=0){
      for (int i=0;i<label_pt[cens_index].size();i++)
        cluster_radar(label_pt[cens_index].at(i),cens_index);
    }
    if( firstFrame &&cluster_cens.size()!=0){

    int current_id = cluster_cens.size();
    // float dt = 0.1f; 
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
        cout<<"( "<<cluster_cens.at(i).x<<","<<cluster_cens.at(i).y<<","<<cluster_cens.at(i).z<<")\n";
        ka.statePost.at<float>(0)=cluster_cens.at(i).x;
        ka.statePost.at<float>(1)=cluster_cens.at(i).y;
        ka.statePost.at<float>(2)=cluster_cens.at(i).z;
        ka.statePost.at<float>(3)=cluster_cens.at(i).x_v;// initial v_x
        ka.statePost.at<float>(4)=cluster_cens.at(i).y_v;// initial v_y
        ka.statePost.at<float>(5)=cluster_cens.at(i).z_v;// initial v_z
          // ka.statePost.at<float>(5)=0;// initial v_z
        tk.kf = ka;
        tk.state = "tracking";
        tk.lose_frame = 0;

        // uuid_t uu;
        // uuid_generate(uu);
        tk.uuid = i;
        id_count = i;
        filters.push_back(tk);
    }

    cout<<"Initiate "<<filters.size()<<" tracks."<<endl;

    cout << m_s.markers.size() <<endl;
    m_s.markers.clear();
    cout<< "\033[1;33mMarkerarray is empty(1):\033[0m"<<m_s.markers.empty()<< endl;
    std::vector<int> no_use_id;
    marker_id(firstFrame,no_use_id);
          
    firstFrame=false;
    // return;//////////////////////////first initialization down 
    
  }
  if(cluster_cens.size()!=0 && !firstFrame){
    KFT();
  }
  
  return;
}

void tf_listen(const tf::tfMessage& msg){
  if(!(lidar_tf_check && radar_front_tf_check)){
    for(int i=0;i<msg.transforms.size();i++){
    // cout<<msg.transforms.at(i).child_frame_id<<endl;
    if(((msg.transforms.at(i).child_frame_id)=="nuscenes_lidar") && ((msg.transforms.at(i).header.frame_id)=="car")){
      tf::transformMsgToTF(msg.transforms.at(i).transform,transform_lidar);
      lidar_tf_check = true;
      cout<<"Lidar transform:"<<transform_lidar.getOrigin().getX() <<endl;
      }
    else if(((msg.transforms.at(i).child_frame_id)=="nuscenes_radar_front") && ((msg.transforms.at(i).header.frame_id)=="car")){
      tf::transformMsgToTF(msg.transforms.at(i).transform,transform_radar_front);
      radar_front_tf_check = true;
      }
    
    }
  }
  return;

}
int main(int argc, char** argv){
  ros::init(argc,argv,"radar_track");
  ros::NodeHandle nh;
  ros::Subscriber tf_sub;
  // tf_sub = nh.subscribe("tf",100,&tf_listen);
  
  sub = nh.subscribe("radar_front_outlier",10,&callback);
  pub_marker = nh.advertise<visualization_msgs::MarkerArray>("marker", 1);
  ros::Subscriber lidr_label;
  
  // lidr_label = nh.subscribe("lidar_label",50,&marker_label);

  ros::Rate r(10);
  while(ros::ok()){
    ros::spinOnce();
    pub_marker.publish(m_s);
    r.sleep();
  }

  return 0;
}


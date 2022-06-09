#include <ros/ros.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>

tf::TransformBroadcaster *br;

void odom_callback(const nav_msgs::OdometryConstPtr& odom_msg){
  // ROS_INFO("Get Odometry");
  // tf::TransformBroadcaster br;
  tf::Transform carTransform;
  carTransform.setOrigin(tf::Vector3(odom_msg->pose.pose.position.x,odom_msg->pose.pose.position.y,odom_msg->pose.pose.position.z));
  tf::Quaternion q;
  q.setValue(odom_msg->pose.pose.orientation.x,odom_msg->pose.pose.orientation.y,odom_msg->pose.pose.orientation.z,odom_msg->pose.pose.orientation.w);
  carTransform.setRotation(q);
  br->sendTransform(tf::StampedTransform(carTransform,odom_msg->header.stamp,odom_msg->header.frame_id,odom_msg->child_frame_id));
  // ROS_INFO("%s %s",odom_msg->header.frame_id.c_str(), odom_msg->child_frame_id.c_str());
}

int main(int argc, char * argv[])
{
  ros::init(argc, argv, "radarScenes_Odom_Sub");
  ros::NodeHandle nh;
  ros::Subscriber odom_sub = nh.subscribe("odometry",10,&odom_callback);
  br = new tf::TransformBroadcaster;
  while(ros::ok()){
    ros::spinOnce();
  }
  return 0;
}

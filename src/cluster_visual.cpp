#include "cluster_visual/cluster_visual.h"

Cluster_visual::Cluster_visual()
{
    valid_cluster_ = true;
}

geometry_msgs::PolygonStamped Cluster_visual::GetPolygon()
{
    return polygon_;
}

jsk_recognition_msgs::BoundingBox Cluster_visual::GetBoundingBox()
{
    return bounding_box_;
}

std::vector<cv::Point2f> Cluster_visual::GetHull()
{
    return hull_;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Cluster_visual::GetCloud()
{
    return pointcloud_;
}

pcl::PointXYZ Cluster_visual::GetMinPoint()
{
    return min_point_;
}

pcl::PointXYZ Cluster_visual::GetMaxPoint()
{
    return max_point_;
}

pcl::PointXYZ Cluster_visual::GetCentroid()
{
    return centroid_;
}

pcl::PointXYZ Cluster_visual::GetAveragePoint()
{
    return average_point_;
}

double Cluster_visual::GetOrientationAngle()
{
    return orientation_angle_;
}

float Cluster_visual::GetLength()
{
    return bounding_box_.dimensions.x;
}

float Cluster_visual::GetWidth()
{
    return bounding_box_.dimensions.y;
}

float Cluster_visual::GetHeight()
{
    return height_;
}

float Cluster_visual::GetVolume()
{
    return cv::contourArea(hull_) * height_;
}

Eigen::Matrix3f Cluster_visual::GetEigenVectors()
{
    return eigen_vectors_;
}

Eigen::Vector3f Cluster_visual::GetEigenValues()
{
    return eigen_values_;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr Cluster_visual::GetColoredPointCloud()
{
    return pointcloud_colored_;
}

void Cluster_visual::SetColoredPointCloud(int intensity)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr output_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  for(int i = 0; i < pointcloud_->points.size(); i++){
    pcl::PointXYZI pt;
    pt.x = pointcloud_->points[i].x;
    pt.y = pointcloud_->points[i].y;
    pt.z = pointcloud_->points[i].z;
    pt.intensity = intensity;
    output_ptr->points.push_back(pt);
  }
  pointcloud_colored_ = output_ptr;
}

void Cluster_visual::SetCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr in_origin_cloud_ptr,
                       std_msgs::Header in_ros_header, bool in_estimate_pose)
{

    pcl::PointCloud<pcl::PointXYZ>::Ptr current_cluster (new pcl::PointCloud<pcl::PointXYZ>);
    float min_x =  std::numeric_limits<float>::max();
    float max_x = -std::numeric_limits<float>::max();
    float min_y =  std::numeric_limits<float>::max();
    float max_y = -std::numeric_limits<float>::max();
    float min_z =  std::numeric_limits<float>::max();
    float max_z = -std::numeric_limits<float>::max();
    float average_x = 0, average_y = 0, average_z = 0;

    for(size_t i = 0; i < in_origin_cloud_ptr->points.size (); ++i)
    {
        pcl::PointXYZ p;
        p.x = in_origin_cloud_ptr->points[i].x;
        p.y = in_origin_cloud_ptr->points[i].y;
        p.z = in_origin_cloud_ptr->points[i].z;

        average_x += p.x;
        average_y += p.y;
        average_z += p.z;
        centroid_.x += p.x;
        centroid_.y += p.y;
        centroid_.z += p.z;
        current_cluster->points.push_back(p);

        if (p.x < min_x)
          min_x = p.x;
        if (p.y < min_y)
          min_y = p.y;
        if (p.z < min_z)
          min_z = p.z;
        if (p.x > max_x)
          max_x = p.x;
        if (p.y > max_y)
          max_y = p.y;
        if (p.z > max_z)
          max_z = p.z;

    }

    min_point_.x = min_x;
    min_point_.y = min_y;
    min_point_.z = min_z;
    max_point_.x = max_x;
    max_point_.y = max_y;
    max_point_.z = max_z;

    if (in_origin_cloud_ptr->points.size () > 0)
    {
        centroid_.x /= in_origin_cloud_ptr->points.size ();
        centroid_.y /= in_origin_cloud_ptr->points.size ();
        centroid_.z /= in_origin_cloud_ptr->points.size ();

        average_x /= in_origin_cloud_ptr->points.size ();
        average_y /= in_origin_cloud_ptr->points.size ();
        average_z /= in_origin_cloud_ptr->points.size ();
    }

    average_point_.x = average_x;
    average_point_.y = average_y;
    average_point_.z = average_z;

    length_ = max_point_.x - min_point_.x;
    width_ = max_point_.y - min_point_.y;
    height_ = max_point_.z - min_point_.z;

    if(height_ == 0) height_ = 0.01;

    bounding_box_.header = in_ros_header;

    bounding_box_.pose.position.x = min_point_.x + length_ / 2;
    bounding_box_.pose.position.y = min_point_.y + width_ / 2;
    bounding_box_.pose.position.z = min_point_.z + height_ / 2;

    bounding_box_.dimensions.x = ((length_ < 0) ? -1 * length_ : length_);
    bounding_box_.dimensions.y = ((width_ < 0) ? -1 * width_ : width_);
    bounding_box_.dimensions.z = ((height_ < 0) ? -1 * height_ : height_);

    std::vector<cv::Point2f> points;

    for (unsigned int i = 0; i < current_cluster->points.size(); i++)
    {
        cv::Point2f pt;
        pt.x = current_cluster->points[i].x;
        pt.y = current_cluster->points[i].y;
        points.push_back(pt);

    }

    cv::convexHull(points, hull_);
    polygon_.header = in_ros_header;
    for (size_t i = 0; i < hull_.size() + 1; i++)
    {
        geometry_msgs::Point32 point;
        point.x = hull_[i % hull_.size()].x;
        point.y = hull_[i % hull_.size()].y;
        point.z = min_point_.z - 0.001;
        polygon_.polygon.points.push_back(point);
    }

    for (size_t i = 0; i < hull_.size() + 1; i++)
    {
        geometry_msgs::Point32 point;
        point.x = hull_[i % hull_.size()].x;
        point.y = hull_[i % hull_.size()].y;
        point.z = max_point_.z + 0.001;
        polygon_.polygon.points.push_back(point);
    }

    //pose estimation
    double rz = 0;

    if (in_estimate_pose)
    {
        cv::RotatedRect box = minAreaRect(hull_);;
        rz = box.angle * M_PI / 180;
        bounding_box_.pose.position.x = box.center.x;
        bounding_box_.pose.position.y = box.center.y;
        bounding_box_.dimensions.x = box.size.width;
        bounding_box_.dimensions.y = box.size.height;

        centroid_.x = box.center.x;
        centroid_.y = box.center.y;
    }

    tf::Quaternion quat = tf::createQuaternionFromRPY(0.0, 0.0, rz);
    tf::quaternionTFToMsg(quat, bounding_box_.pose.orientation);

    current_cluster->width = current_cluster->points.size();
    current_cluster->height = 1;
    current_cluster->is_dense = true;

    valid_cluster_ = true;
    pointcloud_ = current_cluster;

}
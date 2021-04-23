#include <iostream>
#include <stdlib.h>
#include <string> 
#include <algorithm>
#include <Eigen/Dense>
// pcl for kdtree
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>

#include <dbscan/dbscan.h>

#define stage_one_num 5
#define stage_one_vel 0.3
#define center_vel_thres 0.05

dbscan::dbscan()
{
    // initial input output data
    in_points.clear();
    out_points.clear();
    center.clear();
    var_list.clear();
    center_list_info.clear();
    stage_one_cluster.clear();
    // initial dbscan parameter (from slow to fast)
    param.eps[0] = 2.5;
    param.MinPts[0] = 2;
    param.eps[1] = 2.0;
    param.MinPts[1] = 2;
    param.eps[2] = 1.5;
    param.MinPts[2] = 2;
    param.eps[3] = 1;
    param.MinPts[3] = 1;
    // initial all cluster index to -1
    cluster_count = 0;
    state = 1;
    // vel_scaling = data_period*1.5;
    vel_scaling = 1;
    output_info = true;
    output_param_optimize = true;
    use_vel_scaling = false;
    frame_state = FRAME_STATE::first;
}

dbscan::~dbscan(){}


// cluster points without RANSAC
std::vector< std::vector<cluster_point> > dbscan::cluster(std::vector<cluster_point> data){
    // initial dbscan parameter (from slow to fast)
    points.clear();
    points = data;
    analysis_data();
    if(frame_state == FRAME_STATE::first){
        history_points = data;
    }
    else if(frame_state == FRAME_STATE::second){
        points.insert(points.end(),history_points.begin(),history_points.end());
        history_back = history_points;
        oldest_points = history_points;
        history_points = data;
    }
    else{
        points.insert(points.end(),history_points.begin(),history_points.end());
        points.insert(points.end(),oldest_points.begin(),oldest_points.end());
        history_back = oldest_points;
        history_back.insert(history_back.end(),history_points.begin(),history_points.end());
        oldest_points = history_points;
        history_points = data;
    }
    cluster_idx.assign(points.size(),-1);
    final_center.clear();
    final_var_list.clear();
    center_list_info.clear();
    cluster_count = 0;
    if(output_info){
        std::string sep = "===========================\n";
        cout << sep;
        cout << "DBSCAN scan vel analysis:" << endl;
        cout << "\tVmax = " << data_vel.max << endl;
        cout << "\tVmin = " << data_vel.min << endl;
        cout << "\tVrange = " << data_vel.range << endl;
        cout << "\tVave = " << data_vel.ave << endl;
        cout << "\tScan Eps = " << scan_eps << endl;
        switch (frame_state){
            case FRAME_STATE::first:
                cout << "\tFrame State = first" << endl;
                break;
            case FRAME_STATE::second:
                cout << "\tFrame State = second" << endl;
                break;
            default:
                cout << "\tFrame State = more" << endl;
                break;
        }
        cout << sep;
    }
    use_vel_scaling = true;
    param.eps[0] = 2.5;
    param.MinPts[0] = 2;
    param.eps[1] = 2.5;
    param.MinPts[1] = 2;
    param.eps[2] = 2.5;
    param.MinPts[2] = 2;
    param.eps[3] = 2.5;
    param.MinPts[3] = 2;
    for(int i=0;i<points.size();i++){
        if(points.at(i).visited && cluster_idx.at(i)!=-1)
            continue;
        points.at(i).visited = true;
        cluster_point core_pt = points.at(i);
        int core_level = decide_vel_level_test(core_pt.vel);
        // double Nmin = (core_pt.vel > data_vel.ave ? 1 : 3);
        double Nmin = 2;
        std::vector<int> neighbor = find_neighbor(core_pt, core_level);
        if(neighbor.size() >= Nmin){
            cluster_idx.at(i) = cluster_count;
            expand_neighbor(neighbor);
            cluster_count ++;
        }
    }
    std::vector< std::vector<cluster_point> > cluster_list(cluster_count);
    std::vector< std::vector<cluster_point> > non_cluster_list, final_cluster_list;
    for(int j=0;j<cluster_idx.size();j++){
        if(cluster_idx.at(j)!=-1)
            cluster_list.at(cluster_idx.at(j)).push_back(points.at(j));
        else{
            if(points.at(j).scan_time != scan_num)
                continue;
            std::vector<cluster_point> non_cluster_temp;
            non_cluster_temp.push_back(points.at(j));
            non_cluster_list.push_back(non_cluster_temp);
        }
    }
    final_cluster_list = cluster_list;
    // split(non_cluster_list);
    remove(cluster_list);
    remove(non_cluster_list);
    final_cluster_list.insert(final_cluster_list.end(),non_cluster_list.begin(),non_cluster_list.end());

    // cluster_center(cluster_list);
    cluster_center(final_cluster_list);
    // merge(cluster_list);
    switch (frame_state){
        case FRAME_STATE::first:
            frame_state = FRAME_STATE::second;
            break;
        case FRAME_STATE::second:
            split(cluster_list);
            frame_state = FRAME_STATE::more;
            break;
        default:
            split(cluster_list);
            break;
    }
    cluster_list.insert(cluster_list.end(),non_cluster_list.begin(),non_cluster_list.end());
    // cluster_center(cluster_list);
    if(output_info){
        cout<<"\nDBSCAN cluster index (total):\n";
        cout<<"==============================\n";
        int cluster_num = 0;
        for(int k=0;k<cluster_list.size();k++){
            if(k != 0)
                cout<<"----------------------\n";
            cout<<"cluster index: "<<k<<endl;
            cout<<"\033[1;33mcenter position: (" << center.at(k).x << "," << center.at(k).y << ")\033[0m";
            cout<<endl;
            cout<<"\033[1;33mcenter velocity: (" << center.at(k).x_v << "," << center.at(k).y_v << ")\033[0m";
            cout<<endl;
            cout<<"\033[1;33mcenter var: (" << var_list.at(k)(0) << "," << var_list.at(k)(1) << ")\033[0m";
            cout<<endl<<endl;
            for(int idx=0;idx<cluster_list.at(k).size();idx++){
                cout<<"\tPosition: ("<<cluster_list.at(k).at(idx).x<<","<<cluster_list.at(k).at(idx).y<<")"<<endl;
                cout<<"\tVelocity: "<<cluster_list.at(k).at(idx).vel<<" ("<<cluster_list.at(k).at(idx).x_v<<","<<cluster_list.at(k).at(idx).y_v<<")"<<endl;
                cout<<endl;
                cluster_num ++;
            }
        }

    }
    
    remove(cluster_list);
    cluster_center(cluster_list);

    if(output_param_optimize){
        std::vector<double> eps_optimize_min;
        // std::cout << "============================================" << std::endl;
        for(int i=0;i<cluster_list.size();i++){
            // std::cout << "--------------------- Cluster " << i << " ---------------------" << std::endl;
            if(cluster_list.at(i).size() <= 1){
                // std::cout << "Cluster size = "<<cluster_list.at(i).size()<<"!!" << std::endl;
                continue;
            }
            eps_optimize_min.push_back(find_near_min(cluster_list.at(i),1));
        }
        double ave_min_distance = std::accumulate(eps_optimize_min.begin(),eps_optimize_min.end(),0.0) / eps_optimize_min.size();
        // std::cout << "============================================" << std::endl;
        std::cout << "Average eps = " << ave_min_distance << "(n = " << eps_optimize_min.size() << ")" << std::endl;
    }
    // cluster_list.insert(cluster_list.end(),non_cluster_list.begin(),non_cluster_list.end());
    return cluster_list;
}

// return the cluster number after cluster function
int dbscan::cluster_num(void){
    return cluster_count;
}

double dbscan::vel_function(double delta_v){
    // return delta_v * vel_scaling;
    // return std::exp(std::fabs(delta_v)) - 1;
    return std::pow(2,std::fabs(delta_v)) - 1;
}

std::vector<int> dbscan::find_neighbor(cluster_point pt, int vel_level){
    std::vector<int> neighbor_cluster;
    neighbor_cluster.clear();
    if(use_vel_scaling){
        // Eigen::Vector3f core_pt(pt.x,pt.y,pt.vel*vel_scaling);
        Eigen::Vector2f core_pt(pt.x,pt.y);
        // double eps = 2.0;
        double eps = 2.5;
        // double eps = 2.5  + 1.2*(data_vel.ave - pt.vel)/data_vel.range;
        // double eps = scan_eps - 1.2*abs(data_vel.ave - pt.vel)/data_vel.range;
        // double eps = scan_eps;
        for(int i=0;i<points.size();i++){
            cluster_point temp = points.at(i);
            // Eigen::Vector3f neighbor(temp.x,temp.y,temp.vel*vel_scaling);
            Eigen::Vector2f neighbor(temp.x,temp.y);
            if(((core_pt-neighbor).squaredNorm() + vel_function(pt.vel-temp.vel)) <= eps*eps){
                neighbor_cluster.push_back(i);
            }
        }
    }
    else{
        Eigen::Vector3f core_pt(pt.x,pt.y,pt.z);
        for(int i=0;i<points.size();i++){
            cluster_point temp = points.at(i);
            Eigen::Vector3f neighbor(temp.x,temp.y,temp.z);
            if((core_pt-neighbor).norm() <= param.eps[vel_level]){
                neighbor_cluster.push_back(i);
            }
        }
    }
    return neighbor_cluster;
}

void dbscan::expand_neighbor(std::vector<int> neighbor){
    for(int i=0;i<neighbor.size();i++){
        if(points.at(neighbor.at(i)).visited)
            continue;
        points.at(neighbor.at(i)).visited = true;
        cluster_point n_pt = points.at(neighbor.at(i));
        int n_pt_level;
        double Nmin;
        if(use_vel_scaling){
            // Nmin = (n_pt.vel > data_vel.ave ? 1 : 3);
            Nmin = 2;
        }
        else{
            n_pt_level = decide_vel_level(n_pt.vel);
            Nmin = param.MinPts[n_pt_level];
        }
        std::vector<int> expand_n = find_neighbor(n_pt,n_pt_level);
        if(expand_n.size() >= Nmin){
            cluster_idx.at(neighbor.at(i)) = cluster_count;
            expand_neighbor(expand_n);
        }
    }
}

int dbscan::decide_vel_level(double vel){
    switch (state){
        case 1:
            if(vel <= 0.05)
                return 0;
            else if(vel <= 0.1)
                return 1;
            else if(vel <= 0.25)
                return 2;
            else
                return 3;
            break;
        case 2:
            if(vel <= 0.1)
                return 0;
            else if(vel <= 0.2)
                return 1;
            else if(vel <= 0.3)
                return 2;
            else
                return 3;
        
        default:
            break;
    }
}

int dbscan::decide_vel_level_test(double vel){
    if(vel <= 1)
        return 0;
    else if(vel <= 3)
        return 1;
    else if(vel <= 5)
        return 2;
    else
        return 3;
               
}

std::vector<cluster_point> dbscan::delist(std::vector< std::vector<cluster_point> > cluster_l){
    std::vector<cluster_point> output_list;
    for(int i=0;i<cluster_l.size();i++){
        for(int j=0;j<cluster_l.at(i).size();j++){
            cluster_point pt;
            pt = cluster_l.at(i).at(j);
            pt.cluster_id = -1;
            pt.visited = false;
            output_list.push_back(pt);
        }
    }
    return output_list;
}


/*
 * calculate the center and the variance of the cluster_list
 */ 
void dbscan::cluster_center(std::vector< std::vector<cluster_point> > cluster_list){
    center_list_info.assign(cluster_list.size(),0);
    center.clear();
    final_center.clear();
    for(int cluster_idx=0;cluster_idx<cluster_list.size();cluster_idx++){
        cluster_point pt;
        pt.x = 0;
        pt.y = 0;
        pt.z = 0;
        pt.x_v = 0;
        pt.y_v = 0;
        pt.z_v = 0;
        for(int i=0;i<cluster_list.at(cluster_idx).size();i++){
            pt.x += cluster_list.at(cluster_idx).at(i).x;
            pt.y += cluster_list.at(cluster_idx).at(i).y;
            pt.z += cluster_list.at(cluster_idx).at(i).z;
            pt.x_v += cluster_list.at(cluster_idx).at(i).x_v;
            pt.y_v += cluster_list.at(cluster_idx).at(i).y_v;
            pt.z_v += cluster_list.at(cluster_idx).at(i).z_v;
        }
        center_list_info.at(cluster_idx) = cluster_list.at(cluster_idx).size();
        pt.x /= cluster_list.at(cluster_idx).size();
        pt.y /= cluster_list.at(cluster_idx).size();
        pt.z /= cluster_list.at(cluster_idx).size();
        pt.x_v /= cluster_list.at(cluster_idx).size();
        pt.y_v /= cluster_list.at(cluster_idx).size();
        pt.z_v /= cluster_list.at(cluster_idx).size();
        pt.cluster_id = cluster_idx;
        pt.vel = sqrt(pt.x_v*pt.x_v+pt.y_v*pt.y_v+pt.z_v*pt.z_v);
        center.push_back(pt);

    }
    final_center = center;
    var_list.clear();
    final_var_list.clear();
    for(int cluster_idx=0;cluster_idx<cluster_list.size();cluster_idx++){
        Eigen::VectorXd var(6), mean(6);
        var.fill(0);
        mean(0) = center.at(cluster_idx).x;
        mean(1) = center.at(cluster_idx).y;
        mean(2) = center.at(cluster_idx).z;
        mean(3) = center.at(cluster_idx).x_v;
        mean(4) = center.at(cluster_idx).y_v;
        mean(5) = center.at(cluster_idx).z_v;
        for(int i=0;i<cluster_list.at(cluster_idx).size();i++){
            Eigen::VectorXd temp(6);
            temp(0) = cluster_list.at(cluster_idx).at(i).x;
            temp(1) = cluster_list.at(cluster_idx).at(i).y;
            temp(2) = cluster_list.at(cluster_idx).at(i).z;
            temp(3) = cluster_list.at(cluster_idx).at(i).x_v;
            temp(4) = cluster_list.at(cluster_idx).at(i).y_v;
            temp(5) = cluster_list.at(cluster_idx).at(i).z_v;
            temp = temp - mean;
            for(int k=0;k<temp.size();k++)
                var(k) += temp(k)*temp(k);
            // var += ((temp - mean).transpose() * (temp - mean).asDiagonal());
        }
        var = (var / cluster_list.at(cluster_idx).size());
        for(int j=0;j<var.size();j++)
            var(j) = sqrt(var(j));
        var_list.push_back(var);
    }
    final_var_list = var_list;
    // std::vector<Eigen::VectorXd>::iterator var_iter = final_var_list.begin();
    // for(std::vector<cluster_point>::iterator it=final_center.begin();it!=final_center.end();){
    //     Eigen::VectorXd var_temp = *var_iter;
    //     float radius = sqrt(var_temp(0)*var_temp(0)+var_temp(1)*var_temp(1));
    //     if(radius > 5){
    //         it = final_center.erase(it);
    //         var_iter = final_var_list.erase(var_iter);
    //     }
    //     // else if((*it).vel < center_vel_thres){
    //     //     it = final_center.erase(it);
    //     //     var_iter = final_var_list.erase(var_iter);
    //     // }
    //     else{
    //         it++;
    //         var_iter++;
    //     }
    // }
}

std::vector<cluster_point> dbscan::get_center(void){
    return final_center;
}

void dbscan::merge(std::vector< std::vector<cluster_point> > &cluster_list){
    std::vector< std::vector<cluster_point> > merge_list = cluster_list;
    std::vector<int> merge_list_info;
    merge_list_info.assign(center.size(),-1);
    for(int idx = 0;idx<center.size();idx++){
        Eigen::Vector3f core_pos(center.at(idx).x, center.at(idx).y, center.at(idx).z);
        Eigen::Vector3f core_vel(center.at(idx).x_v, center.at(idx).y_v, center.at(idx).z_v);
        float merge_dist = 100;
        for(int i=idx+1;i<center.size();i++){
            Eigen::Vector3f temp_pos(center.at(i).x, center.at(i).y, center.at(i).z);
            Eigen::Vector3f temp_vel(center.at(i).x_v, center.at(i).y_v, center.at(i).z_v);
            if((core_pos-temp_pos).norm() < 1 || ((core_pos-temp_pos).norm() < 2.5 && (core_vel-temp_vel).norm() < 0.8)){
                if(merge_dist > (core_pos-temp_pos).norm()){
                    merge_list_info.at(i) = idx;
                    merge_dist = (core_pos-temp_pos).norm();
                }
                // break;
            }
        }
    }
    if(output_info){
        cout<<"***************************"<<endl;
        cout<<"In Merge State:\n";
        cout<<"Before merge: "<<center.size()<<" clusters\n\n";
    }
    for(int i=center.size()-1;i>=0;i--){
        if(merge_list_info.at(i)!=-1){
            if(output_info){
                cout<<"Merge the "<<i<<"-th cluster to "<<merge_list_info.at(i)<<endl;
            }
            merge_list.at(merge_list_info.at(i)).insert(merge_list.at(  merge_list_info.at(i)).end(),
                                                                        merge_list.at(i).begin(),
                                                                        merge_list.at(i).end());
            merge_list.erase(merge_list.begin()+i);
        }
    }
    center.clear();
    cluster_center(merge_list);
    if(output_info){
        cout<<endl<<"After merge: "<<center.size()<<" clusters\n";
        cout<<"***************************"<<endl;
    }
    cluster_list = merge_list;
}

/*
 * Split out the history points from the cluster_list.
 * Let the cluster_list contain only the current scan points.
 */
void dbscan::split(std::vector< std::vector<cluster_point> > &cluster_list){
    std::vector< std::vector<cluster_point> > result;
    for(int i=0;i<cluster_list.size();i++){
        std::vector<cluster_point> temp;
        for(int j = 0;j<cluster_list.at(i).size();j++){
            if(cluster_list.at(i).at(j).scan_time == scan_num){
                temp.push_back(cluster_list.at(i).at(j));
            }
        }
        if(temp.size()!=0)
            result.push_back(temp);
    }
    cluster_list = result;
}

// remove the cluster that contains only past points
void dbscan::remove(std::vector< std::vector<cluster_point> > &cluster_list){
    for(std::vector< std::vector<cluster_point> >::iterator it=cluster_list.begin();it != cluster_list.end();){
        int count = 0;
        for(int i=0;i<(*it).size();i++){
            if((*it).at(i).scan_time != scan_num)
                count ++;
        }
        if(count == (*it).size()){
            it = cluster_list.erase(it);
        }
        else{
            it++;
        }
    }
}

void dbscan::analysis_data(void){
    data_vel.max = -1;
    data_vel.min = 100;
    data_vel.ave = 0;
    scan_eps = 0;
    for(int i=0;i<points.size();i++){
        data_vel.ave += points.at(i).vel;
        if(points.at(i).vel > data_vel.max)
            data_vel.max = points.at(i).vel;
        else if(points.at(i).vel < data_vel.min)
            data_vel.min = points.at(i).vel;
        double temp_Nmin = 100;
        Eigen::Vector3f core_pt(points.at(i).x,points.at(i).y,points.at(i).vel*vel_scaling);
        for(int j=0;j<points.size();j++){
            if(i==j)
                continue;
            Eigen::Vector3f temp_pt(points.at(j).x,points.at(j).y,points.at(j).vel*vel_scaling);
            if(temp_Nmin>(core_pt-temp_pt).norm())
                temp_Nmin = (core_pt-temp_pt).norm();
            
        }
        scan_eps += temp_Nmin;
    }
    scan_eps /= points.size();
    data_vel.ave /= points.size();
    data_vel.range = data_vel.max - data_vel.min;
    return;
}

std::vector<cluster_point> dbscan::get_history_points(void){
    std::vector<cluster_point> multi_frame_pts;
    if(frame_state != FRAME_STATE::first)
        multi_frame_pts = history_back;
    return multi_frame_pts;
}

void dbscan::grid(double grid_size){
    // Eigen::MatrixXd grid_mat();
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for(int i=0;i<points.size();i++){
        pcl::PointXYZ temp_pt;
        temp_pt.x = points.at(i).x;
        temp_pt.y = points.at(i).y;
        temp_pt.z = points.at(i).vel;
        input_cloud->points.push_back(temp_pt);
    }
    kdtree.setInputCloud(input_cloud);
    // kdtree.set
}

double dbscan::find_near_min(std::vector<cluster_point> kd_cloud, int k=1){
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for(int i=1;i<kd_cloud.size();i++){
        pcl::PointXYZ temp_pt;
        temp_pt.x = kd_cloud.at(i).x;
        temp_pt.y = kd_cloud.at(i).y;
        temp_pt.z = kd_cloud.at(i).vel*vel_scaling;
        input_cloud->points.push_back(temp_pt);
    }
    kdtree.setInputCloud(input_cloud);

    pcl::PointXYZ searchPoint;
	searchPoint.x = kd_cloud.at(0).x;
	searchPoint.y = kd_cloud.at(0).y;
	searchPoint.z = kd_cloud.at(0).vel*vel_scaling;

    std::vector<int> pointIdxNKNSearch(k);
	std::vector<float> pointNKNSquaredDistance(k);
    // std::cout << "K nearest neighbor search at (" << searchPoint.x
	// 	<< " " << searchPoint.y	<< " " << searchPoint.z	<< ") with K=" << k << std::endl;
	if (kdtree.nearestKSearch(searchPoint, k, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
	{
		for (size_t i = 0; i < pointIdxNKNSearch.size(); ++i)
		{

			// std::cout << "    " << input_cloud->points[pointIdxNKNSearch[i]].x << " " << input_cloud->points[pointIdxNKNSearch[i]].y
			// 	<< " " << input_cloud->points[pointIdxNKNSearch[i]].z << " (distance: " << std::sqrt(pointNKNSquaredDistance[i]) << ")" << std::endl;
            // std::cout << "    " << "squared distance: " << pointNKNSquaredDistance[i] << std::endl;
		}
	}
    
    return std::sqrt(pointNKNSquaredDistance[0]);

}
#include "dbpda/dbpda.h"

dbpda::dbpda(double eps, int Nmin){
    dbpda_param.eps = eps;
    dbpda_param.Nmin = Nmin;
    history_frame_num = 3;
    vel_scale = 1;
    input_cloud = pcl::PointCloud<pcl::PointXYZ>().makeShared();
    output_info = true;
}

dbpda::~dbpda(){}

std::vector< std::vector<cluster_point> > dbpda::cluster(std::vector<cluster_point> data){
    input_cloud->clear();
    points_history.push_back(data);
    if(points_history.size()>history_frame_num){
        points_history.erase(points_history.begin());
    }
    switch (points_history.size())
    {
    case 1:
        frame_state = FRAME_STATE::first;
        break;
    case 2:
        frame_state = FRAME_STATE::second;
    default:
        frame_state = FRAME_STATE::more;
        break;
    }
    cluster_queue.clear();
    cluster_queue.shrink_to_fit();
    std::vector< cluster_point > process_data;
    for(int i=0;i<points_history.size();i++){
        for(int j=0;j<points_history.at(i).size();j++){
            pcl::PointXYZ temp_pt;
            temp_pt.x = points_history.at(i).at(j).x;
            temp_pt.y = points_history.at(i).at(j).y;
            temp_pt.z = points_history.at(i).at(j).vel * vel_scale;
            input_cloud->points.push_back(temp_pt);
            process_data.push_back(points_history.at(i).at(j));
            // initial the reachable_dist
            dbpda_info cluster_pt;
            cluster_pt.reachable_dist = -1; // undefined distance
            cluster_queue.push_back(cluster_pt);
        }
    }

    cluster_order.clear();
    cluster_order.shrink_to_fit();
    std::vector< std::vector<int> > final_cluster_order;
    
    for(int i=0;i<process_data.size();i++){
        // check if the pt has been visited
        if(process_data.at(i).vistited)
            continue;
        else
            process_data.at(i).vistited = true;
        std::vector<int> temp_cluster;
        
        // find neighbor and get the core distance
        cluster_queue.at(i).neighbor_info = find_neighbors(process_data.at(i));
        cluster_order.push_back(i);
        temp_cluster.push_back(i);
        // satisfy the Nmin neighbors
        if(cluster_queue.at(i).neighbor_info.pointIdxNKNSearch.size() != 0){
            cluster_queue.at(i).core_dist = std::sqrt(*std::max_element(cluster_queue.at(i).neighbor_info.pointNKNSquaredDistance.begin(),
                                                     cluster_queue.at(i).neighbor_info.pointNKNSquaredDistance.end()));
            expand_neighbor(process_data, temp_cluster, i);
        }
        else{
            cluster_queue.at(i).core_dist = -1;      // undefined distance (not a core point)
        }
        final_cluster_order.push_back(temp_cluster);
    }
    std::vector<int> current_order = split_past(process_data, final_cluster_order);
    std::vector< std::vector<cluster_point> > cluster_result;
    for(int i=0;i<final_cluster_order.size();i++){
        std::vector<cluster_point> temp_cluster_unit;
        for(int j=0;j<final_cluster_order.at(i).size();j++){
            temp_cluster_unit.push_back(process_data.at(final_cluster_order.at(i).at(j)));
        }
        cluster_result.push_back(temp_cluster_unit);
    }
    cluster_center(cluster_result);
    return cluster_result;
}

double dbpda::distance(cluster_point p1, cluster_point p2){
    Eigen::Vector3d d(p1.x-p2.x,p1.y-p2.y,(p1.vel-p2.vel)*vel_scale);
    return d.norm();
}


dbpda_neighbor_info dbpda::find_neighbors(cluster_point p){
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(input_cloud);

    pcl::PointXYZ search_pt;
    search_pt.x = p.x;
    search_pt.y = p.y;
    search_pt.z = p.vel * vel_scale;

    // find the eps radius neighbors
    dbpda_neighbor_info info;
    info.search_radius = dbpda_param.eps;
    info.search_k = dbpda_param.Nmin;
    // find the eps neighbors
    if(kdtree.radiusSearch(search_pt,info.search_radius,info.pointIdxRadiusSearch,info.pointRadiusSquaredDistance) >= info.search_k){
        // for (size_t i = 0; i < info.pointIdxRadiusSearch.size (); ++i){
            // std::cout << "    "  <<   input_cloud->points[ info.pointIdxRadiusSearch[i] ].x 
            //             << " " << input_cloud->points[ info.pointIdxRadiusSearch[i] ].y 
            //             << " " << input_cloud->points[ info.pointIdxRadiusSearch[i] ].z 
            //             << " (squared distance: " << info.pointRadiusSquaredDistance[i] << ")" << std::endl;
            // }
        // if the point has more than Nmin neighbors, find the min distance that contains only Nmin neighbors
        if(kdtree.nearestKSearch(search_pt, info.search_k,info.pointIdxNKNSearch,info.pointNKNSquaredDistance)>0){
            // for (size_t i = 0; i < info.pointIdxNKNSearch.size (); ++i){
            //     std::cout << "    "  <<   input_cloud->points[ info.pointIdxNKNSearch[i] ].x 
            //                 << " " << input_cloud->points[ info.pointIdxNKNSearch[i] ].y 
            //                 << " " << input_cloud->points[ info.pointIdxNKNSearch[i] ].z 
            //                 << " (squared distance: " << info.pointNKNSquaredDistance[i] << ")" << std::endl;
            // }
        }
    }
    return info;
}

void dbpda::expand_neighbor(std::vector< cluster_point > &process_data, std::vector<int> &temp_cluster, int core_index){
    for(int i=0;i<cluster_queue.at(core_index).neighbor_info.pointIdxRadiusSearch.size();i++){
        int neighbor_index = cluster_queue.at(core_index).neighbor_info.pointIdxRadiusSearch.at(i);
        if(process_data.at(neighbor_index).vistited)
            continue;
        else
            process_data.at(neighbor_index).vistited = true;
        temp_cluster.push_back(neighbor_index);
        cluster_queue.at(neighbor_index).neighbor_info = find_neighbors(process_data.at(neighbor_index));
        if(cluster_queue.at(neighbor_index).neighbor_info.pointIdxNKNSearch.size() != 0){
            cluster_queue.at(neighbor_index).core_dist = std::sqrt(*std::max_element(cluster_queue.at(neighbor_index).neighbor_info.pointNKNSquaredDistance.begin(),
                                                     cluster_queue.at(neighbor_index).neighbor_info.pointNKNSquaredDistance.end()));
            expand_neighbor(process_data, temp_cluster, neighbor_index);
        }
        else{
            cluster_queue.at(neighbor_index).core_dist = -1;      // undefined distance (not a core point)
        }
    }
}

std::vector<int> dbpda::split_past(std::vector< cluster_point > process_data, std::vector< std::vector<int> > &final_cluster_order){
    std::vector<int> current_order;
    for(int i=0;i<cluster_order.size();i++){
        if(process_data.at(cluster_order.at(i)).scan_id == scan_num){
            current_order.push_back(cluster_order.at(i));
        }
    }
    // remove the cluster that only contains past data
    for(std::vector< std::vector<int> >::iterator it=final_cluster_order.begin();it != final_cluster_order.end();){
        int count = 0;
        for(int i=0;i<(*it).size();i++){
            if(process_data.at((*it).at(i)).scan_id != scan_num)
                count ++;
        }
        if(count == (*it).size()){
            it = final_cluster_order.erase(it);
        }
        else{
            it++;
        }
    }
    std::vector< std::vector<int> > temp_final_list;
    for(int i=0;i<final_cluster_order.size();i++){
        std::vector<int> temp;
        for(int j=0;j<final_cluster_order.at(i).size();j++){
            if(process_data.at(final_cluster_order.at(i).at(j)).scan_id == scan_num){
                temp.push_back(final_cluster_order.at(i).at(j));
            }
        }
        if(temp.size() != 0)
            temp_final_list.push_back(temp);
    }
    final_cluster_order = temp_final_list;
    return current_order;
}

void dbpda::cluster_center(std::vector< std::vector<cluster_point> > cluster_list){
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
        pt.cluster_flag = cluster_idx;
        pt.vel = sqrt(pt.x_v*pt.x_v+pt.y_v*pt.y_v+pt.z_v*pt.z_v);
        center.push_back(pt);

    }
    final_center = center;
}

std::vector<cluster_point> dbpda::get_center(void){
    return final_center;
}

std::vector<cluster_point> dbpda::get_history_points(void){
    std::vector<cluster_point> multi_frame_pts;
    if(frame_state != FRAME_STATE::first){
        for(int i=0;i<points_history.size()-1;i++){
            for(int j=0;j<points_history.at(i).size();j++){
                multi_frame_pts.push_back(points_history.at(i).at(j));
            }
        }
    }
    return multi_frame_pts;
}
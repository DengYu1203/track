#include "dbpda/dbpda.h"

dbpda::dbpda(double eps, int Nmin){
    dbpda_param.eps = eps;
    dbpda_param.Nmin = Nmin;
    history_frame_num = 4;
    points_history.clear();
    vel_scale = 1;
    input_cloud = pcl::PointCloud<pcl::PointXYZ>().makeShared();
    output_info = true;
    show_kf_info = true;
    use_kf = false;
}

dbpda::~dbpda(){}

std::vector< std::vector<cluster_point> > dbpda::cluster(std::vector<cluster_point> data){
    input_cloud->clear();
    points_history.push_back(data);
    if(points_history.size()>history_frame_num){
        points_history.erase(points_history.begin());
    }
    std::cout << "points_history size : " << points_history.size() << std::endl;
    switch (points_history.size())
    {
    case 1:
        frame_state = FRAME_STATE::first;
        break;
    case 2:
        frame_state = FRAME_STATE::second;
        break;
    default:
        frame_state = FRAME_STATE::more;
        break;
    }
    cluster_queue.clear();
    cluster_queue.shrink_to_fit();
    // ready to process the cluster points
    std::vector< cluster_point > process_data;
    if(frame_state == FRAME_STATE::more && use_kf){
        std::vector<cluster_point> encode_centers = time_encode();
        for(int i=0;i<encode_centers.size();i++){
            pcl::PointXYZ temp_pt;
            temp_pt.x = encode_centers.at(i).x;
            temp_pt.y = encode_centers.at(i).y;
            temp_pt.z = encode_centers.at(i).vel * vel_scale;
            input_cloud->points.push_back(temp_pt);
            process_data.push_back(encode_centers.at(i));
            // initial the reachable_dist
            dbpda_info cluster_pt;
            cluster_pt.reachable_dist = -1; // undefined distance
            cluster_queue.push_back(cluster_pt);
        }
        for(int i=0;i<data.size();i++){
            pcl::PointXYZ temp_pt;
            temp_pt.x = data.at(i).x;
            temp_pt.y = data.at(i).y;
            temp_pt.z = data.at(i).vel * vel_scale;
            input_cloud->points.push_back(temp_pt);
            process_data.push_back(data.at(i));
            // initial the reachable_dist
            dbpda_info cluster_pt;
            cluster_pt.reachable_dist = -1; // undefined distance
            cluster_queue.push_back(cluster_pt);
        }
    }
    else{
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
    }
    cluster_order.clear();
    cluster_order.shrink_to_fit();
    std::vector< std::vector<int> > final_cluster_order;
    
    // DBSCAN cluster
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
    
    std::vector< std::vector<cluster_point> > cluster_result;
    // init the kalman filter from the third frame
    if(frame_state == FRAME_STATE::second && use_kf){
        for(int i=0;i<final_cluster_order.size();i++){
            std::vector<cluster_point> temp_cluster_unit;
            for(int j=0;j<final_cluster_order.at(i).size();j++){
                temp_cluster_unit.push_back(process_data.at(final_cluster_order.at(i).at(j)));
            }
            cluster_result.push_back(temp_cluster_unit);
        }
        cluster_center(cluster_result);
        for(int j=0;j<cluster_result.size();j++){
            kalman_filter_init(cluster_result.at(j),j);
        }
    }
    else if(frame_state == FRAME_STATE::more && use_kf){
        for(int i=0;i<final_cluster_order.size();i++){
            std::vector<cluster_point> temp_cluster_unit;
            for(int j=0;j<final_cluster_order.at(i).size();j++){
                temp_cluster_unit.push_back(process_data.at(final_cluster_order.at(i).at(j)));
            }
            cluster_result.push_back(temp_cluster_unit);
        }
        cluster_center(cluster_result);
        std::cout << "before kalman association\n";
        kalman_association(process_data,final_cluster_order);
        std::cout << "after kalman association\n";
    }
    cluster_result.clear();
    std::vector<int> current_order = split_past(process_data, final_cluster_order);
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

double mahalanobis_distance(Eigen::VectorXd v, cluster_filter cluster_kf){
    cv::KalmanFilter kf = cluster_kf.kf;
    Eigen::MatrixXd S;
    cv::Mat Cov = kf.measurementMatrix * kf.errorCovPre * kf.measurementMatrix.t() + kf.measurementNoiseCov;
    cv::cv2eigen(Cov,S);
    return std::sqrt(v.transpose()*S.inverse()*cluster_kf.pred_state);
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
    // get the history data with time series
    std::vector< std::vector< std::vector<cluster_point> > > points_cluster_vec;    // contains the cluster result with all datas(need current data)
    std::vector< std::vector< std::vector<cluster_point> > > past_points_cluster_vec;   // the cluster result that only contain past data
    for(int i=0;i<final_cluster_order.size();i++){
        std::vector< std::vector<cluster_point> > temp_past_data(history_frame_num);
        int past_scan_id_count = 0;
        for(int j=0;j<final_cluster_order.at(i).size();j++){
            int past_timestamp = process_data.at(final_cluster_order.at(i).at(j)).scan_id;
            if(past_timestamp != scan_num){
                past_scan_id_count++;
            }
            temp_past_data.at(history_frame_num-1-(scan_num-past_timestamp)).push_back(process_data.at(final_cluster_order.at(i).at(j)));
        }
        if(past_scan_id_count != final_cluster_order.at(i).size()){
            points_cluster_vec.push_back(temp_past_data);
        }
        else{
            past_points_cluster_vec.push_back(temp_past_data);
        }
    }
    points_cluster_vec.insert(points_cluster_vec.end(),past_points_cluster_vec.begin(),past_points_cluster_vec.end());
    history_points_with_cluster_order.clear();
    history_points_with_cluster_order = points_cluster_vec;

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
    std::vector<cluster_point> center;
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

std::vector< std::vector< std::vector<cluster_point> > > dbpda::get_history_points_with_cluster_order(void){
    return history_points_with_cluster_order;
}

void dbpda::set_parameter(double eps, int Nmin, int frames_num){
    dbpda_param.eps = eps;
    dbpda_param.Nmin = Nmin;
    history_frame_num = frames_num;
}

void dbpda::kalman_filter_init(std::vector<cluster_point> cluster_list, int init_index){
    cluster_filter cluster_kf;
    cv::KalmanFilter kf;
    kf.init(stateDim, measureDim,controlDim,CV_64F);
    // std::cout << "Init kf" << std::endl;
    kf.transitionMatrix = (cv::Mat_<double>(4,4) << 1,0,data_period,0,
                                                    0,1,0,data_period,
                                                    0,0,1,0,
                                                    0,0,0,1);
    cv::setIdentity(kf.measurementMatrix);
    cv::setIdentity(kf.processNoiseCov, cv::Scalar(0.01));
    // cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(0.099,0.002,0.028,0.028));
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(0.1));
    cv::Mat new_state;
    kf.statePost.at<double>(0) = final_center.at(init_index).x;
    kf.statePost.at<double>(1) = final_center.at(init_index).y;
    kf.statePost.at<double>(2) = final_center.at(init_index).x_v;
    kf.statePost.at<double>(3) = final_center.at(init_index).y_v;

    // update the cluster points to one cluster
    kalman_filter_update_clusterPt(cluster_list,kf);
    // std::cout << "Finish update kf" << std::endl;
    cv::Mat pred = kf.predict();
    Eigen::VectorXd init_pred(4);
    init_pred << pred.at<double>(0),
                 pred.at<double>(1),
                 pred.at<double>(2),
                 pred.at<double>(3);
    cluster_kf.kf = kf;
    cluster_kf.pred_state = init_pred;
    cluster_kf.lose_frame = cluster_kf.track_frame = 0;
    if(final_center.at(init_index).vel < 0.1)
        cluster_kf.motion = MOTION_STATE::stop;
    else
        cluster_kf.motion = MOTION_STATE::move;
    cluster_kf.cluster_state = CLUSTER_STATE::unstable;
    cluster_kf.cluster_id = cluster_count++;
    cluster_past.push_back(cluster_kf);
}

void dbpda::kalman_filter_update_clusterPt(std::vector<cluster_point> cluster_update, cv::KalmanFilter &kf){
    // std::cout << "In update kf func" << std::endl;
    for(int i=0;i<cluster_update.size();i++){
        cluster_point temp_cluster = cluster_update.at(i);
        double measurement[4] = {temp_cluster.x,temp_cluster.y,temp_cluster.x_v,temp_cluster.y_v};
        cv::Mat measureMat = cv::Mat(4,1,CV_64F,measurement);
        // double measurement[2] = {temp_cluster.x,temp_cluster.y};
        // cv::Mat measureMat = cv::Mat(2,1,CV_64F,measurement);
        cv::Mat estimateMat = kf.correct(measureMat);
    }
}

void dbpda::kalman_association(std::vector< cluster_point > process_data, std::vector< std::vector<int> > final_cluster_order){
    std::cout << "In association kf func\nCluster kf filter size: " << cluster_past.size() << std::endl;
    std::vector<int> hit_track(cluster_past.size(),0);
    for(int i=0;i<final_cluster_order.size();i++){
        int past_kalman_id = -1;
        std::vector<cluster_point> temp_list;
        // cluster_point past_kf;
        for(int j=0;j<final_cluster_order.at(i).size();j++){
            if(process_data.at(final_cluster_order.at(i).at(j)).cluster_flag != -1){
                past_kalman_id = process_data.at(final_cluster_order.at(i).at(j)).cluster_flag;
                // past_kf = process_data.at(final_cluster_order.at(i).at(j));
            }
            else{
                temp_list.push_back(process_data.at(final_cluster_order.at(i).at(j)));
            }
        }
        if(past_kalman_id != -1){
            int update_order;
            for(int k=0;k<cluster_past.size();k++){
                if(past_kalman_id == cluster_past.at(k).cluster_id){
                    hit_track.at(k)++;
                    update_order = k;
                    std::cout << "update kf with cluster_id:" << past_kalman_id << ", filter order: " << k << std::endl;
                }
            }
            // hit_track.at(past_kalman_id)++;
            kalman_filter_update_clusterPt(temp_list,cluster_past.at(update_order).kf);
            double current_v = Eigen::Vector2d(cluster_past.at(update_order).pred_state(2),
                                            cluster_past.at(update_order).pred_state(3)).norm();
            // update the kalman filter
            cv::Mat pred = cluster_past.at(update_order).kf.predict();
            Eigen::VectorXd kf_pred(4);
            kf_pred << pred.at<double>(0),
                       pred.at<double>(1),
                       pred.at<double>(2),
                       pred.at<double>(3);
            cluster_past.at(update_order).pred_state = kf_pred;
            double pred_v = Eigen::Vector2d(kf_pred(2),kf_pred(3)).norm();
            cluster_past.at(update_order).track_frame++;
            if(current_v > pred_v){
                if(cluster_past.at(update_order).motion == MOTION_STATE::move || cluster_past.at(update_order).motion == MOTION_STATE::slow_down){
                    cluster_past.at(update_order).motion = MOTION_STATE::slow_down;
                }
            }
            else if(pred_v < 0.1){
                cluster_past.at(update_order).motion = MOTION_STATE::stop;
            }
            else{
                cluster_past.at(update_order).motion = MOTION_STATE::move;
            }
            // std::cout << "Done the update\n";
        }
        // initialize the new kalman filter
        else{
            kalman_filter_init(temp_list,i);
        }
        
    }
    for(int i=0;i<hit_track.size();i++){
        if(hit_track.at(i)==0){
            cluster_past.at(i).lose_frame++;
        }
        else{
            cluster_past.at(i).lose_frame = 0;
        }
    }
    // delete the lose kalman filters
    for(std::vector< cluster_filter >::iterator it=cluster_past.begin();it != cluster_past.end();){
        if((*it).lose_frame >= lose_max){
            // std::cout << "delete kf\n";
            it = cluster_past.erase(it);
        }
        else{
            it++;
        }
    }

}

std::vector<cluster_point> dbpda::time_encode(){
    std::cout << "In time encode func" << std::endl;
    std::vector<cluster_point> encode_centers;
    for(int i=0;i<cluster_past.size();i++){
        cluster_point temp;
        temp.x = cluster_past.at(i).pred_state(0);
        temp.y = cluster_past.at(i).pred_state(1);
        temp.x_v = cluster_past.at(i).pred_state(2);
        temp.y_v = cluster_past.at(i).pred_state(3);
        temp.vel = Eigen::Vector2d(temp.x_v,temp.y_v).norm();
        temp.vistited = false;
        temp.scan_id = scan_num-1;
        temp.cluster_flag = cluster_past.at(i).cluster_id;  // assign the past cluster id
        encode_centers.push_back(temp);
        if(show_kf_info){
            std::cout << "============================================\n";
            std::cout << "Cluster KF @" << i << " with id:" << temp.cluster_flag << std::endl;
            std::cout << "Pos: (" << temp.x << ", " << temp.y
                      << "), Vel: " << temp.vel << "(" << temp.x_v << ", " << temp.y_v << ")\n";
        }
    }
    return encode_centers;
}
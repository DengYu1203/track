#include "dbtrack/dbtrack.h"

#define MOTION_OPT_MSG(msg) if(output_motion_eq_optimizer_msg) {std::cout << "[Motion eq Optimizer] " << msg << std::endl;}
#define CLUSTER_TRACK_MSG(msg) if (output_cluster_track_msg) {std::cout << "\033[34m" << msg << "\033[0m";}
#define CLUSTER_TRACK_MSG_BOLD(msg) if (output_cluster_track_msg) {std::cout << "\033[1;34m" << msg << "\033[0m";}
#define RLS_MSG(msg) if (RLS_msg_flag) {std::cout << "[RLS]" << msg;}
#define keep_cluster_id_frame 5
#define assign_cluster_id_weight 0.8  // decide the percentage of the unclustered points
#define noise_point_frame 2   // the point that is not clustered over this number would be removed from the points_history vec

dbtrack::dbtrack(double eps, int Nmin){
  dbtrack_param.eps = eps;
  dbtrack_param.eps_min = 1.0;
  dbtrack_param.eps_max = 2.0;
  dbtrack_param.Nmin = Nmin;
  dbtrack_param.time_threshold = 3;
  history_frame_num = 4;
  dt_threshold_vel = 0.0;
  points_history.clear();
  vel_scale = 1;
  input_cloud = pcl::PointCloud<pcl::PointXYZ>().makeShared();
  dynamic_dbscan_result.Nmin = 0;
  dynamic_dbscan_result.Vthresh = 0;
}

dbtrack::~dbtrack(){}

/* 
 * Input: The current radar points 
 * Output: The current radar cluster result
 */
std::vector< std::vector<cluster_point> > dbtrack::cluster(std::vector<cluster_point> data){
  cluster_queue.clear();
  cluster_queue.shrink_to_fit();
  // ready to process the cluster points
  pcl::PointCloud<pcl::PointXYZ>::Ptr current_pc = pcl::PointCloud<pcl::PointXYZ>().makeShared();
  int moving_pt_num = 0;
  Nmin_training_record.vel_slot.clear();
  Nmin_training_record.vel_slot.assign(vel_slot_size,0);
  for(auto pt:data){
    pcl::PointXYZ temp_pt;
    temp_pt.x = pt.x;
    temp_pt.y = pt.y;
    if(pt.vel>=0.2){
      moving_pt_num++;
    }
    int slot_num = int(pt.vel/5);
    slot_num = slot_num>(vel_slot_size-1) ? (vel_slot_size-1):slot_num;
    Nmin_training_record.vel_slot[slot_num]++;
    temp_pt.z = vel_function(pt.vel,scan_num-pt.scan_time);
    current_pc->points.push_back(temp_pt);
  }
  input_cloud_vec.push_back(current_pc);
  points_history.push_back(data);
  if(points_history.size()>history_frame_num){
    points_history.erase(points_history.begin());
    input_cloud_vec.erase(input_cloud_vec.begin());
  }
  std::vector< cluster_point > process_data;
  input_cloud->clear();
  for(int i=0;i<points_history.size();i++){
    process_data.insert(process_data.end(),points_history.at(i).begin(),points_history.at(i).end());
    *input_cloud += *input_cloud_vec.at(i);
  }
  dbtrack_info cluster_pt;
  cluster_pt.reachable_dist = -1; // undefined distance
  cluster_queue.assign(process_data.size(),cluster_pt);
  
  double CLUSTER_START, CLUSTER_END;
  if(use_dynamic_Nmin||use_dynamic_vel_threshold){
    Eigen::MatrixXd Nmin_input_vec_(7,1);
    Nmin_input_vec_ <<  double(Nmin_training_record.vel_slot[0])/data.size(),
                        double(Nmin_training_record.vel_slot[1])/data.size(),
                        double(Nmin_training_record.vel_slot[2])/data.size(),
                        double(Nmin_training_record.vel_slot[3])/data.size(),
                        double(Nmin_training_record.vel_slot[4])/data.size(),
                        double(Nmin_training_record.vel_slot[5])/data.size(),
                        data.size();
    if(use_dynamic_Nmin){
      CLUSTER_START = clock();
      std::cout << "Input vector for Nmin: " << Nmin_input_vec_.transpose() << std::endl;
      dbtrack_param.Nmin = MLP_classfication_Nmin_v1(Nmin_input_vec_);
      if(dbtrack_param.Nmin<2){
        dbtrack_param.Nmin = 2;
      }
      else if(dbtrack_param.Nmin>10){
        dbtrack_param.Nmin = 10;
      }
      dynamic_dbscan_result.Nmin = dbtrack_param.Nmin;
      CLUSTER_END = clock();
      std::cout << "\033[42mGet Nmin Parameters time: " << (CLUSTER_END - CLUSTER_START) / CLOCKS_PER_SEC << "s\033[0m" << std::endl;
      std::cout << "Use Dynamic Nmin: " << dbtrack_param.Nmin << std::endl;
    }
    if(use_dynamic_vel_threshold){
      MLP_vel_v1(Nmin_input_vec_);
      dynamic_dbscan_result.Vthresh = cluster_vel_threshold;
    }
  }
  CLUSTER_START = clock();
  std::vector<dbtrack_para> parameters_vec;
  if(use_dynamic_eps){
    std::cout<<"Use Dynamic Parameter eps\n";
    Eigen::MatrixXd input_mat_(4,process_data.size());
    int col = 0;
    for(auto pt:process_data){
      Eigen::MatrixXd input_vec_(4,1);
      input_vec_ << pt.vel, pt.r, double(scan_num-pt.scan_time)*data_period, 
                    double(moving_pt_num)/data.size();
      input_mat_.col(col++) << input_vec_;
    }
    Eigen::MatrixXd eps_mat = MLP_eps_v3(input_mat_);
    dbtrack_para temp_para = dbtrack_param;
    dynamic_dbscan_result.eps.clear();dynamic_dbscan_result.eps.shrink_to_fit();
    for(int i=0;i<eps_mat.cols();i++){
      temp_para.eps = eps_mat(0,i);
      if(i<data.size()){
        dynamic_dbscan_result.eps.push_back(eps_mat(0,i));
      }
      parameters_vec.push_back(temp_para);
    }
    

  }
  else{
    std::cout<<"Use Fixed Parameter eps "<<dbtrack_param.eps<<"\n";
    parameters_vec.assign(process_data.size(),dbtrack_param);
  }
  CLUSTER_END = clock();
  std::cout << "\033[42mGet Eps Parameters time: " << (CLUSTER_END - CLUSTER_START) / CLOCKS_PER_SEC << "s\033[0m" << std::endl;
  std::cout << "Nmin: "<<dbtrack_param.Nmin<<", cluster velocity: "<< cluster_vel_threshold<<std::endl;
  // Set the kd-tree input
  // std::cout<<"kdtree input cloud size: "<< input_cloud->points.size()<< std::endl;
  kdtree.setInputCloud(input_cloud);

  std::cout << "DBTRACK Cluster points size : " << data.size() << "/" << process_data.size() << "(" << points_history.size() << " frames)" << std::endl;
  std::vector< std::vector<int> > final_cluster_order;
    
  // DBSCAN cluster
  for(int i=0;i<process_data.size();i++){
    // check if the pt has been visited
    if(process_data.at(i).visited)
      continue;
    else if(process_data.at(i).noise_detect.size()>=noise_point_frame){
      process_data.at(i).visited = true;
      continue;
    }
    else
      process_data.at(i).visited = true;
    std::vector<int> temp_cluster;
    
    // find neighbor and get the core distance
    cluster_queue.at(i).neighbor_info = find_neighbors(process_data.at(i), &parameters_vec[i]);
    temp_cluster.push_back(i);
    // satisfy the Nmin neighbors (!= 0: find all cluster even the single point;>=Nmin : remove the noise)
    // if(cluster_queue.at(i).neighbor_info.pointIdxNKNSearch.size() != 0){
    if(cluster_queue.at(i).neighbor_info.pointIdxNKNSearch.size() >= dbtrack_param.Nmin){
      cluster_queue.at(i).core_dist = std::sqrt(*std::max_element(cluster_queue.at(i).neighbor_info.pointNKNSquaredDistance.begin(),
                                                cluster_queue.at(i).neighbor_info.pointNKNSquaredDistance.end()));
      if(process_data.at(i).noise_detect.size()==0)
        expand_neighbor(process_data, temp_cluster, i, &parameters_vec);
      // final_cluster_order.push_back(temp_cluster); // filter out the outlier that only contains one point in one cluster
    }
    else{
      cluster_queue.at(i).core_dist = -1;      // undefined distance (not a core point)
    }
    final_cluster_order.push_back(temp_cluster);    // push the cluster that contains only one point to the final cluster result
  }
    
  std::vector< std::vector<cluster_point> > cluster_result;

  split_past(process_data, final_cluster_order);
  cluster_result = current_final_cluster_vec;

  return cluster_result;
}

void dbtrack::Nmin_training_cluster(std::vector<std::vector<cluster_point>>GroundTruthCluster, std::vector<cluster_point> data){
  // ready to process the cluster points
  // Nmin_train = true;
  if(GroundTruthCluster.size()<1){
    return;
  }
  pcl::PointCloud<pcl::PointXYZ>::Ptr current_pc = pcl::PointCloud<pcl::PointXYZ>().makeShared();
  int moving_pt_num = 0;
  Nmin_training_record.vel_slot.clear();
  Nmin_training_record.vel_slot.assign(vel_slot_size,0);
  for(auto pt:data){
    pcl::PointXYZ temp_pt;
    temp_pt.x = pt.x;
    temp_pt.y = pt.y;
    if(pt.vel>=0.2){
      moving_pt_num++;
    }
    int slot_num = int(pt.vel/5);
    slot_num = slot_num>(vel_slot_size-1) ? (vel_slot_size-1):slot_num;
    Nmin_training_record.vel_slot[slot_num]++;
    temp_pt.z = vel_function(pt.vel,scan_num-pt.scan_time);
    current_pc->points.push_back(temp_pt);
  }
  input_cloud_vec.push_back(current_pc);
  points_history.push_back(data);
  if(points_history.size()>history_frame_num){
    points_history.erase(points_history.begin());
    input_cloud_vec.erase(input_cloud_vec.begin());
  }
  std::vector< cluster_point > process_data;
  input_cloud->clear();
  for(int i=0;i<points_history.size();i++){
    process_data.insert(process_data.end(),points_history.at(i).begin(),points_history.at(i).end());
    *input_cloud += *input_cloud_vec.at(i);
  }
  
  double CLUSTER_START, CLUSTER_END;
  CLUSTER_START = clock();
  std::cout<<"Use Dynamic Parameter eps\n";
  Eigen::MatrixXd input_mat_(4,process_data.size());
  int col = 0;
  for(auto pt:process_data){
    Eigen::MatrixXd input_vec_(4,1);
    input_vec_ << pt.vel, pt.r, double(scan_num-pt.scan_time)*data_period, 
                  double(moving_pt_num)/data.size();
    input_mat_.col(col++) << input_vec_;
  }
  Eigen::MatrixXd eps_mat = MLP_eps_v3(input_mat_);
  
  CLUSTER_END = clock();
  if(Nmin_train){
    Nmin_training_record.current_scan_points = data.size();
    Nmin_training_record.moving_obj_percentage = double(moving_pt_num)/data.size();
    Nmin_training_record.past_scan_points = process_data.size()-data.size();
  }
  std::cout << "\033[42mGet Eps Parameters time: " << (CLUSTER_END - CLUSTER_START) / CLOCKS_PER_SEC << "s\033[0m" << std::endl;
  // Set the kd-tree input
  kdtree.setInputCloud(input_cloud);
  std::cout << "DBTRACK Cluster points size : " << data.size() << "/" << process_data.size() << "(" << points_history.size() << " frames)" << std::endl;
  
  // train Nmin
  std::vector<double> f1_score_vec_Nmin_train, f1_score_vec_Nmin_train_2;
  for(int Nmin_iter=2;Nmin_iter<11;Nmin_iter++){
    for(auto &pt:process_data){
      pt.visited = false;
    }
    cluster_queue.clear();
    cluster_queue.shrink_to_fit();
    dbtrack_info cluster_pt;
    cluster_pt.reachable_dist = -1; // undefined distance
    cluster_queue.assign(process_data.size(),cluster_pt);
    std::vector< std::vector<int> > final_cluster_order;
    std::vector<dbtrack_para> parameters_vec;
    dbtrack_para temp_para = dbtrack_param;
    temp_para.Nmin = Nmin_iter;
    for(int i=0;i<eps_mat.cols();i++){
      temp_para.eps = eps_mat(0,i);
      parameters_vec.push_back(temp_para);
    }
    // DBSCAN cluster
    for(int i=0;i<process_data.size();i++){
      // check if the pt has been visited
      if(process_data.at(i).visited)
        continue;
      else if(process_data.at(i).noise_detect.size()>=noise_point_frame){
        process_data.at(i).visited = true;
        continue;
      }
      else
        process_data.at(i).visited = true;
      std::vector<int> temp_cluster;
      
      // find neighbor and get the core distance
      cluster_queue.at(i).neighbor_info = find_neighbors(process_data.at(i), &parameters_vec[i]);
      temp_cluster.push_back(i);
      // satisfy the Nmin neighbors (!= 0: find all cluster even the single point;>=Nmin : remove the noise)
      if(cluster_queue.at(i).neighbor_info.pointIdxNKNSearch.size() >= parameters_vec[i].Nmin){
        cluster_queue.at(i).core_dist = std::sqrt(*std::max_element(cluster_queue.at(i).neighbor_info.pointNKNSquaredDistance.begin(),
                                                  cluster_queue.at(i).neighbor_info.pointNKNSquaredDistance.end()));
        if(process_data.at(i).noise_detect.size()==0)
          expand_neighbor(process_data, temp_cluster, i, &parameters_vec);
        final_cluster_order.push_back(temp_cluster); // filter out the outlier that only contains one point in one cluster
      }
      else{
        cluster_queue.at(i).core_dist = -1;      // undefined distance (not a core point)
      }
      // final_cluster_order.push_back(temp_cluster);    // push the cluster that contains only one point to the final cluster result
    }
      
    std::vector< std::vector<cluster_point> > cluster_result;
    // split_past(process_data, final_cluster_order);
    // cluster_result = current_final_cluster_vec;
    for(auto final_cluster_iter:final_cluster_order){
      std::vector<cluster_point> temp_cluster;
      for(auto cluster_idx_iter:final_cluster_iter){
        if(process_data.at(cluster_idx_iter).scan_time==scan_num){
          temp_cluster.push_back(process_data.at(cluster_idx_iter));
        }
      }
      if(temp_cluster.size()>0){
        cluster_result.push_back(temp_cluster);
      }
    }
    std::cout<<"\nNmin: "<<Nmin_iter<<std::endl;
    std::pair<double, double> f1_score_pair = f1_score(GroundTruthCluster,cluster_result);
    f1_score_vec_Nmin_train.push_back(f1_score_pair.first);
    f1_score_vec_Nmin_train_2.push_back(f1_score_pair.second);
  }
  std::vector<double> gt_cluster_vel;
  for(auto cluster_group:GroundTruthCluster){
    Eigen::Vector2d vel(0, 0);
    for(auto cluster_pt:cluster_group){
      vel += Eigen::Vector2d(cluster_pt.x_v, cluster_pt.y_v);
    }
    vel /= cluster_group.size();
    gt_cluster_vel.push_back(vel.norm());
  }
  // train the Nmin
  if(Nmin_train){
    string output_path_dir = "/home/user/deng/catkin_deng/src/track/DBSCAN_Train/radarScenes_v6_Nmin/";
    string output_path = output_path_dir + training_scene_name + ".csv";
    if(! boost::filesystem::exists(output_path_dir)){
      boost::filesystem::create_directories(output_path_dir);   
    }
    if(scan_num == 1){
      ofstream file;
      file.open(output_path, ios::out);
      file << "frame" << ",";
      file << "time stamp" << ",";
      file << "Best Nmin(iou>=0.3)" << ",";
      file << "Best Nmin(iou>=0.5)" << ",";
      for(int i=0;i<vel_slot_size;i++){
        file << "vel_slot " << std::fixed << setprecision(0) << i << ",";
      }
      file << "moving_obj_percentage" << ",";
      file << "current_scan_points" << ",";
      file << "past_scan_points" << ",";
      file << "GT min velocity" << ",";
      file << "f1_score(iou>=0.3)->Nmin=2" << ",";
      file << "f1_score(iou>=0.3)->Nmin=3" << ",";
      file << "f1_score(iou>=0.3)->Nmin=4" << ",";
      file << "f1_score(iou>=0.3)->Nmin=5" << ",";
      file << "f1_score(iou>=0.3)->Nmin=6" << ",";
      file << "f1_score(iou>=0.3)->Nmin=7" << ",";
      file << "f1_score(iou>=0.3)->Nmin=8" << ",";
      file << "f1_score(iou>=0.3)->Nmin=9" << ",";
      file << "f1_score(iou>=0.3)->Nmin=10" << ",";
      file << "f1_score(iou>=0.5)->Nmin=2" << ",";
      file << "f1_score(iou>=0.5)->Nmin=3" << ",";
      file << "f1_score(iou>=0.5)->Nmin=4" << ",";
      file << "f1_score(iou>=0.5)->Nmin=5" << ",";
      file << "f1_score(iou>=0.5)->Nmin=6" << ",";
      file << "f1_score(iou>=0.5)->Nmin=7" << ",";
      file << "f1_score(iou>=0.5)->Nmin=8" << ",";
      file << "f1_score(iou>=0.5)->Nmin=9" << ",";
      file << "f1_score(iou>=0.5)->Nmin=10" << std::endl;
      file.close();
    }
    int best_Nmin = std::max_element(f1_score_vec_Nmin_train.begin(),f1_score_vec_Nmin_train.end())-f1_score_vec_Nmin_train.begin();
    int best_Nmin_2 = std::max_element(f1_score_vec_Nmin_train_2.begin(),f1_score_vec_Nmin_train_2.end())-f1_score_vec_Nmin_train_2.begin();
    ofstream file;
    file.open(output_path, ios::out|ios::app);
    file << std::fixed << setprecision(0) << scan_num << ",";
    file << std::fixed << setprecision(3) << time_stamp << ",";
    file << std::fixed << setprecision(0) << best_Nmin+2 << ",";
    file << std::fixed << setprecision(0) << best_Nmin_2+2 << ",";
    for(int i=0;i<vel_slot_size;i++){
      file << std::fixed << setprecision(0) << Nmin_training_record.vel_slot[i] << ",";
    }
    file << std::fixed << setprecision(3) << Nmin_training_record.moving_obj_percentage << ",";
    file << std::fixed << setprecision(0) << Nmin_training_record.current_scan_points << ",";
    file << std::fixed << setprecision(0) << Nmin_training_record.past_scan_points << ",";
    file << std::fixed << setprecision(3) << *std::min_element(gt_cluster_vel.begin(),gt_cluster_vel.end()) << ",";
    for(auto train_score:f1_score_vec_Nmin_train){
      file << std::fixed << setprecision(3) << train_score << ",";
    }
    for(int score_idx=0;score_idx<f1_score_vec_Nmin_train_2.size();){
      file << std::fixed << setprecision(3) << f1_score_vec_Nmin_train_2[score_idx];
      if(++score_idx<f1_score_vec_Nmin_train_2.size()){
        file << ",";
      }
      else{
        file << std::endl;
      }
    }
    file.close();
  }
}

void dbtrack::cluster_track(std::vector< cluster_point > process_data, std::vector< std::vector<int> > final_cluster_order){
  int points_history_size = points_history.size();
  bool use_vote_func = true;
  final_cluster_idx.clear();
  final_cluster_idx.shrink_to_fit();
  std::map<int,std::vector<cluster_point*>> cluster_result_ptr;   // cluster index, cluster data list
  if(!use_dbscan_only){
    std::cout<<"Use VoteID\n";
    for(int cluster_idx=0;cluster_idx<final_cluster_order.size();cluster_idx++){
      std::map<int,std::vector<cluster_point*>> cluster_id_map;
      using pair_type_test = decltype(cluster_id_map)::value_type;
      // CLUSTER_TRACK_MSG("Searching process data" << std::endl);
      for(int element_idx=0;element_idx<final_cluster_order.at(cluster_idx).size();element_idx++){
        int data_idx = final_cluster_order.at(cluster_idx).at(element_idx); // get the cluster element index
        int time_diff = scan_num - process_data.at(data_idx).scan_time;     // calculate the time layer of the element
        points_history.at(points_history_size-time_diff-1).at(process_data.at(data_idx).id).current_cluster_order = cluster_idx;
        cluster_id_map[points_history.at(points_history_size-time_diff-1).at(process_data.at(data_idx).id).cluster_id].push_back(&points_history.at(points_history_size-time_diff-1).at(process_data.at(data_idx).id));
        // CLUSTER_TRACK_MSG("   idx:" << data_idx << ", time diff:" << time_diff << ", cluster id:" << points_history.at(points_history_size-time_diff-1).at(process_data.at(data_idx).id).cluster_id << std::endl);
        
      }
      // get the cluster id of this cluster
      int assign_cluster_id;
      if(use_vote_func){
        assign_cluster_id = voting_id(cluster_result_ptr, cluster_id_map, final_cluster_order.at(cluster_idx).size());
      }
      else{
        if(cluster_id_map[-1].size()>=final_cluster_order.at(cluster_idx).size()*assign_cluster_id_weight){
          assign_cluster_id = tracking_id_max++;
          // CLUSTER_TRACK_MSG("In cluster-tracking function-> new tracking id:" << assign_cluster_id << std::endl << std::endl);
        }
        else{
          std::map<int,std::vector<cluster_point*>> temp_map = cluster_id_map;
          temp_map.erase(-1); // erase the points that don't have cluster id
          auto max_cluster_id = std::max_element(temp_map.begin(),temp_map.end(),[] (const pair_type_test & p1, const pair_type_test & p2) {return p1.second.size() < p2.second.size();});
          assign_cluster_id = max_cluster_id->first;  // assign the cluster id with the maximum cluster id in this group
          CLUSTER_TRACK_MSG("\nIn cluster-tracking function-> find max cluster key:" << max_cluster_id->first << "(" << max_cluster_id->second.size() << ")");
        }
        // assign cluster id
        CLUSTER_TRACK_MSG_BOLD("\nClsuter idx: " << assign_cluster_id << "(" << final_cluster_order.at(cluster_idx).size() << ")\n");
        for(auto data_idx=cluster_id_map.begin();data_idx!=cluster_id_map.end();data_idx++){
          // if((*data_idx).first==assign_cluster_id)
          //   continue;
          for(auto it=(*data_idx).second.begin();it!=(*data_idx).second.end();it++){
            bool keep_id = false;

            // temporal method to solve the wrong cluster tracking keep
            if((*it)->tracking_history.size()>=keep_cluster_id_frame){
              std::map<int,int> find_max_freq;
              // using pair_two = decltype(find_max_freq)::value_type;
              for(auto vi=(*it)->tracking_history.begin();vi!=(*it)->tracking_history.end();vi++){
                find_max_freq[*vi]++;
              }
              auto max_info = std::max_element(find_max_freq.begin(),find_max_freq.end());
              int max_id = std::count((*it)->tracking_history.begin(),(*it)->tracking_history.end(),(*it)->tracking_history.back());
              // if(max_id>=(*it)->tracking_history.size()*0.6 && (*it)->tracking_history.back()!=assign_cluster_id){
              if(max_info->second>=(*it)->tracking_history.size()*0.6 && max_info->first!=assign_cluster_id){
                keep_id = true;
                (*it)->tracking_history.push_back(max_info->first);
                CLUSTER_TRACK_MSG_BOLD("Max history id: "<<max_info->first<<", size: "<<max_info->second<<std::endl);
                CLUSTER_TRACK_MSG_BOLD("Original assign id: "<<assign_cluster_id<<std::endl);
              }
            }
            // temporal method to solve the wrong cluster tracking keep

            if(keep_id){
              (*it)->cluster_id = (*it)->tracking_history.back();
              // (*it)->tracking_history.push_back((*it)->tracking_history.back()); // record the tracking history
              cluster_result_ptr[(*it)->tracking_history.back()].push_back(*it);
            }
            else{
              (*it)->cluster_id = assign_cluster_id;
              (*it)->tracking_history.push_back(assign_cluster_id); // record the tracking history
              cluster_result_ptr[assign_cluster_id].push_back(*it);
            }
            (*it)->current_cluster_order = cluster_idx;

            if((*it)->scan_time!=scan_num){
              CLUSTER_TRACK_MSG("   History list:");
              for(auto i:(*it)->tracking_history){
                CLUSTER_TRACK_MSG(" "<< i);
              }
            }
            else{
              CLUSTER_TRACK_MSG_BOLD("   History list:");
              for(auto i:(*it)->tracking_history){
                CLUSTER_TRACK_MSG_BOLD(" "<< i);
              }
            }
            CLUSTER_TRACK_MSG(std::endl);
          }
        }
      }
      final_cluster_idx.push_back(assign_cluster_id);
    }
  }
  // Use DBSCAN result without VoteID
  else{
    std::cout<<"No voteID!"<<std::endl;
    int cluster_id_for_dbscan_only = 0;
    for(int cluster_idx=0;cluster_idx<final_cluster_order.size();cluster_idx++){
      std::map<int,std::vector<cluster_point*>> cluster_id_map;
      for(int element_idx=0;element_idx<final_cluster_order.at(cluster_idx).size();element_idx++){
        int data_idx = final_cluster_order.at(cluster_idx).at(element_idx); // get the cluster element index
        int time_diff = scan_num - process_data.at(data_idx).scan_time;     // calculate the time layer of the element
        points_history.at(points_history_size-time_diff-1).at(process_data.at(data_idx).id).current_cluster_order = cluster_idx;
        cluster_id_map[points_history.at(points_history_size-time_diff-1).at(process_data.at(data_idx).id).cluster_id].push_back(&points_history.at(points_history_size-time_diff-1).at(process_data.at(data_idx).id));
      }
      // get the cluster id of this cluster
      int assign_cluster_id = cluster_id_for_dbscan_only++;
      // assign cluster id
      for(auto data_idx=cluster_id_map.begin();data_idx!=cluster_id_map.end();data_idx++){
        for(auto it=(*data_idx).second.begin();it!=(*data_idx).second.end();it++){
          (*it)->cluster_id = assign_cluster_id;
          (*it)->tracking_history.push_back(assign_cluster_id); // record the tracking history
          cluster_result_ptr[assign_cluster_id].push_back(*it);
          (*it)->current_cluster_order = cluster_idx;
        }
      }
      final_cluster_idx.push_back(assign_cluster_id);
    }
  }
  merge_cluster(cluster_result_ptr);
  MOTION_OPT_MSG("Ready to Optimizer!");
  motion_equation_optimizer(cluster_result_ptr);
  tracking_id_adjust();
  // use the RLS velocity estimation to get the tracker velocity
  if(use_RLS)
    updateTrackerVelocity(cluster_based_on_now, final_cluster_idx);
}

void dbtrack::merge_cluster(std::map<int,std::vector<cluster_point*>> cluster_result_ptr){
  cluster_based_on_now.clear();
  cluster_based_on_now.shrink_to_fit();
  final_cluster_idx.clear();
  final_cluster_idx.shrink_to_fit();
  current_final_cluster_vec.clear();
  current_final_cluster_vec.shrink_to_fit();
  cluster_track_vector_address.clear();cluster_track_vector_address.shrink_to_fit();
  ambiguous_cluster_vec.clear();ambiguous_cluster_vec.shrink_to_fit();
  for(auto idx = cluster_result_ptr.begin();idx != cluster_result_ptr.end();idx++){
    std::vector<cluster_point> temp_cluster_vec;
    std::vector<cluster_point*> diff_cluster_vec;
    std::vector<cluster_point> temp_current_vec;  // store the current cluster point list
    std::vector<cluster_point*> temp_current_vec_address;  // store the current cluster point list
    int cluster_order = idx->second.front()->current_cluster_order;
    std::map<int,int> cluster_order_map;
    for(auto data_ptr=idx->second.begin();data_ptr!=idx->second.end();data_ptr++){
      if(cluster_order_map.find((*data_ptr)->current_cluster_order)!=cluster_order_map.end()){
        cluster_order_map[(*data_ptr)->current_cluster_order]++;
      }
      else{
        cluster_order_map[(*data_ptr)->current_cluster_order]=1;
      }
      // temporal method to solve the wrong cluster tracking keep
      // if((*data_ptr)->current_cluster_order!=cluster_order){
      //   (*data_ptr)->cluster_state.push_back(1);
      // }
      // else{
      //   (*data_ptr)->cluster_state.clear();
      // }
      // temporal method to solve the wrong cluster tracking keep

      // if((*data_ptr)->cluster_state.size()>=(keep_cluster_id_frame-2)){
      //   diff_cluster_vec.push_back(*data_ptr);
      // }
      // else{
      temp_cluster_vec.push_back(**data_ptr);
      if((*data_ptr)->scan_time==scan_num){
        temp_current_vec.push_back(**data_ptr);
        temp_current_vec_address.push_back(*data_ptr);
      }
      // }
    }
    bool ambiguous = false;
    if(cluster_order_map.size()>1){
      int max_order = -1;
      ambiguous = true;
      // std::cout<<"Find split cluster case! Cluster ID: "<<idx->first<<"\n";
      for(auto order_iter:cluster_order_map){
        max_order = (max_order<order_iter.second?order_iter.second:max_order);
      }
      temp_cluster_vec.clear();
      temp_cluster_vec.shrink_to_fit();
      temp_current_vec.clear();
      temp_current_vec.shrink_to_fit();
      temp_current_vec_address.clear();temp_current_vec.shrink_to_fit();
      for(auto data_ptr=idx->second.begin();data_ptr!=idx->second.end();data_ptr++){
        if((*data_ptr)->current_cluster_order!=max_order){
          (*data_ptr)->cluster_state.push_back(1);
        }
        // else{
        //   (*data_ptr)->cluster_state.clear();
        // }
        // split 2 clusters which is wrong merged for 0.1s(0.05 for radarscenes)
        // if((*data_ptr)->cluster_state.size()*data_period>=0.05){
        if((*data_ptr)->cluster_state.size()*data_period>=0.1){
          diff_cluster_vec.push_back(*data_ptr);
        }
        else{
          temp_cluster_vec.push_back(**data_ptr);
          if((*data_ptr)->scan_time==scan_num){
            temp_current_vec.push_back(**data_ptr);
            temp_current_vec_address.push_back(*data_ptr);
          }
        }
      }
    }
    // if(temp_current_vec.size()!=0){
    if(temp_current_vec.size()>1){
      cluster_based_on_now.push_back(temp_cluster_vec);
      final_cluster_idx.push_back(idx->first);
      if(ambiguous)
        ambiguous_cluster_vec.push_back(idx->first);
      current_final_cluster_vec.push_back(temp_current_vec);
      cluster_track_vector_address.push_back(temp_current_vec_address);
    }
    else if(points_history.size()==history_frame_num && temp_current_vec_address.size()==1){
      temp_current_vec_address.at(0)->noise_detect.push_back(1);
    }
    temp_cluster_vec.clear();
    temp_current_vec.clear();
    temp_current_vec_address.clear();
    for(auto it=diff_cluster_vec.begin();it!=diff_cluster_vec.end();it++){
      (*it)->tracking_history.pop_back(); // remove the wrong cluster id from the history vec
      (*it)->cluster_id = tracking_id_max;
      (*it)->cluster_state.clear();
      (*it)->tracking_history.push_back((*it)->cluster_id);
      temp_cluster_vec.push_back(**it);
      if((*it)->scan_time==scan_num){
        temp_current_vec.push_back(**it);
        temp_current_vec_address.push_back(*it);
      }
    }
    if(diff_cluster_vec.size()>0){
      // if(temp_current_vec.size()!=0){
      // temp_cluster_vec.clear();
      // temp_current_vec.clear();
      // temp_current_vec_address.clear();
      // for(auto it=diff_cluster_vec.begin();it!=diff_cluster_vec.end();it++){
      //   (*it)->tracking_history.pop_back(); // remove the wrong cluster id from the history vec
      //   (*it)->cluster_id = tracking_id_max;
      //   (*it)->cluster_state.clear();
      //   (*it)->tracking_history.push_back((*it)->cluster_id);
      //   temp_cluster_vec.push_back(**it);
      //   if((*it)->scan_time==scan_num){
      //     temp_current_vec.push_back(**it);
      //     temp_current_vec_address.push_back(*it);
      //   }
      // }
      if(temp_current_vec.size()>1){
        cluster_based_on_now.push_back(temp_cluster_vec);
        final_cluster_idx.push_back(tracking_id_max);
        if(ambiguous)
          ambiguous_cluster_vec.push_back(tracking_id_max);
        current_final_cluster_vec.push_back(temp_current_vec);
        cluster_track_vector_address.push_back(temp_current_vec_address);
      }
      else if(points_history.size()==history_frame_num && temp_current_vec_address.size()==1){
        temp_current_vec_address.at(0)->noise_detect.push_back(1);
      }
      tracking_id_max++;
    }
    
  }
  cluster_center(current_final_cluster_vec);
}

void dbtrack::motion_equation_optimizer(std::map<int,std::vector<cluster_point*>> cluster_result_ptr){
  /* Optimizer parameter
   * Use the iterator num to limit the gradient size
   */
  double x_bias = 0;
  double y_bias = 0;
  for(auto cluster_idx=cluster_result_ptr.begin();cluster_idx!=cluster_result_ptr.end();cluster_idx++){
    int n =cluster_idx->second.size();
    if(n<2){
      // MOTION_OPT_MSG("One Point Only!!");
      // MOTION_OPT_MSG("Point:("<<process_cluster.at(cluster_idx).at(0).x<<","<<process_cluster.at(cluster_idx).at(0).y<<"), t: "<<process_cluster.at(cluster_idx).at(0).scan_time << ", dt: " << history_frame_num - (scan_num - process_cluster.at(cluster_idx).at(0).scan_time));
      continue;
    }
    MOTION_OPT_MSG("\n-------------------------------------------");
    MOTION_OPT_MSG("Cluster index: "<< cluster_idx->first << "(point size="<<n<<", t="<< scan_num <<")");
    // close form regression
    int regression_order = 2;
    Eigen::VectorXd output_x(n),output_y(n);
    Eigen::MatrixXd t_mat(n,regression_order),x_mat(n,regression_order); // y = a1 x^2 + a2 x + a3
    int index = 0;
    for(auto it=cluster_idx->second.begin();it!=cluster_idx->second.end();it++){
      MOTION_OPT_MSG("Point:("<<(*it)->x<<","<<(*it)->y<<"), t: "<<(*it)->scan_time << ", dt: " << history_frame_num - (scan_num - (*it)->scan_time));
      output_x(index) = (*it)->x;
      output_y(index) = (*it)->y;
      for(int i=0;i<regression_order;i++){
        t_mat(index,i) = std::pow(history_frame_num - (scan_num - (*it)->scan_time),i);
        x_mat(index,i) = std::pow((*it)->x,i);
      }
      index++;
    }
    Eigen::VectorXd a(regression_order),b(regression_order),c(regression_order);
    a = (t_mat.transpose()*t_mat).inverse()*t_mat.transpose()*output_x;
    b = (t_mat.transpose()*t_mat).inverse()*t_mat.transpose()*output_y;
    c = (x_mat.transpose()*x_mat).inverse()*x_mat.transpose()*output_y;
    // MOTION_OPT_MSG("\nInput:\nX:\n"<<output_x<<"\nY:\n"<<output_y<<"\nT mat:\n"<<t_mat<<"\nOutput:\na:\n"<<a<<"\nb:\n"<<b<<"\nc:\n"<<c);
    double error = 0, error_c=0;
    std::vector<double> error_list;
    for(auto it=cluster_idx->second.begin();it!=cluster_idx->second.end();it++){
      int t =  history_frame_num - (scan_num - (*it)->scan_time);
      double pred_x = 0;
      double pred_y = 0;
      double pred_c = 0;
      for(int i=0;i<regression_order;i++){
        pred_x += std::pow(t,i) * a(i);
        pred_y += std::pow(t,i) * b(i);
        pred_c += std::pow((*it)->x,i) * c(i);
      }
      double error_c_ = std::fabs(pred_c-(*it)->y);
      error_list.push_back(error_c_);
      error_c += error_c_*error_c_;
      // if(error_list.back()>dbtrack_param.eps){
      //   outlier_vec.push_back(*it);
      // }
      error += std::sqrt((pred_x - ((*it)->x - x_bias))*(pred_x - ((*it)->x - x_bias)) + (pred_y - ((*it)->y - y_bias))*(pred_y - ((*it)->y - y_bias)));
    }
    double c_mean = std::accumulate(error_list.begin(),error_list.end(),0)/error_list.size();
    double c_std = std::sqrt(error_c/error_list.size() - c_mean*c_mean);
    std::vector<cluster_point*> outlier_vec;
    for(int i=0;i<cluster_idx->second.size();i++){
      if(std::fabs(error_list.at(i)-c_mean)>=2*c_std){
        outlier_vec.push_back(cluster_idx->second.at(i));
      }
    }
    // motion_eq.push_back(func_para);
    MOTION_OPT_MSG("error: "<< error/n<<", error_c: "<<std::sqrt(error_c)/n);
    // MOTION_OPT_MSG("X-t function: " << func_para.x1 << "*t^2 + " << func_para.x2 << "*t + " << func_para.x3);
    // MOTION_OPT_MSG("Y-t function: " << func_para.y1 << "*t^2 + " << func_para.y2 << "*t + " << func_para.y3);
    string function_para_str;
    for(int i=0;i<regression_order;i++){
      function_para_str += std::to_string(c(i)) + "*x^" + std::to_string(i);
    }
    MOTION_OPT_MSG("Y-X function: " << function_para_str);
    if(outlier_vec.size()>0){
      MOTION_OPT_MSG("Outlier:")
      for(auto it=outlier_vec.begin();it!=outlier_vec.end();it++){
        MOTION_OPT_MSG("\tPoint:("<<(*it)->x<<","<<(*it)->y<<"), t: "<<(*it)->scan_time);
        // for(int i=0;i<noise_point_frame;i++)
          (*it)->noise_detect.push_back(1);
      }
    }
  }
}

int dbtrack::voting_id(std::map<int,std::vector<cluster_point*>> &cluster_result_ptr, std::map<int,std::vector<cluster_point*>> cluster_id_map, int cluster_size){
  int assign_cluster_id = -1;
  bool output_cluster_track_msg_latch = output_cluster_track_msg;
  if(cluster_size==1)
    output_cluster_track_msg = false;
  std::map<int,std::vector<double>> voting_list;  // id, weight list
  int new_born_count = 0;
  for(auto data_idx=cluster_id_map.begin();data_idx!=cluster_id_map.end();data_idx++){
    // CLUSTER_TRACK_MSG_BOLD("Cluster ID Map -> "<<(*data_idx).first<<"\n");
    for(auto it=(*data_idx).second.begin();it!=(*data_idx).second.end();it++){
      if((*it)->tracking_history.size()>0){
        int count_ = std::count((*it)->tracking_history.begin(),(*it)->tracking_history.end(),(*it)->tracking_history.back());
        double weight_ = (double)count_/points_history.size();
        voting_list[(*it)->tracking_history.back()].push_back(weight_);
        // output msg
        if(output_cluster_track_msg){
          CLUSTER_TRACK_MSG_BOLD("Voting Cluster id: "<<(*it)->tracking_history.back()<<", weight: "<<weight_<<", count: "<<count_<<" @("<<(*it)->x<<", "<<(*it)->y<<") "<<" ->");
          if((*it)->scan_time!=scan_num){
            CLUSTER_TRACK_MSG("   History list:");
            for(auto i:(*it)->tracking_history){
              CLUSTER_TRACK_MSG(" "<< i);
            }
          }
          else{
            CLUSTER_TRACK_MSG_BOLD("   History list:");
            for(auto i:(*it)->tracking_history){
              CLUSTER_TRACK_MSG_BOLD(" "<< i);
            }
          }
          CLUSTER_TRACK_MSG(std::endl);
        }
      }
      else{
        new_born_count++;
      }
    }
  }
  
  CLUSTER_TRACK_MSG("------------------------------------\n");
  double max_voting_result = -1;
  int max_voting_size = -1;
  double voting_size_base;
  if(voting_list.size()>0){
    voting_size_base = (double)(cluster_size-new_born_count)/(voting_list.size()+0.5);
    CLUSTER_TRACK_MSG("Voting Size Base: "<<voting_size_base<<"\n");
  }
  std::vector<int> multiple_voting_;  // record the cluster id that voting size are larger than setting(10)
  for(auto vi=voting_list.begin();vi!=voting_list.end();vi++){
    int voting_size = (*vi).second.size();
    // if(voting_size>10){
    if(voting_size>=voting_size_base){
      multiple_voting_.push_back((*vi).first);
    }
    double voting_w = (double)std::accumulate((*vi).second.begin(),(*vi).second.end(),0.0)/voting_size;
    if(tracking_id_history_map.find(vi->first)!=tracking_id_history_map.end()){
      voting_w *= tracking_id_history_map[vi->first].size();
      CLUSTER_TRACK_MSG("Tracking hit: "<<tracking_id_history_map[vi->first].size()<<"-> ");
    }
    // double voting_w = (double)std::accumulate((*vi).second.begin(),(*vi).second.end(),0.0);
    if((max_voting_result<voting_w) || ((max_voting_result==voting_w) && (max_voting_size<voting_size))){
      assign_cluster_id = (*vi).first;
      max_voting_result = voting_w;
      max_voting_size = voting_size;
    }
    CLUSTER_TRACK_MSG("Voting id: "<<(*vi).first<<", weight: "<<voting_w<<", size: "<<voting_size<<std::endl);
  }
  if(multiple_voting_.size()<=1){
    CLUSTER_TRACK_MSG("After voting-> id: "<<assign_cluster_id<<", weight: "<<max_voting_result<<", size:"<<max_voting_size<<std::endl);
    if(max_voting_result==-1){
      assign_cluster_id = tracking_id_max++;
    }
    CLUSTER_TRACK_MSG("==============================================\n");
    // avoid assigning the same ID to cluster result
    if(cluster_result_ptr.find(assign_cluster_id)!=cluster_result_ptr.end()){
      assign_cluster_id = tracking_id_max++;
    }
    // assign cluster id
    CLUSTER_TRACK_MSG_BOLD("Clsuter idx: " << assign_cluster_id << "(old:" << cluster_size-new_born_count << "/new:" << new_born_count << ")\n\n");
    for(auto data_idx=cluster_id_map.begin();data_idx!=cluster_id_map.end();data_idx++){
      for(auto it=(*data_idx).second.begin();it!=(*data_idx).second.end();it++){
        (*it)->cluster_id = assign_cluster_id;
        (*it)->tracking_history.push_back(assign_cluster_id); // record the tracking history
        cluster_result_ptr[assign_cluster_id].push_back(*it);
        
        // if((*it)->scan_time!=scan_num){
        //   CLUSTER_TRACK_MSG("   History list:");
        //   for(auto i:(*it)->tracking_history){
        //     CLUSTER_TRACK_MSG(" "<< i);
        //   }
        // }
        // else{
        //   CLUSTER_TRACK_MSG_BOLD("   History list:");
        //   for(auto i:(*it)->tracking_history){
        //     CLUSTER_TRACK_MSG_BOLD(" "<< i);
        //   }
        // }
        // CLUSTER_TRACK_MSG(std::endl);
      }
    }
  }
  else{
    CLUSTER_TRACK_MSG_BOLD("\nFind multiple clusters!(old:" << cluster_size-new_born_count << "/new:" << new_born_count << ")\n");
    if(output_cluster_track_msg){
      CLUSTER_TRACK_MSG_BOLD("->");
      for(int i=0;i<multiple_voting_.size();i++){
        CLUSTER_TRACK_MSG_BOLD(" "<<multiple_voting_.at(i));
      }
      CLUSTER_TRACK_MSG_BOLD("\n\n");
    }
    std::vector<cluster_point *> child_pts;
    std::vector<cluster_point> voting_center(multiple_voting_.size());
    std::vector<int> voting_size(multiple_voting_.size());
    for(int i=0;i<multiple_voting_.size();i++){
      voting_center[i].x = voting_center[i].y = 0;
      voting_center[i].x_v = voting_center[i].y_v = 0;
      voting_size[i] = 0;
    }
    for(auto data_idx=cluster_id_map.begin();data_idx!=cluster_id_map.end();data_idx++){
      for(auto  it=(*data_idx).second.begin();it!=(*data_idx).second.end();it++){
        std::vector<int>::iterator cluster_iter;
        cluster_iter = std::find(multiple_voting_.begin(),multiple_voting_.end(),(*it)->cluster_id);
        if(cluster_iter!=multiple_voting_.end()){
          (*it)->tracking_history.push_back((*it)->cluster_id);
          cluster_result_ptr[(*it)->cluster_id].push_back(*it);
          int center_index = std::distance(multiple_voting_.begin(),cluster_iter);
          voting_center[center_index].x += (*it)->x;
          voting_center[center_index].y += (*it)->y;
          voting_center[center_index].x_v += (*it)->x_v;
          voting_center[center_index].y_v += (*it)->y_v;
          voting_size[center_index]++;
        }
        else{
          child_pts.push_back(*it);
        }
      }
    }
    // caluclate center
    pcl::KdTreeFLANN<pcl::PointXYZ> center_kdtree;
    pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud;
    center_cloud = pcl::PointCloud<pcl::PointXYZ>().makeShared();
    for(int i=0;i<multiple_voting_.size();i++){
      // voting_center[i].x /= voting_size[i];
      // voting_center[i].y /= voting_size[i];
      voting_center[i].x_v /= voting_size[i];
      voting_center[i].y_v /= voting_size[i];
      voting_center[i].vel = std::sqrt(voting_center[i].x_v*voting_center[i].x_v+voting_center[i].y_v*voting_center[i].y_v);
      pcl::PointXYZ pt;
      pt.x = voting_center[i].x / voting_size[i];
      pt.y = voting_center[i].y / voting_size[i];
      pt.z = voting_center[i].vel;
      center_cloud->points.push_back(pt);
      // std::cout<<"("<<pt.x<<", "<<pt.y<<", "<<pt.z<<")\n";
    }
    // std::cout<<"center kdtree input cloud size: "<< center_cloud->size()<<", points: "<<center_cloud->points.size()<<std::endl;
    center_kdtree.setInputCloud(center_cloud);
    for(auto it=child_pts.begin();it!=child_pts.end();it++){
      pcl::PointXYZ search_pt;
      dbtrack_neighbor_info info;
      search_pt.x = (*it)->x;
      search_pt.y = (*it)->y;
      search_pt.z = (*it)->vel;
      if(center_kdtree.nearestKSearch(search_pt, 1,info.pointIdxNKNSearch,info.pointNKNSquaredDistance)>0){
        (*it)->cluster_id = multiple_voting_.at(info.pointIdxNKNSearch.at(0));
        (*it)->tracking_history.push_back((*it)->cluster_id); // record the tracking history
        cluster_result_ptr[(*it)->cluster_id].push_back(*it);
      }
    }
  }
  output_cluster_track_msg = output_cluster_track_msg_latch;
  return assign_cluster_id;
}

/*
 * update_pair:(cluster_order, update_tracker_id)
 */
void dbtrack::update_tracker_association(std::vector<std::pair<int,int>> update_pair){
  std::cout<<"Update cluster tracking pair\n";
  for(auto p:update_pair){
    for(auto pt:cluster_track_vector_address.at(p.first)){
      // std::cout<<"cluster idx: "<< final_cluster_idx[p.first]<<", tracking uuid: "<< pt->tracking_history.at(pt->tracking_history.size()-1)<<", update: "<<p.second<<std::endl;
      pt->tracking_history.at(pt->tracking_history.size()-1) = p.second;
      pt->cluster_id = p.second;
    }
  }
}

/*
 * Add the core points to improve the DBSCAN cluster
 * Record the tracking hit to provide the weights for the clsuter voting
 */
void dbtrack::tracking_id_adjust(){
  int count_idx = 0;
  for(auto id=final_cluster_idx.begin();id!=final_cluster_idx.end();id++,count_idx++){
    tracking_id_history_map[*id].push_back(scan_num);
    if(use_RLS){
      cluster_point tracker_core_obj;
      tracker_core_obj.x = final_center[count_idx].x;
      tracker_core_obj.y = final_center[count_idx].y;
      tracker_core_obj.z = final_center[count_idx].z;
      tracker_core_obj.x_v = final_center[count_idx].x_v;
      tracker_core_obj.y_v = final_center[count_idx].y_v;
      tracker_core_obj.z_v = 0;
      tracker_core_obj.vel = final_center[count_idx].vel;
      tracker_core_obj.cluster_id = *id;
      tracker_core_obj.scan_time = scan_num;
      tracker_core_obj.visited = false;
      Eigen::Vector3d temp_center;
      temp_center << tracker_core_obj.x , tracker_core_obj.y, tracker_core_obj.vel;
      if(tracker_cores.count(*id)==0){
        track_core init_core;
        init_core.point = tracker_core_obj;
        init_core.vel << 0,0;
        tracker_cores[*id] = init_core;
        // std::cout << "init tracker cores on index: " <<*id<<std::endl;
      }
      else{
        // double dist = (Eigen::Vector2d(tracker_core_obj.x,tracker_core_obj.y) - Eigen::Vector2d(tracker_cores[*id].point.x,tracker_cores[*id].point.y)).norm();
        // if(dist>tracker_cores[*id].vel.norm()){

        // }
        Eigen::Vector2d pred_v = Eigen::Vector2d(tracker_core_obj.x,tracker_core_obj.y) - Eigen::Vector2d(tracker_cores[*id].hitsory.back().x(),tracker_cores[*id].hitsory.back().y());
        pred_v /= data_period;
        // tracker_core_obj.x += pred_v(0);
        // tracker_core_obj.y += pred_v(1);
        // tracker_core_obj.x_v = pred_v(0);
        // tracker_core_obj.y_v = pred_v(1);
        tracker_cores[*id].point = tracker_core_obj;
        tracker_cores[*id].vel << tracker_core_obj.x_v,tracker_core_obj.y_v;
        // std::cout << "update tracker cores on index: " <<*id<<std::endl;

      }
      tracker_cores[*id].hitsory.push_back(temp_center);
    }
  }
  for(auto search_id=tracking_id_history_map.begin();search_id!=tracking_id_history_map.end();search_id++){
    int loss_frame_num = scan_num - search_id->second.back();
    if(loss_frame_num>10){
      search_id = tracking_id_history_map.erase(search_id);
      
    }
  }
}

std::vector<int> dbtrack::cluster_tracking_result(){
  return final_cluster_idx;
}

std::vector<int> dbtrack::ambiguous_cluster(){
  return ambiguous_cluster_vec;
}

dynamic_param dbtrack::dynamic_viz(){
  return dynamic_dbscan_result;
}

void dbtrack::updateTrackerVelocity(std::vector<std::vector<cluster_point>> cluster_result, std::vector<int> tracker_idx){
  if(cluster_result.size()!=tracker_idx.size()){
    RLS_MSG("\033[1;41m"<<"Cluster result is not comapred with tracker index"<<"\033[0m"<<std::endl);
    return;
  }
  RLS_MSG("RLS velocity estimation START:" << std::endl);
  RLS_MSG("============================================" << std::endl);
  auto idx = tracker_idx.begin();
  for(auto cluster_idx=cluster_result.begin();cluster_idx!=cluster_result.end();cluster_idx++,idx++){
    if(tracking_id_history_map[*idx].size()<=1){
      std::srand(time(NULL));
      double random_delta = (double) std::rand() / (RAND_MAX + 0.01);
      rls_est init_rls_core;
      init_rls_core.P = random_delta*init_rls_core.P.Identity();
      // init_rls_core.vel << tracker_cores[*idx].point.x_v, tracker_cores[*idx].point.y_v;
      init_rls_core.vel << 0, 0;
      tracker_vel_map[*idx] = init_rls_core;
      continue;
    }
    else if(tracker_cores[*idx].hitsory.size()==2){
      Eigen::Vector3d dist_diff = tracker_cores[*idx].hitsory.at(1) - tracker_cores[*idx].hitsory.at(0);
      // dist_diff /= 0.08;
      tracker_vel_map[*idx].vel << dist_diff(0) , dist_diff(1);
      // if(tracker_vel_map[*idx].vel.norm()>2)
      if(tracker_cores[*idx].point.vel>2){
        tracker_vel_map[*idx].vel = tracker_vel_map[*idx].vel*13;
      }
      else{
        tracker_vel_map[*idx].vel << tracker_cores[*idx].point.x_v, tracker_cores[*idx].point.y_v;
      }
      RLS_MSG("Init rls velocity:"<<tracker_vel_map[*idx].vel.transpose()<<std::endl);
      Eigen::Vector2d cluster_vel(tracker_cores[*idx].point.x_v, tracker_cores[*idx].point.y_v);
      RLS_vel(*cluster_idx,tracker_vel_map[*idx],cluster_vel);
    }
    else if(tracker_cores[*idx].hitsory.size()>2){
      Eigen::Vector2d ref_vel(tracker_cores[*idx].point.x_v, tracker_cores[*idx].point.y_v);
      if(tracker_cores[*idx].point.vel>2){
        int hit_number = tracker_cores[*idx].hitsory.size();
        Eigen::Vector3d dist_diff = tracker_cores[*idx].hitsory.back() - tracker_cores[*idx].hitsory.at(hit_number-2);
        // dist_diff = dist_diff*13;
        ref_vel << dist_diff(0), dist_diff(1);
        ref_vel = tracker_cores[*idx].vel.dot(ref_vel)*ref_vel.normalized();
      }
      RLS_vel(*cluster_idx,tracker_vel_map[*idx],ref_vel);
    }
    // else{
    //   continue;
    // }
    
    // RLS_vel(*cluster_idx,tracker_vel_map[*idx]);
    
    if(tracker_vel_map[*idx].vel.norm() <= 1)
      continue;
    RLS_MSG("Tracker ID " << *idx << ", hit " << tracking_id_history_map[*idx].size() << " frames" << std::endl);
    RLS_MSG("  Cluster Vel: " << tracker_cores[*idx].point.vel << "(" << tracker_cores[*idx].point.x_v << ", "
                                                                           << tracker_cores[*idx].point.y_v << ")" << std::endl);
    RLS_MSG("  RLS Vel: " << tracker_vel_map[*idx].vel.norm() << "(" << tracker_vel_map[*idx].vel.transpose() << ")" << std::endl);
    RLS_MSG("  RLS P:\n" << tracker_vel_map[*idx].P << std::endl);
    if(cluster_idx!=cluster_result.end())
      RLS_MSG("------------------------------------------" << std::endl);
  }
  RLS_MSG("============================================" << std::endl);

}


/*
 * Using the RLS to estimate the acutual velocity of one cluster
 */
void dbtrack::RLS_vel(std::vector<cluster_point> cluster_group, rls_est &rls_v, Eigen::Vector2d ref_vel){
  bool update_P_only = false;
  Eigen::Matrix2d P = rls_v.P;
  Eigen::Vector2d v = rls_v.vel;
  std::vector<cluster_point> shuffle_cluster_group = cluster_group;
  for(auto element=shuffle_cluster_group.begin();element!=shuffle_cluster_group.end();){
    if((scan_num-element->scan_time)>3){
      shuffle_cluster_group.erase(element);
    }
    else
      element++;
  }
  if(shuffle_cluster_group.size()<3){
    update_P_only = true;
  }
  int random_num = 6;
  Eigen::Matrix2d random_P[random_num];
  Eigen::Vector2d random_v[random_num];
  double reproject_error[random_num];
  
  for(int rand_i=0;rand_i<random_num;rand_i++){
    random_P[rand_i] = rls_v.P;
    random_v[rand_i] = rls_v.vel;
    std::random_shuffle(shuffle_cluster_group.begin(),shuffle_cluster_group.end());
    for(auto it=shuffle_cluster_group.begin();it!=shuffle_cluster_group.end();it++){
      double r_norm = it->vel * (it->vel_dir?1.0:-1.0);
      double theta = it->vel_ang;
      Eigen::Vector2d phi(std::cos(theta),std::sin(theta));
      // P = P - P*phi*phi.transpose()*P/(1+phi.transpose()*P*phi);
      random_v[rand_i] = random_v[rand_i] + random_P[rand_i]*phi*(r_norm-phi.transpose()*random_v[rand_i])/(1+phi.transpose()*random_P[rand_i]*phi);
      random_P[rand_i] = random_P[rand_i] - random_P[rand_i]*phi*phi.transpose()*random_P[rand_i]/(1+phi.transpose()*random_P[rand_i]*phi);
    }
    reproject_error[rand_i] = 0.0;
    for(auto it:shuffle_cluster_group){
      reproject_error[rand_i] += std::fabs(random_v[rand_i](0)*std::cos(it.vel_ang)+random_v[rand_i](1)*std::sin(it.vel_ang)-it.vel);
    }
  }
  auto min_iter = std::min_element(reproject_error,reproject_error+random_num);
  int min_idx = std::distance(reproject_error,min_iter);
  RLS_MSG("\nReproject Error:"<<*min_iter<<" at index:"<<min_idx<<std::endl);
  if(*min_iter>20){
    RLS_MSG("---------------------------------\n");
    RLS_MSG("Refresh RLS parameters"<<std::endl);
    std::srand(time(NULL));
    double random_delta = (double) std::rand() / (RAND_MAX + 0.01);
    for(int rand_i=0;rand_i<random_num;rand_i++){
      random_P[rand_i] = random_delta*P.Identity();
      random_v[rand_i] = ref_vel;
      std::random_shuffle(shuffle_cluster_group.begin(),shuffle_cluster_group.end());
      for(auto it=shuffle_cluster_group.begin();it!=shuffle_cluster_group.end();it++){
      double r_norm = it->vel * (it->vel_dir?1.0:-1.0);
      double theta = it->vel_ang;
      Eigen::Vector2d phi(std::cos(theta),std::sin(theta));
      // P = P - P*phi*phi.transpose()*P/(1+phi.transpose()*P*phi);
      random_v[rand_i] = random_v[rand_i] + random_P[rand_i]*phi*(r_norm-phi.transpose()*random_v[rand_i])/(1+phi.transpose()*random_P[rand_i]*phi);
      random_P[rand_i] = random_P[rand_i] - random_P[rand_i]*phi*phi.transpose()*random_P[rand_i]/(1+phi.transpose()*random_P[rand_i]*phi);
      }
      reproject_error[rand_i] = 0.0;
      for(auto it:shuffle_cluster_group){
        reproject_error[rand_i] += std::fabs(random_v[rand_i](0)*std::cos(it.vel_ang)+random_v[rand_i](1)*std::sin(it.vel_ang)-it.vel);
      }
    }
    min_iter = std::min_element(reproject_error,reproject_error+random_num);
    min_idx = std::distance(reproject_error,min_iter);
    RLS_MSG("\nReproject Error:"<<*min_iter<<" at index:"<<min_idx<<std::endl);
    RLS_MSG("---------------------------------\n");

  }
  
  P = random_P[min_idx];
  v = random_v[min_idx];
  
  
  // for(auto it=cluster_group.begin();it!=cluster_group.end();it++){
  //   if(it->scan_time!=scan_num)
  //     continue;
  //   Eigen::Vector2d r(it->x_v,it->y_v);
  //   double r_norm = r.norm() * (it->vel_dir?1.0:-1.0);
  //   double theta = it->vel_ang;
  //   Eigen::Vector2d phi(std::cos(theta),std::sin(theta));
  //   // P = P - P*phi*phi.transpose()*P/(1+phi.transpose()*P*phi);
  //   v = v + P*phi*(r_norm-phi.transpose()*v)/(1+phi.transpose()*P*phi);
  //   P = P - P*phi*phi.transpose()*P/(1+phi.transpose()*P*phi);
  // }
  
  rls_v.P = P;
  if(!update_P_only){
    rls_v.vel = v;
  }
}

double dbtrack::distance(cluster_point p1, cluster_point p2){
  Eigen::Vector3d d(p1.x-p2.x,p1.y-p2.y,(p1.vel-p2.vel)*vel_scale);
  return d.norm();
}

dbtrack_neighbor_info dbtrack::find_neighbors(cluster_point p, dbtrack_para *Optional_para){
  pcl::PointXYZ search_pt;
  search_pt.x = p.x;
  search_pt.y = p.y;
  search_pt.z = vel_function(p.vel, scan_num - p.scan_time);

  // find the eps radius neighbors
  dbtrack_neighbor_info info;
  if(Optional_para == NULL){
    // info.search_radius = dbtrack_param.eps;
    // info.search_radius = dbtrack_param.eps + p.vel*data_period; // consider the velocity into the distance search
    info.search_radius = dbtrack_param.eps;
    info.search_k = dbtrack_param.Nmin;
  }
  else{
    // info.search_radius = Optional_para->eps + p.vel*data_period;
    info.search_radius = Optional_para->eps;
    info.search_k = Optional_para->Nmin;
  }
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

void dbtrack::expand_neighbor(std::vector< cluster_point > &process_data, std::vector<int> &temp_cluster, int core_index, std::vector<dbtrack_para> *Optional_para){
  for(int i=0;i<cluster_queue.at(core_index).neighbor_info.pointIdxRadiusSearch.size();i++){
    int neighbor_index = cluster_queue.at(core_index).neighbor_info.pointIdxRadiusSearch.at(i);
    if(process_data.at(neighbor_index).visited)
      continue;
    // set the time stamp threshold
    // else if(std::abs(process_data.at(core_index).scan_time-process_data.at(neighbor_index).scan_time)>dbtrack_param.time_threshold)
    else if(std::abs(process_data.at(core_index).scan_time-process_data.at(neighbor_index).scan_time)*data_period>0.3)
      continue;
    else if(process_data.at(neighbor_index).noise_detect.size()>=noise_point_frame)
      continue;
    else
      process_data.at(neighbor_index).visited = true;
    temp_cluster.push_back(neighbor_index);
    if(Optional_para!=NULL)
      cluster_queue.at(neighbor_index).neighbor_info = find_neighbors(process_data.at(neighbor_index), &Optional_para->at(neighbor_index));
    else
      cluster_queue.at(neighbor_index).neighbor_info = find_neighbors(process_data.at(neighbor_index));
    // if(cluster_queue.at(neighbor_index).neighbor_info.pointIdxNKNSearch.size() != 0){
    if(cluster_queue.at(neighbor_index).neighbor_info.pointIdxNKNSearch.size() >= dbtrack_param.Nmin){
      cluster_queue.at(neighbor_index).core_dist = std::sqrt(*std::max_element(cluster_queue.at(neighbor_index).neighbor_info.pointNKNSquaredDistance.begin(),
                                                cluster_queue.at(neighbor_index).neighbor_info.pointNKNSquaredDistance.end()));
      if(process_data.at(neighbor_index).noise_detect.size()==0)
        expand_neighbor(process_data, temp_cluster, neighbor_index, Optional_para);
    }
    else{
        cluster_queue.at(neighbor_index).core_dist = -1;      // undefined distance (not a core point)
    }
  }
}

void dbtrack::split_past(std::vector< cluster_point > process_data, std::vector< std::vector<int> > &final_cluster_order){
  // get the history data with time series
  std::vector< std::vector< std::vector<cluster_point> > > points_cluster_vec;    // contains the cluster result with all datas(need current data)
  std::vector< std::vector< std::vector<cluster_point> > > past_points_cluster_vec;   // the cluster result that only contain past data
  
  // remove the cluster that only contains past data
  std::vector< std::vector<int> > temp_final_list;  // store the current points only
  std::vector< std::vector<int> > temp_final_vec = final_cluster_order;  // temp vec
  for(std::vector< std::vector<int> >::iterator it=final_cluster_order.begin();it != final_cluster_order.end();){
    int count = 0;
    std::vector<int> temp;
    std::vector< std::vector<cluster_point> > temp_past_data(history_frame_num);    // store the cluster
    std::vector< cluster_point> temp_past_cluster;
    for(int i=0;i<(*it).size();i++){
      if(process_data.at((*it).at(i)).scan_time != scan_num)
        count ++;
      else{
        temp.push_back((*it).at(i));  // store the current cluster result that would be calculated center later
      }
      // vector that contains time series
      temp_past_data.at(history_frame_num-1-(scan_num-process_data.at((*it).at(i)).scan_time)).push_back(process_data.at((*it).at(i)));
      // vector that just record the cluster points
      temp_past_cluster.push_back(process_data.at((*it).at(i)));
    }
    if(temp.size() != 0)
      temp_final_list.push_back(temp);
    
    // the points in this cluster are all PAST data
    if(count == (*it).size()){
      past_points_cluster_vec.push_back(temp_past_data);
      for(int i=0;i<(*it).size();i++){
        int data_idx_in_process = (*it).at(i);
        int time_diff = scan_num - process_data.at(data_idx_in_process).scan_time;     // calculate the time layer of the element
        cluster_point *point_ptr = &points_history.at(points_history.size()-time_diff-1).at(process_data.at(data_idx_in_process).id);
        // double tracking_stability = (double)std::accumulate(point_ptr->tracking_history.begin(),point_ptr->tracking_history.end(),point_ptr->tracking_history.back())/point_ptr->tracking_history.size();
        // if(tracking_stability<0.9)
        // point_ptr->noise_detect.push_back(1);
      }
      it = final_cluster_order.erase(it);
    }
    else{
      it++;
      points_cluster_vec.push_back(temp_past_data); // contains current scan point
    }
  }
  points_cluster_vec.insert(points_cluster_vec.end(),past_points_cluster_vec.begin(),past_points_cluster_vec.end());
  history_points_with_cluster_order.clear();
  history_points_with_cluster_order = points_cluster_vec;
  
  // cluster_track(process_data,final_cluster_order);
  cluster_track(process_data,temp_final_vec);
  final_cluster_order = temp_final_list;
}

std::vector< std::vector<cluster_point> > dbtrack::cluster_with_past_now(){
  return cluster_based_on_now;
}

void dbtrack::cluster_center(std::vector< std::vector<cluster_point> > cluster_list){
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
    pt.cluster_id = cluster_idx;
    pt.vel = sqrt(pt.x_v*pt.x_v+pt.y_v*pt.y_v+pt.z_v*pt.z_v);
    center.push_back(pt);

  }
  final_center = center;
}

double dbtrack::vel_function(double vel, int frame_diff){
  // return vel_scale * vel * (dt_threshold_vel / history_frame_num * frame_diff + 1);
  return vel_scale * std::fabs(vel);
  // return vel_scale * vel;
}

std::vector<cluster_point> dbtrack::get_center(void){
  if(use_RLS){
    if(final_center.size()!=final_cluster_idx.size()){
      std::cout <<"\033[1;41m"<<"Cluster result is not comapred with tracker index"<<"\033[0m"<<std::endl;
      return final_center;
    }
    auto center_item = final_center.begin();
    for(auto id=final_cluster_idx.begin();id!=final_cluster_idx.end();id++,center_item++){
      center_item->x_v = tracker_vel_map[*id].vel(0);
      center_item->y_v = tracker_vel_map[*id].vel(1);
      center_item->z_v = 0;
      center_item->vel = tracker_vel_map[*id].vel.norm();
    }
  }
  return final_center;
}

std::vector< std::vector< std::vector<cluster_point> > > dbtrack::get_history_points_with_cluster_order(void){
  return history_points_with_cluster_order;
}

void dbtrack::set_parameter(double eps, double eps_min, double eps_max, int Nmin, int frames_num, double dt_weight, double v_threshold){
  dbtrack_param.eps = eps;
  dbtrack_param.eps_min = eps_min;
  dbtrack_param.eps_max = eps_max;
  dbtrack_param.Nmin = Nmin;
  history_frame_num = frames_num;
  dt_threshold_vel = dt_weight;
  cluster_vel_threshold = v_threshold;
}

void dbtrack::set_dynamic(bool dynamic_eps, bool dynamic_Nmin, bool dynamic_vel_thrshold){
  use_dynamic_eps = dynamic_eps; use_dynamic_Nmin = dynamic_Nmin; use_dynamic_vel_threshold = dynamic_vel_thrshold;
}

void dbtrack::set_output_info(bool cluster_track_msg, bool motion_eq_optimizer_msg, bool rls_msg){
  output_cluster_track_msg = cluster_track_msg;
  output_motion_eq_optimizer_msg = motion_eq_optimizer_msg;
  RLS_msg_flag = rls_msg;
}

void dbtrack::set_clustr_score_name(std::string score_file_name){
  if(score_file_name!=""){
    output_score_dir_name = score_file_name;
  }
}

void dbtrack::training_dbtrack_para(bool training_, std::string filepath){
  parameter_training = training_;
  parameter_path = filepath;
  check_path(parameter_path);
  dbscan_para_path = boost::filesystem::path(parameter_path.c_str())/boost::filesystem::path(para_dataload);
  
  if(parameter_training){
    // traing the parameters
    use_dbscan_training_para = false;
    std::cout << "Use default parameter setting for Cluster Algo\n";
    std::cout << "Start to training the PARAMETERS\n";
    trainingInit();
    // test save matrix function
    // Eigen::Matrix4d testMat;
    // testMat = testMat.Random();
    // boost::filesystem::path test_path = boost::filesystem::path(parameter_path.c_str())/boost::filesystem::path("test_mat.csv");
    // save_parameter(testMat,test_path.c_str(),false);
    // std::cout << "Test mat output:\n"<<testMat<<std::endl;

  }
  else if(!boost::filesystem::exists(dbscan_para_path)){
    // wanna use training parameters but no pre-train parameters file
    use_dbscan_training_para = false;
    std::cout << "Use default parameter setting for Cluster Algo\n";
  }
  else{
    // use the training parameters
    use_dbscan_training_para = true;
    std::cout << "Use learning parameter setting for Cluster Algo\n";
    dbscan_parameter_matrix = load_parameter(dbscan_para_path.c_str());
    std::cout << "Loading Matrix\n"<<dbscan_parameter_matrix<<std::endl;
  }
}

void dbtrack::get_training_name(std::string scene_name){
  training_scene_name = scene_name;
}

void dbtrack::save_parameter(Eigen::MatrixXd parameter_matrix, std::string filepath, bool appendFlag){
  const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
  ofstream file;
  if(appendFlag){
    file.open(filepath.c_str(), ios::out|ios::app);
    file << "\n";
  }
  else
    file.open(filepath.c_str(), ios::out);
  
  file << parameter_matrix.format(CSVFormat);
  file.close();
  // std::cout << "Save Data Path: " << filepath << std::endl;
}

Eigen::MatrixXd dbtrack::load_parameter(std::string filepath){
  vector<double> matrixEntries;
  ifstream matrixDataFile(filepath.c_str());
  // this variable is used to store the row of the matrix that contains commas 
  string matrixRowString;
  // this variable is used to store the matrix entry;
  string matrixEntry;
  // this variable is used to track the number of rows
  int matrixRowNumber = 0;

  while (getline(matrixDataFile, matrixRowString))
  {
    stringstream matrixRowStringStream(matrixRowString);
    while (getline(matrixRowStringStream, matrixEntry, ','))
    {
      matrixEntries.push_back(std::stod(matrixEntry));
    }
    matrixRowNumber++; //update the column numbers
  }
  return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
}

void dbtrack::check_path(std::string path){
  if(boost::filesystem::create_directories(path.c_str())){
    std::cout << "Create directories\nPath: " << path.c_str() << std::endl;
  }
}

void dbtrack::get_gt_cluster(std::vector< std::vector<cluster_point> > gt_cluster){
  if(gt_cluster.size()<1){
    return;
  }
  int gt_scan_num = gt_cluster.at(0).at(0).scan_time;
  int delta_scan_num = scan_num - gt_scan_num;
  while(delta_scan_num>=points_history.size()){
    gt_cluster.erase(gt_cluster.begin());
    gt_scan_num = gt_cluster.at(0).at(0).scan_time;
    delta_scan_num = scan_num - gt_scan_num;
  }
  std::cout << "DBSCAN scan num: " << scan_num << std::endl;
  std::cout << "gt_cluster scan num: " << gt_scan_num << std::endl;
  std::cout << "gt_cluster delta scan num: " << delta_scan_num << std::endl;
  for(auto &cluster_obj:gt_cluster){
    for(auto &pt:cluster_obj){
      Eigen::Vector3d clusterPoint(pt.x,pt.y,pt.vel);
      for(auto &history_pt:points_history.at(points_history.size()-delta_scan_num-1)){
        Eigen::Vector3d historyPoint(history_pt.x,history_pt.y,history_pt.vel);
        historyPoint -= clusterPoint;
        if(historyPoint.norm()==0){
          history_pt.tracking_id = pt.tracking_id;
          break;
        }
      }
    }
  }
  boost::filesystem::path radarInputPath = boost::filesystem::path(parameter_path.c_str())/boost::filesystem::path(training_scene_name);
  check_path(radarInputPath.c_str());
  // std::cout << "Radar input path: " << radarInputPath.c_str() << std::endl;
  for(auto history_vec:points_history){
    // the data has been recorded
    if(history_vec.at(0).scan_time<=radarInput_scanTime){
      continue;
    }
    radarInput_scanTime = history_vec.at(0).scan_time;
    bool firstWrite = true;
    radarInputPath = boost::filesystem::path(parameter_path.c_str())/boost::filesystem::path(training_scene_name)/boost::filesystem::path(std::to_string(radarInput_scanTime)+".csv");
    for(auto history_pt:history_vec){
      Eigen::VectorXd radarInputVec(9);
      radarInputVec << history_pt.x, history_pt.y, history_pt.scan_time,
                       history_pt.r, history_pt.vel, history_pt.vel_ang,
                       history_pt.x_v, history_pt.y_v, history_pt.tracking_id;
      save_parameter(radarInputVec.transpose().matrix(),radarInputPath.c_str(),!firstWrite);
      if(firstWrite)
        firstWrite = false;
      // std::cout << radarInputVec.transpose().matrix() << std::endl;
    }
  }
}

void dbtrack::trainingInit(){
  boost::filesystem::path(parameter_path.c_str());
  std::vector< std::pair<std::string, int> > trainingPathVec;
  for(auto filename:boost::filesystem::directory_iterator(parameter_path)){
    if(boost::filesystem::is_directory(filename)){
      // std::cout << filename.path() << std::endl;
      int scene_frames = std::count_if(boost::filesystem::directory_iterator(filename.path()),
                                       boost::filesystem::directory_iterator(),
                                       static_cast<bool(*)(const boost::filesystem::path&)>(boost::filesystem::is_regular_file) );
      trainingPathVec.push_back(std::pair<std::string,int>(filename.path().c_str(),scene_frames));
      std::cout << "file numbers: " << scene_frames << std::endl;
    }
  }
  std::vector<std::string> skipList;
  std::string skip_train_path = std::string(parameter_path.c_str()) + "/train_list.txt";
  if(!boost::filesystem::exists(skip_train_path)){
    ofstream skipTrainTxtfile;
    skipTrainTxtfile.open(skip_train_path,ios::out);
    skipTrainTxtfile.close();
  }
  else{
    std::ifstream skipTrainTxt(skip_train_path, std::ios::in);
    std::string s;
    while (std::getline(skipTrainTxt, s)) {
      std::cout << s << "\n";
      skipList.push_back(s);
    }
    skipTrainTxt.close();
  }

  for(auto train_pair:trainingPathVec){
    auto compare_index = std::find(skipList.begin(),skipList.end(),train_pair.first);
    if(compare_index!=skipList.end()){
      continue;
    }
    // read training data
    std::vector<int> files_vec;
    for(auto scene_file:boost::filesystem::directory_iterator(train_pair.first)){
      files_vec.push_back(std::stoi(scene_file.path().stem().c_str()));
    }
    std::sort(files_vec.begin(),files_vec.end());
    int count = 0;
    for(auto scene_index:files_vec){
      /* InputDataMat
       * Row: points, Col: 9 elements
       */
      Eigen::MatrixXd InputDataMat = load_parameter(train_pair.first+"/"+std::to_string(scene_index)+".csv");
      // std::cout << "Get file: " << train_pair.first+"/"+std::to_string(scene_index)+".csv\n"
      //           << "Input Matrix\n\tRow: " << InputDataMat.rows()
      //           << "\n\tCol:" << InputDataMat.cols() << std::endl;
      std::vector<cluster_point> scene_data;
      for(int r=0;r<InputDataMat.rows();r++){
        cluster_point pt;
        pt.x = InputDataMat(r,0);
        pt.y = InputDataMat(r,1);
        pt.scan_time = InputDataMat(r,2);
        pt.r = InputDataMat(r,3);
        pt.vel = InputDataMat(r,4);
        pt.vel_ang = InputDataMat(r,5);
        pt.x_v = InputDataMat(r,6);
        pt.y_v = InputDataMat(r,7);
        pt.tracking_id = InputDataMat(r,8);
        pt.z = pt.z_v = 0;
        pt.cluster_id = -1;
        pt.visited = false;
        pt.id = r;
        scene_data.push_back(pt);
      }
      scan_num = scene_index;
      count++;
      // std::cout << "trainingCluster Function at " << count << std::endl;
      if(count%10==0)
        trainingCluster(scene_data, true);
      else
        trainingCluster(scene_data, false);
    }

    // write out the training scene name
    ofstream skipTrainTxtfile;
    skipTrainTxtfile.open(skip_train_path,ios::out|ios::app);
    skipTrainTxtfile << train_pair.first;
    skipTrainTxtfile << "\n";
    skipTrainTxtfile.close();

  }

}

void dbtrack::trainingCluster(std::vector<cluster_point> data, bool skip){
  cluster_queue.clear();
  cluster_queue.shrink_to_fit();
  // ready to process the cluster points
  pcl::PointCloud<pcl::PointXYZ>::Ptr current_pc = pcl::PointCloud<pcl::PointXYZ>().makeShared();
  for(auto pt:data){
    pcl::PointXYZ temp_pt;
    temp_pt.x = pt.x;
    temp_pt.y = pt.y;
    temp_pt.z = vel_function(pt.vel,scan_num-pt.scan_time);
    current_pc->points.push_back(temp_pt);
  }
  input_cloud_vec.push_back(current_pc);
  points_history.push_back(data);
  if(points_history.size()>history_frame_num){
    points_history.erase(points_history.begin());
    input_cloud_vec.erase(input_cloud_vec.begin());
  }
  std::vector< cluster_point > process_data;
  std::vector<int> PointsHistoryNumVec; // record numbers per element in vec->points_history 
  input_cloud->clear();
  for(int i=0;i<points_history.size();i++){
    process_data.insert(process_data.end(),points_history.at(i).begin(),points_history.at(i).end());
    *input_cloud += *input_cloud_vec.at(i);
    PointsHistoryNumVec.push_back(points_history.at(i).size());
  }
  // Skip the DBSCAN cluster part and just store the input data
  if(skip)
    return;

  dbtrack_info cluster_pt;
  cluster_pt.reachable_dist = -1; // undefined distance
  cluster_queue.assign(process_data.size(),cluster_pt);

  // Set the kd-tree input
  kdtree.setInputCloud(input_cloud);

  std::cout << "DBTRACK Cluster points size : " << data.size() << "/" << process_data.size() << "(" << points_history.size() << " frames)" << std::endl;
  std::vector< std::vector<int> > final_cluster_order;
    
  // DBSCAN cluster
  for(int i=0;i<process_data.size();i++){
    // check if the pt has been visited
    if(process_data.at(i).visited)
      continue;
    else if(process_data.at(i).noise_detect.size()>=noise_point_frame){
      process_data.at(i).visited = true;
      continue;
    }
    else
      process_data.at(i).visited = true;
    std::vector<int> temp_cluster;
    
    // find neighbor and get the core distance
    cluster_queue.at(i).neighbor_info = find_neighbors(process_data.at(i));
    temp_cluster.push_back(i);
    // satisfy the Nmin neighbors (!= 0: find all cluster even the single point;>=Nmin : remove the noise)
    if(cluster_queue.at(i).neighbor_info.pointIdxNKNSearch.size() != 0){
      cluster_queue.at(i).core_dist = std::sqrt(*std::max_element(cluster_queue.at(i).neighbor_info.pointNKNSquaredDistance.begin(),
                                                cluster_queue.at(i).neighbor_info.pointNKNSquaredDistance.end()));
      if(process_data.at(i).noise_detect.size()==0)
        expand_neighbor(process_data, temp_cluster, i);
      // final_cluster_order.push_back(temp_cluster); // filter out the outlier that only contains one point in one cluster
    }
    else{
      cluster_queue.at(i).core_dist = -1;      // undefined distance (not a core point)
    }
    final_cluster_order.push_back(temp_cluster);    // push the cluster that contains only one point to the final cluster result
  }

  std::vector<dbtrack_para> trainParameterVec(process_data.size(),dbtrack_param);
  std::vector< std::vector<cluster_point*> > clusterVec;
  std::map< int,std::vector<cluster_point*> > trackMap;
  std::vector<std::vector<cluster_point>> dbscan_cluster;
  for(auto cluster_vec:final_cluster_order){
    std::vector<cluster_point*> tempVec;
    std::vector<cluster_point> tempCluster;
    for(auto cluster_pt:cluster_vec){
      tempVec.push_back(&process_data.at(cluster_pt));
      tempCluster.push_back(process_data.at(cluster_pt));
      trackMap[process_data.at(cluster_pt).tracking_id].push_back(&process_data.at(cluster_pt));
    }
    clusterVec.push_back(tempVec);
    dbscan_cluster.push_back(tempCluster);

  }
  if(trackMap.size()<2){
    return;
  }

  /* Optimize the training parameters
   * 
   */
  std::vector<std::vector<cluster_point>> gt_cluster;
  for(auto clusterList:trackMap){
    if(clusterList.first!=-1){
      std::vector<cluster_point> temp_cluster;
      for(auto pt:clusterList.second){
        temp_cluster.push_back(*pt);
      }
      gt_cluster.push_back(temp_cluster);
    }
  }

  double VMeasureScore = cluster_score(gt_cluster,dbscan_cluster);
  int epoch = 0;
  std::cout << setprecision(3);
  std::cout << "\nStart to training Optimize\n";
  std::cout << "Before Optimize -> V Measure Score: " << "\033[1;46m" << VMeasureScore << "\033[0m"  << std::endl;
  while(VMeasureScore<0.95&&epoch<20){
    VMeasureScore = trainingOptimize(process_data,trainParameterVec,clusterVec,trackMap);
    epoch++;
    // std::cout << "Epoch " << epoch << std::endl;
  }
  std::cout << setprecision(3);
  std::cout << "After Optimize! Epoch: " << epoch << ", V Measure Score: " << "\033[1;46m" << VMeasureScore << "\033[0m"  << std::endl << std::endl;
}

double dbtrack::trainingOptimize(std::vector< cluster_point > &process_data,
                               std::vector<dbtrack_para> &trainParameterVec,
                               std::vector< std::vector<cluster_point*> > &clusterVec,
                               std::map< int,std::vector<cluster_point*> > &trackMap){
  // Decide the parameters
  std::cout << setprecision(3);
  for(auto clusterGroup:clusterVec){
    std::map<int,std::vector<cluster_point*>> checkMap;
    for(auto clusterPt:clusterGroup){
      if(clusterPt->tracking_id==-1)
        continue;
      checkMap[clusterPt->tracking_id].push_back(clusterPt);
    }
    if(checkMap.size()<1){
      // no gt
      continue;
    }
    else if(checkMap.size()==1){
      if(checkMap.begin()->second.size()==trackMap[checkMap.begin()->first].size()){
        // perfect match
        // for(auto pt:checkMap.at(0)){
        //   int index = std::distance(process_data.begin(),std::find(process_data.begin(),process_data.end(),pt));
        //   trainParameterVec.at(index) = dbtrack_param;
        // }
      }
      else{
        // over seg(one gt obj is splited to more than one cluster)
        for(auto pt:checkMap.begin()->second){
          int index = std::distance(process_data.begin(),std::find(process_data.begin(),process_data.end(),*pt));
          trainParameterVec.at(index).eps += 0.2;
          // std::cout << "OverSeg at point "<<index<<", eps: "<<trainParameterVec.at(index).eps<<"\n";
        }
      }
    }
    else{
      // under seg(more than one gt obj in this cluster)
      for(auto pt:checkMap.begin()->second){
          int index = std::distance(process_data.begin(),std::find(process_data.begin(),process_data.end(),*pt));
          trainParameterVec.at(index).eps -= 0.2;
          // std::cout << "UnderSeg at point "<<index<<", eps: "<<trainParameterVec.at(index).eps<<"\n";
        }
    }
  }
  // reset the visited flag
  for(auto &pt:process_data){
    pt.visited = false;
  }
  std::vector<std::vector<cluster_point>> gt_cluster, dbscan_cluster;
  for(auto clusterList:trackMap){
    if(clusterList.first!=-1){
      std::vector<cluster_point> temp_cluster;
      for(auto pt:clusterList.second){
        temp_cluster.push_back(*pt);
      }
      gt_cluster.push_back(temp_cluster);
    }
  }
  // get the training result
  std::vector< std::vector<int> > final_cluster_order = trainingDBSCAN(process_data, trainParameterVec);
  

  // reset the vector
  clusterVec.clear();
  clusterVec.shrink_to_fit();
  trackMap.clear();
  for(auto cluster_vec:final_cluster_order){
    std::vector<cluster_point*> tempVec;
    std::vector<cluster_point> tempCluster;
    for(auto cluster_pt:cluster_vec){
      tempVec.push_back(&process_data.at(cluster_pt));
      tempCluster.push_back(process_data.at(cluster_pt));
      trackMap[process_data.at(cluster_pt).tracking_id].push_back(&process_data.at(cluster_pt));
    }
    clusterVec.push_back(tempVec);
    dbscan_cluster.push_back(tempCluster);
  }
  return cluster_score(gt_cluster,dbscan_cluster);
  
}

std::vector< std::vector<int> > dbtrack::trainingDBSCAN(std::vector< cluster_point > &process_data,
                             std::vector<dbtrack_para> &trainParameterVec){
  dbtrack_info cluster_pt;
  cluster_pt.reachable_dist = -1; // undefined distance
  cluster_queue.clear();
  cluster_queue.shrink_to_fit();
  cluster_queue.assign(process_data.size(),cluster_pt);
  std::vector< std::vector<int> > final_cluster_order;
  // DBSCAN cluster
  for(int i=0;i<process_data.size();i++){
    // check if the pt has been visited
    if(process_data.at(i).visited)
      continue;
    else if(process_data.at(i).noise_detect.size()>=noise_point_frame){
      process_data.at(i).visited = true;
      continue;
    }
    else
      process_data.at(i).visited = true;
    std::vector<int> temp_cluster;
    
    // find neighbor and get the core distance
    cluster_queue.at(i).neighbor_info = find_neighbors(process_data.at(i),&trainParameterVec.at(i));
    temp_cluster.push_back(i);
    // satisfy the Nmin neighbors (!= 0: find all cluster even the single point;>=Nmin : remove the noise)
    if(cluster_queue.at(i).neighbor_info.pointIdxNKNSearch.size() != 0){
      cluster_queue.at(i).core_dist = std::sqrt(*std::max_element(cluster_queue.at(i).neighbor_info.pointNKNSquaredDistance.begin(),
                                                cluster_queue.at(i).neighbor_info.pointNKNSquaredDistance.end()));
      if(process_data.at(i).noise_detect.size()==0)
        expand_neighbor(process_data, temp_cluster, i);
      // final_cluster_order.push_back(temp_cluster); // filter out the outlier that only contains one point in one cluster
    }
    else{
      cluster_queue.at(i).core_dist = -1;      // undefined distance (not a core point)
    }
    final_cluster_order.push_back(temp_cluster);    // push the cluster that contains only one point to the final cluster result
  }
  return final_cluster_order;
}

double dbtrack::cluster_score(std::vector<std::vector<cluster_point>>GroundTruthCluster, std::vector<std::vector<cluster_point>>ResultCluster, double beta){
  if(GroundTruthCluster.size()==0){
    return 0;
  }
  Eigen::MatrixXd v_measure_mat(GroundTruthCluster.size(),ResultCluster.size()+1); // +1 for the noise cluster from dbscan
  Eigen::MatrixXd cluster_list_mat(1,ResultCluster.size()+1); // check the cluster performance
  v_measure_mat.setZero();
  cluster_list_mat.setZero();
  // std::cout << "Ground truth cluster:\n";
  for(int idx=0;idx<GroundTruthCluster.size();idx++){
    std::vector<kf_tracker_point> temp_gt = GroundTruthCluster.at(idx);
    for(int i=0;i<temp_gt.size();i++){
      bool find_cluster =false;

      for(int cluster_idx=0;cluster_idx<ResultCluster.size()+1;cluster_idx++){
        cluster_list_mat(0,cluster_idx) = cluster_idx;
        if(cluster_idx == ResultCluster.size()){
          if(!find_cluster){
            v_measure_mat(idx,cluster_idx) += 1;
          }
          continue;
        }
        std::vector<kf_tracker_point> temp_dbcluster = ResultCluster.at(cluster_idx);
        for(int j=0;j<temp_dbcluster.size();j++){
          Eigen::Vector3d distance = Eigen::Vector3d(temp_gt.at(i).x,temp_gt.at(i).y,0) - Eigen::Vector3d(temp_dbcluster.at(j).x,temp_dbcluster.at(j).y,0);
          if(distance.norm() <= 0.001){
            v_measure_mat(idx,cluster_idx) += 1;
            find_cluster = true;
          }
        }
      }
    }
  }
  int n = v_measure_mat.sum();
  
  // v-measure score calculation
  Eigen::RowVectorXd row_sum = v_measure_mat.rowwise().sum();
  Eigen::RowVectorXd col_sum = v_measure_mat.colwise().sum();

  // std::cout << "Ground true (row): " << GroundTruthCluster.size() << " ,DBSCAN cluster (col):" << ResultCluster.size() << endl;
  // std::cout << "v_measure_mat sum = " << n;
  // std::cout << endl;

  clusterScore result;
  result.object_num = GroundTruthCluster.size();
  std::cout << setprecision(0);
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
          result.under_cluster ++;
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
  result.frame = scan_num;
  std::cout<<"*****************************************************\n";
  std::cout << setprecision(3);
  std::cout << "Homogeneity:\t" << result.homo << "\t-> ";
  std::cout << "H(C|k) = " << result.h_ck << ", H(C) = " << result.h_c << endl;
  std::cout << "Completeness:\t" << result.comp << "\t-> ";
  std::cout << "H(K|C) = " << result.h_kc << ", H(K) = " << result.h_k << endl;
  std::cout << "\033[1;46mV-measure score: " << result.v_measure_score*100 << "% \033[0m" << endl;
  std::cout << "Total object: " << result.object_num << endl;
  std::cout << "Correct object: " << result.good_cluster << endl;
  std::cout << "Over Seg in one object: " << result.multi_cluster << endl;
  std::cout << "Under Seg objects: " << result.under_cluster << endl;
  std::cout << "Bad object: " << result.no_cluster << endl;
  std::cout<<"*****************************************************\n" << endl;
  outputScore(result);
  return result.v_measure_score;
}

void dbtrack::f1_score_Vth_test(std::vector<std::vector<cluster_point>>GroundTruthCluster, std::vector<std::vector<cluster_point>>ResultCluster, double v_test){
  if(GroundTruthCluster.size()==0){
    std::cout<<"No ground truth to score!\n";
    return ;
  }
  double iou_1 = 0.3; // iou 0.3
  double iou_2 = 0.5; // iou 0.5
  std::cout << "Before removing stationary cluster, Objects' size: "<<ResultCluster.size()<<std::endl;
  std::cout << "Vthershold: "<<v_test <<std::endl;
  for(auto cluster_it=ResultCluster.begin();cluster_it!=ResultCluster.end();){
    Eigen::Vector2d vel(0, 0);
    for(auto pt:*cluster_it){
      vel += Eigen::Vector2d(pt.x_v,pt.y_v);
    }
    vel /= (*cluster_it).size();
    // 0.8->2.88km/h
    // 0.2
    if(vel.norm()<v_test){
      ResultCluster.erase(cluster_it);
      // std::cout<<"Cluster velocity: "<<vel.norm()<<"("<<vel.x()<<", "<<vel.y()<<")";
      // std::cout<<" with "<<(*cluster_it).size()<<" points -> erase!\n";
    }
    else if((*cluster_it).size()==1){
      ResultCluster.erase(cluster_it);
    }
    else{
      std::cout<<"Cluster velocity: "<<vel.norm()<<"("<<vel.x()<<", "<<vel.y()<<")";
      std::cout<<" with "<<(*cluster_it).size()<<" points\n";
      cluster_it++;
    }
  }
  std::cout << "After removing stationary cluster, Objects' size: "<<ResultCluster.size()<<std::endl;
  std::cout << "Groud Truth Objects' size: "<<GroundTruthCluster.size()<<std::endl;

  Eigen::MatrixXd TP_Mat(ResultCluster.size(),GroundTruthCluster.size());
  Eigen::MatrixXd FP_Mat(ResultCluster.size(),GroundTruthCluster.size());
  Eigen::MatrixXd GT_Mat(ResultCluster.size(),GroundTruthCluster.size());
  Eigen::MatrixXd IOU_Mat(ResultCluster.size(),GroundTruthCluster.size());
  TP_Mat.setZero();FP_Mat.setZero();GT_Mat.setZero();IOU_Mat.setZero();

  for(int gt_obj_idx=0;gt_obj_idx<GroundTruthCluster.size();gt_obj_idx++){
    GT_Mat.col(gt_obj_idx).setConstant(GroundTruthCluster.at(gt_obj_idx).size());
    for(int cluster_obj_idx=0;cluster_obj_idx<ResultCluster.size();cluster_obj_idx++){
      for(auto cluster_pt:ResultCluster.at(cluster_obj_idx)){
        Eigen::Vector2d cluster_pt_vec(cluster_pt.x,cluster_pt.y);
        bool match_pt = false;
        for(auto gt_pt:GroundTruthCluster.at(gt_obj_idx)){
          Eigen::Vector2d gt_pt_vec(gt_pt.x,gt_pt.y);
          // find points -> TP
          if((cluster_pt_vec-gt_pt_vec).norm()==0){
            // TP_Mat(cluster_obj_idx,gt_obj_idx)++;
            match_pt = true;
            break;
          }
        }
        match_pt ? TP_Mat(cluster_obj_idx,gt_obj_idx)++:FP_Mat(cluster_obj_idx,gt_obj_idx)++;
      }
    }
  }
  // std::cout<<"TP:\n"<<TP_Mat<<std::endl;
  // std::cout<<"FP:\n"<<FP_Mat<<std::endl;
  // std::cout<<"GT:\n"<<GT_Mat<<std::endl;
  for(int r = 0;r<IOU_Mat.rows();r++){
    for(int c = 0;c<IOU_Mat.cols();c++)
      IOU_Mat(r,c) = TP_Mat(r,c)/(GT_Mat(r,c)+FP_Mat(r,c));
  }
  // std::cout<<"IOU:\n"<<IOU_Mat<<std::endl;
  Eigen::MatrixXd match_Mat = Eigen::MatrixXd(ResultCluster.size(),GroundTruthCluster.size()).setConstant(1)-IOU_Mat;
  std::vector<std::vector<double>> matchMat;
  for(int c=0;c<match_Mat.cols();c++){
    std::vector<double> tempVec(match_Mat.col(c).data(),match_Mat.col(c).data()+match_Mat.col(c).size());
    matchMat.push_back(tempVec);
  }
  // std::cout<<"IOU cost:\n"<<match_Mat<<std::endl;
  std::vector<int> hungPair;
  HungarianAlgorithm hungAlgo;
  double cost = hungAlgo.Solve(matchMat,hungPair);
  F1Score IOU1_score, IOU2_score;
  
  IOU1_score.TP = IOU2_score.TP = hungPair.size();
  for(int c=0;c<hungPair.size();c++){
    std::cout<<"Pair: "<<c<<", "<<hungPair[c];
    if(hungPair[c]==-1){
      IOU1_score.TP--;IOU2_score.TP--;
      std::cout<<std::endl;
      continue;
    }
    std::cout<<" ->IOU:"<<IOU_Mat(hungPair[c],c)<<std::endl;
    if(IOU_Mat(hungPair[c],c)<iou_1){
      IOU1_score.TP--;IOU2_score.TP--;
    }
    else if(IOU_Mat(hungPair[c],c)<iou_2){
      IOU2_score.TP--;
    }
  }
  IOU1_score.FN = GroundTruthCluster.size() - IOU1_score.TP;
  IOU1_score.FP = ResultCluster.size() - IOU1_score.TP;
  IOU1_score.f1_score = (double)2*IOU1_score.TP/(2*IOU1_score.TP+IOU1_score.FP+IOU1_score.FN);
  IOU1_score.precision = (double)IOU1_score.TP/(IOU1_score.TP+IOU1_score.FP);
  IOU1_score.recall = (double)IOU1_score.TP/(IOU1_score.TP+IOU1_score.FN);
  
  IOU2_score.FN = GroundTruthCluster.size() - IOU2_score.TP;
  IOU2_score.FP = ResultCluster.size() - IOU2_score.TP;
  IOU2_score.f1_score = (double)2*IOU2_score.TP/(2*IOU2_score.TP+IOU2_score.FP+IOU2_score.FN);
  IOU2_score.precision = (double)IOU2_score.TP/(IOU2_score.TP+IOU2_score.FP);
  IOU2_score.recall = (double)IOU2_score.TP/(IOU2_score.TP+IOU2_score.FN);
  std::cout<<"===============================\n";
  std::cout << "\033[1;46mIOU>="<<iou_1<<"  F1-score: "<<IOU1_score.f1_score*100<< "% \033[0m"<<std::endl
            <<"TP: "<<IOU1_score.TP<<", FP: "<<IOU1_score.FP<<", FN: "<<IOU1_score.FN
            <<"\nPrecision: "<<IOU1_score.precision<<", Recall: "<<IOU1_score.recall<<"\n\n";
  std::cout << "\033[1;46mIOU>="<<iou_2<<"  F1-score: "<<IOU2_score.f1_score*100<< "% \033[0m"<<std::endl
            <<"TP: "<<IOU2_score.TP<<", FP: "<<IOU2_score.FP<<", FN: "<<IOU2_score.FN
            <<"\nPrecision: "<<IOU2_score.precision<<", Recall: "<<IOU2_score.recall<<"\n";
  std::cout<<"===============================\n";
  
  outputScore(IOU1_score,IOU2_score);
  return ;
}

std::pair<double, double> dbtrack::f1_score(std::vector<std::vector<cluster_point>>GroundTruthCluster, std::vector<std::vector<cluster_point>>ResultCluster){
  if(GroundTruthCluster.size()==0){
    std::cout<<"No ground truth to score!\n";
    return std::pair<double, double>(-1, -1);
  }
  std::vector<std::vector<cluster_point>>GroundTruthCluster_copy = GroundTruthCluster;
  std::vector<std::vector<cluster_point>>ResultCluster_copy = ResultCluster;
  double iou_1 = 0.3; // iou 0.3
  double iou_2 = 0.5; // iou 0.5
  std::cout << "Before removing stationary cluster, Objects' size: "<<ResultCluster.size()<<std::endl;
  for(auto cluster_it=ResultCluster.begin();cluster_it!=ResultCluster.end();){
    Eigen::Vector2d vel(0, 0);
    for(auto pt:*cluster_it){
      vel += Eigen::Vector2d(pt.x_v,pt.y_v);
    }
    vel /= (*cluster_it).size();
    // 0.8->2.88km/h
    // 0.2
    if(vel.norm()<cluster_vel_threshold){
      ResultCluster.erase(cluster_it);
      // std::cout<<"Cluster velocity: "<<vel.norm()<<"("<<vel.x()<<", "<<vel.y()<<")";
      // std::cout<<" with "<<(*cluster_it).size()<<" points -> erase!\n";
    }
    else if((*cluster_it).size()==1){
      ResultCluster.erase(cluster_it);
    }
    else{
      std::cout<<"Cluster velocity: "<<vel.norm()<<"("<<vel.x()<<", "<<vel.y()<<")";
      std::cout<<" with "<<(*cluster_it).size()<<" points\n";
      cluster_it++;
    }
  }
  std::cout << "After removing stationary cluster, Objects' size: "<<ResultCluster.size()<<std::endl;
  std::cout << "Groud Truth Objects' size: "<<GroundTruthCluster.size()<<std::endl;

  Eigen::MatrixXd TP_Mat(ResultCluster.size(),GroundTruthCluster.size());
  Eigen::MatrixXd FP_Mat(ResultCluster.size(),GroundTruthCluster.size());
  Eigen::MatrixXd GT_Mat(ResultCluster.size(),GroundTruthCluster.size());
  Eigen::MatrixXd IOU_Mat(ResultCluster.size(),GroundTruthCluster.size());
  TP_Mat.setZero();FP_Mat.setZero();GT_Mat.setZero();IOU_Mat.setZero();

  for(int gt_obj_idx=0;gt_obj_idx<GroundTruthCluster.size();gt_obj_idx++){
    GT_Mat.col(gt_obj_idx).setConstant(GroundTruthCluster.at(gt_obj_idx).size());
    for(int cluster_obj_idx=0;cluster_obj_idx<ResultCluster.size();cluster_obj_idx++){
      for(auto cluster_pt:ResultCluster.at(cluster_obj_idx)){
        Eigen::Vector2d cluster_pt_vec(cluster_pt.x,cluster_pt.y);
        bool match_pt = false;
        for(auto gt_pt:GroundTruthCluster.at(gt_obj_idx)){
          Eigen::Vector2d gt_pt_vec(gt_pt.x,gt_pt.y);
          // find points -> TP
          if((cluster_pt_vec-gt_pt_vec).norm()==0){
            // TP_Mat(cluster_obj_idx,gt_obj_idx)++;
            match_pt = true;
            break;
          }
        }
        match_pt ? TP_Mat(cluster_obj_idx,gt_obj_idx)++:FP_Mat(cluster_obj_idx,gt_obj_idx)++;
      }
    }
  }
  // std::cout<<"TP:\n"<<TP_Mat<<std::endl;
  // std::cout<<"FP:\n"<<FP_Mat<<std::endl;
  // std::cout<<"GT:\n"<<GT_Mat<<std::endl;
  for(int r = 0;r<IOU_Mat.rows();r++){
    for(int c = 0;c<IOU_Mat.cols();c++)
      IOU_Mat(r,c) = TP_Mat(r,c)/(GT_Mat(r,c)+FP_Mat(r,c));
  }
  // std::cout<<"IOU:\n"<<IOU_Mat<<std::endl;
  Eigen::MatrixXd match_Mat = Eigen::MatrixXd(ResultCluster.size(),GroundTruthCluster.size()).setConstant(1)-IOU_Mat;
  std::vector<std::vector<double>> matchMat;
  for(int c=0;c<match_Mat.cols();c++){
    std::vector<double> tempVec(match_Mat.col(c).data(),match_Mat.col(c).data()+match_Mat.col(c).size());
    matchMat.push_back(tempVec);
  }
  // std::cout<<"IOU cost:\n"<<match_Mat<<std::endl;
  std::vector<int> hungPair;
  HungarianAlgorithm hungAlgo;
  double cost = hungAlgo.Solve(matchMat,hungPair);
  F1Score IOU1_score, IOU2_score;
  
  IOU1_score.TP = IOU2_score.TP = hungPair.size();
  for(int c=0;c<hungPair.size();c++){
    std::cout<<"Pair: "<<c<<", "<<hungPair[c];
    if(hungPair[c]==-1){
      IOU1_score.TP--;IOU2_score.TP--;
      std::cout<<std::endl;
      continue;
    }
    std::cout<<" ->IOU:"<<IOU_Mat(hungPair[c],c)<<std::endl;
    if(IOU_Mat(hungPair[c],c)<iou_1){
      IOU1_score.TP--;IOU2_score.TP--;
    }
    else if(IOU_Mat(hungPair[c],c)<iou_2){
      IOU2_score.TP--;
    }
  }
  IOU1_score.FN = GroundTruthCluster.size() - IOU1_score.TP;
  IOU1_score.FP = ResultCluster.size() - IOU1_score.TP;
  IOU1_score.f1_score = (double)2*IOU1_score.TP/(2*IOU1_score.TP+IOU1_score.FP+IOU1_score.FN);
  IOU1_score.precision = (double)IOU1_score.TP/(IOU1_score.TP+IOU1_score.FP);
  IOU1_score.recall = (double)IOU1_score.TP/(IOU1_score.TP+IOU1_score.FN);
  
  IOU2_score.FN = GroundTruthCluster.size() - IOU2_score.TP;
  IOU2_score.FP = ResultCluster.size() - IOU2_score.TP;
  IOU2_score.f1_score = (double)2*IOU2_score.TP/(2*IOU2_score.TP+IOU2_score.FP+IOU2_score.FN);
  IOU2_score.precision = (double)IOU2_score.TP/(IOU2_score.TP+IOU2_score.FP);
  IOU2_score.recall = (double)IOU2_score.TP/(IOU2_score.TP+IOU2_score.FN);
  std::cout<<"===============================\n";
  std::cout << "\033[1;46mIOU>="<<iou_1<<"  F1-score: "<<IOU1_score.f1_score*100<< "% \033[0m"<<std::endl
            <<"TP: "<<IOU1_score.TP<<", FP: "<<IOU1_score.FP<<", FN: "<<IOU1_score.FN
            <<"\nPrecision: "<<IOU1_score.precision<<", Recall: "<<IOU1_score.recall<<"\n\n";
  std::cout << "\033[1;46mIOU>="<<iou_2<<"  F1-score: "<<IOU2_score.f1_score*100<< "% \033[0m"<<std::endl
            <<"TP: "<<IOU2_score.TP<<", FP: "<<IOU2_score.FP<<", FN: "<<IOU2_score.FN
            <<"\nPrecision: "<<IOU2_score.precision<<", Recall: "<<IOU2_score.recall<<"\n";
  std::cout<<"===============================\n";
  if(!Nmin_train){
    outputScore(IOU1_score,IOU2_score);
    if(Vthreshold_test_flag){
      std::string original_path = output_score_dir_name;
      std::stringstream ss(output_score_dir_name);
      std::vector<std::string> split_str_vec;
      std::string tok;
      while(std::getline(ss,tok,'_')){
        split_str_vec.push_back(tok);
      }
      for(int i=1;i<6;i++){
        double v_test = double(i)/10.0;
        output_score_dir_name = split_str_vec[0];
        output_score_dir_name[output_score_dir_name.size()-1]+=(i-1);
        for(int c=1;c<split_str_vec.size()-1;c++){
          output_score_dir_name+=("_"+split_str_vec[c]);
        }
        output_score_dir_name+="_p"+to_string(i);
        std::cout<<"Output dir name: "<<output_score_dir_name<<std::endl;
        f1_score_Vth_test(GroundTruthCluster_copy, ResultCluster_copy, v_test);
      }
      output_score_dir_name = original_path;
    }
  }
  return std::pair<double, double>(IOU1_score.f1_score, IOU2_score.f1_score);
}

void dbtrack::outputScore(F1Score f1_1, F1Score f1_2){
  string dir_path;
  if(training_scene_name.find("sequence")!=training_scene_name.npos){
    dir_path = "/home/user/deng/catkin_deng/src/track/cluster_score/radar_scenes/f1score_"+output_score_dir_name+"/";
  }
  else{
    dir_path = "/home/user/deng/catkin_deng/src/track/cluster_score/nuscenes/f1score_"+output_score_dir_name+"/";
  }
  string csv_name_with_para;
  stringstream eps_s,Nmin_s;
  eps_s << dbtrack_param.eps;
  Nmin_s << dbtrack_param.Nmin;
  if(!use_dynamic_eps){
    csv_name_with_para += "Eps-"+eps_s.str();
  }
  else{
    csv_name_with_para += "Eps-dynamic";
  }
  if(!use_dynamic_Nmin){
    csv_name_with_para += "_Nmin-"+Nmin_s.str()+"_";
  }
  else{
    csv_name_with_para += "_Nmin-dynamic_";
  }
  string csv_file_name = csv_name_with_para + training_scene_name + ".csv";
  string output_path = dir_path + csv_file_name;
  if(! boost::filesystem::exists(dir_path)){
    boost::filesystem::create_directories(dir_path);   
  }
  if(scan_num == 1){
    ofstream file;
    file.open(output_path, ios::out);
    file << "frame" << ",";
    file << "f1_score(iou>=0.3)" << ",";
    file << "precision" << ",";
    file << "recall" << ",";
    file << "TP" << ",";
    file << "FP" << ",";
    file << "FN" << ",";
    file << "f1_score(iou>=0.5)" << ",";
    file << "precision" << ",";
    file << "recall" << ",";
    file << "TP" << ",";
    file << "FP" << ",";
    file << "FN" << std::endl;
    file.close();
  }
  ofstream file;
  file.open(output_path, ios::out|ios::app);
  file << setprecision(0) << scan_num << ",";
  file << setprecision(3) << f1_1.f1_score << ",";
  file << setprecision(3) << f1_1.precision << ",";
  file << setprecision(3) << f1_1.recall << ",";
  file << setprecision(0) << f1_1.TP << ",";
  file << setprecision(0) << f1_1.FP << ",";
  file << setprecision(0) << f1_1.FN << ",";
  file << setprecision(3) << f1_2.f1_score << ",";
  file << setprecision(3) << f1_2.precision << ",";
  file << setprecision(3) << f1_2.recall << ",";
  file << setprecision(0) << f1_2.TP << ",";
  file << setprecision(0) << f1_2.FP << ",";
  file << setprecision(0) << f1_2.FN << std::endl;
  file.close();
  
  return;
}

void dbtrack::outputScore(clusterScore result, double beta){
  string dir_path;
  if(training_scene_name.find("sequence")!=training_scene_name.npos){
    dir_path = "/home/user/deng/catkin_deng/src/track/cluster_score/radar_scenes/vmeasure_"+output_score_dir_name+"/";
  }
  else{
    dir_path = "/home/user/deng/catkin_deng/src/track/cluster_score/nuscenes/vmeasure_"+output_score_dir_name+"/";
  }
  string csv_name_with_para;
  stringstream eps_s,Nmin_s;
  eps_s << dbtrack_param.eps;
  Nmin_s << dbtrack_param.Nmin;
  if(!use_dynamic_eps){
    csv_name_with_para += "Eps-"+eps_s.str();
  }
  else{
    csv_name_with_para += "Eps-dynamic";
  }
  if(!use_dynamic_Nmin){
    csv_name_with_para += "_Nmin-"+Nmin_s.str()+"_";
  }
  else{
    csv_name_with_para += "_Nmin-dynamic_";
  }
  string csv_file_name = csv_name_with_para + training_scene_name + ".csv";
  string output_path = dir_path + csv_file_name;
  if(! boost::filesystem::exists(dir_path)){
    boost::filesystem::create_directories(dir_path);   
  }
  if(scan_num == 1){
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
  file << 0 << ",";
  file << 0 << endl;
  file.close();
}

void dbtrack::train_radarScenes(std::map<std::string,std::vector<kf_tracker_point>> GroundTruthMap, int scan_points){
  string output_path_dir = "/home/user/deng/catkin_deng/src/track/DBSCAN_Train/radarScenes_v2/";
  string output_path = output_path_dir + training_scene_name;
  if(scan_num==1){
    // output csv
    if(! boost::filesystem::exists(output_path_dir)){
      boost::filesystem::create_directories(output_path_dir);   
    }
    ofstream file;
    file.open(output_path, ios::out);
    // main focus
    file << "Eps" << ",";
    file << "Nmin" << ",";
    file << "x" << ",";
    file << "y" << ",";
    file << "vel" << ",";
    file << "r" << ",";
    file << "dt" << ",";
    file << "Scan Num" << ",";
    // other information
    file << "x_v" << ",";
    file << "y_v" << ",";
    file << "rcs" << ",";
    file << "vel_ang" << ",";
    file << "vel_dir" << std::endl;
    file.close();
  }
  
  GTMap_vec.push_back(GroundTruthMap);
  if(GTMap_vec.size()>history_frame_num){
    GTMap_vec.erase(GTMap_vec.begin());
  }
  int moving_points_num = 0;
  std::map<std::string,std::vector<std::vector<kf_tracker_point>>> process_data;
  for(auto time_it:GTMap_vec){
    for(auto gtMap:time_it){
      process_data[gtMap.first].push_back(gtMap.second);
      moving_points_num += gtMap.second.size();
    }
  }
  std::cout << "\nstart to train!\n";
  for(auto track_group:process_data){
    std::string track_token = track_group.first;
    pcl::PointCloud<pcl::PointXYZ>::Ptr tracker_pc = pcl::PointCloud<pcl::PointXYZ>().makeShared();
    tracker_pc->clear();
    std::vector< cluster_point > data;
    if(track_group.second.back().at(0).scan_time!=scan_num){
      continue;
    }
    int current_scan_obj_num = track_group.second.back().size();
    if(track_group.second.size()>1)
      current_scan_obj_num++;
    for(auto same_time_cluster:track_group.second){
      pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_pc = pcl::PointCloud<pcl::PointXYZ>().makeShared();
      for(auto pt:same_time_cluster){
        data.push_back(pt);
        pcl::PointXYZ temp_pt;
        temp_pt.x = pt.x;
        temp_pt.y = pt.y;
        temp_pt.z = vel_function(pt.vel,scan_num-pt.scan_time);
        cluster_pc->points.push_back(temp_pt);
      }
      *tracker_pc += *cluster_pc;
    }
    kdtree.setInputCloud(tracker_pc);
    dbtrack_para trainParameters;
    trainParameters.eps = 1.1;
    trainParameters.Nmin = 2;
    
    std::vector< int > final_cluster_order;
    final_cluster_order.push_back(0);
    
    int iteration = 0;
    // Start to Train Parameters
    std::cout<<"Parameters Optimization\n";
    while(*std::max_element(final_cluster_order.begin(),final_cluster_order.end())<current_scan_obj_num && iteration<25){
      trainParameters.eps += 0.1;
      iteration++;
      cluster_queue.clear();
      final_cluster_order.clear();final_cluster_order.shrink_to_fit();
      std::vector< cluster_point > training_data = data;

      std::vector<dbtrack_para> parameters_vec;
      parameters_vec.assign(training_data.size(),trainParameters);
      dbtrack_info cluster_pt_info;
      cluster_pt_info.reachable_dist = -1;
      cluster_queue.assign(training_data.size(),cluster_pt_info);
      // DBSCAN
      for(int i=0;i<training_data.size();i++){
        if(training_data.at(i).visited)
          continue;
        else
          training_data.at(i).visited = true;
        std::vector<int> temp_cluster;
        // find neighbor and get the core distance
        cluster_queue.at(i).neighbor_info = find_neighbors(training_data.at(i),&parameters_vec.at(i));
        temp_cluster.push_back(i);
        // satisfy the Nmin neighbors (!= 0: find all cluster even the single point;>=Nmin : remove the noise)
        if(cluster_queue.at(i).neighbor_info.pointIdxNKNSearch.size() != 0){
          cluster_queue.at(i).core_dist = std::sqrt(*std::max_element(cluster_queue.at(i).neighbor_info.pointNKNSquaredDistance.begin(),
                                                    cluster_queue.at(i).neighbor_info.pointNKNSquaredDistance.end()));
          if(training_data.at(i).noise_detect.size()==0)
            expand_neighbor(training_data, temp_cluster, i, &parameters_vec);
        }
        else{
          cluster_queue.at(i).core_dist = -1;      // undefined distance (not a core point)
        }
        final_cluster_order.push_back(temp_cluster.size());
      }
      // std::cout<<"cluster result: "<<temp_cluster.size()<<"/GT: "<<data.size()<<std::endl;
    }
    std::cout << "Tracker Token: " << track_token << std::endl;
    std::cout << "Iteration: " << iteration << ", cluster: " << "cluster result: "<<*std::max_element(final_cluster_order.begin(),final_cluster_order.end())<<"/GT: "<<data.size() << "(" <<current_scan_obj_num << ")" << std::endl;
    std::cout << "Eps: " <<trainParameters.eps << std::endl;
    std::cout << "write to file: "<<output_path<<std::endl;
    std::cout << "------------------------------------------\n";
    // output csv
    ofstream file;
    file.open(output_path, ios::out|ios::app);
    for(auto pt:data){
      // main focus
      file << std::fixed << setprecision(2) << trainParameters.eps << ",";
      file << std::fixed << setprecision(0) << trainParameters.Nmin << ",";
      file << std::fixed << setprecision(3) << double(pt.x) << ",";
      file << std::fixed << setprecision(3) << double(pt.y) << ",";
      file << std::fixed << setprecision(3) << double(pt.vel) << ",";
      file << std::fixed << setprecision(2) << double(pt.r) << ",";
      file << std::fixed << setprecision(3) << (scan_num-pt.scan_time)*data_period << ",";
      file << std::fixed << setprecision(2) << moving_points_num/double(scan_points) << ",";
      // other information
      file << std::fixed << setprecision(2) << pt.x_v << ",";
      file << std::fixed << setprecision(2) << pt.y_v << ",";
      file << std::fixed << setprecision(2) << pt.rcs << ",";
      file << std::fixed << setprecision(2) << pt.vel_ang << ",";
      file << std::fixed << setprecision(2) << pt.vel_dir << std::endl;
    }
    file.close();
  }
}

void dbtrack::reset(){
  GTMap_vec.clear();GTMap_vec.shrink_to_fit();
  scan_num = 0;
  input_cloud_vec.clear();input_cloud_vec.shrink_to_fit();
  points_history.clear();points_history.shrink_to_fit();
  tracking_id_max = 0;
}

double sigmoid(double x){
  return 1.0/(1.0+std::exp(-x));
}

Eigen::MatrixXd dbtrack::MLP_eps_v3(Eigen::MatrixXd input){
  Eigen::MatrixXd node1_Weight(200,1);
  node1_Weight << -1.78548878e-05, -2.26799870e-05, -1.75120827e-05, -1.92100492e-05,
       -1.77951659e-05, -1.77142313e-05, -1.77395271e-05, -2.00186730e-05,
       -2.23582077e-05, -1.92729437e-05, -1.77107102e-05, -1.75434920e-05,
       -1.81578538e-05, -2.35243523e-05, -1.93613614e-05, -1.97074738e-05,
       -1.76710211e-05, -1.79118266e-05, -1.95254693e-05, -1.75496689e-05,
       -1.77498545e-05, -1.80541336e-05, -1.77485241e-05, -1.94764852e-05,
       -1.76426046e-05, -1.80214693e-05, -1.74863864e-05, -1.76104780e-05,
       -2.04749567e-05, -1.74851637e-05, -1.74452221e-05, -2.18457418e-05,
       -1.94743173e-05, -1.79553469e-05, -1.98025594e-05, -1.83183846e-05,
       -1.86965081e-05, -1.92977802e-05, -2.06937990e-05, -1.78230208e-05,
       -1.81120230e-05, -1.94502119e-05, -1.95249172e-05, -2.17999353e-05,
       -1.79293733e-05, -2.16923404e-05, -1.76296942e-05, -1.76689071e-05,
       -1.96188203e-05, -1.75796128e-05, -4.40563068e-05, -1.75931228e-05,
       -1.89755114e-05, -1.77659745e-05, -1.77431357e-05,  1.00158374e+00,
       -1.82523605e-05, -1.92710827e-05, -1.82203538e-05, -2.35091253e-01,
       -5.44626373e-02, -1.76462066e-05, -2.02850563e-05,  3.27078946e-01,
       -2.39942013e-05, -1.81837939e-05,  5.75096099e-01, -1.78731580e-05,
       -1.91270680e-05, -2.14349600e-05, -1.75626522e-05, -1.88753566e-05,
       -2.28724791e-05, -1.77046947e-05, -2.00245583e-05, -1.77605678e-05,
        4.30926832e-02, -5.96382134e-02, -1.79203681e-05, -1.76286243e-05,
       -1.96472743e-05, -1.97439727e-05, -2.10210569e-05, -1.77469914e-05,
       -1.77444065e-05, -1.81320040e-05, -1.81496089e-05, -1.75969208e-05,
       -1.79130994e-05, -1.77318757e-05, -1.79458563e-05, -1.75114412e-05,
       -1.91060136e-05, -1.76961308e-05, -1.74743528e-05, -1.77552019e-05,
       -1.77258610e-05, -1.75810141e-05, -1.79579157e-05, -2.13747864e-05,
       -1.74630291e-05, -1.77393136e-05, -1.83078112e-05, -1.75569085e-05,
       -1.82870886e-05, -1.96633833e-05, -1.77738503e-05, -1.77814356e-05,
       -1.98162676e-05, -2.03917698e-05, -1.74788700e-05, -1.80918036e-05,
       -1.77549700e-05, -1.77916612e-05, -2.51251631e-05, -1.81655912e-05,
       -1.77910343e-05,  5.17975298e-01, -1.77992791e-05, -1.77712945e-05,
       -2.20119986e-05, -1.74817897e-05, -1.76723667e-05, -1.92924213e-05,
       -1.95357658e-05, -1.81693546e-05, -1.76781879e-05, -1.92389934e-05,
       -1.78591168e-05,  1.26937168e-01, -1.77347754e-05, -2.79814666e-01,
       -2.01624228e-05, -1.77226157e-05, -1.79003057e-05, -1.75288359e-05,
       -1.91816903e-05, -2.00497201e-05, -1.76495079e-05, -1.91606570e-05,
       -1.78351553e-05, -2.18437025e-05, -1.91094481e-05, -1.94616366e-05,
       -1.92228150e-05, -1.76349204e-05, -2.04988541e-05, -1.78822638e-05,
       -1.79169363e-05, -1.76527502e-05, -2.00173652e-05, -1.74414216e-05,
       -1.80344997e-05, -2.28608076e-05,  1.72354745e-01, -1.75186873e-05,
       -1.79088550e-05, -1.76768483e-05, -2.09437918e-05, -1.75457454e-05,
       -2.18761823e-05, -1.07642795e-04, -9.33043797e-01, -2.51362149e-05,
       -1.77749580e-05, -1.98019156e-05, -2.58268167e-05, -9.02567527e-01,
       -1.90103019e-05, -2.07624533e-05, -1.80025099e-05, -1.77050378e-05,
       -1.90157076e-05, -1.77483810e-05, -2.43895503e-05, -1.75621434e-05,
       -2.52814614e-05, -1.99513575e-05, -1.75942930e-05, -1.77823691e-05,
       -1.97455846e-05, -1.76622211e-05, -1.76667297e-05, -1.98196582e-05,
       -2.15747119e-05, -1.78245588e-05, -1.76807073e-05,  7.54738570e-01,
       -1.76625804e-05, -1.81482012e-05, -1.85909818e-05, -1.79219460e-05,
       -2.10976434e-05, -1.80117600e-05, -1.90669575e-05, -1.96272122e-05,
       -2.21652121e-05, -1.93873682e-05, -1.94677189e-01, -1.74770530e-05;
  Eigen::MatrixXd node2_Weight(200,1);
  node2_Weight << -1.97216802e-04, -2.23690464e-04, -1.95276454e-04, -2.04803102e-04,
       -1.96879406e-04, -1.96421740e-04, -1.96564835e-04, -2.09269922e-04,
       -2.21968077e-04, -2.05152077e-04, -1.96401817e-04, -1.95454611e-04,
       -1.98924289e-04, -2.28184520e-04, -2.05642221e-04, -2.07555915e-04,
       -1.96177193e-04, -1.97538225e-04, -2.06550575e-04, -1.95489639e-04,
       -1.96623242e-04, -1.98340498e-04, -1.96615718e-04, -2.06279629e-04,
       -1.96016293e-04, -1.98156482e-04, -1.95130644e-04, -1.95834310e-04,
       -2.11771949e-04, -1.95123705e-04, -1.94896957e-04, -2.19213473e-04,
       -2.06267635e-04, -1.97783736e-04, -2.08080283e-04, -1.99826287e-04,
       -2.01943574e-04, -2.05289811e-04, -2.12967411e-04, -1.97036804e-04,
       -1.98666428e-04, -2.06134239e-04, -2.06547522e-04, -2.18966551e-04,
       -1.97637228e-04, -2.18386093e-04, -1.95943171e-04, -1.96165225e-04,
       -2.07066489e-04, -1.95659399e-04, -3.29645537e-04, -1.95735968e-04,
       -2.03499386e-04, -1.96714393e-04, -1.96585245e-04, -1.43462220e-01,
       -1.99455535e-04, -2.05141755e-04, -1.99275691e-04, -3.25730644e-01,
       -5.18637620e-01, -1.96036691e-04, -2.10732208e-04, -8.90290520e-01,
       -2.30669775e-04, -1.99070170e-04, -5.64339430e-02, -1.97319965e-04,
       -2.04342265e-04, -2.16994917e-04, -1.95563253e-04, -2.02941512e-04,
       -2.24718208e-04, -1.96367780e-04, -2.09302277e-04, -1.96683823e-04,
        4.99647867e-02,  4.76140169e-01, -1.97586421e-04, -1.95937110e-04,
       -2.07223630e-04, -2.07757264e-04, -2.14749731e-04, -1.96607050e-04,
       -1.96592432e-04, -1.98778867e-04, -1.98877912e-04, -1.95757492e-04,
       -1.97545407e-04, -1.96521557e-04, -1.97730209e-04, -1.95272814e-04,
       -2.04225265e-04, -1.96319318e-04, -1.95062344e-04, -1.96653481e-04,
       -1.96487533e-04, -1.95667341e-04, -1.97798223e-04, -2.16669126e-04,
       -1.94998063e-04, -1.96563627e-04, -1.99766934e-04, -1.95530688e-04,
       -1.99650587e-04, -2.07312570e-04, -1.96758919e-04, -1.96801800e-04,
       -2.08155832e-04, -2.11316759e-04, -1.95087984e-04, -1.98552616e-04,
       -1.96652170e-04, -1.96859598e-04, -2.36608917e-04, -1.98967808e-04,
       -1.96856055e-04, -7.05771392e-02, -1.96902652e-04, -1.96744470e-04,
       -2.20108712e-04, -1.95104555e-04, -1.96184810e-04, -2.05260096e-04,
       -2.06607507e-04, -1.98988973e-04, -1.96217763e-04, -2.04963734e-04,
       -1.97240683e-04, -2.51287853e-01, -1.96537959e-04,  1.13072440e-01,
       -2.10059576e-04, -1.96469175e-04, -1.97473208e-04, -1.95371489e-04,
       -2.04645662e-04, -2.09440582e-04, -1.96055386e-04, -2.04528858e-04,
       -1.97105354e-04, -2.19202483e-04, -2.04244353e-04, -2.06197466e-04,
       -2.04873955e-04, -1.95972772e-04, -2.11902634e-04, -1.97371371e-04,
       -1.97567057e-04, -1.96073746e-04, -2.09262732e-04, -1.94875376e-04,
       -1.98229899e-04, -2.24655947e-04, -3.17009498e-01, -1.95313922e-04,
       -1.97521456e-04, -1.96210180e-04, -2.14329503e-04, -1.95467390e-04,
       -2.19377500e-04, -6.05498929e-04,  3.15384399e-01, -2.36666664e-04,
       -1.96765182e-04, -2.08076734e-04, -2.40264439e-04, -2.00162870e-01,
       -2.03693011e-04, -2.13341848e-04, -1.98049637e-04, -1.96369721e-04,
       -2.03723089e-04, -1.96614909e-04, -2.32752703e-04, -1.95560369e-04,
       -2.37425091e-04, -2.08899691e-04, -1.95742600e-04, -1.96807076e-04,
       -2.07766155e-04, -1.96127372e-04, -1.96152898e-04, -2.08174516e-04,
       -2.17750761e-04, -1.97045494e-04, -1.96232024e-04,  1.15733705e-01,
       -1.96129406e-04, -1.98869993e-04, -2.01353707e-04, -1.97595324e-04,
       -2.15165922e-04, -1.98101769e-04, -2.04008149e-04, -2.07112840e-04,
       -2.20932371e-04, -2.05786290e-04,  7.11072667e-02, -1.95077671e-04;
  Eigen::MatrixXd node3_Weight(200,1);
  node3_Weight << -2.33508484e-07, -2.69027931e-07, -2.30932151e-07, -2.43617942e-07,
       -2.33060223e-07, -2.32452357e-07, -2.32642391e-07, -2.49596991e-07,
       -2.66697102e-07, -2.44084362e-07, -2.32425902e-07, -2.31168540e-07,
       -2.35778821e-07, -2.75121894e-07, -2.44739661e-07, -2.47300424e-07,
       -2.32127642e-07, -2.33935633e-07, -2.45954708e-07, -2.31215021e-07,
       -2.32719963e-07, -2.35002258e-07, -2.32709970e-07, -2.45592197e-07,
       -2.31914029e-07, -2.34757550e-07, -2.30738706e-07, -2.31672458e-07,
       -2.52954462e-07, -2.30729500e-07, -2.30428721e-07, -2.62974961e-07,
       -2.45576151e-07, -2.34261971e-07, -2.48002715e-07, -2.36979345e-07,
       -2.39800589e-07, -2.44268481e-07, -2.54560750e-07, -2.33269326e-07,
       -2.35435769e-07, -2.45397702e-07, -2.45950623e-07, -2.62641645e-07,
       -2.34067223e-07, -2.61858314e-07, -2.31816959e-07, -2.32111752e-07,
       -2.46645172e-07, -2.31440306e-07, -4.16720186e-07, -2.31541929e-07,
       -2.41876531e-07, -2.32841030e-07, -2.32669498e-07, -6.21223667e-02,
       -2.36485789e-07, -2.44070564e-07, -2.36246425e-07, -3.96053579e-01,
        1.35513891e-01, -2.31941109e-07, -2.51558510e-07,  8.26647723e-02,
       -2.78499473e-07, -2.35972927e-07, -1.16642382e+00, -2.33645569e-07,
       -2.43002197e-07, -2.59982167e-07, -2.31312709e-07, -2.41131875e-07,
       -2.70419987e-07, -2.32380703e-07, -2.49640370e-07, -2.32800426e-07,
       -3.60334854e+00, -6.86730330e-02, -2.33999692e-07, -2.31808915e-07,
       -2.46855530e-07, -2.47570062e-07, -2.56958065e-07, -2.32698458e-07,
       -2.32679043e-07, -2.35585348e-07, -2.35717117e-07, -2.31570496e-07,
       -2.33945179e-07, -2.32584915e-07, -2.34190817e-07, -2.30927322e-07,
       -2.42845902e-07, -2.32316352e-07, -2.30648100e-07, -2.32760126e-07,
       -2.32539730e-07, -2.31450847e-07, -2.34281230e-07, -2.59543062e-07,
       -2.30562830e-07, -2.32640788e-07, -2.36900323e-07, -2.31269494e-07,
       -2.36745430e-07, -2.46974601e-07, -2.32900174e-07, -2.32957132e-07,
       -2.48103919e-07, -2.52343199e-07, -2.30682113e-07, -2.35284379e-07,
       -2.32758385e-07, -2.33033910e-07, -2.86592382e-07, -2.35836724e-07,
       -2.33029203e-07,  7.37491796e-01, -2.33091104e-07, -2.32880981e-07,
       -2.64183894e-07, -2.30704096e-07, -2.32137756e-07, -2.44228758e-07,
       -2.46030890e-07, -2.35864886e-07, -2.32181508e-07, -2.43832618e-07,
       -2.33540218e-07, -1.16415739e+00, -2.32606698e-07,  8.74862056e-01,
       -2.50655984e-07, -2.32515350e-07, -2.33849222e-07, -2.31058246e-07,
       -2.43407557e-07, -2.49825810e-07, -2.31965927e-07, -2.43251487e-07,
       -2.33360402e-07, -2.62960124e-07, -2.42871399e-07, -2.45482282e-07,
       -2.43712631e-07, -2.31856255e-07, -2.53129992e-07, -2.33713883e-07,
       -2.33973954e-07, -2.31990301e-07, -2.49587351e-07, -2.30400096e-07,
       -2.34855177e-07, -2.70335630e-07,  2.90282982e-01, -2.30981863e-07,
       -2.33913346e-07, -2.32171440e-07, -2.56392569e-07, -2.31185497e-07,
       -2.63196409e-07, -8.26328970e-07, -7.50254162e-02, -2.86671217e-07,
       -2.32908492e-07, -2.47997961e-07, -2.91588193e-07,  2.05877911e-01,
       -2.42135056e-07, -2.55064141e-07, -2.34615481e-07, -2.32383281e-07,
       -2.42175219e-07, -2.32708896e-07, -2.81334356e-07, -2.31308882e-07,
       -2.87706848e-07, -2.49100685e-07, -2.31550731e-07, -2.32964141e-07,
       -2.47581968e-07, -2.32061496e-07, -2.32095386e-07, -2.48128950e-07,
       -2.61001282e-07, -2.33280871e-07, -2.32200443e-07, -2.28428417e-01,
       -2.32064197e-07, -2.35706582e-07, -2.39014151e-07, -2.34011525e-07,
       -2.57518290e-07, -2.34684798e-07, -2.42555901e-07, -2.46707218e-07,
       -2.65296807e-07, -2.44932320e-07,  5.33159533e-02, -2.30668432e-07;
  Eigen::MatrixXd node4_Weight(200,1);
  node4_Weight << -5.57876495e-07, -6.50584201e-07, -5.51177646e-07, -5.84198382e-07,
       -5.56710673e-07, -5.55129945e-07, -5.55624097e-07, -5.99791405e-07,
       -6.44482454e-07, -5.85414121e-07, -5.55061153e-07, -5.51792134e-07,
       -5.63782849e-07, -6.66547945e-07, -5.87122370e-07, -5.93799942e-07,
       -5.54285623e-07, -5.58987511e-07, -5.90290368e-07, -5.51912963e-07,
       -5.55825817e-07, -5.61762267e-07, -5.55799831e-07, -5.89345112e-07,
       -5.53730221e-07, -5.61125619e-07, -5.50674816e-07, -5.53102161e-07,
       -6.08555336e-07, -5.50650888e-07, -5.49869103e-07, -6.34743326e-07,
       -5.89303272e-07, -5.59836391e-07, -5.95631852e-07, -5.66907218e-07,
       -5.74252649e-07, -5.85894066e-07, -6.12750123e-07, -5.57254487e-07,
       -5.62890202e-07, -5.88837989e-07, -5.90279715e-07, -6.33871487e-07,
       -5.59329799e-07, -6.31822760e-07, -5.53477847e-07, -5.54244309e-07,
       -5.92090956e-07, -5.52498620e-07, -1.04051085e-06, -5.52762813e-07,
       -5.79660350e-07, -5.56140647e-07, -5.55694584e-07, -7.95987400e-01,
       -5.65622640e-07, -5.85378154e-07, -5.64999695e-07, -1.41566264e+00,
        2.03926844e-01, -5.53800629e-07, -6.04910844e-07,  1.18358183e-01,
       -6.75402304e-07, -5.64287956e-07,  2.12888142e+00, -5.58233041e-07,
       -5.82593600e-07, -6.26917003e-07, -5.52166912e-07, -5.77720302e-07,
       -6.54229483e-07, -5.54943626e-07, -5.99904602e-07, -5.56035056e-07,
        3.77326368e+00,  1.00683046e-01, -5.59154138e-07, -5.53456931e-07,
       -5.92639573e-07, -5.94503256e-07, -6.19012939e-07, -5.55769894e-07,
       -5.55719406e-07, -5.63279407e-07, -5.63622286e-07, -5.52837080e-07,
       -5.59012342e-07, -5.55474637e-07, -5.59651297e-07, -5.51165094e-07,
       -5.82186288e-07, -5.54776297e-07, -5.50439308e-07, -5.55930257e-07,
       -5.55357142e-07, -5.52526024e-07, -5.59886489e-07, -6.25769056e-07,
       -5.50217672e-07, -5.55619927e-07, -5.66701540e-07, -5.52054571e-07,
       -5.66298392e-07, -5.92950123e-07, -5.56294451e-07, -5.56442575e-07,
       -5.95895862e-07, -6.06959361e-07, -5.50527716e-07, -5.62496293e-07,
       -5.55925729e-07, -5.56642243e-07, -6.96635800e-07, -5.63933522e-07,
       -5.56630002e-07,  9.91746279e-02, -5.56790984e-07, -5.56244541e-07,
       -6.37905884e-07, -5.50584856e-07, -5.54311921e-07, -5.85790517e-07,
       -5.90489023e-07, -5.64006806e-07, -5.54425681e-07, -5.84757929e-07,
       -5.57959029e-07, -3.00129204e+00, -5.55531281e-07, -1.70906974e+00,
       -6.02555074e-07, -5.55293745e-07, -5.58762749e-07, -5.51505422e-07,
       -5.83650045e-07, -6.00388511e-07, -5.53865157e-07, -5.83243287e-07,
       -5.57491356e-07, -6.34704516e-07, -5.82252734e-07, -5.89058519e-07,
       -5.84445183e-07, -5.53580013e-07, -6.09013669e-07, -5.58410726e-07,
       -5.59087192e-07, -5.53928530e-07, -5.99766251e-07, -5.49794705e-07,
       -5.61379609e-07, -6.54008558e-07,  2.06446585e+00, -5.51306868e-07,
       -5.58929541e-07, -5.54399502e-07, -6.17535376e-07, -5.51836214e-07,
       -6.35322584e-07, -2.13286329e-06,  5.14775136e-01, -6.96842759e-07,
       -5.56316083e-07, -5.95619451e-07, -7.09755413e-07, -8.77494452e-02,
       -5.80333952e-07, -6.14064972e-07, -5.60756019e-07, -5.54950328e-07,
       -5.80438602e-07, -5.55797037e-07, -6.82837413e-07, -5.52156962e-07,
       -6.99561748e-07, -5.98496380e-07, -5.52785696e-07, -5.56460803e-07,
       -5.94534313e-07, -5.54113638e-07, -5.54201756e-07, -5.95961159e-07,
       -6.29581592e-07, -5.57284512e-07, -5.54474914e-07, -1.34251371e-01,
       -5.54120661e-07, -5.63594870e-07, -5.72204631e-07, -5.59184919e-07,
       -6.20476875e-07, -5.60936350e-07, -5.81430567e-07, -5.92252769e-07,
       -6.40817819e-07, -5.87624639e-07, -4.30837493e-01, -5.50492156e-07;
  Eigen::MatrixXd layer1(200,4);
  layer1 << node1_Weight, node2_Weight, node3_Weight, node4_Weight;
  // Eigen::MatrixXd input_matrix(6,1);
  // input_matrix << input[0], input[1], input[2], input[3], input[4], input[6];
  // Eigen::MatrixXd layer1_intput(200,6);
  Eigen::MatrixXd layer1_intput;
  layer1_intput = (layer1*input).unaryExpr(&sigmoid);
  Eigen::MatrixXd layer2(1,200);
  layer2 << -1.07282145e-04, -1.21931254e-04, -1.06189598e-04, -1.11530959e-04,
       -1.07092332e-04, -1.06834746e-04, -1.06915297e-04, -1.14014540e-04,
       -1.20994242e-04, -1.11725492e-04, -1.06823530e-04, -1.06290007e-04,
       -1.08241671e-04, -1.24364640e-04, -1.11998577e-04, -1.13063200e-04,
       -1.06697057e-04, -1.07462907e-04, -1.12504229e-04, -1.06309746e-04,
       -1.06948172e-04, -1.07913812e-04, -1.06943937e-04, -1.12353462e-04,
       -1.06606444e-04, -1.07810424e-04, -1.06107406e-04, -1.06503940e-04,
       -1.15399458e-04, -1.06103494e-04, -1.05975653e-04, -1.19490728e-04,
       -1.12346787e-04, -1.07600935e-04, -1.13354467e-04, -1.08747812e-04,
       -1.09933844e-04, -1.11802247e-04, -1.16059543e-04, -1.07180890e-04,
       -1.08096881e-04, -1.12272539e-04, -1.12502531e-04, -1.19355659e-04,
       -1.07518572e-04, -1.19037953e-04, -1.06565260e-04, -1.06690317e-04,
       -1.12791167e-04, -1.06405401e-04, -1.74307040e-04, -1.06448540e-04,
       -1.10803483e-04, -1.06999473e-04, -1.06926785e-04, -8.81029712e-02,
       -1.08539833e-04, -1.11719739e-04, -1.08438915e-04, -7.70510953e-01,
        2.59158397e-01, -1.06617933e-04, -1.14824497e-04, -1.08089173e-01,
       -1.25703073e-04, -1.08323564e-04,  2.95219429e-01, -1.07340168e-04,
       -1.11273943e-04, -1.18275439e-04, -1.06351227e-04, -1.10491838e-04,
       -1.22489215e-04, -1.06804368e-04, -1.14032479e-04, -1.06982268e-04,
        1.82038905e-01,  2.93488421e-02, -1.07490006e-04, -1.06561846e-04,
       -1.12878527e-04, -1.13175065e-04, -1.17041681e-04, -1.06939058e-04,
       -1.06930830e-04, -1.08160021e-04, -1.08215633e-04, -1.06460665e-04,
       -1.07466945e-04, -1.06890936e-04, -1.07570845e-04, -1.06187547e-04,
       -1.11208667e-04, -1.06777083e-04, -1.06068902e-04, -1.06965191e-04,
       -1.06871784e-04, -1.06409876e-04, -1.07609079e-04, -1.18096654e-04,
       -1.06032660e-04, -1.06914617e-04, -1.08714524e-04, -1.06332878e-04,
       -1.08649262e-04, -1.12927965e-04, -1.07024531e-04, -1.07048662e-04,
       -1.13396415e-04, -1.15147843e-04, -1.06083357e-04, -1.08032963e-04,
       -1.06964453e-04, -1.07081186e-04, -1.28880146e-04, -1.08266102e-04,
       -1.07079192e-04, -2.23462174e-01, -1.07105412e-04, -1.07016400e-04,
       -1.19980029e-04, -1.06092699e-04, -1.06701346e-04, -1.11785689e-04,
       -1.12535903e-04, -1.08277984e-04, -1.06719902e-04, -1.11620512e-04,
       -1.07295577e-04,  4.72279350e-01, -1.06900169e-04, -1.36556878e-01,
       -1.14452121e-04, -1.06861449e-04, -1.07426348e-04, -1.06243162e-04,
       -1.11443169e-04, -1.14109148e-04, -1.06628462e-04, -1.11378026e-04,
       -1.07219454e-04, -1.19484717e-04, -1.11219317e-04, -1.12307733e-04,
       -1.11570462e-04, -1.06581933e-04, -1.15471669e-04, -1.07369079e-04,
       -1.07479119e-04, -1.06638801e-04, -1.14010554e-04, -1.05963483e-04,
       -1.07851675e-04, -1.22455438e-04, -3.48601216e-01, -1.06210716e-04,
       -1.07453478e-04, -1.06715632e-04, -1.16810333e-04, -1.06297208e-04,
       -1.19580426e-04, -2.65052044e-04,  4.84924601e-02, -1.28910886e-04,
       -1.07028055e-04, -1.13352496e-04, -1.30820276e-04,  3.83443716e-01,
       -1.10911599e-04, -1.16266073e-04, -1.07750384e-04, -1.06805460e-04,
       -1.10928392e-04, -1.06943482e-04, -1.26820782e-04, -1.06349602e-04,
       -1.29314347e-04, -1.13809225e-04, -1.06452276e-04, -1.07051631e-04,
       -1.13180003e-04, -1.06669001e-04, -1.06683376e-04, -1.13406788e-04,
       -1.18689910e-04, -1.07185778e-04, -1.06727932e-04,  2.61538915e-02,
       -1.06670147e-04, -1.08211187e-04, -1.09603714e-04, -1.07495012e-04,
       -1.17270675e-04, -1.07779680e-04, -1.11087511e-04, -1.12816937e-04,
       -1.20429645e-04, -1.12078814e-04, -8.23161754e-02, -1.06077543e-04;
  Eigen::MatrixXd result = (layer2*layer1_intput).unaryExpr(&sigmoid);
  // double eps = result(0,0);
  result = result*(dbtrack_param.eps_max-dbtrack_param.eps_min);
  result.array() += dbtrack_param.eps_min;
  // result = result*(2.0-1.0);
  // result.array() += 1.0;
  return result;
}

int dbtrack::MLP_classfication_Nmin_v1(Eigen::MatrixXd input){
  Eigen::MatrixXd node1_w(200,1);
  node1_w << 3.23093721e-05,  1.34118138e-04,  6.57132985e-05,  2.92654247e-04,
       -1.99709470e-02, -5.45690242e-02,  9.33455709e-04,  2.99891050e-05,
        2.10639215e-05,  9.76073011e-03,  8.32761256e-03,  5.66732849e-04,
       -3.03681818e-05,  1.02105572e-04,  2.81938611e-04, -5.72869144e-05,
        5.07463542e-02,  1.73739295e-02, -2.16468666e-05,  7.73222931e-03,
        5.40209321e-04,  5.15709159e-05,  7.32907442e-05,  2.05969121e-03,
        5.07875246e-04, -1.43236765e-04,  1.60914865e-02, -3.68932665e-05,
        1.13893926e-05,  2.24180524e-04,  2.23959007e-04,  3.80989520e-04,
        4.76502888e-04, -1.70004836e-05, -2.96380787e-05,  3.26488833e-03,
        1.45790392e-03,  2.34497387e-04,  3.75129571e-04,  4.05713216e-04,
       -1.11829301e-03,  7.81141885e-05,  1.19070921e-02,  4.69823731e-04,
       -2.02968391e-05, -8.61082511e-07, -2.83803549e-04, -5.44967530e-04,
        3.13715776e-05,  2.21932494e-06,  5.87711598e+00, -1.75674563e+00,
        7.25006745e-05,  4.17778734e-04,  7.55209567e+00,  4.26754721e-04,
        5.12594844e-05, -1.59731872e+00,  1.28530587e-04,  3.19417413e-05,
        2.59266636e-05,  2.63068318e-04,  4.10017012e-04,  1.83452782e-04,
        1.69966264e-04, -7.88316008e-03,  1.05784030e-03, -5.73356522e-05,
        6.40031841e-05,  2.26238523e-05,  2.18622208e-05,  9.93835940e-06,
        2.93859607e-04,  2.62063180e-04,  5.04045864e-04, -1.15587923e-03,
        3.27064693e-07, -7.12107450e-06,  7.05252418e-02, -4.78009381e-04,
       -3.12059334e-05,  7.86275671e-05, -2.14516637e-05,  5.19414063e+00,
       -1.26562867e-06,  4.13621989e-04,  3.22811589e-05,  8.15105402e+00,
        5.92021854e-04,  6.50336398e-05,  2.09788205e-04, -2.86155972e-05,
       -3.16235821e-05,  8.98200692e-05, -1.56228543e+00,  4.63378993e-03,
        2.50703081e-03,  4.74088088e-04,  6.24747012e-05, -2.94809925e-05,
        2.01845362e-03,  1.11383509e-04,  2.76989884e-03,  1.30755500e-04,
        4.10616724e-04, -3.05680967e-03,  6.70276137e-06, -1.22044945e+00,
        1.04667869e-04,  2.61166909e-05, -2.77578431e-04,  4.06725519e-04,
        7.65633827e-05, -3.17434128e-04, -1.65527187e-06,  5.41354405e+00,
        2.67116028e-04, -3.66362482e-05, -2.43689629e-05, -3.16041837e-03,
       -9.30672298e-04,  3.27646809e+00, -9.84032519e-06, -2.47529545e-06,
       -1.40674226e-05,  4.75176537e-04, -1.05607436e-07,  7.99362220e-04,
        3.28860405e-05, -3.35295998e+00,  1.82234385e-04, -8.17967347e-06,
       -2.74770870e-05,  1.79691396e-04, -9.46628784e-05, -1.01068961e-05,
       -1.25080227e-03,  2.05370363e-03, -8.72357303e-05,  6.37238689e-05,
        4.70468246e-03,  2.18792189e-05,  1.64095936e-04, -3.83497190e-03,
        1.00801396e-07,  6.39030646e+00,  4.79273033e-04,  3.76694344e-04,
       -5.71315487e-01,  7.95838708e-04,  1.80768686e-03,  3.41110625e-05,
       -3.20203839e-05, -2.77519986e-05,  6.52937472e-05,  2.98257457e-04,
        4.75746424e-07,  1.75943871e-04, -5.37523948e-03,  8.59175379e-05,
       -5.19998911e-05,  3.59924029e-04,  9.87272060e-05,  2.76665996e-02,
       -1.83315574e-03,  4.92662311e+00,  1.91106845e-03,  8.11909537e-05,
        2.08401199e-02, -1.67716085e-03,  2.39481188e-04, -5.43047306e-04,
        1.47072873e-04,  2.30735363e-04,  3.28230563e-05,  7.35651088e-05,
        3.40377145e-04,  7.75525410e-04,  3.20799638e-05,  9.56705726e-05,
        6.55185577e-05, -1.40209178e-04, -2.01728995e-05,  2.23957024e-04,
        1.82330161e-04,  5.97591276e-05,  4.17438401e-04,  3.59397635e-05,
        1.50256996e-03,  3.38771383e-03,  3.94251245e-04,  3.70535734e-05,
        1.67713153e-02, -3.96642969e-03, -1.40183214e-04,  6.78536377e-05,
       -3.51565211e-05,  6.31123133e-05,  3.35321028e-05,  5.86562002e-04;
  Eigen::MatrixXd node2_w(200,1);
  node2_w << -3.89423780e-06, -1.99051780e-06, -2.81119023e-06, -3.94893955e-06,
       -4.44667090e-03, -5.17635068e-04, -8.95500082e-06, -3.97089431e-06,
       -4.62240886e-06, -5.66347725e-04, -8.73268429e-05, -9.16020683e-06,
       -6.49435322e-06, -7.85643146e-06,  4.33918457e-06, -7.36858983e-06,
        8.72972150e-02, -7.59532951e-04, -5.19713819e-06,  1.05968948e-04,
        2.69469804e-05, -2.74967139e-06, -2.33002240e-06,  5.14871568e-07,
        1.22201067e-05,  1.72801748e-05, -9.95954545e-06, -6.08538420e-06,
       -4.27443751e-06, -2.64550622e-06,  7.51402741e-06,  9.23817373e-06,
        4.30646462e-06, -4.41196280e-05, -7.85402015e-06,  2.96663314e-04,
       -6.31377468e-06, -9.17941537e-07,  1.00802600e-05,  3.52844080e-06,
       -3.68112547e-06, -2.70718114e-06,  2.17075716e-04, -2.45910105e-06,
       -5.14676490e-06, -4.21583592e-06, -4.74784788e-06,  4.96861565e-06,
       -3.83281865e-06, -4.08504701e-06,  1.65202448e+01,  6.62834773e+01,
       -2.56770553e-06, -3.17418289e-07,  1.63049930e+00, -1.60839344e-05,
       -4.94227402e-06,  4.87073000e+01, -1.17488746e-07, -3.80718127e-06,
       -8.86865201e-06, -2.37939459e-06,  1.58352802e-05,  8.87864444e-06,
       -1.80776363e-06, -3.19179589e-04,  1.11282716e-04, -7.37072011e-06,
       -2.79617201e-06, -3.96660920e-06, -4.15327249e-06, -4.04367477e-06,
       -3.44130431e-06, -6.47168735e-06,  7.79486220e-06,  9.50392110e-06,
       -4.16314242e-06, -4.55082495e-06,  1.01258484e-01,  2.71998029e-06,
       -6.57847766e-06, -2.71058323e-06, -5.18944650e-06, -2.93772115e+01,
       -4.23482799e-06, -1.76851227e-05, -3.75487167e-06, -5.74810512e-01,
        4.24251856e-08, -2.80661694e-06,  3.98282438e-06, -5.76101588e-06,
       -5.87295611e-06, -2.40488364e-06,  7.91356470e+00,  4.93154610e-05,
        9.69657855e-05, -1.70390478e-06, -2.77638314e-06, -6.47326492e-06,
        2.48346951e-06, -2.17183242e-06,  2.06606726e-06, -1.92939005e-06,
        1.04788573e-05,  2.55363657e-04, -4.13499334e-06,  2.26731628e+00,
       -2.49695626e-06, -4.51839051e-06,  2.04703776e-05,  2.51771235e-06,
       -2.61104872e-06, -3.28827142e-05, -4.25470525e-06,  3.42619624e-01,
        3.53983341e-06, -6.06663236e-06, -5.32596943e-06, -6.44596318e-05,
       -8.21694154e-06,  1.44250883e-01,  1.52200878e-06, -4.30501046e-06,
       -4.87963573e-06,  1.01104231e-05, -4.18366604e-06, -1.52279875e-05,
       -3.98031287e-06,  1.38801975e+01, -2.31741641e-06, -4.60350640e-06,
       -6.49619646e-06, -1.80228370e-06, -3.35509453e-06, -1.00498708e-06,
       -8.96741479e-06,  5.35841913e-06,  1.83765985e-07, -2.79405741e-06,
        2.64487711e-04, -4.00743765e-06, -2.18685964e-06,  7.01523993e-06,
       -4.17421779e-06, -1.20940234e+01, -1.12249959e-06,  1.70941545e-05,
        2.39942727e-01, -7.12035929e-07,  8.42290665e-07, -3.74329507e-06,
       -5.87791181e-06, -5.70236099e-06, -2.80850786e-06,  5.75987653e-06,
       -4.15533191e-06, -1.58981483e-06, -2.53995889e-03, -2.52360731e-06,
       -7.02973486e-06, -2.32008138e-06, -2.32177851e-06,  4.23908127e-02,
       -1.02333994e-05,  2.12442169e+01, -3.48169033e-06, -2.46059929e-06,
        2.12462962e-03,  1.44992807e-04,  2.72841769e-06,  5.79555149e-05,
       -7.75202226e-06,  7.06732782e-07, -3.66688370e-06, -2.48373930e-06,
        1.21307235e-05, -7.33415231e-07, -3.72502953e-06, -2.33690855e-06,
       -2.80822365e-06,  8.39464256e-06, -5.14240496e-06, -2.64440687e-06,
        1.96341655e-06, -2.77222977e-06,  1.46707160e-05, -3.78961001e-06,
        4.28095906e-05, -1.21103983e-04,  5.04675774e-06, -3.79418780e-06,
        6.01128981e-04,  1.76376832e-04, -1.33160826e-05, -2.81869387e-06,
       -5.96904353e-06, -2.79109459e-06, -3.96257071e-06, -1.90468975e-06;
  Eigen::MatrixXd node3_w(200,1);
  node3_w << -8.76565019e-06, -4.24723420e-06, -7.15554397e-06, -8.39995112e-07,
       -3.68573403e-03, -4.93552170e-03,  6.39396653e-06, -8.87687609e-06,
       -9.40925639e-06, -1.68515622e-03,  6.30799905e-04, -4.74801060e-06,
       -1.19036651e-05, -1.25627232e-05,  8.99804866e-06, -1.17630885e-05,
        4.01521268e-02,  6.32707499e-04, -1.09933890e-05,  7.19969000e-04,
        3.36431864e-05, -7.35185366e-06, -5.52010443e-06,  5.99236323e-05,
        1.32178918e-05, -7.51725636e-05,  1.26047070e-05, -1.13819929e-05,
       -9.64889685e-06, -5.73235209e-06,  1.16188281e-05,  5.35593429e-05,
        2.45017914e-05, -1.10009025e-05, -1.26132851e-05,  4.66716353e-04,
        9.85559999e-06, -1.87045535e-06,  3.44546813e-05,  9.77640476e-06,
        5.49417144e-05, -6.69138473e-06,  8.62553445e-04,  6.74720192e-06,
       -1.09349373e-05, -1.00474929e-05, -3.73181657e-06, -2.00557629e-05,
       -8.79982952e-06, -9.83859549e-06, -7.39835107e+00,  2.37757737e+01,
       -6.51057289e-06, -3.60499621e-06,  5.05003366e+00, -2.28738359e-05,
       -7.67825871e-06,  1.76155097e+00,  5.84638967e-08, -8.75402561e-06,
       -1.94692865e-05, -4.94190180e-06,  4.93521606e-05,  1.93400149e-05,
       -3.73791518e-06,  1.30470754e-04,  5.04143595e-04, -1.17654102e-05,
       -7.19782836e-06, -9.05473122e-06, -9.22212815e-06, -9.64189523e-06,
       -4.37794967e-06, -1.13582458e-05,  1.07853448e-05, -1.72695058e-05,
       -9.97014745e-06, -1.03766342e-05,  2.66378230e-02,  1.99289584e-06,
       -1.19992267e-05, -6.69334416e-06, -1.09844219e-05, -1.44670637e+01,
       -1.00704025e-05, -1.14470650e-05, -8.73111290e-06, -1.83330500e+00,
        2.03669687e-06, -7.17838025e-06,  3.78265111e-06, -1.11282817e-05,
       -1.12137534e-05, -5.83900785e-06,  1.66399389e+01,  2.37420664e-04,
       -3.91190020e-05, -2.10620952e-06, -7.27012121e-06, -1.20966343e-05,
        2.70172296e-05, -4.89519504e-06,  1.05556681e-04, -4.65728185e-06,
        1.13313203e-05,  2.35534540e-04, -9.80422124e-06,  4.60990441e+00,
       -6.10031257e-06, -9.22296107e-06,  3.74604627e-06,  7.74234815e-06,
       -6.56838990e-06, -5.28729085e-05, -1.00912919e-05, -9.79663655e+00,
        1.45714915e-05, -1.13792416e-05, -1.10837442e-05, -1.03741345e-04,
        1.13998810e-05, -1.81010483e-01, -3.46802268e-05, -1.01380392e-05,
       -1.07503498e-05,  2.41457887e-05, -1.00010404e-05, -4.75398211e-06,
       -8.77454582e-06,  1.16905625e+01, -4.86190086e-06, -1.04315785e-05,
       -1.19922322e-05, -3.82795120e-06, -1.35385944e-05, -4.57127224e-07,
        1.31267631e-05,  2.92818189e-05,  8.84413197e-06, -7.20929663e-06,
        8.08561550e-05, -9.08614375e-06, -4.55090633e-06, -3.27182314e-05,
       -9.98673939e-06, -1.43695053e+01,  2.97003548e-06,  2.23417150e-05,
        3.17841561e-01,  1.88432669e-06,  2.08269022e-05, -8.66670431e-06,
       -1.12275916e-05, -1.11076077e-05, -7.16988870e-06,  1.11351242e-05,
       -9.95859216e-06, -3.50465684e-06, -2.30469681e-03, -6.18861051e-06,
       -1.16969542e-05, -4.22044300e-06, -5.51518405e-06, -2.56593375e-03,
       -6.01734434e-05, -9.60350028e-01,  4.72767452e-05, -5.92079061e-06,
        1.43392009e-03, -1.82864435e-03,  9.96104477e-06,  3.09386765e-05,
       -1.39611281e-05,  2.46994804e-06, -8.71678869e-06, -5.99725472e-06,
        2.06462394e-05,  1.82899233e-06, -8.76043683e-06, -5.57471311e-06,
       -7.16094998e-06, -1.41078743e-06, -1.09297735e-05, -5.73145888e-06,
        5.71832655e-06, -7.31415200e-06,  3.14199351e-05, -8.63291332e-06,
        1.92261198e-05, -3.38588319e-04,  1.54499345e-05, -8.60738373e-06,
       -2.99471473e-04,  6.20191547e-05, -4.66100047e-06, -7.14338071e-06,
       -1.13480615e-05, -7.23178870e-06, -8.75119825e-06, -2.66662047e-06;
  Eigen::MatrixXd node4_w(200,1);
  node4_w << -1.31442706e-06, -9.94597917e-07, -7.01723860e-07, -9.43227613e-07,
        3.36743299e-04,  5.37574018e-04, -3.52790001e-08, -1.32294124e-06,
       -1.44193871e-06,  1.93432846e-04, -3.22080776e-04, -5.13423974e-08,
       -2.16300336e-06, -3.10847298e-06, -3.74444461e-06, -4.21893667e-07,
        3.81666134e-02, -2.82053874e-04, -6.86815960e-07, -2.84630393e-04,
        3.15723180e-05, -7.06980264e-07, -8.51602746e-07, -7.10631671e-06,
        1.51501190e-06, -2.10695536e-05, -3.72579579e-06, -5.84897978e-07,
       -1.17844714e-06, -1.24876746e-06, -4.46011652e-06, -1.13867083e-05,
       -3.88907322e-06, -2.35796298e-05, -8.53305473e-07, -4.32391033e-05,
       -8.78232409e-06, -1.44086927e-06, -4.92419785e-06, -1.05844092e-06,
       -6.91859958e-06, -1.26788503e-06, -3.39311938e-04,  5.91157731e-07,
       -7.06461274e-07, -9.75367590e-07,  1.42770264e-05, -1.74068556e-05,
       -1.29591689e-06, -9.84700456e-07,  4.25232227e+00,  1.34353161e+01,
       -1.31376609e-06,  5.84654824e-07, -1.66025176e+00, -1.75386872e-06,
       -2.17335492e-06, -2.37070592e+01, -4.69122600e-07, -1.29047791e-06,
        1.24138174e-06, -1.32664401e-06, -3.87720771e-06, -1.42703923e-06,
       -9.66173190e-07,  1.39289859e-03, -1.09312504e-04, -4.18855175e-07,
       -7.05329766e-07, -1.26754538e-06, -1.27545674e-06, -8.88966538e-07,
       -1.76783423e-06,  1.57171647e-06,  8.41469319e-07,  8.80529408e-06,
       -9.88142061e-07, -8.50168287e-07,  2.46121256e-02,  3.47428017e-06,
       -2.02336670e-06, -1.27439009e-06, -6.89996105e-07, -8.24836710e+00,
       -9.67298742e-07, -3.13264105e-06, -1.28145746e-06, -8.71185908e-01,
       -6.41510787e-06, -7.06841878e-07,  1.78754308e-06, -6.72392580e-07,
       -6.44817870e-07, -8.19752083e-07, -2.95941649e+00, -4.71856482e-05,
       -1.78533443e-05, -2.90853449e-06, -7.24267241e-07, -1.54206668e-06,
       -1.05065554e-05, -9.27831958e-07, -1.93802138e-05, -1.31579517e-06,
       -3.75243341e-06,  7.73674734e-05, -8.83708383e-07,  6.07934973e-03,
       -8.60295819e-07, -1.45258597e-06, -2.04257903e-05, -2.49817684e-06,
       -1.31955732e-06,  9.88070413e-06, -9.58461283e-07, -2.18640546e-01,
       -2.97402046e-06, -5.85162921e-07, -6.59305705e-07,  1.39112875e-04,
        9.32250401e-06,  5.45951672e-02,  1.70437466e-07, -9.38771876e-07,
       -7.60571600e-07, -4.18860908e-07, -9.85936961e-07,  1.16520402e-06,
       -1.34435039e-06,  1.09003386e+01, -9.38895089e-07, -8.35253733e-07,
       -1.86589529e-06, -9.34035119e-07,  2.36750707e-06, -1.96234512e-07,
        4.33267924e-05,  3.37537105e-06,  9.44908346e-06, -7.06578152e-07,
        1.96311267e-04, -1.27401424e-06, -8.50754394e-07,  3.43009392e-05,
       -9.87331241e-07, -6.75747608e+00,  5.92876548e-07,  4.09193219e-05,
        4.07616600e-02, -4.01957778e-06, -8.96555529e-06, -1.28340282e-06,
       -6.39184199e-07, -6.79303522e-07, -7.05355684e-07, -4.13349489e-06,
       -9.88209504e-07, -1.32597353e-06,  1.62361156e-03, -1.21004180e-06,
       -4.60839379e-07, -2.48520309e-06, -8.53595922e-07,  1.24895982e-02,
        7.55477265e-08,  3.51595922e+00, -8.15159606e-06, -1.08165349e-06,
       -4.66516235e-05, -3.59202435e-05, -1.82025732e-06,  9.10164958e-05,
       -4.40750526e-06, -1.92762166e-06, -1.27441423e-06, -1.01215397e-06,
        1.96146513e-06, -3.98079085e-06, -1.27761659e-06, -8.46461313e-07,
       -7.04172156e-07,  4.07070944e-06, -7.08109838e-07, -1.24716331e-06,
       -8.72668466e-07, -7.22418840e-07, -1.07720319e-06, -1.30376946e-06,
        2.18552755e-04, -2.49670699e-04, -1.06160189e-06, -1.31090921e-06,
        1.51553448e-04,  3.14421133e-04,  1.59944486e-06, -7.13805172e-07,
       -5.93085982e-07, -7.08492943e-07, -1.34227703e-06, -2.78299337e-06;
  Eigen::MatrixXd node5_w(200,1);
  node5_w << -1.36436655e-06, -1.08603945e-06, -1.25166160e-06, -1.02743494e-06,
        2.54324588e-06,  4.96324875e-04, -3.10584276e-07, -1.37415310e-06,
       -1.43262919e-06, -6.53038790e-05,  7.63933337e-06, -1.65218081e-06,
       -1.67877306e-06, -3.11336104e-06,  6.37935865e-07, -1.56630199e-06,
        3.27722102e-02, -1.06487386e-04, -1.51161984e-06,  1.77232211e-05,
        6.45381919e-07, -1.22593743e-06, -1.04986038e-06, -6.75232266e-06,
        6.57008317e-07, -7.73696946e-06,  5.58549197e-07, -1.54077988e-06,
       -1.42023597e-06, -1.63257685e-06,  1.89022338e-06,  8.19452843e-06,
        2.40461140e-06,  2.67492278e-06, -1.74891929e-06,  1.70522334e-05,
       -4.39400412e-06, -1.05801361e-06,  5.58531392e-06,  1.85919591e-07,
        1.19921481e-05, -1.19670393e-06,  1.38708312e-05,  3.36382875e-07,
       -1.50662126e-06, -1.42036456e-06,  2.58658778e-06,  1.31456433e-06,
       -1.36233263e-06, -1.40018189e-06, -3.90005250e-01,  9.42320310e+00,
       -1.14758196e-06, -2.10267097e-06,  4.48176816e-01, -4.11643510e-06,
       -1.30497420e-06, -2.93881819e+01, -3.15738775e-07, -1.35689561e-06,
       -3.56047002e-06, -1.65724112e-06,  7.21068188e-06,  3.34902923e-06,
       -1.14793752e-06, -2.80664006e-05, -8.25689585e-06, -1.56670645e-06,
       -1.25143034e-06, -1.36942732e-06, -1.39771516e-06, -1.40366564e-06,
       -1.14424846e-06, -1.57774190e-06,  7.40714841e-07, -6.78152129e-08,
       -1.41389993e-06, -1.45425820e-06,  1.75859702e-02,  2.61866627e-06,
       -1.68904762e-06, -1.19891764e-06, -1.51083959e-06, -4.13325605e+00,
       -1.42207313e-06, -4.37642689e-06, -1.35199450e-06,  7.14045232e-01,
       -2.59676471e-07, -1.25223605e-06,  9.71619486e-07, -1.52378189e-06,
       -1.52909692e-06, -1.15835862e-06,  3.92732478e+00, -8.35981701e-06,
       -1.05657066e-05, -1.96072801e-06, -1.25547518e-06, -1.69597578e-06,
       -4.79433020e-06, -1.09943119e-06, -1.08851378e-05, -1.09409842e-06,
        4.98074764e-07,  6.27606596e-05, -1.41641632e-06,  3.85705377e-01,
       -1.25901935e-06, -1.42356946e-06,  1.54388164e-06, -2.59860218e-07,
       -1.16776615e-06, -7.82007440e-06, -1.42373523e-06, -7.13717034e-01,
        1.81216668e-06, -1.54059289e-06, -1.51839391e-06,  8.82727351e-06,
        5.90751657e-06,  8.95183874e-03, -4.99248472e-06, -1.42836553e-06,
       -1.49462831e-06,  2.41323306e-06, -1.41679190e-06, -9.08026274e-07,
       -1.37148022e-06,  5.45933045e+00, -1.35410689e-06, -1.46029131e-06,
       -1.69536977e-06, -1.20435975e-06, -2.44274011e-06, -2.27315798e-08,
        1.49350290e-05, -2.47147466e-06,  4.13864412e-06, -1.25199973e-06,
       -5.54492154e-05, -1.37282431e-06, -1.25225088e-06,  1.13676150e-05,
       -1.41550570e-06, -2.78486276e+00, -1.12827519e-07,  1.50136365e-05,
        5.01108724e-02, -2.55730392e-06, -4.49657548e-06, -1.34947414e-06,
       -1.52981959e-06, -1.52185123e-06, -1.25201842e-06,  1.82497939e-06,
       -1.41270099e-06, -1.09345232e-06, -1.41869331e-04, -1.15925602e-06,
       -1.56322105e-06, -1.85067500e-06, -1.14486359e-06,  5.15959820e-03,
       -6.74153793e-06, -3.53659630e-01, -6.15079543e-06, -1.11816575e-06,
        2.38824652e-04, -1.73592696e-04,  1.09081293e-06,  1.00976732e-05,
       -2.29421569e-06,  7.29809885e-07, -1.34642753e-06, -1.10609967e-06,
        2.87315362e-06, -2.49495915e-06, -1.35341442e-06, -1.14188469e-06,
       -1.25165018e-06,  3.55597376e-07, -1.50618192e-06, -1.63154332e-06,
        1.69775757e-07, -1.25141529e-06,  3.78054065e-06, -1.35465600e-06,
        4.62672211e-06, -3.07156445e-05,  1.55460018e-06, -1.35562711e-06,
       -1.55497078e-05,  8.38623673e-05,  8.70789939e-07, -1.25708245e-06,
       -1.53813102e-06, -1.25285725e-06, -1.36998198e-06, -2.42672580e-06;
  Eigen::MatrixXd node6_w(200,1);
  node6_w << -2.59531933e-07, -1.01706678e-07, -6.87985914e-08, -3.31046953e-08,
       -5.97494217e-06, -2.05497933e-05, -1.60900846e-06, -2.73113449e-07,
       -3.90615124e-07, -9.28958988e-05, -9.00004768e-05, -9.54530243e-07,
       -7.43997496e-07,  8.96231024e-08, -1.51202374e-07, -3.37755664e-07,
        1.52747572e-03, -4.01748789e-06, -1.69629219e-07, -8.47629881e-05,
        9.13137795e-06, -5.90919092e-08, -7.16101701e-08, -2.30780462e-06,
       -3.40713421e-07,  4.04010245e-06, -2.51570651e-06, -2.33268253e-07,
       -2.45336829e-07, -2.38064714e-07, -1.22036601e-06, -1.77598078e-07,
        3.19263898e-07, -3.19441329e-06, -6.92287220e-07, -1.62034689e-05,
       -1.83694945e-06, -2.02892183e-07, -2.08797347e-06, -2.99128735e-08,
       -9.56438352e-06, -1.06000234e-07, -8.65472180e-05, -1.38496113e-07,
       -1.66874266e-07, -1.43407948e-07, -1.46426720e-06, -3.01858802e-06,
       -2.41643412e-07, -1.23732100e-07, -1.19251778e+01,  2.93504000e+01,
       -1.04326304e-07, -1.78820682e-07,  1.63951990e+00, -6.48866015e-07,
       -6.50795617e-07,  1.41680709e+01, -5.10379118e-09, -2.36515074e-07,
       -1.17477486e-06, -2.46347852e-07, -1.14683187e-06, -1.04043958e-06,
       -1.11388022e-07,  4.60116535e-04,  4.00315412e-05, -3.39142586e-07,
       -6.62086605e-08, -2.54257233e-07, -2.92477673e-07, -8.19387777e-08,
       -1.62016054e-07, -8.84870569e-07, -6.12899614e-08,  4.04016147e-06,
       -1.45067416e-07, -1.28126688e-07,  5.44992247e-03, -4.90166879e-07,
       -7.18284037e-07, -1.08366477e-07, -1.69146839e-07, -9.97132230e+00,
       -1.39704526e-07, -1.13773611e-06, -2.23882907e-07,  8.66128980e+00,
       -1.17250006e-06, -6.82138988e-08, -5.14260125e-07, -2.16807687e-07,
       -2.21738469e-07, -7.66558985e-08,  3.66836824e+00, -1.10199243e-05,
       -7.88077556e-06, -5.28418888e-07, -6.62619205e-08, -5.99450352e-07,
       -2.16815681e-06, -8.99641809e-08, -5.33495278e-06, -1.28242046e-07,
       -8.64226822e-08,  1.58121909e-05, -8.69378712e-08,  3.28260892e-01,
       -9.32099854e-08, -3.93306523e-07, -1.58169308e-06,  4.16183693e-08,
       -1.03479079e-07, -3.37431199e-06, -1.35915745e-07, -1.16354626e+00,
       -7.60879210e-07, -2.31891238e-07, -1.79226256e-07,  1.38888339e-05,
        1.11256237e-06, -1.53602177e-02,  5.37470848e-07, -1.30427194e-07,
       -1.44335080e-07, -6.47185137e-07, -1.46954792e-07, -2.31231398e-06,
       -2.82674230e-07,  6.46714771e+00, -1.32414981e-07, -1.30230488e-07,
       -6.70356341e-07, -1.17419965e-07,  5.47400472e-08, -2.14279517e-08,
       -6.01149939e-06,  1.29906776e-06,  4.30875241e-07, -6.59608295e-08,
        2.76762340e-06, -2.63225111e-07, -1.06440510e-07,  5.37387737e-06,
       -1.46428720e-07, -1.06439678e+01, -4.08929682e-07,  6.33978293e-08,
        1.21889521e-02, -7.15815478e-07, -1.90363489e-06, -2.22918666e-07,
       -2.21502449e-07, -2.12712937e-07, -6.85217260e-08, -9.60414469e-07,
       -1.43646881e-07, -1.45268178e-07,  2.33858483e-04, -9.64575382e-08,
       -3.10641113e-07, -4.78784954e-07, -8.16858474e-08, -2.94077037e-03,
        5.46332657e-06, -3.73445205e+00, -2.32927140e-06, -8.49051203e-08,
       -3.67802846e-05, -3.61735656e-05, -6.15779433e-07,  7.65150627e-06,
       -4.87179946e-07, -9.92936276e-07, -2.03960095e-07, -7.79671942e-08,
       -1.68186682e-06, -7.12232592e-07, -2.15870476e-07, -7.99449980e-08,
       -6.85405070e-08, -5.60711706e-07, -1.66671280e-07, -2.37396478e-07,
       -2.28249641e-07, -6.49644464e-08, -5.35997519e-07, -2.38782999e-07,
        8.22919741e-05, -7.63568628e-05, -1.20514056e-07, -2.41999078e-07,
       -5.50264623e-05,  4.77653055e-05, -2.85315232e-06, -7.25956893e-08,
       -2.24920352e-07, -6.56024843e-08, -2.79291656e-07, -4.66251521e-07;
  Eigen::MatrixXd node7_w(200,1);
  node7_w << -0.05414997, -0.06775448, -0.06035873,  0.05842089, -0.02787812,
       -0.04236588,  0.05273012, -0.05387808, -0.05086809, -0.05238873,
       -0.06391593,  0.05457717, -0.04006499,  0.04998896,  0.05138772,
       -0.03961897, -0.04823459,  0.02045022, -0.04460397, -0.06230641,
        0.05623041, -0.05884278, -0.06439915, -0.07573514,  0.05431355,
        0.0325342 ,  0.05421349, -0.04205856, -0.05088467, -0.06452674,
        0.04714826,  0.04293657,  0.0550364 ,  0.03779631, -0.03849476,
       -0.0702489 , -0.07108573, -0.07051373,  0.0489675 ,  0.06434282,
        0.04426032, -0.06058385, -0.06734655,  0.06440217, -0.04478204,
       -0.04874158,  0.06131323,  0.04022737, -0.05436877, -0.04945452,
       -0.03028671, -0.00149177, -0.06093774,  0.05545857, -0.04877223,
        0.04936665, -0.05199719,  0.00119106,  0.0605466 , -0.05435724,
        0.0475256 , -0.06607279,  0.05048014,  0.0475034 , -0.06885102,
        0.03801985,  0.04526214, -0.03959846, -0.05998705, -0.05304008,
       -0.05177099, -0.05070103,  0.05358897,  0.04796892,  0.05472955,
        0.04770868, -0.04912382, -0.04702495, -0.04818011,  0.0695203 ,
       -0.040101  , -0.06056314, -0.04463165, -0.02280708, -0.04858243,
        0.05148518, -0.05451136, -0.05499045,  0.04664196, -0.06021713,
        0.0547977 , -0.04306063, -0.04270138, -0.0633965 , -0.03251025,
       -0.07109164,  0.03172412, -0.06633167, -0.05938672, -0.04067801,
       -0.07269638, -0.06622786, -0.07386553, -0.06574833,  0.04764922,
        0.05270882, -0.05008718, -0.04369765, -0.06228616, -0.05118048,
        0.03419439,  0.05758993, -0.0609342 ,  0.04665368, -0.04842855,
       -0.0638519 ,  0.05246065, -0.04209796, -0.04412287,  0.05116188,
        0.06394547, -0.07034202,  0.04399165, -0.04812377, -0.04604771,
        0.05548664, -0.04900426,  0.04008805, -0.05396268,  0.00899115,
       -0.06714205, -0.04683716, -0.04055325, -0.06814951,  0.05259236,
        0.067744  ,  0.05010695,  0.05765484,  0.05994738, -0.05990135,
        0.03806157, -0.05287162, -0.06772225,  0.05258276, -0.04906382,
       -0.03078797,  0.06421768,  0.04461453, -0.05103627, -0.06923429,
       -0.07245394, -0.05459507, -0.04266351, -0.04320696, -0.06027443,
        0.04875856, -0.04915955, -0.06838693,  0.04531256, -0.06202451,
       -0.04014345, -0.06400744, -0.06436809, -0.05207103,  0.0590648 ,
       -0.04666251, -0.07232033, -0.06284469, -0.06260413,  0.0246198 ,
        0.05264652,  0.05956798,  0.04822482,  0.05187029, -0.05490829,
       -0.06268727,  0.05101059, -0.06910249, -0.05474581, -0.06421409,
       -0.06031077,  0.05279779, -0.04479711, -0.06453642,  0.05897303,
       -0.05913087,  0.05581334, -0.05464766,  0.04415859,  0.02543577,
        0.06052488, -0.05472772,  0.03180249,  0.05362512,  0.04876872,
       -0.06051122, -0.04231579, -0.05974657, -0.05405548, -0.06819213;
  Eigen::MatrixXd layer1_bias(200,1);
  layer1_bias << -0.40723189,  1.57807708,  0.43312013,  2.11483888, -3.3997435 ,
        1.97679551,  2.38640121, -0.44821123, -0.92813005,  1.8672958 ,
        4.10972216,  2.65703229, -2.84679767,  3.34622384,  3.2600386 ,
       -2.91061294,  5.15807864,  2.48808183, -2.02743435,  3.74713574,
       -0.40343711,  0.15626515,  0.98350608,  3.82128556,  2.57545211,
        5.49061714,  2.60215603, -2.46392181, -0.95128497,  1.35487212,
        4.15113393,  3.30084234,  2.24991429,  3.51562977, -3.17659932,
        3.67272852,  3.32304249,  2.1490945 ,  3.30354418,  1.35968859,
        2.07111639,  0.56251838,  4.47814977,  1.17132548, -1.99603572,
       -1.32231545,  0.02238733,  4.36742443, -0.37242707, -1.21306235,
        5.84525442,  1.00029141,  0.58272301,  2.31979589,  8.61338463,
        3.10546544, -0.68451491, -0.2748149 ,  2.92208054, -0.37681945,
        3.25470279,  1.62577235,  2.58195288,  3.50010829,  1.77956046,
       -0.2588267 ,  0.19968776, -2.91550573,  0.36573713, -0.59548784,
       -0.80860589, -1.02556308,  3.0665549 ,  3.26564226,  2.4087573 ,
        2.52381439, -1.26126648, -1.61439891,  4.68598695, -0.48536666,
       -2.84371594,  0.56213935, -2.02256315,  3.06971594, -1.34865471,
        2.3256509 , -0.35289457,  8.47282641,  3.21219513,  0.40945885,
        2.59856855, -2.28367298, -2.34783314,  0.86192614,  5.47879015,
        3.98375134,  3.44633545,  1.95523323,  0.26447638, -2.74949418,
        3.59824568,  1.31598495,  3.929065  ,  1.33228554,  3.34739674,
       -0.15078739, -1.12117708,  5.28899984,  0.7395599 , -0.88249448,
        5.23100376,  2.37270478,  0.60120393,  2.22662496, -1.37445929,
        4.63648821,  2.94618727, -2.457161  , -2.11022521,  0.08880992,
       -0.35470688,  6.33369938,  3.62867148, -1.42583371, -1.77914299,
        2.07069599, -1.27978393,  5.42209016, -0.43304387, -0.66439596,
        1.624886  , -1.64637214, -2.76381715,  1.70122963,  2.30205751,
        1.9774212 ,  1.02334078,  0.91262616,  0.38298501,  0.35095056,
        2.09110079, -0.62256277,  1.63308018,  1.08339867, -1.27046645,
        4.9140981 ,  1.44791099,  2.04662725,  4.94639314,  2.64881141,
        3.49327172, -0.33937232, -2.35515427, -2.2582062 ,  0.41933717,
        3.79893893, -1.25592909,  1.7764921 , -1.61267636,  0.75339676,
       -2.81135225,  1.51431746,  1.01759975,  4.83497785,  0.80557998,
        6.52911862,  3.26653193,  0.82729353,  4.71560191,  3.13091209,
        3.25115339, -1.46980334,  3.7995886 ,  3.14852084, -0.28781799,
        0.77921729,  2.65216793,  2.60765274, -0.31316976,  0.99003288,
        0.42487368,  2.07655085, -1.99336627,  1.3558691 ,  2.76372083,
        0.21904595,  2.05815106, -0.32762731,  0.11114605,  4.95071779,
        1.757128  , -0.3131787 ,  3.24553339, -1.3660399 ,  2.14598357,
        0.4676467 , -2.4195617 ,  0.32450931, -0.41818934,  2.37560189;
  Eigen::MatrixXd layer1(200,7);
  layer1 << node1_w, node2_w, node3_w, node4_w, node5_w, node6_w, node7_w;
  Eigen::MatrixXd layer1_intput;
  layer1_intput = (layer1*input + layer1_bias).unaryExpr(&sigmoid);
  
  Eigen::MatrixXd l2_node1_w(1,200);
  l2_node1_w << 5.17331189e-03,  7.68277240e-03,  5.66216078e-03,  9.21382366e-02,
       -6.91212020e-03,  1.35731559e-02,  1.00294952e-01,  5.13507877e-03,
        5.35226554e-03,  1.09371172e-01,  5.54213408e-02,  1.07948933e-01,
        5.97559293e-03,  5.54446269e-02,  1.24170889e-01,  4.45453462e-03,
       -7.86451706e-02, -3.80083863e-02,  4.48130242e-03,  4.69280450e-02,
       -2.11849192e-02,  5.30847514e-03,  5.89852045e-03,  3.63028684e-02,
        8.11171914e-02, -4.54451168e-02,  1.18048858e-01,  4.48802906e-03,
        4.83187488e-03,  9.93465740e-03,  1.08049607e-01,  3.18033074e-02,
        4.91472452e-02,  6.23744355e-02,  6.01114137e-03,  2.09671947e-02,
        2.20001260e-02,  9.76072566e-03,  7.57124357e-02,  9.08541438e-02,
       -6.10434483e-02,  6.36152270e-03,  7.75002393e-02,  1.10949523e-01,
        4.51086303e-03,  4.61258928e-03, -2.67057038e-02, -1.05713573e-01,
        5.09728087e-03,  4.65988379e-03,  1.18552698e+00, -1.77321053e+00,
        6.15893330e-03,  5.65308929e-02, -2.26502780e-01,  1.06038037e-01,
        6.47203153e-03, -6.00821814e-01,  1.27448846e-01,  5.11595067e-03,
       -3.52901711e-03,  1.07223687e-02,  2.48791837e-02, -1.10526877e-02,
        8.62204438e-03, -2.18994801e-02,  5.28079991e-02,  4.45175515e-03,
        5.64425823e-03,  4.94371309e-03,  5.10093423e-03,  4.90371453e-03,
        1.11913222e-01,  4.14760887e-02,  9.02622723e-02, -2.00481907e-01,
        4.59389652e-03,  4.68610670e-03,  9.69804539e-03, -6.05253017e-02,
        5.88856276e-03,  6.37054078e-03,  4.48541757e-03,  4.61747417e-01,
        4.62956382e-03,  1.55779984e-02,  5.10155562e-03, -1.92771666e+00,
        6.19845456e-02,  5.64858782e-03,  5.16949908e-02,  4.62157569e-03,
        4.57294362e-03,  6.47109554e-03, -2.22099245e+00,  5.67249640e-02,
       -2.44707193e-02,  1.39741863e-02,  5.65028179e-03,  5.51745753e-03,
        3.08183119e-02,  7.04950274e-03,  4.55959675e-02,  7.57991104e-03,
        7.24611353e-02, -7.09701356e-02,  4.84943036e-03, -1.02722687e+00,
        6.94163438e-03,  5.46678590e-03, -8.00186733e-02,  7.72367586e-02,
        6.33410033e-03, -3.69796517e-02,  4.64720205e-03,  2.91507541e+00,
        6.08221897e-02,  4.48481062e-03,  4.45939478e-03, -9.26201005e-02,
       -4.74249850e-02,  1.33560576e-01, -3.05978585e-02,  4.67911216e-03,
        4.55578220e-03,  8.79481567e-02,  4.59385958e-03,  2.45638131e-01,
        5.23932291e-03, -1.55546305e-01,  8.92501350e-03,  4.67290847e-03,
        5.82901533e-03,  8.75290940e-03, -6.20689852e-02, -1.59091659e-02,
       -3.77114232e-02,  6.18205643e-02, -6.97935692e-04,  5.64330510e-03,
        4.78691561e-03,  4.94503241e-03,  8.49419287e-03, -3.46712692e-01,
        4.59295759e-03,  7.86767216e-01,  9.24481191e-02, -6.40165150e-03,
       -1.84622816e-01,  1.78363886e-02,  2.86443193e-02,  5.15900893e-03,
        4.55909622e-03,  4.61645019e-03,  5.65268710e-03,  1.15873616e-01,
        4.59587586e-03,  8.66676196e-03, -5.03921984e-02,  6.58891983e-03,
        4.46735427e-03,  1.22622363e-02,  6.71591630e-03,  6.56675143e-02,
       -1.65538425e-01,  1.38511375e+00,  3.32133948e-02,  6.38789439e-03,
        4.32741938e-02,  8.56311686e-02,  1.11752169e-01, -5.58187203e-02,
        1.11790742e-01,  8.60233082e-02,  5.06359177e-03,  6.12337683e-03,
        5.07772877e-02,  1.76739360e-02,  5.06100490e-03,  6.62727817e-03,
        5.65929046e-03, -1.88672120e-02,  4.51374381e-03,  9.92912115e-03,
        1.07101915e-01,  5.57642900e-03,  2.45049752e-02,  5.22933373e-03,
       -2.77096309e-02, -1.57838042e-02,  1.16555152e-01,  5.26236920e-03,
        9.38811764e-02, -7.40123295e-02, -1.39031693e-03,  5.70885867e-03,
        4.47818951e-03,  5.63689975e-03,  5.24809926e-03,  1.51749783e-02;
  Eigen::MatrixXd l2_node2_w(1,200);
  l2_node2_w << -1.26053010e-03,  8.05884109e-04,  5.66942531e-05,  1.62442312e-02,
        2.88685068e-02, -1.05984309e-02,  9.29040620e-02, -1.25608379e-03,
       -1.73776020e-03,  3.48103233e-02, -1.66909042e-03,  5.25809572e-02,
       -2.04933268e-03,  8.00389341e-02,  8.26544893e-02, -1.19027351e-03,
       -6.68597635e-02, -3.55604689e-02, -8.43095023e-04,  3.63227309e-03,
       -9.76868078e-03, -3.33402967e-05,  8.36001970e-04,  7.99687756e-04,
       -4.14873365e-02, -6.26456981e-03, -7.66387644e-02, -9.49715812e-04,
       -1.45538220e-03,  1.88628348e-04, -3.25390776e-02, -5.89135202e-02,
        4.11657171e-02,  1.05835191e-01, -2.25147255e-03,  2.76505999e-02,
        1.38884395e-02,  8.04178920e-04, -2.72206124e-02, -2.71991006e-02,
       -1.07307935e-02,  4.08664969e-04, -1.93509799e-02,  6.42965247e-02,
       -8.34399339e-04, -8.15730054e-04,  7.02978750e-02, -1.35713733e-01,
       -1.13699863e-03, -6.79029880e-04,  3.78544218e-01,  3.02267548e-01,
        6.88895212e-04, -3.59928862e-02,  2.60304423e-01,  7.33027940e-02,
       -2.14783347e-03, -8.93219097e-01, -3.78564217e-02, -1.15985402e-03,
       -5.14933584e-02,  2.53650197e-04, -3.75553455e-03, -2.42886683e-02,
        3.97203269e-04,  6.69530465e-02, -7.16252771e-02, -1.19501219e-03,
       -1.17338806e-05, -1.26023117e-03, -1.52837113e-03, -4.51031376e-04,
        7.66207330e-02, -7.05948930e-02, -3.87027930e-02, -9.77342290e-02,
       -8.52681757e-04, -6.54465279e-04, -9.51748673e-02,  9.24309172e-02,
       -2.02903870e-03,  3.73625332e-04, -8.41786995e-04, -1.57377615e+00,
       -7.76630975e-04,  9.43132494e-02, -1.09641730e-03,  7.72274530e-01,
        5.98936793e-02,  3.55323552e-05,  7.32514611e-02, -9.24977393e-04,
       -9.31548837e-04,  2.01918687e-04, -3.05103079e+00,  2.58692721e-03,
       -4.84523174e-02,  1.44531154e-04, -1.10005191e-04, -1.84531709e-03,
        3.47675215e-03,  6.80815752e-04, -9.56561419e-04,  6.37236831e-04,
        1.07190227e-02,  9.01720451e-03, -4.82622040e-04, -1.56334616e+00,
       -1.66329020e-04, -1.73230443e-03, -2.21242022e-01, -7.58349706e-02,
        6.86210472e-04, -2.68940140e-03, -7.37452728e-04,  7.96844261e-01,
       -1.05959388e-02, -9.47145731e-04, -8.60979488e-04,  7.75612615e-02,
        1.12012118e-01,  5.48215045e-02,  1.53944202e-02, -6.79947991e-04,
       -7.71677698e-04,  5.49767690e-03, -8.59972182e-04,  1.78404650e-01,
       -1.33620534e-03,  3.02683651e-01,  3.64311434e-04, -6.71003567e-04,
       -1.97339831e-03,  4.46704940e-06, -3.43468309e-02,  1.06190090e-01,
        1.04311515e-01, -6.27549153e-02,  1.29420810e-02, -2.67993981e-05,
        9.08764327e-02, -1.29056137e-03,  2.92721347e-04,  6.46448996e-02,
       -8.59649577e-04, -6.84705412e-01,  6.44253294e-03, -6.30688347e-02,
       -3.98066568e-01,  4.35329670e-04,  2.88910526e-03, -1.13385604e-03,
       -9.30797305e-04, -9.18264936e-04,  4.50092565e-05, -5.72852406e-02,
       -8.43491799e-04,  6.62112967e-04,  5.01537157e-02,  6.41526146e-04,
       -1.11535594e-03,  4.61750550e-05,  3.09641660e-04, -8.66611255e-02,
       -1.30208837e-01,  1.44558404e-01,  1.19099365e-03,  7.52339143e-04,
       -4.51906222e-02,  1.07584297e-01, -4.57367364e-02, -7.84811260e-03,
        2.45656299e-02,  2.99902672e-02, -9.24150992e-04,  7.47200135e-04,
        1.52993240e-02,  4.30407787e-04, -9.91666599e-04,  3.21155735e-04,
        4.81523249e-05,  7.05622808e-02, -8.33595075e-04,  1.88529678e-04,
       -4.44251970e-02, -1.14924176e-04, -6.03328075e-02, -1.21217388e-03,
       -5.64949636e-02, -1.04441928e-01,  4.18833090e-02, -1.22750836e-03,
       -2.43637419e-03,  1.03250219e-01,  9.27670361e-02,  7.92458115e-05,
       -9.34998073e-04, -5.09466396e-05, -1.33172711e-03,  3.21426130e-04;
  Eigen::MatrixXd l2_node3_w(1,200);
  l2_node3_w << -2.12726339e-03, -3.92966184e-03, -1.40946616e-03,  1.65390783e-02,
       -6.07255097e-02, -4.43630518e-04,  5.54410085e-02, -2.07816716e-03,
       -1.45819717e-03,  7.02594960e-02, -6.24949557e-02,  5.43700589e-02,
       -7.60503855e-04, -7.35913316e-02, -2.76148726e-02,  1.94226261e-03,
       -1.91717998e-01,  9.80880887e-04,  2.90848144e-04, -6.21972246e-02,
        4.98739781e-02, -1.53520009e-03, -3.28600885e-03, -1.34251881e-02,
        1.05024340e-01, -2.20888337e-01, -6.61308958e-02,  9.87666233e-04,
       -1.43150072e-03, -1.35907378e-03, -4.00267986e-02, -1.72644531e-02,
       -6.70437601e-02, -5.00414054e-02,  2.89477257e-03, -5.45167515e-02,
       -9.40476179e-03, -5.68251229e-03, -1.25334804e-02, -1.93873388e-02,
       -5.43775820e-02, -3.50617491e-03, -1.69469402e-01,  8.18577939e-02,
        2.27552443e-04, -1.02512376e-03,  1.11571464e-01, -2.20792984e-01,
       -2.20503160e-03, -1.20002543e-03,  3.04788051e-02,  4.50247172e-01,
       -4.10887920e-03,  3.91215483e-02, -3.81073258e-01,  2.86274389e-02,
       -2.13654146e-03, -2.78061100e-01,  1.12345962e-02, -2.19985804e-03,
        2.52013866e-02, -1.65933511e-03, -2.25180304e-02, -4.03195952e-02,
       -3.50159596e-03,  1.00232366e-01,  1.13079977e-02,  1.94993356e-03,
       -1.41457491e-03, -1.95962291e-03, -1.61998911e-03, -1.00243242e-03,
       -1.95259847e-02,  6.60438247e-02,  8.95326717e-02, -1.53568700e-02,
       -1.12014988e-03, -4.45018273e-04, -3.08582068e-01,  5.61144390e-02,
       -5.61407354e-04, -3.50764190e-03,  2.80524043e-04, -1.49411361e-01,
       -9.83909412e-04, -6.19544753e-02, -2.25017124e-03,  1.46051582e-01,
       -6.03106491e-02, -1.42365773e-03,  7.70564338e-02,  6.68108966e-04,
        7.73698878e-04, -2.41502047e-03, -1.24015827e+00, -3.54454830e-02,
       -2.18004592e-02, -3.82634233e-03, -1.43382088e-03, -5.58467882e-05,
       -1.16387577e-02, -3.46260431e-03, -2.79057891e-02, -4.85454400e-03,
       -5.40810357e-03,  5.72357651e-02, -9.13912334e-04, -1.56129103e+00,
       -1.86792215e-03, -1.49981082e-03, -2.72064020e-01, -6.70368302e-02,
       -3.96445451e-03,  1.50634479e-02, -9.41235034e-04, -1.08193309e+00,
       -4.03524872e-02,  9.77070085e-04,  4.22147576e-04,  1.04670018e-01,
        8.81359369e-02, -9.68304437e-01, -4.33823536e-02, -8.44093289e-04,
       -7.04117368e-05,  6.94813906e-02, -1.08972942e-03,  1.18034304e-01,
       -2.07007176e-03,  1.27086025e-01, -1.96240692e-03, -3.77220005e-04,
       -4.31979788e-04, -2.77012116e-03, -1.64396094e-02, -7.17454253e-02,
        8.50842597e-02,  6.04436179e-02,  7.96264356e-02, -1.41301686e-03,
        9.98681430e-02, -1.92469543e-03, -2.26982566e-03,  6.81692036e-02,
       -1.10466464e-03, -4.12938433e-01,  7.57509006e-02,  1.05685115e-01,
       -8.41341371e-01, -5.07389272e-03, -1.00430127e-02, -2.24288794e-03,
        7.85356023e-04,  6.22441967e-04, -1.41997945e-03, -2.03742550e-02,
       -1.12974036e-03, -5.12133151e-03,  1.09557758e-01, -3.69139742e-03,
        1.70294936e-03, -2.73822070e-03, -2.70387709e-03, -1.60359776e-01,
       -5.35416422e-02,  1.10051611e-01, -1.14919447e-02, -3.59365700e-03,
       -2.55658537e-01, -5.80582980e-02,  3.01863087e-02,  4.98786932e-02,
       -5.60667054e-02, -2.63502767e-03, -2.38073069e-03, -3.41335612e-03,
        8.13963004e-02, -5.10667722e-03, -2.31925257e-03, -2.68599488e-03,
       -1.41774264e-03,  3.59407962e-02,  2.22363298e-04, -1.36181723e-03,
       -1.27272625e-02, -1.44323022e-03,  3.65716126e-03, -2.21695778e-03,
        4.40574640e-02, -2.46060746e-01,  3.91353996e-02, -2.22119714e-03,
        1.06838955e-01,  1.03756358e-01, -9.10076654e-03, -1.42369710e-03,
        9.10413381e-04, -1.40955943e-03, -2.08514467e-03, -3.29447589e-03;
  Eigen::MatrixXd l2_node4_w(1,200);
  l2_node4_w << -3.10618040e-03, -5.64948281e-03, -4.74576396e-03, -2.41958502e-02,
       -4.59793762e-03,  5.00712588e-02,  1.52325907e-02, -3.03585467e-03,
       -2.47266759e-03,  2.50138499e-02, -3.09332949e-03, -4.14723435e-02,
       -1.91083066e-03, -7.52384349e-02,  2.49872862e-02, -2.16390608e-03,
       -4.48275592e-03, -2.17801125e-02, -2.94973538e-03, -3.78783979e-03,
        1.46585078e-02, -4.58639680e-03, -5.22011835e-03, -1.99958809e-02,
        6.23502759e-02, -1.63088541e-01,  4.21500209e-03, -2.58368283e-03,
       -3.00166508e-03, -5.02954518e-03,  9.28758297e-02,  5.45931005e-02,
        6.28858551e-03,  6.72407761e-03, -1.76207708e-03,  7.41789860e-03,
       -6.20019396e-03, -4.80502691e-03,  1.05706676e-01, -5.63091909e-03,
        2.65691520e-02, -4.57583635e-03, -2.65846645e-02,  7.15475114e-03,
       -2.96816856e-03, -3.26324264e-03,  3.51698857e-02, -2.72194355e-03,
       -3.20511647e-03, -3.39273100e-03, -2.86457926e-01,  5.92997310e-01,
       -4.65577070e-03, -7.73136580e-02, -3.31505266e-01, -6.47367198e-02,
       -1.40275988e-03, -7.55518355e-02, -3.33197764e-02, -3.22178532e-03,
       -4.32867352e-02, -5.29644425e-03,  9.14145893e-02,  3.48768588e-02,
       -5.84887739e-03, -7.53294805e-03,  4.21789263e-02, -2.16107108e-03,
       -4.71837982e-03, -3.04799929e-03, -2.87232387e-03, -3.69044215e-03,
       -6.21792673e-02, -3.46301226e-02,  4.06884267e-02,  3.85522948e-03,
       -3.28153437e-03, -3.22461357e-03,  5.66197227e-02,  2.73697339e-02,
       -1.96004499e-03, -4.55687707e-03, -2.95261307e-03, -3.21658784e-02,
       -3.26773257e-03, -7.29957148e-02, -3.29040294e-03,  3.37470054e-01,
       -4.29245680e-02, -4.73078048e-03,  3.88854476e-02, -2.69956713e-03,
       -2.65901211e-03, -5.15660952e-03, -1.17806042e+00, -1.24113268e-02,
       -7.14791671e-03, -4.52744771e-03, -4.64698358e-03, -2.16220842e-03,
       -7.73361631e-03, -5.50178645e-03, -2.45215137e-02, -5.02312391e-03,
        8.16559928e-03,  7.16902724e-02, -3.61875957e-03,  8.29016911e-01,
       -4.90455838e-03, -2.48554610e-03, -4.73530577e-03, -5.40996129e-02,
       -4.67330382e-03, -7.37544946e-02, -3.27291253e-03, -1.62005256e+00,
        2.16198035e-02, -2.59070191e-03, -2.88923622e-03, -2.90432872e-02,
        8.97737499e-02, -4.40731779e-01, -8.18937407e-02, -3.27566891e-03,
       -3.12917622e-03,  7.87783870e-02, -3.26609583e-03, -4.58998737e-02,
       -2.99653754e-03,  2.19724645e-01, -5.69695287e-03, -3.20853333e-03,
       -2.05833879e-03, -5.60821168e-03, -5.29993060e-02, -5.56362102e-02,
        4.84301213e-02, -4.43326614e-02,  6.55963385e-03, -4.70979334e-03,
       -3.74512818e-02, -3.00047392e-03, -5.85446834e-03,  1.12920895e-01,
       -3.27209539e-03, -2.05375096e-01, -3.41701276e-02,  6.17221821e-03,
        1.23693901e-01, -5.38244817e-03, -7.60983519e-03, -3.29882749e-03,
       -2.65695033e-03, -2.72107588e-03, -4.73584837e-03,  5.73059831e-02,
       -3.28965286e-03, -5.25269236e-03, -5.76324343e-02, -4.83975109e-03,
       -2.26702895e-03, -4.14592314e-03, -5.27885338e-03,  7.64635948e-02,
       -9.02329929e-03, -4.36853848e-02, -1.54341401e-02, -5.01150987e-03,
       -6.34246162e-02, -5.22202461e-02,  8.44674966e-02, -4.89194850e-02,
       -6.42070334e-02, -7.16604881e-03, -3.41901538e-03, -5.01618707e-03,
        6.75355477e-02, -5.28868406e-03, -3.35218458e-03, -5.26272953e-03,
       -4.74138088e-03,  2.11633295e-02, -2.96968684e-03, -5.03335046e-03,
        5.47779889e-02, -4.61502457e-03,  7.45616459e-02, -3.23202689e-03,
       -5.34359262e-02, -2.29759756e-02,  8.19283214e-02, -3.22447373e-03,
       -2.88832232e-02,  3.79517271e-02, -1.01893945e-02, -4.74130447e-03,
       -2.62576614e-03, -4.69263125e-03, -3.01652363e-03, -5.86256076e-03;
  Eigen::MatrixXd l2_node5_w(1,200);
  l2_node5_w << -2.34859894e-03, -5.24477879e-03, -5.13355683e-03,  4.11512290e-03,
        3.27928683e-02, -2.10771687e-02,  3.87168978e-02, -2.27347043e-03,
       -1.82293396e-03,  4.59565114e-02,  1.88728588e-02,  8.02060394e-02,
       -4.75594453e-03,  2.86817370e-02, -5.66731446e-02, -3.84608477e-03,
       -5.08787051e-03,  4.33376631e-03, -3.77306372e-03,  8.10651779e-03,
       -2.95054064e-02, -5.06608843e-03, -4.93631593e-03,  1.72421734e-02,
       -8.47064924e-02, -8.85656861e-02,  2.28908471e-02, -3.81191623e-03,
       -2.60834005e-03, -5.84371502e-03, -2.88909048e-02,  1.12927326e-02,
       -9.92494219e-02,  5.87784242e-02, -4.64081790e-03, -2.34804820e-02,
       -9.58200377e-03, -5.32862902e-03,  1.25710834e-02, -6.48280001e-02,
        7.32701870e-02, -4.36982672e-03,  2.18437462e-02,  1.17748552e-02,
       -3.77663992e-03, -3.61572796e-03,  5.89994860e-02,  3.06595118e-02,
       -2.50457949e-03, -3.66339269e-03, -6.58720687e-01,  5.68276878e-01,
       -4.51245326e-03, -3.89648588e-02, -6.65370042e-02,  4.26955517e-02,
        7.88744637e-05,  2.49863869e-01, -5.11405044e-02, -2.53469114e-03,
        7.35025709e-02, -5.93710088e-03, -3.38782145e-02, -7.19461938e-02,
       -5.62411207e-03,  1.71774897e-02,  5.44799259e-03, -3.85231931e-03,
       -5.14343274e-03, -2.44641481e-03, -2.41396388e-03, -3.83259027e-03,
       -8.03445019e-02,  2.64289092e-02, -7.21151258e-02,  4.82287359e-02,
       -3.54539357e-03, -3.84727952e-03, -5.16070952e-02, -3.71734489e-03,
       -4.73367362e-03, -4.33523414e-03, -3.77304541e-03,  7.24991434e-02,
       -3.66363085e-03,  4.83184106e-02, -2.64551838e-03,  7.32942138e-01,
       -9.22184219e-02, -5.14185803e-03, -9.44412640e-02, -3.81485922e-03,
       -3.80650093e-03, -5.27340132e-03, -5.35081271e-01,  8.38123422e-03,
       -6.07370592e-02, -5.60897274e-03, -5.15169417e-03, -4.65410700e-03,
       -2.39451146e-03, -5.26718738e-03,  2.30790259e-02, -4.95719072e-03,
       -8.29451689e-02, -7.43880572e-02, -3.82227568e-03, -2.89938314e-01,
       -5.46034755e-03, -1.89818417e-03, -5.11932654e-02, -7.99233821e-02,
       -4.51730717e-03,  6.21964032e-02, -3.70871449e-03, -8.49519622e-01,
       -2.03865068e-02, -3.81137249e-03, -3.78063732e-03,  5.42077354e-03,
        4.37428948e-02, -1.33843587e-01, -7.22981969e-02, -3.77682058e-03,
       -3.76171728e-03,  2.43290875e-02, -3.55060574e-03, -5.79739217e-02,
       -2.18242694e-03,  9.45147528e-02, -5.47955067e-03, -3.83974147e-03,
       -4.71774881e-03, -5.88915860e-03, -7.57084607e-03,  8.47205207e-03,
        7.41530002e-02, -2.51677217e-02, -3.60772204e-02, -5.14473380e-03,
       -5.28908973e-02, -2.38725879e-03, -5.57468560e-03,  1.29194651e-01,
       -3.54498876e-03, -2.12254654e-01,  3.27027867e-02, -7.73818150e-02,
       -2.19502584e-01, -5.55009953e-03, -1.81568689e-03, -2.63796830e-03,
       -3.80514292e-03, -3.80903907e-03, -5.13884649e-03, -4.08653349e-02,
       -3.55020693e-03, -5.06304335e-03,  6.49646981e-02, -4.65511614e-03,
       -3.86845135e-03, -5.69564574e-03, -5.32787727e-03,  2.41741039e-02,
        6.77544725e-02, -1.04966067e+00,  1.38792079e-02, -4.73550420e-03,
        4.57235635e-02, -4.01267142e-02,  5.22789014e-02, -7.83842583e-02,
        7.12759406e-02, -2.32115835e-02, -2.85314443e-03, -4.71770491e-03,
        1.69820619e-02, -5.50859362e-03, -2.73553574e-03, -5.30556783e-03,
       -5.13754562e-03, -3.10345587e-02, -3.77729800e-03, -5.84236015e-03,
        6.83726223e-02, -5.13300837e-03, -4.24194084e-02, -2.49142601e-03,
       -9.57031793e-02, -2.64766717e-03, -4.92255848e-02, -2.46130033e-03,
        7.74495076e-02, -1.87953864e-02,  2.59801740e-02, -5.14224287e-03,
       -3.80759090e-03, -5.14496025e-03, -2.20029296e-03, -6.36685336e-03;
  Eigen::MatrixXd l2_node6_w(1,200);
  l2_node6_w << 5.87870977e-04,  1.28180526e-03,  1.12581549e-03, -9.16834780e-02,
       -7.87388191e-03, -7.40941578e-02, -8.34349811e-02,  5.54162556e-04,
        2.30545394e-04, -2.12223921e-02,  1.94817639e-03, -8.56146821e-02,
       -2.52439890e-04, -9.53697580e-02,  4.74279235e-02, -1.09694381e-04,
        6.54461295e-02,  8.83057811e-03,  6.11053993e-04, -1.62419230e-02,
       -4.23872685e-02,  1.23809883e-03,  1.73220993e-03, -5.21475011e-03,
        8.10573091e-03,  1.00720427e-01, -1.14006215e-01,  2.79656272e-04,
        6.70266397e-04, -1.83927677e-06,  4.65178626e-02, -5.69884717e-02,
       -1.17766898e-01, -2.14879883e-02, -6.38244435e-04, -1.53419841e-02,
       -5.67832035e-03,  3.94783876e-04, -6.26665274e-03, -1.07256520e-01,
       -4.71653151e-03,  1.78288647e-03,  1.54206410e-02, -1.14659316e-01,
        6.23184491e-04,  9.78850104e-04, -1.36310826e-02,  1.23072562e-01,
        6.92833184e-04,  1.06234300e-03, -8.50722504e-01,  5.18946725e-01,
        1.99559167e-03,  8.51194886e-03,  1.00841164e-01, -2.62264756e-03,
       -5.89754099e-04,  5.64048350e-01, -2.19589720e-02,  6.98812832e-04,
        2.05543867e-02, -9.72420154e-05, -9.44986622e-02, -6.11956169e-02,
        7.11794553e-04,  1.00805162e-02, -4.56704022e-02, -1.15305658e-04,
        1.12128630e-03,  6.47221571e-04,  4.77318318e-04,  1.14438531e-03,
       -5.39555749e-02,  2.96432158e-02,  5.18224145e-03,  8.79835204e-02,
        9.89046145e-04,  8.69768282e-04,  5.40351904e-02, -8.06933520e-03,
       -2.48494342e-04,  1.75977512e-03,  6.12991263e-04,  1.86316394e-01,
        9.79970659e-04, -1.15282021e-01,  7.63423042e-04,  6.55033552e-01,
       -3.73777972e-02,  1.12329583e-03, -1.08208028e-02,  3.31126336e-04,
        3.14647259e-04,  1.16471366e-03,  8.35099590e-01, -5.84657219e-03,
        3.30950885e-02, -6.80999105e-04,  1.09713546e-03, -1.30403916e-04,
        6.98285937e-04,  1.34647342e-03, -4.50235534e-04,  1.66029904e-03,
        2.21426306e-02, -1.27889986e-04,  1.10397236e-03,  1.14654032e+00,
        6.02508463e-04,  1.73579011e-04,  1.42402371e-01, -8.83547326e-02,
        2.01025440e-03,  4.18896989e-02,  9.79800068e-04, -4.50353727e-01,
       -2.80539568e-02,  2.87436579e-04,  5.63323423e-04, -2.45060101e-04,
       -9.74993577e-02,  1.71408063e-01,  7.22363971e-02,  9.68391087e-04,
        7.50534996e-04,  1.89736281e-02,  9.79980604e-04, -2.61976337e-01,
        4.89557642e-04,  1.59461017e-01,  6.17647314e-04,  8.48031692e-04,
       -2.13959350e-04,  1.96595546e-04, -3.50687920e-02, -6.87795784e-02,
        2.60039237e-02, -6.69297568e-02,  9.59566168e-03,  1.11824459e-03,
        7.24833046e-02,  6.14056727e-04,  6.08202359e-04,  1.40156321e-01,
        9.83356734e-04, -1.18263420e-01, -4.88084585e-02,  6.28683522e-02,
        2.51565491e-01, -1.22243007e-03,  5.97466515e-04,  7.43603701e-04,
        3.18529559e-04,  3.56194298e-04,  1.12428968e-03,  2.90523368e-02,
        9.94204994e-04,  1.28056335e-03, -7.25247025e-02,  1.89254974e-03,
       -2.04750772e-05, -4.19949690e-04,  1.16461647e-03,  4.50095324e-02,
        1.74420915e-01, -1.60409748e+00, -6.77398219e-03,  1.86951769e-03,
       -3.30931165e-02, -2.40535677e-02, -2.74547240e-02, -3.00286929e-02,
        1.73733283e-02,  5.87622789e-02,  9.07795825e-04,  1.87011727e-03,
        7.39469570e-02, -1.18645774e-03,  8.37931588e-04,  1.20073655e-03,
        1.12488926e-03,  4.86399654e-02,  6.24160572e-04,  2.47689571e-06,
       -8.16093223e-02,  1.11825802e-03, -7.03273012e-02,  6.58499443e-04,
       -2.36054950e-02,  7.16601890e-03, -7.51277020e-02,  6.39344021e-04,
        5.50999075e-02, -7.20296487e-02,  8.16249880e-03,  1.10745794e-03,
        3.22101210e-04,  1.11398638e-03,  4.98791594e-04, -8.53031362e-04;
  Eigen::MatrixXd l2_node7_w(1,200);
  l2_node7_w << -5.23604219e-03, -4.38495245e-03, -4.36192669e-03,  3.61941651e-02,
        2.61191092e-02,  4.70897773e-02, -9.38168497e-02, -5.34649543e-03,
       -5.70386710e-03, -2.99768880e-02, -7.20253160e-02, -9.46120921e-02,
       -6.41007336e-03, -1.38387746e-02,  5.82757472e-02, -4.06390716e-03,
       -4.41376904e-01, -5.64371419e-02, -3.62483117e-03, -7.87052198e-02,
        4.49766432e-02, -4.25951242e-03, -3.95721480e-03, -1.80989380e-02,
       -6.32801543e-02,  9.74495530e-02, -1.24505037e-01, -3.74884550e-03,
       -4.74103443e-03, -6.51242563e-03, -7.11499370e-02,  5.43893841e-02,
        4.22089196e-02,  3.53990658e-02, -6.26978288e-03, -9.85835903e-02,
       -1.54429727e-02, -6.46843582e-03, -4.55756629e-02,  2.48869436e-03,
       -8.13951603e-02, -3.54878447e-03, -1.11810948e-01,  6.14825610e-02,
       -3.60644090e-03, -3.72191753e-03, -9.53051235e-02, -6.16990380e-02,
       -5.12255504e-03, -3.56685887e-03, -1.21631566e+00,  4.38882914e-01,
       -3.56017934e-03, -5.91840984e-02,  5.45975901e-01, -1.85154649e-02,
       -6.73849937e-03,  9.67949088e-01,  5.87701054e-02, -5.05283583e-03,
       -9.43643104e-02, -6.58487935e-03, -2.44521301e-02, -1.07659433e-01,
       -4.84127915e-03,  7.72595548e-02,  1.33386792e-01, -4.07963597e-03,
       -4.33876381e-03, -5.09061387e-03, -5.29728061e-03, -3.17252314e-03,
       -2.42570403e-02, -3.59126173e-02,  2.08782972e-02,  1.39315259e-01,
       -3.77771898e-03, -3.43106966e-03, -2.12450464e-01, -1.09005031e-01,
       -6.34849633e-03, -3.58624222e-03, -3.62095426e-03,  3.18910864e-01,
       -3.66758325e-03, -1.07658050e-01, -4.94318727e-03,  5.17931398e-01,
       -6.83911122e-02, -4.35639310e-03, -1.07462556e-01, -3.73076729e-03,
       -3.72316604e-03, -4.36303062e-03,  2.12218625e+00, -3.20863187e-02,
       -5.69111583e-02, -7.97445080e-03, -4.35452790e-03, -6.09302107e-03,
       -1.41971390e-02, -4.34117932e-03, -2.45906188e-02, -4.37338415e-03,
        1.29783561e-02,  5.23814544e-02, -3.19550829e-03, -5.05972267e-01,
       -4.73569168e-03, -5.82028539e-03, -3.88013544e-02,  2.45264067e-02,
       -3.45850302e-03, -8.91688211e-02, -3.61338139e-03, -1.11097905e+00,
       -2.03567546e-02, -3.74778643e-03, -3.66205934e-03,  1.48811202e-02,
        7.65929737e-02, -7.75099640e-01, -5.17447164e-02, -3.52804259e-03,
       -3.51509158e-03,  1.72076487e-02, -3.78503245e-03, -3.42257186e-01,
       -5.40348606e-03, -3.78971090e-02, -4.96958038e-03, -3.43872048e-03,
       -6.27516467e-03, -5.14510435e-03, -8.69934436e-02, -1.38330925e-02,
       -8.01192015e-02,  3.82314883e-02,  4.62118159e-02, -4.33840240e-03,
       -1.12563810e-01, -5.14997026e-03, -4.74276219e-03,  1.38892574e-01,
       -3.78567131e-03, -2.27397542e-01, -1.14256042e-02, -1.46385213e-02,
       -5.25855708e-01, -8.85360120e-03, -1.35135479e-02, -4.91753091e-03,
       -3.72115925e-03, -3.71987645e-03, -4.35912087e-03, -1.85041277e-02,
       -3.76659040e-03, -4.76345575e-03,  4.61076016e-02, -3.50493106e-03,
       -4.00363470e-03, -7.66413027e-03, -4.40954960e-03, -1.86887368e-01,
        2.84365945e-01, -1.72215183e+00, -1.55924201e-02, -3.52351784e-03,
       -1.84224992e-01, -1.42948240e-02, -2.40534149e-04, -3.41067239e-02,
        2.20719944e-02, -7.06671359e-02, -4.79180547e-03, -3.52727366e-03,
       -7.04098039e-02, -8.85786354e-03, -4.91286727e-03, -4.37785786e-03,
       -4.35864911e-03, -9.53439467e-02, -3.60535536e-03, -6.50487096e-03,
       -1.34497117e-02, -4.34209059e-03, -2.55799723e-02, -5.07357230e-03,
        4.86106547e-02, -2.21310026e-01,  5.49616142e-02, -5.10791713e-03,
       -3.97546417e-02,  4.70091763e-02, -8.12652259e-02, -4.40555773e-03,
       -3.73626238e-03, -4.33860808e-03, -5.38290540e-03, -7.53840039e-03;
  Eigen::MatrixXd l2_node8_w(1,200);
  l2_node8_w << 7.04188097e-03,  6.55799429e-03,  7.46490640e-03,  4.17163477e-02,
       -3.85632045e-02,  5.01685109e-02, -2.76016805e-02,  7.10663713e-03,
        6.78815734e-03, -1.99540451e-02,  1.36864692e-02, -9.19383056e-03,
        5.07695008e-03,  5.89460169e-02, -1.01580392e-01,  3.86955815e-03,
       -8.32079540e-02, -2.99421175e-02,  4.81250694e-03,  7.49368387e-03,
        3.93613276e-02,  7.52969982e-03,  6.82862125e-03, -7.54222439e-03,
        5.45042459e-02,  5.47797023e-02, -5.55334250e-02,  4.27666129e-03,
        6.54438977e-03,  7.47904804e-03,  4.87243440e-03,  1.17171114e-02,
       -1.28423144e-01, -6.19097843e-02,  4.55724001e-03,  8.43452340e-03,
        9.62538092e-03,  7.37101913e-03, -1.09490380e-01, -6.44050984e-02,
       -7.95169217e-02,  5.86142445e-03, -1.15228046e-02, -7.38763403e-02,
        4.81637258e-03,  5.55025125e-03,  3.14864088e-03,  1.08738065e-01,
        7.05035949e-03,  5.47843618e-03, -1.20161542e+00,  2.02303222e-01,
        6.00438156e-03, -7.08036356e-02,  5.03365794e-01,  1.84598782e-02,
        6.73354529e-03,  1.06998616e+00,  4.70613983e-02,  6.97421005e-03,
        4.70308611e-02,  7.41036045e-03, -3.55834591e-02, -3.20764957e-02,
        6.98111735e-03,  2.87194108e-02,  7.78379357e-02,  3.87575159e-03,
        7.50551683e-03,  6.89734149e-03,  6.75562455e-03,  5.36313309e-03,
       -3.46120778e-03,  2.36113890e-04,  3.27849790e-02,  1.05821551e-01,
        5.66069224e-03,  5.01364674e-03,  3.39365371e-02, -1.02616436e-01,
        5.09485690e-03,  5.86949516e-03,  4.81332535e-03,  3.02963633e-01,
        5.47172028e-03, -7.93591072e-02,  6.92055857e-03,  7.54058431e-01,
       -9.81356001e-02,  7.48819902e-03,  4.76659505e-02,  4.34704172e-03,
        4.31665542e-03,  7.43106839e-03,  2.21012635e+00, -6.16800987e-03,
        5.11397900e-02,  7.63972310e-03,  7.54580277e-03,  5.25014650e-03,
        6.90098471e-03,  6.93419300e-03, -1.19503610e-02,  6.59155266e-03,
       -4.91542440e-02, -6.50650026e-02,  5.31725635e-03,  4.44323868e-01,
        7.73379153e-03,  6.78806288e-03,  6.58575091e-02, -9.55162277e-02,
        5.86333356e-03, -9.93938721e-02,  5.39403191e-03, -1.28649522e+00,
       -1.33124909e-01,  4.28808326e-03,  4.73926110e-03, -9.12414862e-02,
        3.69036201e-02,  1.33709458e-01,  3.02120630e-02,  5.26132888e-03,
        5.00350412e-03, -7.22645424e-02,  5.65171107e-03, -1.47238537e-01,
        7.08717447e-03, -2.79955873e-01,  6.70984993e-03,  4.99774027e-03,
        5.19006198e-03,  7.46348306e-03,  4.19048286e-02, -1.18360955e-01,
       -6.07858323e-02, -8.78289946e-02, -7.47222412e-02,  7.51213686e-03,
        4.87413050e-02,  6.91178580e-03,  7.01364720e-03,  9.73060944e-02,
        5.66060013e-03, -7.14571190e-01, -1.24529271e-01, -7.79275912e-02,
       -5.13652361e-03,  7.72582371e-03,  6.84905943e-03,  6.87252972e-03,
        4.32300856e-03,  4.38097169e-03,  7.47984927e-03, -2.60617939e-02,
        5.65410005e-03,  6.51807805e-03, -6.76126444e-02,  5.91977524e-03,
        3.96849804e-03,  7.61241777e-03,  7.37598105e-03, -8.05103468e-02,
        2.64568438e-01, -1.44687645e+00, -2.89139680e-03,  6.08598655e-03,
       -1.53403850e-02, -6.33666169e-02,  3.03042959e-02, -1.20169416e-01,
        3.79860732e-02, -1.07400585e-01,  6.90144240e-03,  6.20260241e-03,
       -8.98585682e-02,  7.71671291e-03,  6.97831566e-03,  7.36675863e-03,
        7.47448201e-03, -9.57761570e-04,  4.81641441e-03,  7.47768059e-03,
        2.10762474e-02,  7.54771709e-03, -5.66472189e-02,  6.96690363e-03,
       -6.62593074e-02,  8.53365885e-02, -3.22779151e-03,  6.98821359e-03,
       -1.24552182e-02, -6.12085673e-02, -9.67297143e-02,  7.45438339e-03,
        4.33935777e-03,  7.52211270e-03,  7.08358234e-03,  7.39079254e-03;
  Eigen::MatrixXd l2_node9_w(1,200);
  l2_node9_w << -7.63471442e-04, -1.91465917e-03, -1.79252572e-03, -5.52508945e-02,
       -3.25912664e-02, -6.91598158e-03, -3.46910000e-02, -7.51276912e-04,
       -5.07791831e-04,  2.74588665e-04, -4.51373366e-02, -4.50689571e-02,
        1.53040543e-04,  4.22327490e-02,  4.66190554e-02,  2.91405329e-04,
       -4.64726411e-01,  1.71004630e-03, -3.29968401e-04, -4.67867256e-02,
        2.40127051e-02, -1.80869601e-03, -1.67533975e-03, -8.01914748e-03,
        6.16981852e-02, -6.88709884e-02, -4.91477424e-02, -5.35679803e-05,
       -5.81450355e-04, -1.51641246e-03, -6.57380781e-02, -9.84105481e-02,
       -5.11091555e-02, -7.54488347e-02,  4.75473484e-04, -3.98190588e-02,
       -5.95846385e-03, -1.86911261e-03, -9.73262504e-02,  4.89761596e-04,
        6.28604779e-03, -1.36799957e-03, -6.26808527e-02, -9.63496133e-02,
       -3.52607855e-04, -6.42207211e-04, -3.39965787e-02, -8.92410198e-02,
       -8.01280265e-04, -7.34860361e-04, -1.99144875e+00,  1.32117335e-01,
       -1.41228703e-03,  5.46570370e-02,  8.67132888e-04, -1.58375569e-02,
       -6.31815523e-04,  1.42228011e+00, -6.35814509e-02, -7.92494337e-04,
        5.45187148e-02, -1.63703998e-03, -7.39788441e-02, -1.16923737e-01,
       -2.19087632e-03, -1.87006032e-02,  1.39077872e-01,  2.98147985e-04,
       -1.83246322e-03, -6.85430852e-04, -5.37174837e-04, -9.07242465e-04,
       -9.32345058e-02, -5.00626796e-02, -1.75128536e-02,  1.20465817e-01,
       -6.60804916e-04, -6.04024391e-04, -2.57153840e-01, -5.03689987e-02,
        1.85038232e-04, -1.35051328e-03, -3.33256963e-04,  6.54717190e-01,
       -6.43404269e-04,  4.25891888e-02, -8.12586460e-04,  7.64186919e-02,
       -5.98769886e-02, -1.81751599e-03, -1.07988489e-01, -1.44247601e-04,
       -1.18599739e-04, -1.98067157e-03,  2.96449376e+00, -1.80186932e-02,
        1.09352310e-02, -2.63840581e-03, -1.85409231e-03,  2.24746907e-04,
       -6.69612743e-03, -1.91748400e-03, -1.47565346e-02, -1.61878778e-03,
        3.74163036e-02, -1.54781541e-02, -8.63353790e-04,  1.51334645e-01,
       -2.08344186e-03, -4.89565402e-04, -1.14365041e-02, -1.03649861e-02,
       -1.43071632e-03,  2.86295826e-02, -6.46125334e-04, -5.30924337e+00,
       -5.74367162e-02, -5.82878211e-05, -2.80814895e-04, -8.94876376e-02,
       -5.17449864e-02, -1.63608680e+00,  2.88753141e-02, -6.47223881e-04,
       -4.77779820e-04,  6.02440561e-02, -6.48712843e-04, -2.75929437e-01,
       -7.41798942e-04, -4.95810977e-01, -1.70234122e-03, -5.87610804e-04,
        1.80305105e-04, -2.38339579e-03,  2.08532927e-02, -7.64366838e-02,
       -4.93257598e-02,  4.73957152e-02, -1.03244228e-01, -1.83661947e-03,
        1.19506581e-02, -6.67583956e-04, -2.11377210e-03,  7.40751024e-02,
       -6.53698819e-04,  2.23160053e-01, -5.48445379e-02, -9.46252190e-02,
       -6.89542846e-01, -3.31574274e-03, -6.01747039e-03, -8.07416472e-04,
       -1.16941510e-04, -1.63184965e-04, -1.80892933e-03, -1.18489006e-01,
       -6.66554498e-04, -1.78562030e-03, -7.76697881e-02, -1.47108309e-03,
        2.00107272e-04, -2.31595784e-03, -2.00257943e-03, -8.09193559e-02,
        2.98432765e-01, -4.28390967e-01, -6.69021469e-03, -1.50486765e-03,
       -1.16392890e-01, -3.86803399e-02, -4.46196203e-02, -6.49933236e-02,
       -3.48087266e-02, -9.48578574e-02, -8.73968290e-04, -1.50226874e-03,
        5.23734909e-02, -3.29733957e-03, -8.51079533e-04, -1.98838822e-03,
       -1.80474703e-03,  5.90044738e-02, -3.54644396e-04, -1.51687935e-03,
        4.03409524e-02, -1.84217771e-03, -1.25340892e-02, -8.00433936e-04,
       -2.60339137e-02, -1.96588922e-01, -4.87328425e-02, -8.03964509e-04,
       -1.12582955e-01, -4.36629495e-02, -1.93960453e-02, -1.78234176e-03,
       -8.39637824e-05, -1.84103698e-03, -7.48136703e-04, -2.52425688e-03;
  Eigen::MatrixXd layer2_bias(9,1);
  layer2_bias << 0.01668477,  0.02692392,  0.02630559, -0.01386332,  0.01512696,
       -0.07395734, -0.03719363,  0.05980836,  0.03385783;
  Eigen::MatrixXd layer2(9,200);
  layer2 << l2_node1_w, l2_node2_w, l2_node3_w, l2_node4_w, l2_node5_w, l2_node6_w, l2_node7_w, l2_node8_w, l2_node9_w;
  Eigen::MatrixXd result;
  result = (layer2*layer1_intput + layer2_bias).unaryExpr(&sigmoid);
  std::cout<<"Nmin output prob: "<<result.transpose()<<std::endl;
  double max_value = -1;
  int max_idx;
  for(int i=0;i<9;i++){
    if(max_value<result(i,0)){
      max_idx = i;
      max_value = result(i,0);
    }
  }
  return max_idx+2;
}

void dbtrack::MLP_vel_v1(Eigen::MatrixXd input){
  Eigen::MatrixXd node1_Weight(200,1);
  node1_Weight << 1.40526740e-04,  6.40829798e-04,  5.45176339e-04, -1.41000470e-04,
       -5.57905912e-05, -1.69590052e-04,  3.69495456e+00,  1.21088844e-02,
        1.83418153e+00,  1.21162370e-03,  1.95117006e-01,  1.12711592e+00,
        3.62941612e+00,  3.19919381e-04, -5.78796594e-04, -1.37850871e-03,
        1.64448983e-04,  5.68677126e+00,  1.08041852e-01,  1.37391350e-04,
        6.76498623e-03,  3.00281428e-03,  9.18007010e-04,  1.67328894e-03,
       -1.36609731e-04, -1.27521676e-04,  4.73596002e-03,  1.37276600e-01,
        1.66085816e-02,  9.04899993e-03,  1.87424969e-04,  4.13939832e-03,
        6.26569123e-02,  3.02374428e-02,  1.32638951e-04, -1.47925287e-04,
        6.27915286e+00,  4.83853782e-05, -7.98311461e-05,  1.53042696e-04,
        1.41136457e-04,  8.35053700e-03,  1.01649998e-02,  9.35597434e-02,
        1.36460029e-04,  5.18182324e-02,  5.37823163e-05,  1.35553765e-04,
        1.28235994e-02,  1.21250338e-04,  1.17021253e-02,  1.56948742e-07,
        4.36434513e-02, -2.25227312e-03,  9.86564359e-03,  2.02904307e-03,
        1.49543163e-03,  1.16208527e-04, -7.97550296e-05,  2.64270673e-02,
        1.45103787e-02, -1.64788569e-04,  1.42150202e-04,  4.00432252e-04,
       -1.04271914e-04,  1.06249278e-04,  2.26135120e+00,  1.97727499e-04,
        1.63108293e-03,  3.32177984e-03,  1.21730804e-04,  1.35606208e-04,
       -1.83087925e-03,  2.34794484e-03,  3.15779447e-02,  1.75933962e-02,
       -4.21362417e-05,  2.15481718e-02, -1.71978419e-05,  5.94207292e-01,
       -1.34739633e-04,  1.64931323e-02,  1.92992843e-03, -1.29458460e+00,
       -2.17064468e-05,  1.21881428e-04,  7.43648799e-02,  6.25871887e-04,
       -1.01974967e-04, -1.18876066e-04,  4.05084469e-04,  5.22832274e-04,
        6.24511090e-04,  6.36314707e-04, -1.19985982e-04, -1.62204691e-05,
        1.48206048e-04,  4.02619414e-01,  6.22688707e-04,  1.07500595e-02,
        5.96311807e+00,  3.13788174e-02, -5.56815911e-03, -4.20614840e-05,
        1.65935875e-02,  1.47962895e-04,  1.39127248e-04,  5.48106877e-04,
        9.09794144e-02,  4.59687133e-01, -1.28336156e-04,  1.48832161e-03,
        9.36632564e-02,  1.98221778e-02,  6.56108621e-02,  5.10879575e-04,
        1.17489682e-04,  1.21531316e-04, -4.81064976e-03,  4.29465804e-04,
        2.09094665e-03, -6.25418028e-04,  4.73198986e-04,  4.07025579e-03,
        4.33507191e-04,  2.15322711e-01,  9.17725186e-02,  3.09661135e-03,
        1.67663219e-03, -1.33396599e-04,  5.33826124e-02, -1.28019889e-04,
        1.32806494e-04,  6.25035243e-04, -3.98256843e-01,  1.35079597e-02,
       -1.19717493e-04,  2.39169304e-03,  6.81689420e-04,  6.04765318e-02,
        7.28185729e-04,  1.39250804e-04,  5.00863124e-02,  5.10486906e-05,
        6.86416350e+00,  4.79505604e-05,  1.63604386e-02,  1.33782115e-04,
        1.31474386e-04,  1.19352388e-04,  1.22101185e-04,  8.41547389e-03,
        6.18367030e-01,  8.89965473e-04,  2.41853061e-03,  4.71484169e-04,
        2.20039077e-03, -1.28126742e-04,  1.21463258e-04,  1.43884481e-04,
       -1.65834551e-05,  1.44113508e-04,  2.38688542e-04,  1.16328291e-02,
       -7.89742960e-04, -3.98163794e-04, -4.21407702e+00, -1.54448742e-04,
       -5.49633786e-05,  1.25685900e-04,  1.79882130e-01,  6.70977644e+00,
       -1.40946582e-04,  3.10689033e-01, -7.98341735e-05,  3.99165136e-04,
        8.22126759e-03,  7.76129147e-03,  7.61639455e+00, -1.10283890e-04,
        4.41203902e-04, -1.38315394e-04,  1.06496217e+00, -1.23469844e-04,
        7.07243645e-04, -1.18835210e-04, -1.24199797e-04, -1.55180319e-03,
        4.23680594e-03,  1.22107854e-04, -2.50927284e-03, -7.32363684e-05,
        6.11749214e-04, -1.19744095e-04, -1.26420574e-04, -7.90219636e-05,
        9.87544122e+00,  1.28500325e-04, -1.15631261e-05,  1.62303833e-02;
  Eigen::MatrixXd node2_Weight(200,1);
  node2_Weight << 3.57089428e-06,  1.38433950e-05,  1.02470680e-05, -7.41371401e-06,
       -1.52480087e-04, -7.16052609e-06, -7.48969133e+01, -4.24072506e-04,
        9.80751004e+01, -4.25416358e-05, -2.78857766e-02, -1.65579187e-01,
       -7.44830386e+01, -3.50448463e-04,  5.52248097e-06,  4.35751478e-05,
        3.39855761e-06, -6.94139419e-02, -2.18246670e-02,  3.04305714e-06,
       -8.61457458e-04, -7.83991488e-06,  2.62005045e-05, -3.12675238e-05,
       -7.36738895e-06, -6.77483935e-06, -1.32293382e-04, -9.50087145e-02,
       -4.77141986e-04, -6.93173354e-04,  3.95186337e-06, -1.52569868e-05,
        4.14520270e-03,  3.81307994e-03,  4.19979491e-06, -6.72448496e-06,
        7.39011820e+00,  1.24918124e-05, -6.07592210e-06,  3.04668026e-06,
        3.59297020e-06, -7.35057336e-04, -3.50025989e-04, -1.93132665e-02,
        3.50794082e-06, -1.31396886e-03,  1.12299810e-05,  3.63325038e-06,
       -5.31940515e-04,  3.62542155e-06, -4.08123195e-04,  7.47330327e-06,
        7.35998545e-03, -4.57131787e-06, -7.97026571e-04, -3.81448733e-04,
       -6.15893765e-05,  3.57264328e-06,  4.32596864e-06, -1.41965893e-03,
       -8.08292962e-04, -7.04579250e-06,  4.16183499e-06, -7.80173171e-06,
       -3.20776616e-04,  3.17199225e-06, -6.19877759e-02, -1.98571853e-05,
       -5.08044105e-05, -2.76481535e-04,  3.68003590e-06,  3.03730932e-06,
       -1.30145691e-06, -3.67137485e-04,  4.44161977e-03, -8.73830628e-04,
       -8.09997594e-06, -2.86251379e-04,  5.99756378e-06, -8.41408649e-02,
       -7.08719926e-06, -4.72025704e-04, -7.95743589e-06,  1.02486950e+02,
        5.63506837e-06,  3.98589760e-06, -2.05955849e-02,  1.32422507e-05,
       -1.53621275e-05, -6.97662859e-06, -6.52809058e-06,  9.64796507e-06,
        1.54945813e-05,  1.36266420e-05, -7.00269272e-06,  6.07941412e-06,
        3.02573770e-06, -7.57778223e+01,  1.29916525e-05, -3.85452622e-04,
       -2.64657082e+01,  4.35151522e-03,  2.16575770e-05, -8.11159685e-06,
       -4.76473953e-04,  3.02479722e-06,  3.51841231e-06, -3.79870515e-05,
       -1.54407937e-02, -4.38218576e-02, -6.51916979e-06, -4.63307154e-05,
       -1.95126980e-02, -1.94907973e-03,  2.85098534e-03, -2.22430233e-05,
        3.76071566e-06,  3.88659350e-06,  1.36313772e-05,  7.58737375e-06,
       -5.41401144e-05, -9.07595790e-06,  8.50898561e-06, -2.32970130e-04,
       -6.26705780e-05, -3.20149399e-02, -1.74980175e-02, -4.22257424e-04,
       -4.89240162e-05, -7.03121421e-06,  4.70169301e-03, -6.79466506e-06,
        3.63410238e-06,  1.31009689e-05, -8.32180638e+01, -7.65589742e-04,
       -6.99639254e-06, -4.85290343e-05,  1.58160482e-05, -1.35293797e-02,
        1.80015349e-05,  3.54979033e-06,  6.14406032e-03,  1.10492661e-05,
        1.86218586e+00,  1.08621978e-05, -8.70836776e-04,  3.66984070e-06,
        2.97797115e-06,  3.85105333e-06,  3.68271833e-06, -7.36071977e-04,
       -8.73740385e-02, -4.84689311e-05, -2.32369350e-04, -3.46770810e-05,
       -3.03219473e-04, -6.51786051e-06,  3.67862663e-06,  3.52946982e-06,
        6.00071509e-06,  4.17053264e-06, -1.63683886e-05, -4.39984148e-04,
        4.89047536e-05, -1.37181488e-06,  8.99048976e+01, -6.64583315e-06,
       -1.84106645e-05,  3.81320952e-06, -4.26196121e-02,  5.37646577e+00,
       -7.35526125e-06, -3.05358261e-02, -6.07599251e-06, -5.10943696e-05,
       -7.27639694e-04, -2.93302744e-04,  1.34776604e+01, -6.77475415e-06,
       -1.85869977e-06, -7.39606369e-06,  1.07567685e+02, -6.51127750e-06,
        2.18685348e-07, -6.97566829e-06, -6.50585625e-06,  3.39866380e-05,
        1.46043014e-05,  3.54620715e-06,  2.80559404e-05, -5.91519504e-06,
        2.54889183e-05, -6.99701691e-06, -6.50874387e-06, -6.05701746e-06,
        3.58117747e+00,  3.03716299e-06,  6.48201323e-06, -8.67583778e-04;
  Eigen::MatrixXd node3_Weight(200,1);
  node3_Weight << 6.55747175e-06,  1.92253425e-05,  1.39610320e-05, -1.14720879e-05,
       -8.62531144e-04, -1.05644574e-05, -7.28102348e+01,  1.14567373e-03,
        1.91933715e+00, -1.02876652e-04,  3.56077272e-02,  3.12657952e-01,
       -5.91100398e+01, -1.48414142e-03, -1.62536902e-05,  1.50686520e-04,
        6.06884895e-06, -5.07886703e+01,  1.83004200e-02,  5.53075057e-06,
        1.21089642e-03,  5.52472016e-04,  4.38922005e-05,  2.67702079e-04,
       -1.14414812e-05, -1.03736519e-05,  2.20639807e-04, -2.38780236e-03,
        5.84682719e-04,  8.34405544e-04,  6.99341584e-06,  6.24326965e-04,
        2.46328630e-02,  1.08696704e-02,  7.86636634e-06, -9.99683886e-06,
        1.72759499e+01,  2.75642476e-05, -9.57725439e-06,  5.44988956e-06,
        6.59771216e-06,  7.17239897e-04,  8.09970761e-04,  1.41869844e-02,
        6.45627510e-06,  6.15110289e-03,  2.50113850e-05,  6.71135149e-06,
        2.18707462e-03,  6.77889010e-06,  1.02258051e-03,  1.63194508e-05,
        1.53812243e-02, -3.84434595e-04,  2.14290991e-03, -4.89210138e-04,
       -1.59214616e-04,  6.70109202e-06,  1.00100286e-05,  2.77840660e-03,
        4.08485011e-03, -1.04083291e-05,  7.72849525e-06,  1.37843224e-04,
       -1.41542795e-03,  5.94707404e-06,  5.63798188e-01, -3.70487403e-05,
       -1.20224883e-04, -2.13021333e-04,  6.88598354e-06,  5.52897955e-06,
        4.81852196e-06, -4.29463825e-04,  1.12631572e-02,  5.99622318e-03,
       -1.17913965e-05,  8.03335263e-03,  1.30454100e-05,  1.12704058e-01,
       -1.08921569e-05,  5.82752983e-04, -4.77532195e-05, -2.28221549e+01,
        1.22612663e-05,  7.50162937e-06, -3.69014422e-03,  1.85510216e-05,
       -3.71671950e-05, -1.08858820e-05,  1.36302933e-04,  1.32492269e-05,
        1.34442899e-04,  1.88388066e-05, -1.09239886e-05,  1.32241311e-05,
        5.43685115e-06, -1.07422310e+02,  1.76948026e-05,  7.27927091e-04,
       -5.18304735e+01,  1.12040812e-02,  9.44949290e-05, -1.18082134e-05,
        5.84392075e-04,  5.43639112e-06,  6.46145290e-06, -8.99206701e-05,
        1.44203269e-02,  8.51005566e-02, -9.84739719e-06, -1.10866857e-04,
        1.47571412e-02,  1.22004756e-04,  2.26108747e-02,  1.62270588e-04,
        7.07302769e-06,  7.30329124e-06,  5.14577108e-05,  1.10301486e-05,
       -1.29490821e-04, -2.51667543e-05,  1.20259238e-05, -7.47659619e-05,
       -2.33184939e-04,  4.08028158e-02,  1.29493251e-02, -1.84751444e-03,
       -1.16226417e-04, -1.07989095e-05,  1.79659352e-02, -1.04065042e-05,
        6.72932082e-06,  1.78952133e-05, -7.19747296e+01,  3.48860799e-03,
       -1.09147790e-05, -2.01882602e-04,  2.31343439e-05, -3.09208615e-03,
        2.71318430e-05,  6.52310724e-06,  1.73778167e-02,  2.45853155e-05,
        2.29763533e+00,  2.41380512e-05,  5.23164253e-03,  6.79498328e-06,
        5.43358746e-06,  7.24444143e-06,  6.88924372e-06,  7.30706914e-04,
        1.20044394e-01, -1.15655959e-04, -2.65762778e-04, -8.03190746e-05,
       -3.83120798e-04, -9.84753093e-06,  6.88468940e-06,  6.45506988e-06,
        7.97685969e-06,  7.73303292e-06, -4.15019066e-05,  8.75812622e-04,
        1.32302498e-04, -1.08213669e-06,  7.10144456e+01, -9.74949305e-06,
       -3.10966638e-05,  7.13076148e-06,  3.16545665e-02,  1.15885093e+01,
       -1.13522728e-05,  6.51772296e-02, -9.57735967e-06, -2.53844433e-04,
        6.96954438e-04,  7.04544006e-04, -1.79365660e+00, -1.05911404e-05,
        1.31973154e-04, -1.14757707e-05,  8.68758372e+01, -9.89642069e-06,
       -1.42977326e-04, -1.08844778e-05, -9.87556912e-06,  7.76666339e-05,
        3.70691116e-04,  6.61490318e-06,  1.32518981e-04, -9.33611495e-06,
        4.66645363e-05, -1.09156917e-05, -9.85175787e-06, -9.54888542e-06,
        1.85721726e+00,  5.56591726e-06,  1.41083388e-05,  5.14900050e-03;
  Eigen::MatrixXd node4_Weight(200,1);
  node4_Weight << -1.00561322e-06, -1.54150224e-06, -3.05806516e-06, -2.29648965e-06,
        9.21431562e-05, -1.47630004e-06, -1.50490208e+02,  4.43659103e-04,
       -2.22915170e+01, -1.57477366e-05, -1.27076396e-02, -9.54730738e-02,
       -1.26758291e+02,  1.15967470e-04,  2.49707200e-05,  1.33901043e-05,
       -1.50526157e-06,  2.23748824e+00, -3.60788465e-03, -1.21149924e-06,
        2.58605442e-04,  1.21662235e-04,  4.31476783e-06,  5.59209218e-05,
       -2.36742943e-06, -2.04483587e-06,  2.69768498e-04, -1.07276025e-02,
        5.74036989e-05,  3.70350240e-04, -1.62884459e-06,  1.98617106e-04,
       -6.66423317e-03, -3.47616305e-05, -5.58287369e-07, -1.53233989e-06,
        3.41202117e-01,  3.94332304e-06, -2.81569619e-06, -1.47852789e-06,
       -1.00539532e-06,  3.35052515e-04,  4.28927322e-04, -2.62930283e-03,
       -9.65137990e-07, -1.79389447e-03,  3.50793008e-06, -8.86830774e-07,
        3.90431867e-04, -6.38689096e-07,  4.62845869e-04,  1.90721875e-06,
       -4.84638646e-04, -4.27923116e-05,  3.02268573e-04,  1.89077064e-04,
       -2.75552899e-05, -5.76306788e-07, -5.31797323e-07, -7.48394198e-05,
        3.28258243e-04, -1.46838061e-06, -7.42520423e-07,  3.39064179e-05,
        1.12797197e-04, -5.99146755e-07, -1.21212462e-01,  4.78062554e-06,
       -2.02220940e-05,  2.18564766e-04, -6.20403162e-07, -1.18321382e-06,
       -3.37607131e-06,  1.95168776e-04, -1.22566964e-05,  2.53182048e-04,
       -3.82907983e-06, -1.07693628e-05,  1.26588518e-06, -4.29599926e-02,
       -2.15010804e-06,  5.74719890e-05,  2.78652506e-05, -4.47042737e+01,
        1.08882053e-06, -4.75036918e-07, -7.01974444e-03, -2.21696397e-06,
        2.52149185e-06, -2.47663559e-06,  3.34142166e-05, -3.19259556e-06,
        2.93211839e-05, -1.66196106e-06, -2.47046802e-06,  1.30435245e-06,
       -1.40625887e-06, -2.20887845e+02, -1.83641038e-06,  4.42865967e-04,
       -1.47736370e+01, -1.49319268e-05,  4.36951308e-05, -3.83920154e-06,
        5.74118153e-05, -1.40255708e-06, -1.00701498e-06, -6.87847643e-06,
       -2.44220363e-03, -2.48663559e-02, -1.78993286e-06, -1.82420455e-05,
       -2.67580629e-03,  7.72352384e-05, -6.96601065e-03,  4.07270525e-05,
       -5.07498371e-07, -5.16795992e-07,  4.70377034e-05, -3.30880962e-06,
       -2.33740906e-05,  1.25957752e-05, -3.32467012e-06,  2.38322747e-04,
       -2.65449197e-05, -1.48987441e-02, -2.53615973e-03,  1.15961144e-04,
       -1.99898591e-05, -2.13220619e-06, -3.03396045e-03, -2.05033750e-06,
       -8.37890006e-07, -1.82825030e-06,  5.67155135e+00,  3.35804299e-04,
       -2.47195307e-06, -6.12609332e-05, -1.53931179e-06, -2.72866448e-03,
       -9.00962823e-07, -9.93594316e-07, -1.79832988e-03,  3.42797068e-06,
        5.36614747e+00,  3.34791913e-06,  2.96618889e-04, -8.37399926e-07,
       -1.14018125e-06, -4.96135478e-07, -6.25572548e-07,  3.31436293e-04,
       -4.45656171e-02, -1.33656450e-05,  2.03486961e-04, -6.94187383e-06,
        1.95692967e-04, -1.79385383e-06, -6.16412393e-07, -1.08532730e-06,
        3.35578905e-07, -7.72567730e-07,  3.27900842e-06,  4.84286729e-04,
        2.05608245e-05, -5.25527300e-06,  8.28518176e+01, -1.35333077e-06,
        8.28463034e-06, -6.24651285e-07, -1.46427581e-02,  4.06479603e+00,
       -2.24309965e-06, -3.16270944e-02, -2.81566162e-06, -2.34404303e-05,
        3.28478888e-04,  3.65759740e-04,  1.14504163e+01, -2.52876619e-06,
        3.18693265e-05, -2.34981557e-06, -1.23809272e+01, -1.90485437e-06,
       -8.39204236e-05, -2.47686416e-06, -1.88101153e-06,  5.68748228e-06,
        1.00684814e-04, -6.92680372e-07,  5.47122251e-05, -2.89438473e-06,
        4.15259046e-06, -2.47180575e-06, -1.82761046e-06, -2.82484893e-06,
        4.64304215e+01, -1.05884055e-06,  1.48464325e-06,  3.00055397e-04;
  Eigen::MatrixXd node5_Weight(200,1);
  node5_Weight << 4.00451799e-06,  7.69651073e-06,  6.52273373e-06, -3.46084209e-06,
        2.73357292e-04, -3.33951318e-06, -1.93203174e+02,  1.20026538e-03,
       -2.84945254e+00, -6.58246513e-05, -1.50254520e-02,  7.78336394e-03,
       -1.54484524e+02,  3.34221705e-04, -2.15246832e-05,  1.11367879e-04,
        3.86121744e-06, -8.15326172e-01,  3.60405552e-03,  3.50503935e-06,
        7.25293227e-04,  5.34390200e-04,  1.65610881e-05,  2.78332847e-04,
       -3.41490619e-06, -3.08236619e-06,  7.13974925e-04, -3.76535315e-03,
        9.89171523e-04,  1.48126728e-03,  4.36764767e-06,  5.73423352e-04,
        5.76263401e-04,  3.25442603e-05,  4.56867625e-06, -3.10651871e-06,
       -2.64401359e+00,  1.50440366e-05, -2.26449851e-06,  3.52206941e-06,
        4.02587519e-06,  1.41079667e-03,  1.14312355e-03,  5.39307357e-03,
        3.93965646e-06, -3.51089054e-03,  1.38230602e-05,  4.05406335e-06,
        1.05024045e-03,  4.01638405e-06,  1.25936754e-03,  9.18077297e-06,
        1.51955453e-03, -3.34423435e-04,  8.43110366e-04,  5.06670255e-04,
       -1.02152283e-04,  3.95465650e-06,  7.41694326e-06,  2.74789972e-03,
        9.32870231e-04, -3.28302172e-06,  4.55224823e-06,  1.48955224e-04,
        3.22584666e-04,  3.55399580e-06,  1.23978586e-02, -1.66107014e-05,
       -7.41197488e-05,  5.71213689e-04,  4.06800666e-06,  3.49695397e-06,
        7.48190459e-06,  5.20466308e-04,  1.05267344e-04,  7.68322263e-04,
       -2.38760010e-06,  2.70976079e-04,  7.53779860e-06, -1.14981054e-02,
       -3.27197308e-06,  9.85077640e-04, -5.58723135e-05,  1.03668465e+01,
        7.17543944e-06,  4.34845547e-06, -8.42806361e-03,  7.94233216e-06,
       -2.51134910e-05, -3.10953157e-06,  1.45807535e-04,  6.38147699e-06,
        1.17582267e-04,  7.59004385e-06, -3.13018583e-06,  7.62297343e-06,
        3.49935294e-06, -2.02455095e+02,  7.17469628e-06,  1.24245788e-03,
        1.52143302e+00,  9.37184383e-05,  1.15829096e-04, -2.38884255e-06,
        9.88622017e-04,  3.49828148e-06,  3.95372226e-06, -5.74476051e-05,
        4.19306739e-03, -1.33803605e-02, -2.95432889e-06, -6.95876563e-05,
        5.56215958e-03,  2.02807285e-03,  3.52949934e-04,  1.91328973e-04,
        4.13157238e-06,  4.25729202e-06,  1.26838947e-04,  5.94525644e-06,
       -7.92127976e-05, -1.64973017e-05,  6.16926114e-06,  6.19560760e-04,
       -1.74147271e-04, -1.53050241e-02,  4.42001141e-03,  3.63947888e-04,
       -7.21448033e-05, -3.23849954e-06,  2.92878837e-03, -3.09484048e-06,
        4.04977201e-06,  7.25947357e-06,  3.82551316e+01,  9.43616688e-04,
       -3.12520207e-06, -1.75421611e-04,  9.69643403e-06, -3.97703168e-03,
        1.12706198e-05,  3.98300791e-06,  2.86660783e-03,  1.35926357e-05,
       -1.05181631e+00,  1.33447082e-05,  8.63375882e-04,  4.08463459e-06,
        3.43346702e-06,  4.21894820e-06,  4.07138201e-06,  1.40926731e-03,
       -1.08050160e-02, -7.22134463e-05,  5.28823091e-04, -5.02324521e-05,
        5.15456444e-04, -2.95283910e-06,  4.06605267e-06,  3.97078418e-06,
        8.33876172e-06,  4.56308483e-06, -3.15651951e-05,  1.38190919e-03,
        8.56838937e-05,  6.13382919e-06,  3.76647253e+01, -3.05536647e-06,
       -2.88589037e-05,  4.20017974e-06, -1.32747805e-02, -2.37389252e+00,
       -3.43205743e-06, -3.07537788e-02, -2.26457864e-06, -1.99958211e-04,
        1.38976883e-03,  1.15180345e-03, -2.64160724e+00, -2.94525170e-06,
        1.35619408e-04, -3.43834827e-06, -3.10000577e+01, -2.92892319e-06,
       -2.54437296e-04, -3.10876871e-06, -2.92970052e-06,  5.43040939e-05,
        3.31912489e-04,  3.94526484e-06,  1.39242662e-04, -2.07584982e-06,
        2.18447736e-05, -3.12569622e-06, -2.94118040e-06, -2.24297322e-06,
        1.52993696e+01,  3.48455220e-06,  8.05568210e-06,  8.71135432e-04;
  Eigen::MatrixXd node6_Weight(200,1);
  node6_Weight << -3.09770454e-08, -1.04012391e-06, -1.00851431e-06,  7.45671522e-08,
       -2.00509900e-05,  1.23321077e-07, -2.78004128e+01, -3.30671432e-04,
       -8.86720299e+01, -2.90447284e-06, -1.67616566e-02, -1.46428974e-01,
       -3.09143557e+00, -4.33764849e-05,  1.21312530e-06,  7.84231172e-06,
       -1.20575311e-07, -4.20981800e+00, -1.32948679e-02, -7.33845133e-08,
       -5.92918296e-04,  3.28835807e-05, -5.32589681e-07,  1.35849625e-05,
        7.38181147e-08,  7.66566940e-08, -7.64047630e-05, -2.25011903e-02,
       -2.17248482e-05, -9.77683671e-05, -1.37107730e-07,  1.70898877e-05,
       -7.12991684e-03, -3.13803693e-03,  3.94026601e-08,  1.06760224e-07,
        8.03194679e-01,  6.89398891e-07,  9.01403777e-08, -1.22317516e-07,
       -3.06301986e-08, -4.52955237e-05, -2.35440408e-04, -1.02908406e-02,
       -2.56959516e-08, -5.26960011e-03,  6.24180748e-07, -1.22575038e-08,
       -6.68299355e-04,  2.17215115e-08, -2.87792603e-04,  4.32034899e-07,
       -3.17828124e-03, -1.77436758e-05, -8.04075303e-04, -1.05168526e-04,
       -4.12979914e-06,  2.88495423e-08,  4.57410871e-07, -8.01314555e-04,
       -1.47049234e-03,  1.21811473e-07,  1.48007272e-08,  6.34255052e-06,
       -3.90111936e-05,  1.95782407e-08, -2.28486280e-01,  2.63973366e-07,
       -3.11764619e-06, -9.84368754e-05,  2.48496750e-08, -6.86662529e-08,
       -1.33977872e-07, -1.06471357e-04, -3.06327459e-03, -2.37734678e-03,
       -4.25035068e-07, -3.27075498e-03,  3.68747027e-07, -6.70930518e-02,
        7.37188815e-08, -2.11608111e-05, -1.58650812e-06, -6.84733623e+01,
        3.56676996e-07,  4.69074106e-08, -1.01023664e-02, -1.00850945e-06,
       -1.06589068e-06,  7.15901780e-08,  6.22469769e-06, -9.66671897e-07,
        5.40567931e-06, -1.04792050e-06,  7.16220120e-08,  3.71784927e-07,
       -1.08893614e-07,  2.15975355e+01, -1.07584839e-06, -2.16800313e-04,
        6.09956472e+00, -3.07357541e-03, -1.04416421e-07, -4.26944605e-07,
       -2.16515400e-05, -1.08215259e-07, -3.19526117e-08, -2.83015808e-06,
       -1.06638009e-02, -3.21118024e-02,  8.54516558e-08, -2.95664241e-06,
       -1.02480002e-02, -3.05800296e-04, -7.60170797e-03,  7.98123459e-06,
        4.00005785e-08,  4.05955076e-08, -9.05425346e-07, -7.43877385e-07,
       -3.12183452e-06,  7.72125062e-07, -8.49397216e-07, -9.62805091e-05,
       -9.20107168e-06, -1.90578021e-02, -1.07820637e-02, -4.48215817e-05,
       -3.02870394e-06,  7.38240506e-08, -5.83616751e-03,  7.64983696e-08,
       -5.18433815e-09, -1.07005483e-06,  5.27338536e+01, -1.22116283e-03,
        7.16125634e-08, -2.67640777e-06, -9.10416355e-07, -7.37167121e-03,
       -8.25958373e-07, -2.94282197e-08, -4.55975116e-03,  6.14481420e-07,
        5.51688452e-01,  6.04246181e-07, -1.99854094e-03, -4.61999631e-09,
       -6.26428792e-08,  4.25981740e-08,  2.42261420e-08, -5.25633221e-05,
       -6.88507783e-02, -3.27406414e-06, -7.97788508e-05, -2.53508026e-06,
       -9.13947995e-05,  8.52406193e-08,  2.53391212e-08, -4.41245053e-08,
        3.61331653e-07,  1.07349343e-08, -1.90869384e-06, -2.48705806e-04,
        4.33128133e-06,  9.91314372e-07,  7.54309517e+00,  1.22817771e-07,
       -1.36727069e-07,  2.61491844e-08, -1.93462038e-02,  7.44833977e-01,
        7.46744493e-08, -1.78361205e-02,  9.01368088e-08, -1.19606504e-05,
       -3.81620565e-05, -1.02987401e-04,  4.04930801e+00,  7.20588171e-08,
        5.85388729e-06,  7.41109878e-08, -5.45154653e+01,  8.03306602e-08,
        3.18351463e-06,  7.15893734e-08,  8.12133655e-08,  4.46433119e-06,
        2.32533793e-06,  1.35158057e-08, -9.68593610e-07,  9.98400757e-08,
        3.85650029e-07,  7.16134502e-08,  8.35290873e-08,  9.11002554e-08,
       -5.09165234e+00, -4.81814468e-08,  3.87814294e-07, -1.95838281e-03;
  Eigen::MatrixXd node7_Weight(200,1);
  node7_Weight << 0.04622869,  0.03369559,  0.0340341 , -0.0385921 , -0.0807285 ,
       -0.03331696, -0.00477006, -0.08123539, -0.00407663,  0.06862148,
       -0.08277696, -0.08688007, -0.00719307, -0.0826146 ,  0.0451514 ,
       -0.06605765,  0.04328658, -0.01865901, -0.07480926,  0.04444767,
       -0.08516248, -0.07227122,  0.04025037, -0.0720219 , -0.03913518,
       -0.03714759, -0.08080158, -0.0721532 , -0.0756804 , -0.07402907,
        0.04322064, -0.0754802 , -0.06997591, -0.08203159,  0.04932701,
       -0.03411945, -0.07509324, -0.06924828, -0.04335932,  0.04296176,
        0.04624873, -0.07368835, -0.08121968, -0.07439592,  0.04641498,
       -0.08214579, -0.06847197,  0.04698989, -0.08334998,  0.04856503,
       -0.08074075, -0.06651995, -0.07576417,  0.06617251, -0.08472082,
       -0.08363255,  0.07044748,  0.04895914, -0.06033858, -0.06831064,
       -0.0842622 , -0.03346378,  0.04817385, -0.07390564, -0.08265085,
        0.04853309, -0.0850515 ,  0.04540285,  0.07044304, -0.0831434 ,
        0.04871721,  0.0446043 , -0.04517737, -0.08357751, -0.08166882,
       -0.08395396, -0.03022914, -0.08324398, -0.06522535, -0.08182049,
       -0.03780885, -0.07567432,  0.04840061,  0.00398109, -0.06476612,
        0.04981424, -0.08093423,  0.03446441,  0.05617996, -0.04017392,
       -0.07388027,  0.03432567, -0.07308059,  0.03363579, -0.04011475,
       -0.06532096,  0.0433244 ,  0.00268872,  0.03327892, -0.07960088,
       -0.04723137, -0.08172748, -0.08049448, -0.03023032, -0.07567957,
        0.04334337,  0.04617465,  0.06352369, -0.0743548 , -0.08143007,
       -0.03554386,  0.06977299, -0.07448572, -0.06894081, -0.07021064,
       -0.07416374,  0.049521  ,  0.04950271, -0.08035233,  0.03600681,
        0.07156545,  0.04383928,  0.03520598, -0.08266538,  0.06629242,
       -0.08220268, -0.07431052, -0.08099951,  0.07074895, -0.03771693,
       -0.0710879 , -0.03718891,  0.04729086,  0.03336513, -0.00067848,
       -0.08428978, -0.04012903,  0.07661128,  0.0358354 , -0.08249856,
        0.03701779,  0.04628155, -0.07222248, -0.06839116, -0.07548293,
       -0.06833544, -0.08412461,  0.04731927,  0.04477548,  0.04963062,
        0.04868415, -0.07347167, -0.08199439,  0.06584077, -0.08318142,
        0.06441928, -0.08346724, -0.03556427,  0.04874314,  0.04572727,
       -0.05887892,  0.04799722,  0.05755646, -0.0792554 , -0.07049688,
       -0.05444489,  0.00287222, -0.03303997,  0.04766396,  0.04875603,
       -0.08066081, -0.07594532, -0.03826554, -0.08510664, -0.04335892,
        0.0653284 , -0.07363136, -0.07669799, -0.07340438, -0.04065722,
       -0.07379442, -0.03898627, -0.00313898, -0.03616518,  0.07692463,
       -0.0401761 , -0.03602935, -0.06645946, -0.07522461,  0.0481579 ,
       -0.0817673 , -0.04433003,  0.05477925, -0.04012762, -0.03574088,
       -0.04346651, -0.04617692,  0.04534575, -0.0657262 , -0.08413892;
  Eigen::MatrixXd layer1(200,7);
  layer1 << node1_Weight, node2_Weight, node3_Weight, node4_Weight, node5_Weight, node6_Weight, node7_Weight;
  std::cout << "check row: " << std::fixed << std::setprecision(5) << layer1.row(0) << std::endl;
  
  Eigen::MatrixXd layer1_bias(200,1);
  layer1_bias << 6.95205324,  7.11154417,  7.24475967, -4.97743674,  2.15582262,
       -5.98503626,  0.78934585,  3.47020606,  1.66451403,  0.61925488,
        5.55003489,  7.16366942,  0.95243774,  2.55048263,  0.8990507 ,
        0.2199026 ,  7.43780867,  3.72407715,  4.89846393,  8.82782734,
        3.51879518,  2.09823068,  5.06726762,  1.35746034, -4.88107161,
       -5.2339244 ,  2.9709076 ,  5.03356054,  2.75599625,  2.93765034,
        6.7448406 ,  2.62144674,  4.1781247 ,  3.9665511 ,  6.94308791,
       -5.80846318,  8.07185376, -0.32780182, -4.17857984, 10.58489202,
        6.92377261,  2.8997395 ,  3.41595186,  4.76757933,  7.05026927,
        4.26730181, -0.47703796,  6.91747998,  3.63013361,  7.2416376 ,
        3.4508615 , -0.85482185,  4.03182067, -1.68060956,  3.60733394,
        3.01119731, -0.57169684,  7.52895433, -1.44495359,  2.85293907,
        3.74148176, -5.95176981,  6.55547694,  1.05634501,  2.52603675,
        9.99175923,  6.30721577,  1.42475465, -0.32599202,  3.04370082,
        7.20246843,  8.85410144, -3.47927518,  3.02444923,  3.97458998,
        3.81875774, -6.2303451 ,  3.88536608, -1.05009919,  6.42462483,
       -5.12228821,  2.7511066 ,  1.37708437, -1.51621907, -1.10811507,
        7.51769069,  4.51440956,  6.66439755,  1.08499763, -4.70366169,
        1.04023195,  7.31836017,  0.86861378,  7.18954716, -4.71363211,
       -1.0370558 , 10.49890946, -2.30972434,  7.29467511,  3.30676863,
        2.68039539,  3.9734309 ,  1.45378914, -6.22917687,  2.7553478 ,
       10.49159831,  7.02392021, -0.04164802,  4.71994777,  6.05959714,
       -5.53225725, -0.02166992,  4.79971747,  2.62785748,  4.2246058 ,
        1.24600244,  7.49538903,  7.31569637,  1.53992287,  7.19641438,
       -0.743667  ,  1.4273535 ,  7.29464591,  3.05363264, -0.42960108,
        5.65851823,  4.72571893,  2.43475149,  0.02248775, -5.13833056,
        4.08554698, -5.2267958 ,  6.94835013,  7.30193936, -0.9324275 ,
        3.71149722, -4.71122481, -2.07179713,  6.25580842,  4.37551107,
        5.90779088,  6.9827918 ,  4.06366432, -0.49387952,  8.72695331,
       -0.5074182 ,  3.7915137 ,  6.90591479, 10.34734801,  7.48593131,
        7.18599228,  2.86477582,  6.44992557, -0.64178342,  2.98766039,
        0.53918141,  3.00143563, -5.52839962,  7.21455086,  7.00085428,
       -2.29768765,  6.52403344,  1.32573756,  3.36409874,  0.45033209,
       -1.95177913, -0.31751172, -6.02862303,  2.90932563,  7.02002535,
        5.40968936,  8.36621143, -5.03776435,  5.95284671, -4.17864352,
       -0.48937647,  2.88504101,  2.95724315,  9.2187884 , -4.62241107,
        0.98918005, -4.90718053,  1.73515212, -5.41507643,  1.76300126,
       -4.7032936 , -5.44069002,  0.18237767,  1.77311582,  7.25580061,
        1.8083607 , -4.02588711,  7.08941944, -4.71146344, -5.49505967,
       -4.16152183, 11.14776169,  8.75771672, -0.98001693,  3.78850959;
  Eigen::MatrixXd layer1_intput;

  layer1_intput = (layer1*input + layer1_bias).unaryExpr(&sigmoid);
  Eigen::MatrixXd layer2(1,200);
  layer2 << 1.21209298e-01,  3.05525412e-02,  4.21030511e-02,  1.27596039e-03,
        8.31935783e-02,  3.68724227e-04, -6.59980121e+00,  1.28531926e-01,
       -1.59781303e+00, -6.74425619e-02,  4.05120606e-01,  1.05891656e+00,
       -3.92610479e+00,  9.03988553e-02, -3.51180288e-03,  4.11553212e-02,
        1.35574121e-01,  9.64598153e-01,  2.79817688e-01,  5.75238580e-01,
        1.26592253e-01,  6.75266957e-02,  1.00925768e-02,  6.40756898e-02,
        1.38375518e-03,  1.26737319e-03,  1.03771245e-01,  2.99945499e-01,
        1.04156721e-01,  1.03442828e-01,  7.40206127e-02,  5.28447941e-02,
        2.18411622e-01,  1.73181759e-01,  1.95541489e-01,  6.27658528e-04,
        1.54314895e+00,  1.18385488e-02,  2.74910650e-03,  2.82726183e+00,
        1.18639268e-01,  9.86323579e-02,  1.24287864e-01,  2.57699120e-01,
        1.34829388e-01,  2.06704525e-01,  1.13042866e-02,  1.29523063e-01,
        1.35576413e-01,  2.14591191e-01,  1.27258245e-01,  8.89841671e-03,
        1.90674392e-01, -2.94787020e-02,  1.34380597e-01,  1.03769303e-01,
       -4.00925909e-02,  2.95862001e-01,  7.85882208e-03,  1.27599736e-01,
        1.44813326e-01,  4.03188088e-04,  1.14849638e-01,  5.40663526e-02,
        8.92984294e-02,  3.00166142e+00,  1.03633177e+00, -2.52197725e-03,
       -3.70379254e-02,  1.06179052e-01,  2.12537440e-01,  6.00078990e-01,
        7.19394243e-03,  1.04606892e-01,  1.75085857e-01,  1.52148952e-01,
        4.91036236e-03,  1.59566775e-01,  7.85619205e-03,  7.11494196e-01,
        1.24912073e-03,  1.04020098e-01, -1.29638689e-02, -3.60053649e+00,
        7.60883703e-03,  3.53097369e-01,  2.29572484e-01,  2.44438183e-02,
       -1.05636253e-02,  1.71877534e-03,  5.35895520e-02,  4.80293604e-02,
        4.72798733e-02,  3.30523765e-02,  1.69713668e-03,  7.91137092e-03,
        2.68891463e+00, -1.35006286e+01,  3.45788048e-02,  1.21573760e-01,
       -4.38693402e+00,  1.74806827e-01,  6.39010353e-02,  4.92068077e-03,
        1.04138106e-01,  2.67440331e+00,  1.28027399e-01, -1.77715040e-02,
        2.54486816e-01,  6.32022080e-01,  1.07952698e-03, -4.31672164e-02,
        2.55718014e-01,  1.12058815e-01,  2.22198528e-01,  5.99508668e-02,
        3.18340489e-01,  2.72543117e-01,  6.35196704e-02,  5.53990447e-02,
       -2.96195664e-02, -2.94122471e-03,  5.44713487e-02,  1.07170903e-01,
       -5.19919465e-02,  4.36875909e-01,  2.55977514e-01,  9.12768130e-02,
       -5.27016182e-02,  1.25228679e-03,  2.03338747e-01,  1.26396638e-03,
        1.38346777e-01,  3.54804298e-02,  4.33204588e+00,  1.42022355e-01,
        1.70235076e-03, -3.24858450e-02,  2.06793312e-02,  2.15399066e-01,
        1.80044708e-02,  1.25208819e-01,  1.98777275e-01,  1.12001663e-02,
        3.79153680e-01,  1.10824605e-02,  1.49316107e-01,  1.34030943e-01,
        2.68296694e+00,  3.25595070e-01,  2.08383324e-01,  9.90769993e-02,
        7.32498682e-01, -1.57512194e-02,  1.02913647e-01, -3.05182486e-02,
        1.03506339e-01,  1.08553068e-03,  2.15704523e-01,  1.19100479e-01,
        1.31151489e-02,  1.09143934e-01, -2.05893288e-02,  1.24023719e-01,
        4.04923241e-02,  8.84387475e-03, -7.77858826e+00,  4.15461051e-04,
       -2.08141667e-02,  1.82979663e-01,  3.78654343e-01,  1.04285476e+00,
        1.23932751e-03,  5.32721383e-01,  2.74899864e-03, -5.13693967e-02,
        9.78498605e-02,  1.04868912e-01,  1.78697804e+00,  1.89575540e-03,
        5.18763566e-02,  1.34861458e-03, -3.97905293e+00,  1.23653278e-03,
       -2.28031423e+00,  1.71957619e-03,  1.20849204e-03,  2.35435557e-02,
        7.28859800e-02,  2.04046349e-01,  6.39581552e-02,  3.00774636e-03,
        1.45734665e+00,  1.70183358e-03,  1.13594102e-03,  2.77797142e-03,
       -1.30664612e+00,  5.91589112e-01,  8.19720335e-03,  1.48986195e-01;
  Eigen::MatrixXd layer2_bias(1,1);
  layer2_bias << 0.06323474;

  Eigen::MatrixXd result = (layer2*layer1_intput + layer2_bias).unaryExpr(&sigmoid);
  double result_vel = result(0,0)/2;
  // double result_vel = result(0,0);
  std::cout << "Velocity threshlod output: " << result_vel << std::endl;
  if(result_vel<0){
    cluster_vel_threshold = 0.1;
  }
  else if(result_vel>0.5){
    cluster_vel_threshold = 0.5;
  }
  else{
    cluster_vel_threshold = result_vel;
  }
  return;
}
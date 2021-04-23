#include "dbtrack/dbtrack.h"

#define MOTION_OPT_MSG(msg) if(output_motion_eq_optimizer_msg) {std::cout << "[Motion eq Optimizer] " << msg << std::endl;}
#define CLUSTER_TRACK_MSG(msg) if (output_cluster_track_msg) {std::cout << "\033[34m" << msg << "\033[0m";}
#define CLUSTER_TRACK_MSG_BOLD(msg) if (output_cluster_track_msg) {std::cout << "\033[1;34m" << msg << "\033[0m";}
#define keep_cluster_id_frame 5
#define assign_cluster_id_weight 0.8  // decide the percentage of the unclustered points
#define noise_point_frame 2   // the point that is not clustered over this number would be removed from the points_history vec

dbtrack::dbtrack(double eps, int Nmin){
  dbtrack_param.eps = eps;
  dbtrack_param.Nmin = Nmin;
  dbtrack_param.time_threshold = 3;
  history_frame_num = 4;
  dt_threshold_vel = 0.0;
  points_history.clear();
  vel_scale = 1;
  input_cloud = pcl::PointCloud<pcl::PointXYZ>().makeShared();
  output_info = true;
}

dbtrack::~dbtrack(){}

/* 
 * Input: The current radar points 
 * Output: The current radar cluster result
 */
std::vector< std::vector<cluster_point> > dbtrack::cluster(std::vector<cluster_point> data){
  input_cloud->clear();
  points_history.push_back(data);
  if(points_history.size()>history_frame_num){
    points_history.erase(points_history.begin());
  }
  cluster_queue.clear();
  cluster_queue.shrink_to_fit();
  // ready to process the cluster points
  std::vector< cluster_point > process_data;

  for(int i=0;i<points_history.size();i++){
    for(int j=0;j<points_history.at(i).size();j++){
      pcl::PointXYZ temp_pt;
      temp_pt.x = points_history.at(i).at(j).x;
      temp_pt.y = points_history.at(i).at(j).y;
      temp_pt.z = vel_function(points_history.at(i).at(j).vel, scan_num - points_history.at(i).at(j).scan_time);
      input_cloud->points.push_back(temp_pt);
      process_data.push_back(points_history.at(i).at(j));
      // initialize the reachable_dist
      dbtrack_info cluster_pt;
      cluster_pt.reachable_dist = -1; // undefined distance
      cluster_queue.push_back(cluster_pt);
    }
  }
  
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
    
  std::vector< std::vector<cluster_point> > cluster_result;
  // split_past(process_data, final_cluster_order);
  split_past_new(process_data, final_cluster_order);
  // for(int i=0;i<final_cluster_order.size();i++){
  //   std::vector<cluster_point> temp_cluster_unit;
  //   for(int j=0;j<final_cluster_order.at(i).size();j++){
  //     temp_cluster_unit.push_back(process_data.at(final_cluster_order.at(i).at(j)));
  //   }
  //   cluster_result.push_back(temp_cluster_unit);
  // }
  // cluster center is calculated from the merge_cluster() function
  // cluster_center(cluster_result);
  cluster_result = current_final_cluster_vec;

  return cluster_result;
}

void dbtrack::cluster_track(std::vector< cluster_point > process_data, std::vector< std::vector<int> > final_cluster_order){
  int points_history_size = points_history.size();
  bool use_vote_func = true;
  final_cluster_idx.clear();
  final_cluster_idx.shrink_to_fit();
  std::map<int,std::vector<cluster_point*>> cluster_result_ptr;   // cluster index, cluster data list
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
  merge_cluster(cluster_result_ptr);
  MOTION_OPT_MSG("Ready to Optimizer!");
  motion_equation_optimizer(cluster_result_ptr);
}

void dbtrack::merge_cluster(std::map<int,std::vector<cluster_point*>> cluster_result_ptr){
  cluster_based_on_now.clear();
  cluster_based_on_now.shrink_to_fit();
  final_cluster_idx.clear();
  final_cluster_idx.shrink_to_fit();
  current_final_cluster_vec.clear();
  current_final_cluster_vec.shrink_to_fit();
  for(auto idx = cluster_result_ptr.begin();idx != cluster_result_ptr.end();idx++){
    std::vector<cluster_point> temp_cluster_vec;
    std::vector<cluster_point*> diff_cluster_vec;
    std::vector<cluster_point> temp_current_vec;  // store the current cluster point list
    int cluster_order = idx->second.front()->current_cluster_order;
    for(auto data_ptr=idx->second.begin();data_ptr!=idx->second.end();data_ptr++){
      
      // temporal method to solve the wrong cluster tracking keep
      // if((*data_ptr)->current_cluster_order!=cluster_order){
      //   (*data_ptr)->cluster_state.push_back(1);
      // }
      // else{
      //   (*data_ptr)->cluster_state.clear();
      // }
      // temporal method to solve the wrong cluster tracking keep

      if((*data_ptr)->cluster_state.size()>=(keep_cluster_id_frame-2)){
        diff_cluster_vec.push_back(*data_ptr);
      }
      else{
        temp_cluster_vec.push_back(**data_ptr);
        if((*data_ptr)->scan_time==scan_num){
          temp_current_vec.push_back(**data_ptr);
        }
      }
    }
    if(temp_current_vec.size()!=0){
      cluster_based_on_now.push_back(temp_cluster_vec);
      final_cluster_idx.push_back(idx->first);
      current_final_cluster_vec.push_back(temp_current_vec);
    }
    temp_cluster_vec.clear();
    temp_current_vec.clear();
    for(auto it=diff_cluster_vec.begin();it!=diff_cluster_vec.end();it++){
      (*it)->tracking_history.pop_back(); // remove the wrong cluster id from the history vec
      (*it)->cluster_id = tracking_id_max;
      (*it)->cluster_state.clear();
      (*it)->tracking_history.push_back((*it)->cluster_id);
      temp_cluster_vec.push_back(**it);
      if((*it)->scan_time==scan_num){
        temp_current_vec.push_back(**it);
      }
    }
    if(diff_cluster_vec.size()>0){
      if(temp_current_vec.size()!=0){
        cluster_based_on_now.push_back(temp_cluster_vec);
        final_cluster_idx.push_back(tracking_id_max);
        current_final_cluster_vec.push_back(temp_current_vec);
      }
      tracking_id_max++;
    }
    
  }
  cluster_center(current_final_cluster_vec);
}

void dbtrack::motion_equation_optimizer(std::map<int,std::vector<cluster_point*>> cluster_result_ptr){
  std::vector<regression_line> motion_eq;
  
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

  // motion_model_center.clear();
  // motion_model_center.shrink_to_fit();
  // for(int i=0;i<motion_eq.size();i++){
  //   cluster_point temp_data;
  //   regression_line line_para = motion_eq.at(i);
  //   temp_data.cluster_id = final_cluster_idx.at(i);
  //   temp_data.x = line_para.x1*history_frame_num*history_frame_num + line_para.x2*history_frame_num + line_para.x3;
  //   temp_data.y = line_para.y1*history_frame_num*history_frame_num + line_para.y2*history_frame_num + line_para.y3;
  //   temp_data.z = 0;
  //   temp_data.x_v = (2*line_para.x1*history_frame_num + line_para.x2)*data_period;
  //   temp_data.y_v = (2*line_para.y1*history_frame_num + line_para.y2)*data_period;
  //   temp_data.z_v = 0;
  //   temp_data.vel = std::sqrt(temp_data.x_v*temp_data.x_v+temp_data.y_v*temp_data.y_v+temp_data.z_v*temp_data.z_v);
  //   motion_model_center.push_back(temp_data);
  // }
}

int dbtrack::voting_id(std::map<int,std::vector<cluster_point*>> &cluster_result_ptr, std::map<int,std::vector<cluster_point*>> cluster_id_map, int cluster_size){
  int assign_cluster_id = -1;
  std::map<int,std::vector<double>> voting_list;  // id, weight list
  for(auto data_idx=cluster_id_map.begin();data_idx!=cluster_id_map.end();data_idx++){
    for(auto it=(*data_idx).second.begin();it!=(*data_idx).second.end();it++){
      if((*it)->tracking_history.size()>0){
        int count_ = std::count((*it)->tracking_history.begin(),(*it)->tracking_history.end(),(*it)->tracking_history.back());
        double weight_ = (double)count_/points_history.size();
        voting_list[(*it)->tracking_history.back()].push_back(weight_);
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
  }
  CLUSTER_TRACK_MSG("------------------------------------\n");
  double max_voting_result = -1;
  int max_voting_size = -1;
  std::vector<int> multiple_voting_;  // record the cluster id that voting size are larger than setting(10)
  for(auto vi=voting_list.begin();vi!=voting_list.end();vi++){
    int voting_size = (*vi).second.size();
    if(voting_size>10){
      multiple_voting_.push_back((*vi).first);
    }
    double voting_w = (double)std::accumulate((*vi).second.begin(),(*vi).second.end(),0.0)/voting_size;
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

    // assign cluster id
    CLUSTER_TRACK_MSG_BOLD("Clsuter idx: " << assign_cluster_id << "(" << cluster_size << ")\n\n");
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
    CLUSTER_TRACK_MSG_BOLD("Find multiple clusters!\n");
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
    pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud (new pcl::PointCloud<pcl::PointXYZ>);
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
    }
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
  return assign_cluster_id;
}

std::vector<int> dbtrack::cluster_tracking_result(){
  return final_cluster_idx;
}

double dbtrack::distance(cluster_point p1, cluster_point p2){
  Eigen::Vector3d d(p1.x-p2.x,p1.y-p2.y,(p1.vel-p2.vel)*vel_scale);
  return d.norm();
}

dbtrack_neighbor_info dbtrack::find_neighbors(cluster_point p){
  pcl::PointXYZ search_pt;
  search_pt.x = p.x;
  search_pt.y = p.y;
  search_pt.z = vel_function(p.vel, scan_num - p.scan_time);

  // find the eps radius neighbors
  dbtrack_neighbor_info info;
  // info.search_radius = dbtrack_param.eps;
  info.search_radius = dbtrack_param.eps + p.vel*data_period; // consider the velocity into the distance search
  info.search_k = dbtrack_param.Nmin;
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

void dbtrack::expand_neighbor(std::vector< cluster_point > &process_data, std::vector<int> &temp_cluster, int core_index){
  for(int i=0;i<cluster_queue.at(core_index).neighbor_info.pointIdxRadiusSearch.size();i++){
    int neighbor_index = cluster_queue.at(core_index).neighbor_info.pointIdxRadiusSearch.at(i);
    if(process_data.at(neighbor_index).visited)
      continue;
    // set the time stamp threshold
    else if(std::abs(process_data.at(core_index).scan_time-process_data.at(neighbor_index).scan_time)>dbtrack_param.time_threshold)
      continue;
    else if(process_data.at(neighbor_index).noise_detect.size()>=noise_point_frame)
      continue;
    else
      process_data.at(neighbor_index).visited = true;
    temp_cluster.push_back(neighbor_index);
    cluster_queue.at(neighbor_index).neighbor_info = find_neighbors(process_data.at(neighbor_index));
    if(cluster_queue.at(neighbor_index).neighbor_info.pointIdxNKNSearch.size() != 0){
      cluster_queue.at(neighbor_index).core_dist = std::sqrt(*std::max_element(cluster_queue.at(neighbor_index).neighbor_info.pointNKNSquaredDistance.begin(),
                                                cluster_queue.at(neighbor_index).neighbor_info.pointNKNSquaredDistance.end()));
      if(process_data.at(neighbor_index).noise_detect.size()==0)
        expand_neighbor(process_data, temp_cluster, neighbor_index);
    }
    else{
        cluster_queue.at(neighbor_index).core_dist = -1;      // undefined distance (not a core point)
    }
  }
}

void dbtrack::split_past(std::vector< cluster_point > process_data, std::vector< std::vector<int> > &final_cluster_order){
  // record the cluster result with past data
  cluster_with_history.clear();
  cluster_based_on_now.clear();
  // get the history data with time series
  std::vector< std::vector< std::vector<cluster_point> > > points_cluster_vec;    // contains the cluster result with all datas(need current data)
  std::vector< std::vector< std::vector<cluster_point> > > past_points_cluster_vec;   // the cluster result that only contain past data
  for(int i=0;i<final_cluster_order.size();i++){
    std::vector< std::vector<cluster_point> > temp_past_data(history_frame_num);
    std::vector< cluster_point> temp_past_cluster;
    int past_scan_id_count = 0;
    for(int j=0;j<final_cluster_order.at(i).size();j++){
      int past_timestamp = process_data.at(final_cluster_order.at(i).at(j)).scan_time;
      if(past_timestamp != scan_num){
        past_scan_id_count++;
      }
      temp_past_data.at(history_frame_num-1-(scan_num-past_timestamp)).push_back(process_data.at(final_cluster_order.at(i).at(j)));
      temp_past_cluster.push_back(process_data.at(final_cluster_order.at(i).at(j)));
    }
    if(past_scan_id_count != final_cluster_order.at(i).size()){
      points_cluster_vec.push_back(temp_past_data); // contains current scan point
      cluster_based_on_now.push_back(temp_past_cluster);
    }
    else{
      past_points_cluster_vec.push_back(temp_past_data);
    }
    cluster_with_history.push_back(temp_past_cluster);
  }
  points_cluster_vec.insert(points_cluster_vec.end(),past_points_cluster_vec.begin(),past_points_cluster_vec.end());
  history_points_with_cluster_order.clear();
  history_points_with_cluster_order = points_cluster_vec;

  // remove the cluster that only contains past data
  std::vector< std::vector<int> > temp_final_list;  // store the current points only
  for(std::vector< std::vector<int> >::iterator it=final_cluster_order.begin();it != final_cluster_order.end();){
    int count = 0;
    std::vector<int> temp;
    for(int i=0;i<(*it).size();i++){
      if(process_data.at((*it).at(i)).scan_time != scan_num)
        count ++;
      else{
        temp.push_back((*it).at(i));
      }
    }
    if(temp.size() != 0)
      temp_final_list.push_back(temp);
    
    if(count == (*it).size()){
      it = final_cluster_order.erase(it);
    }
    else{
      it++;
    }
  }
  cluster_track(process_data,final_cluster_order);
  final_cluster_order = temp_final_list;
}

void dbtrack::split_past_new(std::vector< cluster_point > process_data, std::vector< std::vector<int> > &final_cluster_order){
  // record the cluster result with past data
  cluster_with_history.clear();
  cluster_based_on_now.clear();
  // get the history data with time series
  std::vector< std::vector< std::vector<cluster_point> > > points_cluster_vec;    // contains the cluster result with all datas(need current data)
  std::vector< std::vector< std::vector<cluster_point> > > past_points_cluster_vec;   // the cluster result that only contain past data
  
  // remove the cluster that only contains past data
  std::vector< std::vector<int> > temp_final_list;  // store the current points only
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
        point_ptr->noise_detect.push_back(1);
      }
      it = final_cluster_order.erase(it);
    }
    else{
      it++;
      points_cluster_vec.push_back(temp_past_data); // contains current scan point
      cluster_based_on_now.push_back(temp_past_cluster);
    }
    cluster_with_history.push_back(temp_past_cluster);
  }
  points_cluster_vec.insert(points_cluster_vec.end(),past_points_cluster_vec.begin(),past_points_cluster_vec.end());
  history_points_with_cluster_order.clear();
  history_points_with_cluster_order = points_cluster_vec;

  cluster_track(process_data,final_cluster_order);
  final_cluster_order = temp_final_list;
}

std::vector< std::vector<cluster_point> > dbtrack::cluster_with_past(){
  return cluster_with_history;
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
  return vel_scale * vel;
}

std::vector<cluster_point> dbtrack::get_center(void){
  return final_center;
}

std::vector< std::vector< std::vector<cluster_point> > > dbtrack::get_history_points_with_cluster_order(void){
  return history_points_with_cluster_order;
}

void dbtrack::set_parameter(double eps, int Nmin, int frames_num, double dt_weight){
  dbtrack_param.eps = eps;
  dbtrack_param.Nmin = Nmin;
  history_frame_num = frames_num;
  dt_threshold_vel = dt_weight;
}

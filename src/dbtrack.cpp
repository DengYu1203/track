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
  dbtrack_param.Nmin = Nmin;
  dbtrack_param.time_threshold = 3;
  history_frame_num = 4;
  dt_threshold_vel = 0.0;
  points_history.clear();
  vel_scale = 1;
  input_cloud = pcl::PointCloud<pcl::PointXYZ>().makeShared();
}

dbtrack::~dbtrack(){}

/* 
 * Input: The current radar points 
 * Output: The current radar cluster result
 */
std::vector< std::vector<cluster_point> > dbtrack::cluster(std::vector<cluster_point> data){
  return improve_cluster(data);
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
      if(tracker_core_flag && (data_idx>=tracker_core_begin_index))
        continue;
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
  output_cluster_track_msg = output_cluster_track_msg_latch;
  return assign_cluster_id;
}

/*
 * Add the core points to improve the DBSCAN cluster
 * Record the tracking hit to provide the weights for the clsuter voting
 */
void dbtrack::tracking_id_adjust(){
  int count_idx = 0;
  for(auto id=final_cluster_idx.begin();id!=final_cluster_idx.end();id++,count_idx++){
    tracking_id_history_map[*id].push_back(scan_num);
    if(tracker_core_flag || use_RLS){
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
      if(tracker_core_flag){
        auto it=tracker_cores.find(search_id->first);
        it = tracker_cores.erase(it);
      }
      search_id = tracking_id_history_map.erase(search_id);
      
    }
  }
}

std::vector<int> dbtrack::cluster_tracking_result(){
  return final_cluster_idx;
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
    info.search_radius = dbtrack_param.eps + p.vel*data_period; // consider the velocity into the distance search
    info.search_k = dbtrack_param.Nmin;
  }
  else{
    info.search_radius = Optional_para->eps + p.vel*data_period;
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
    else if(std::abs(process_data.at(core_index).scan_time-process_data.at(neighbor_index).scan_time)>dbtrack_param.time_threshold)
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
    if(cluster_queue.at(neighbor_index).neighbor_info.pointIdxNKNSearch.size() != 0){
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
  std::vector< std::vector<int> > temp_final_vec = final_cluster_order;  // temp vec
  for(std::vector< std::vector<int> >::iterator it=final_cluster_order.begin();it != final_cluster_order.end();){
    int count = 0;
    std::vector<int> temp;
    std::vector< std::vector<cluster_point> > temp_past_data(history_frame_num);    // store the cluster
    std::vector< cluster_point> temp_past_cluster;
    for(int i=0;i<(*it).size();i++){
      // if(tracker_core_flag && ((*it).at(i)>=tracker_core_begin_index))
      //   continue;
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
        if(tracker_core_flag && ((*it).at(i)>=tracker_core_begin_index))
          continue;
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
  
  // cluster_track(process_data,final_cluster_order);
  cluster_track(process_data,temp_final_vec);
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

void dbtrack::set_parameter(double eps, int Nmin, int frames_num, double dt_weight){
  dbtrack_param.eps = eps;
  dbtrack_param.Nmin = Nmin;
  history_frame_num = frames_num;
  dt_threshold_vel = dt_weight;
}

void dbtrack::set_output_info(bool cluster_track_msg, bool motion_eq_optimizer_msg, bool rls_msg){
  output_cluster_track_msg = cluster_track_msg;
  output_motion_eq_optimizer_msg = motion_eq_optimizer_msg;
  RLS_msg_flag = rls_msg;
}

std::vector< std::vector<cluster_point> > dbtrack::improve_cluster(std::vector<cluster_point> data){
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
  input_cloud->clear();
  for(int i=0;i<points_history.size();i++){
    process_data.insert(process_data.end(),points_history.at(i).begin(),points_history.at(i).end());
    *input_cloud += *input_cloud_vec.at(i);
  }
  dbtrack_info cluster_pt;
  cluster_pt.reachable_dist = -1; // undefined distance
  cluster_queue.assign(process_data.size(),cluster_pt);

  tracker_core_begin_index = process_data.size();
  
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
  split_past_new(process_data, final_cluster_order);
  cluster_result = current_final_cluster_vec;

  return cluster_result;
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

  // std::cout << setprecision(3);
  // std::cout << "Homogeneity:\t" << result.homo << "\t-> ";
  // std::cout << "H(C|k) = " << result.h_ck << ", H(C) = " << result.h_c << endl;
  // std::cout << "Completeness:\t" << result.comp << "\t-> ";
  // std::cout << "H(K|C) = " << result.h_kc << ", H(K) = " << result.h_k << endl;
  // std::cout << "\033[1;46mV-measure score: " << result.v_measure_score << "\033[0m" << endl;
  // std::cout << "Total object:" << result.object_num << endl;
  // std::cout << "Correct object: " << result.good_cluster << endl;
  // std::cout << "Over Seg in one object: " << result.multi_cluster << endl;
  // std::cout << "Under Seg objects: " << result.under_cluster << endl;
  // std::cout << "Bad object: " << result.no_cluster << endl << endl;

  return result.v_measure_score;
}
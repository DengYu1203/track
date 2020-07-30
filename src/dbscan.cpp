#include <iostream>
#include <stdlib.h>
#include <string> 
#include <algorithm>
#include <Eigen/Dense>

#include <dbscan/dbscan.h>

#define stage_one_num 5
#define stage_one_vel 0.3

dbscan::dbscan(std::vector<cluster_point> &data)
{
    // initial input output data
    points = data;
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
    cluster_idx.assign(points.size(),-1);
    cluster_count = 0;
    state = 1;
    output_info = true;
}

dbscan::~dbscan(){}

// begin to cluster the data
std::vector< std::vector<cluster_point> > dbscan::cluster(void){
    int over_vel_num = 0;
    for(int i=0;i<points.size();i++){
        if(points.at(i).vistited && cluster_idx.at(i)!=-1)
            continue;
        points.at(i).vistited = true;
        cluster_point core_pt = points.at(i);
        // filter out the high speed points
        // if((core_pt.vel>0.3) && (state==1)){
        //     cout<<"*******************************\n";
        //     cout<<"In velocity threshold part\n";
        //     cout<<"vel = "<<core_pt.vel<<endl;
        //     cout<<"*******************************\n";
        //     over_vel_num ++;
        //     continue;
        // }
        int core_level = decide_vel_level(core_pt.vel);
        std::vector<int> neighbor = find_neighbor(core_pt, core_level);
        if(neighbor.size() >= param.MinPts[core_level]){
            cluster_idx.at(i) = cluster_count;
            expand_neighbor(neighbor);
            cluster_count ++;
        }
    }
    std::vector< std::vector<cluster_point> > cluster_list(cluster_count);
    std::vector<cluster_point> non_cluster_list;
    for(int j=0;j<cluster_idx.size();j++){
        if(cluster_idx.at(j)!=-1)
            cluster_list.at(cluster_idx.at(j)).push_back(points.at(j));
        else
            non_cluster_list.push_back(points.at(j));
    }
    if(output_info){
        cout<<"\nDBSCAN cluster index (state "<< state <<"):\n";
        cout<<"==============================\n";
        int cluster_num = 0;
        for(int k=0;k<cluster_list.size();k++){
            if(k != 0)
                cout<<"----------------------\n";
            cout<<"cluster index: "<<k<<endl<<endl;
            for(int idx=0;idx<cluster_list.at(k).size();idx++){
                cout<<"\tPosition:("<<cluster_list.at(k).at(idx).x<<","<<cluster_list.at(k).at(idx).y<<")"<<endl;
                cout<<"\tVelocity: "<<cluster_list.at(k).at(idx).vel<<" ("<<cluster_list.at(k).at(idx).x_v<<","<<cluster_list.at(k).at(idx).y_v<<")\n"<<endl;
                cluster_num ++;
            }
        }
        cout<<"==============================\n\n";
        cout<<"-----------------------------------\n";
        cout<<"At state "<<state<<endl;
        cout<<"Input size:"<<points.size()<<endl;
        cout<<"Over vel number:"<<over_vel_num<<endl;
        cout<<"In cluster list number:"<<cluster_num<<endl;
        cout<<"In non cluster list number:"<<non_cluster_list.size()<<endl;
        cout<<"-----------------------------------\n";

    }
    switch (state){
        case 1:
            stage_one_cluster = stage_one_filter(cluster_list);
            points = delist(cluster_list);
            points.insert(points.end(),non_cluster_list.begin(),non_cluster_list.end());
            state++;
            cluster_idx.assign(points.size(),-1);
            cluster_count = 0;
            param.eps[0] = 2.5;
            param.MinPts[0] = 2;
            param.eps[1] = 2.0;
            param.MinPts[1] = 2;
            param.eps[2] = 1.5;
            param.MinPts[2] = 2;
            param.eps[3] = 2.0;
            param.MinPts[3] = 1;
            cluster_list = cluster();
            break;
        
        default:
            cluster_list.insert(cluster_list.end(),stage_one_cluster.begin(),stage_one_cluster.end());
            cluster_center(cluster_list);
            merge(cluster_list);
            if(output_info){
                cout<<"------------------------------------------\n";
                cout<<"\nVar info:\n";
                cout<<" x     y     z     vx    vy    vz"<<endl;
                for(int i=0;i<var_list.size();i++){
                    cout << var_list.at(i).transpose() << endl;
                }
                cout<<"------------------------------------------\n";
            }
            break;
    }
    
    return cluster_list;
}

std::vector< std::vector<cluster_point> > dbscan::cluster_from_RANSAC(std::vector<cluster_point> inlier,std::vector<cluster_point> outlier){
    in_points = inlier;
    out_points = outlier;
    cluster_idx.assign(in_points.size()+out_points.size(),-1);
    points.clear();
    points = in_points;
    cluster_idx.assign(points.size(),-1);
    cout << endl << "RANSAC + DBSCAN cluster" << endl;
    cout << "Inlier points:" << in_points.size() << " , Outlier points:" << out_points.size() << endl;
    // inlier cluster part
    for(int i=0;i<in_points.size();i++){
        if(in_points.at(i).vistited && cluster_idx.at(i)!=-1)
            continue;
        in_points.at(i).vistited = true;
        cluster_point core_pt = in_points.at(i);
        int core_level = decide_vel_level(core_pt.vel);
        std::vector<int> neighbor = find_neighbor(core_pt, core_level);
        if(neighbor.size() >= param.MinPts[core_level]){
            cluster_idx.at(i) = cluster_count;
            expand_neighbor(neighbor);
            cluster_count ++;
        }
    }
    std::vector< std::vector<cluster_point> > cluster_list(cluster_count);
    std::vector<cluster_point> non_cluster_list;
    for(int j=0;j<cluster_idx.size();j++){
        if(cluster_idx.at(j)!=-1)
            cluster_list.at(cluster_idx.at(j)).push_back(points.at(j));
        else{
            points.at(j).cluster_flag = -1;
            points.at(j).vistited = false;
            non_cluster_list.push_back(points.at(j));
        }
    }
    // if(output_info){
    //     cout<<"\nRANSAC+DBSCAN cluster index (inlier):\n";
    //     cout<<"==============================\n";
    //     int cluster_num = 0;
    //     for(int k=0;k<cluster_list.size();k++){
    //         if(k != 0)
    //             cout<<"----------------------\n";
    //         cout<<"cluster index: "<<k<<endl<<endl;
    //         for(int idx=0;idx<cluster_list.at(k).size();idx++){
    //             cout<<"\tPosition:("<<cluster_list.at(k).at(idx).x<<","<<cluster_list.at(k).at(idx).y<<")"<<endl;
    //             cout<<"\tVelocity: "<<cluster_list.at(k).at(idx).vel<<" ("<<cluster_list.at(k).at(idx).x_v<<","<<cluster_list.at(k).at(idx).y_v<<")\n"<<endl;
    //             cluster_num ++;
    //         }
    //     }
    // }
    
    stage_one_cluster = cluster_list;
    // stage_one_cluster = stage_one_filter(cluster_list);
    // points = delist(cluster_list);
    // non_cluster_list.insert(non_cluster_list.end(),points.begin(),points.end());
    
    non_cluster_list.insert(non_cluster_list.end(),out_points.begin(),out_points.end());
    state++;
    cluster_idx.clear();
    cluster_idx.assign(non_cluster_list.size(),-1);
    cluster_count = 0;
    param.eps[0] = 2.5;
    param.MinPts[0] = 2;
    param.eps[1] = 2.0;
    param.MinPts[1] = 2;
    param.eps[2] = 1.5;
    param.MinPts[2] = 2;
    param.eps[3] = 1.0;
    param.MinPts[3] = 1;

    points.clear();
    points = non_cluster_list;
    // outlier cluster part
    for(int i=0;i<points.size();i++){
        if(points.at(i).vistited && cluster_idx.at(i)!=-1)
            continue;
        points.at(i).vistited = true;
        cluster_point core_pt = points.at(i);
        int core_level = decide_vel_level(core_pt.vel);
        std::vector<int> neighbor = find_neighbor(core_pt, core_level);
        if(neighbor.size() >= param.MinPts[core_level]){
            cluster_idx.at(i) = cluster_count;
            expand_neighbor(neighbor);
            cluster_count ++;
        }
    }
    std::vector< std::vector<cluster_point> > cluster_list2(cluster_count);
    for(int j=0;j<cluster_idx.size();j++){
        if(cluster_idx.at(j)!=-1)
            cluster_list2.at(cluster_idx.at(j)).push_back(points.at(j));
    }
    // if(output_info){
    //     cout<<"\nRANSAC+DBSCAN cluster index (outlier):\n";
    //     cout<<"==============================\n";
    //     int cluster_num = 0;
    //     for(int k=0;k<cluster_list2.size();k++){
    //         if(k != 0)
    //             cout<<"----------------------\n";
    //         cout<<"cluster index: "<<k<<endl<<endl;
    //         for(int idx=0;idx<cluster_list2.at(k).size();idx++){
    //             cout<<"\tPosition:("<<cluster_list2.at(k).at(idx).x<<","<<cluster_list2.at(k).at(idx).y<<")"<<endl;
    //             cout<<"\tVelocity: "<<cluster_list2.at(k).at(idx).vel<<" ("<<cluster_list2.at(k).at(idx).x_v<<","<<cluster_list2.at(k).at(idx).y_v<<")\n"<<endl;
    //             cluster_num ++;
    //         }
    //     }
    // }
    cluster_list.clear();
    cluster_list = stage_one_cluster;
    cluster_list.insert(cluster_list.end(),cluster_list2.begin(),cluster_list2.end());
    cluster_center(cluster_list);
    merge(cluster_list);
    if(output_info){
        cout<<"\nRANSAC+DBSCAN cluster index (total):\n";
        cout<<"Before merge -> Inlier:"<<stage_one_cluster.size()<<" , Outlier:"<<cluster_list2.size()<<endl;
        cout<<"==============================\n";
        int cluster_num = 0;
        for(int k=0;k<cluster_list.size();k++){
            if(k != 0)
                cout<<"----------------------\n";
            cout<<"cluster index: "<<k<<endl;
            cout<<"\033[1;43mcenter position: (" << center.at(k).x << "," << center.at(k).y << ")\033[0m";
            cout<<endl;
            cout<<"\033[1;43mcenter velocity: (" << center.at(k).x_v << "," << center.at(k).y_v << ")\033[0m";
            cout<<endl<<endl;
            for(int idx=0;idx<cluster_list.at(k).size();idx++){
                cout<<"\tPosition: ("<<cluster_list.at(k).at(idx).x<<","<<cluster_list.at(k).at(idx).y<<")"<<endl;
                cout<<"\tVelocity: "<<cluster_list.at(k).at(idx).vel<<" ("<<cluster_list.at(k).at(idx).x_v<<","<<cluster_list.at(k).at(idx).y_v<<")\n"<<endl;
                cluster_num ++;
            }
        }
        cout<<"==============================\n";
        cout<<"------------------------------------------\n";
        cout<<"\nVar info:\n";
        cout<<" x     y     z     vx    vy    vz"<<endl;
        for(int i=0;i<var_list.size();i++){
            cout << var_list.at(i).transpose() << endl;
        }
        cout<<"------------------------------------------\n";
    }
    
    
    return cluster_list;
}

// return stage one cluster
std::vector< std::vector<cluster_point> > dbscan::stage_one_result(void){
    return stage_one_cluster;
}


// return the cluster number after cluster function
int dbscan::cluster_num(void){
    return cluster_count;
}

std::vector<int> dbscan::find_neighbor(cluster_point pt, int vel_level){
    Eigen::Vector3f core_pt(pt.x,pt.y,pt.z);
    std::vector<int> neighbor_cluster;
    neighbor_cluster.clear();
    for(int i=0;i<points.size();i++){
        cluster_point temp = points.at(i);
        Eigen::Vector3f neighbor(temp.x,temp.y,temp.z);
        if((core_pt-neighbor).norm() <= param.eps[vel_level]){
            neighbor_cluster.push_back(i);
        }
    }
    return neighbor_cluster;
}

void dbscan::expand_neighbor(std::vector<int> neighbor){
    for(int i=0;i<neighbor.size();i++){
        if(points.at(neighbor.at(i)).vistited)
            continue;
        points.at(neighbor.at(i)).vistited = true;
        cluster_point n_pt = points.at(neighbor.at(i));
        int n_pt_level = decide_vel_level(n_pt.vel);
        std::vector<int> expand_n = find_neighbor(n_pt,n_pt_level);
        if(expand_n.size() >= param.MinPts[n_pt_level]){
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

int dbscan::decide_vel_level_RANSAC(double vel){
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

std::vector<cluster_point> dbscan::delist(std::vector< std::vector<cluster_point> > cluster_l){
    std::vector<cluster_point> output_list;
    for(int i=0;i<cluster_l.size();i++){
        for(int j=0;j<cluster_l.at(i).size();j++){
            cluster_point pt;
            pt = cluster_l.at(i).at(j);
            pt.cluster_flag = -1;
            pt.vistited = false;
            output_list.push_back(pt);
        }
    }
    return output_list;
}

std::vector< std::vector<cluster_point> > dbscan::stage_one_filter(std::vector< std::vector<cluster_point> > &cluster_list){
    std::vector< std::vector<cluster_point> > copy_list = cluster_list;
    std::vector< std::vector<cluster_point> > stage_one_list;
    std::vector< int > remove_list;
    for(int idx=0;idx<copy_list.size();idx++){
        if(copy_list.at(idx).size() >= stage_one_num){
            bool check_vel = true;
            for(int j=0;j<copy_list.at(idx).size();j++){
                if(copy_list.at(idx).at(j).vel > stage_one_vel){
                    check_vel = false;
                    break;
                }
            }
            if(check_vel){
                stage_one_list.push_back(copy_list.at(idx));
                remove_list.push_back(idx);
            }
        }
    }
    for(int i=remove_list.size()-1;i>=0;i--){
        cluster_list.erase(cluster_list.begin()+remove_list.at(i));
    }
    return stage_one_list;
}

void dbscan::cluster_center(std::vector< std::vector<cluster_point> > cluster_list){
    center_list_info.assign(cluster_list.size(),0);
    center.clear();
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
        center.push_back(pt);

    }
    var_list.clear();
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
}

std::vector<cluster_point> dbscan::get_center(void){
    return center;
}

void dbscan::merge(std::vector< std::vector<cluster_point> > &cluster_list){
    std::vector< std::vector<cluster_point> > merge_list = cluster_list;
    std::vector<int> merge_list_info;
    merge_list_info.assign(center.size(),-1);
    for(int idx = 0;idx<center.size();idx++){
        Eigen::Vector3f core_pos(center.at(idx).x, center.at(idx).y, center.at(idx).z);
        Eigen::Vector3f core_vel(center.at(idx).x_v, center.at(idx).y_v, center.at(idx).z_v);
        for(int i=idx+1;i<center.size();i++){
            Eigen::Vector3f temp_pos(center.at(i).x, center.at(i).y, center.at(i).z);
            Eigen::Vector3f temp_vel(center.at(i).x_v, center.at(i).y_v, center.at(i).z_v);
            if((core_pos-temp_pos).norm() < 3 && (core_vel-temp_vel).norm() < 0.2){
                merge_list_info.at(i) = idx;
                break;
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
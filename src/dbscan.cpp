#include <iostream>
#include <stdlib.h>
#include <string> 
#include <algorithm>
#include <Eigen/Dense>

#include <dbscan/dbscan.h>
dbscan::dbscan(std::vector<cluster_point> &data)
{
    // initial input output data
    points = data;
    center.clear();
    // initial dbscan parameter (from slow to fast)
    param.eps[0] = 2;
    param.MinPts[0] = 5;
    param.eps[1] = 1.5;
    param.MinPts[1] = 3;
    param.eps[2] = 1;
    param.MinPts[2] = 2;
    param.eps[3] = 0.5;
    param.MinPts[3] = 1;
    // initial all cluster index to -1
    cluster_idx.assign(points.size(),-1);
    cluster_count = 0;
}

dbscan::~dbscan(){}

std::vector<cluster_point> dbscan::cluster(void){
    for(int i=0;i<points.size();i++){
        if(points.at(i).vistited)
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
            cluster_idx.at(i) = cluster_count;
            expand_neighbor(expand_n);
        }
    }
}

int dbscan::decide_vel_level(double vel){
    if(vel <= 1.0)
        return 0;
    else if(vel <= 3.0)
        return 1;
    else if(vel <= 3.0)
        return 2;
    else
        return 3;
}
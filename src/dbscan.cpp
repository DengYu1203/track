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
    param.eps[0] = 3.0;
    param.MinPts[0] = 2;
    param.eps[1] = 2.5;
    param.MinPts[1] = 2;
    param.eps[2] = 1.5;
    param.MinPts[2] = 2;
    param.eps[3] = 1;
    param.MinPts[3] = 1;
    // initial all cluster index to -1
    cluster_idx.assign(points.size(),-1);
    cluster_count = 0;
}

dbscan::~dbscan(){}

std::vector< std::vector<cluster_point> > dbscan::cluster(void){
    for(int i=0;i<points.size();i++){
        if(points.at(i).vistited && cluster_idx.at(i)!=-1)
            continue;
        points.at(i).vistited = true;
        cluster_point core_pt = points.at(i);
        // filter out the high speed points
        if(core_pt.vel>0.4)
            continue;
        int core_level = decide_vel_level(core_pt.vel);
        std::vector<int> neighbor = find_neighbor(core_pt, core_level);
        if(neighbor.size() >= param.MinPts[core_level]){
            cluster_idx.at(i) = cluster_count;
            expand_neighbor(neighbor);
            cluster_count ++;
        }
    }
    std::vector< std::vector<cluster_point> > cluster_list(cluster_count);
    cout<<"DBSCAN cluster index:\n";
    cout<<"==============================\n";
    for(int j=0;j<cluster_idx.size();j++){
        // if(j != 0)
        //     cout<<"----------------------\n";
        if(cluster_idx.at(j)!=-1)
            cluster_list.at(cluster_idx.at(j)).push_back(points.at(j));
        // cout<<j<<":\ncluster index: "<<cluster_idx.at(j)<<endl;
        // cout<<"Position:("<<points.at(j).x<<","<<points.at(j).y<<")"<<endl;
        // cout<<"Velocity: "<<points.at(j).vel<<" ("<<points.at(j).x_v<<","<<points.at(j).y_v<<")"<<endl;
    }
    for(int k=0;k<cluster_list.size();k++){
        if(k != 0)
            cout<<"----------------------\n";
        cout<<"cluster index: "<<k<<endl<<endl;
        for(int idx=0;idx<cluster_list.at(k).size();idx++){
            cout<<"\tPosition:("<<cluster_list.at(k).at(idx).x<<","<<cluster_list.at(k).at(idx).y<<")"<<endl;
            cout<<"\tVelocity: "<<cluster_list.at(k).at(idx).vel<<" ("<<cluster_list.at(k).at(idx).x_v<<","<<cluster_list.at(k).at(idx).y_v<<")\n"<<endl;
        }
    }
    cout<<"==============================\n";
    return cluster_list;

}

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
    if(vel <= 0.05)
        return 0;
    else if(vel <= 0.1)
        return 1;
    else if(vel <= 0.2)
        return 2;
    else
        return 3;
    // if(vel <= 0.01)
    //     return 0;
    // else if(vel <= 0.05)
    //     return 1;
    // else if(vel <= 0.1)
    //     return 2;
    // else
    //     return 3;
}
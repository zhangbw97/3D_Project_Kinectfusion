#include <iostream>
#include <fstream>

#include "../../include/Eigen.h"
#include "../../include/data_types.h"
namespace kinectfusion{
namespace internal{	
bool pose_estimation(Eigen::Matrix4f& pose,
                            const std::vector<std::vector<std::vector<Vector3f>>>frame_vertex_pyramid,
                            const std::vector<std::vector<std::vector<Vector3f>>>frame_normal_pyramid,
                            const std::vector<std::vector<std::vector<Vector3f>>>model_vertex_pyramid,
                            const std::vector<std::vector<std::vector<Vector3f>>>model_normal_pyramid,
                            const kinectfusion::CameraParameters& cam_params,
                            const int pyramid_height,
                            const float distance_threshold, const float angle_threshold,
                            const std::vector<int>& iterations);
							}
}
int main(){
Matrix4f current_pose=Matrix4f::Identity();
Eigen::Vector3f point(1,2,3);
std::vector<Vector3f>m_points_1D;
std::vector<std::vector<Vector3f>> m_points_2D;
std::vector<std::vector<std::vector<Vector3f>>>m_points_2D_level;

std::vector<Vector3f>m_normals_1D;
std::vector<std::vector<Vector3f>> m_normals_2D;
std::vector<std::vector<std::vector<Vector3f>>>m_normals_2D_level;

for(int i=0;i<20;i++){
	for(int j=0;j<20;j++){
		m_points_1D.push_back(point);
		m_normals_1D.push_back(point);
	}
	m_points_2D.push_back(m_points_1D);
	m_normals_2D.push_back(m_normals_1D);
}
m_points_2D_level.push_back(m_points_2D);
m_normals_2D_level.push_back(m_normals_2D);

kinectfusion::CameraParameters cam_params{640,480,10.0f,10.0f,10.0f,10.0f};
int current_pyramid_level=1;
float distance_threshold=10.f;
float angle_threshold { 20.f };
std::vector<int> icp_iterations {10, 5, 4};
bool icp_success { true };
icp_success = kinectfusion::internal::pose_estimation(
				current_pose,                                   // 输入: 上一帧的相机位姿; 输出: 当前帧得到的相机位姿
				//frame_data									// 当前帧的深度图/顶点图/法向图数据
				m_points_2D_level,  
				m_normals_2D_level,                                   
				//model_data,                                   // 上一帧图像输入后, 推理出的平面模型，使用顶点图、法向图来表示
				m_points_2D_level,  
				m_normals_2D_level, 

				cam_params,                              		// 相机内参

				//configuration.num_levels,                     // 金字塔层数
				current_pyramid_level,

				//originally defined in configuration					               
				distance_threshold,								// icp 匹配过程中视为 outlier 的距离差
				angle_threshold,                  				// icp 匹配过程中视为 outlier 的角度差 (deg)
				icp_iterations); 
std::cout<<(icp_success)<<"  /n";                
std::cout<<(current_pose);
}
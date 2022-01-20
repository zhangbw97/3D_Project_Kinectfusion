//
// Created by slt on 1/23/2021.
//

#include <iostream>
#include "KinectFusion.h"
using namespace std;
using namespace cv::cuda;
namespace cuda {
void ComputeVertex(const GpuMat& depth_map,                // 输入的一层滤波之后的深度图像
      GpuMat& vertex_map,
       const float fovX, const float fovY,
       const float cX, const float cY ,
       const int width, const int height,
       const float depth_cutoff);
void ComputeNormal(GpuMat& normals_map, const GpuMat& vertex_map, const int width, const int height, const float maxDistanceHalved);
}
     
void surface_measurement(const cv::Mat_<float>& input_depth, InputData& input,
                         const std::vector<CameraParameters>& camera_params,
                         const int num_levels, const float depth_cutoff, const float distance_cutoff,
                         const int kernel_size, const float color_sigma, const float spatial_sigma) {

    for(int level = 0; level < num_levels; level++)
    {
        // 金字塔下一层
        if (level == 0) {
            input.depth_pyramid[level].upload(input_depth);
            
        } else {
            pyrdown(input.depth_pyramid[level-1], input.depth_pyramid[level]);
        }
    }

    for(int level = 0; level < num_levels; level++)
    {
        // 金字塔当前层双边过滤
        bilateral(input.depth_pyramid[level], input.filtered_depth_pyramid[level],kernel_size, color_sigma,spatial_sigma);
    }



    // 生成三层点云和法向量金字塔
    for(int level = 0; level < num_levels; level++)
    {
        // 计算点坐标
        cuda::ComputeVertex(input.filtered_depth_pyramid[level], input.vertex_pyramid[level],
                camera_params[level].fovX, camera_params[level].fovY, camera_params[level].cX,
                camera_params[level].cY, camera_params[level].width, camera_params[level].height,
                depth_cutoff);


        // 计算法向量
        cuda::ComputeNormal(input.normal_pyramid[level],input.vertex_pyramid[level],
                camera_params[level].width, camera_params[level].height, distance_cutoff);

    }
}
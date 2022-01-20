//
// Created by slt on 1/20/2021.
//

#ifndef INC_3D_PROJ_KINECTFUSION_H
#define INC_3D_PROJ_KINECTFUSION_H
#include "data_types.h"
#include <opencv2/core.hpp>
#include <iostream>
#include <fstream>
#include "SimpleMesh.h"

class KinectFusion {
public:
    KinectFusion() {};
   
    KinectFusion(const Configuration& configuration, const CameraParameters& _camera_parameters, const Matrix4f& _init_pos);

    Eigen::Matrix4f GetCurrentPose() {return current_pose;}

    std::vector<Eigen::Matrix4f>& GetPoses() {return poses;}

    void process(const cv::Mat_<float>& depth_map, const cv::Mat_<cv::Vec3b>& color_map,double* time_info);

    bool get_if_output() const {return output_update;}

    const VolumeData* get_volume() const {return &volume;}

    void extract_mesh_export(const std::string& filename);

    /**
     * Extract a dense surface mesh
     * @return A SurfaceMesh representation (see description of SurfaceMesh for more information on the data layout)
     */
    Mesh extract_mesh_cuda() const;

    PointCloud extract_pointcloud_cuda() const;


private:
    // helper function: the function in test1.cpp
    void visual_volume(bool output_file);
    void visual_model(bool output_file);
    void visual_input(InputData& input);

    //all configuration parameters
    Configuration configuration;

    //input data struct, cannot iteratively use
    // InputData input;

    //model data struct
    ModelData model;

    //volume data struct
    VolumeData volume;

    //camera intrinsic paramters
    std::vector<CameraParameters> camera_parameters;

    //camera extrinsic parameters
    Eigen::Matrix4f current_pose;
    std::vector<Eigen::Matrix4f> poses;

    //level information
    int pyramid_level;

    //frame information
    int frame;

    //output information
    cv::Mat output;
    bool output_update { false };

};

void bilateral(cv::cuda::GpuMat& depthMapSrc, cv::cuda::GpuMat& depthMapDst, const int kernel_size,
        const float color_sigma, const float spatial_sigma);
void pyrdown(cv::cuda::GpuMat& depthMapSrc, cv::cuda::GpuMat& depthMapDst);

//Step 1
void surface_measurement(const cv::Mat_<float>& input_depth, InputData& input,
                         const std::vector<CameraParameters>& camera_params,
                         const int num_levels, const float depth_cutoff, const float distance_cutoff,
                         const int kernel_size, const float color_sigma, const float spatial_sigma);

//Step 2
bool pose_estimation(Eigen::Matrix4f& pose,
                     const std::vector<cv::cuda::GpuMat>& frame_vertex_pyramid,
                     const std::vector<cv::cuda::GpuMat>& frame_normal_pyramid,
                     const std::vector<cv::cuda::GpuMat>& model_vertex_pyramid,
                     const std::vector<cv::cuda::GpuMat>& model_normal_pyramid,
                     const std::vector<CameraParameters>& camera_parameters,
                     const int pyramid_height,
                     const float distance_threshold, const float angle_threshold,
                     const float det_threshold,
                     const std::vector<int>& iterations);

namespace cuda {
    // Step 3
    void surface_reconstruction_2d(const cv::cuda::GpuMat& depth_image, const cv::cuda::GpuMat& color_image,
                                VolumeData& volume,
                                const CameraParameters& cam_params, const float truncation_distance,
                                const float zero_threshold,
                                const Eigen::Matrix4f& extrinsic);

    void surface_reconstruction_3d(const cv::cuda::GpuMat& depth_image, const cv::cuda::GpuMat& color_image,
                                   VolumeData& volume,
                                   const CameraParameters& cam_params, const float truncation_distance,
                                   const float zero_threshold,
                                   const Eigen::Matrix4f& extrinsic);

    // Step 4

    void surface_prediction_2d(const VolumeData& volume,
                            cv::cuda::GpuMat& model_vertex,
                            cv::cuda::GpuMat& model_normal,
                            cv::cuda::GpuMat& model_color,
                            const CameraParameters& cam_parameters,
                            const float truncation_distance,
                            const Eigen::Matrix4f& pose);

    void surface_prediction_3d(const VolumeData& volume,
                               cv::cuda::GpuMat& model_vertex,
                               cv::cuda::GpuMat& model_normal,
                               cv::cuda::GpuMat& model_color,
                               const CameraParameters& cam_parameters,
                               const float truncation_distance,
                               const Eigen::Matrix4f& pose);

    PointCloud extract_pointcloud(const VolumeData& volume, const int buffer_size);

    Mesh extract_mesh(const VolumeData& volume, const int buffer_size);

    int extract_mesh_new(const VolumeData& volume, cv::cuda::GpuMat& vertex, cv::cuda::GpuMat& color, cv::cuda::GpuMat& num);
}

// void extract_mesh_export(const std::string& filename, const VolumeData& volume, bool verbose);

//generate point cloud
void export_pointcloud_ply(const std::string& filename, const PointCloud& point_cloud);

//generate point cloud
void export_mesh_ply(const std::string& filename, const Mesh& point_cloud);

#endif //INC_3D_PROJ_KINECTFUSION_H



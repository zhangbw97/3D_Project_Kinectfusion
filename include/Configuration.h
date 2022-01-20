//
// Created by slt on 1/23/2021.
//

#ifndef INC_3D_PROJ_CONFIGURATION_H
#define INC_3D_PROJ_CONFIGURATION_H
#include <vector>

struct Configuration {
    //the number of frame
    int iMax = 150;

    //inital value
    float init_depth = 1.5f;

    // Step 1 surface measurement configuration
    // Number of pyramid
    int used_pyramid_level = 3;

    // filter parameters
    int kernel_size = 5;
    float color_sigma = 1.0;
    float spatial_sigma = 1.0;

    // downsample parameter
    unsigned downsampleFactor = 3;

    // normal/vertex calculation parameter
    float maxDistance = 0.01f;
    float maxDepth = 5.f;

    // Step 2 pose estimation (icp) configuration
    // The distance threshold in m
    float distance_threshold= 0.01f;
    // The angle threshold in degree
    float angle_threshold= 10.0f;
    // Number of iterations for different pyramid level
    std::vector<int> icp_iterations {10, 5, 4};
    // Value of valid determinant of A
    float det_threshold = 1e-5;

    // Step 3 reconstruction surface configuration
    // choose version of step3
    bool step3_3d = false;
    // The truncation distance for both updating and raycasting the TSDF volume
    float truncation_distance = 0.08f;
    // The value for TSDF would be consider as zero-crossing
    float iso_level = 0.04f;
    // storage you have available. Dimensions are (x, y, z).
    int3 volume_size { make_int3(512, 512, 512) };
    // The amount of m one singlevoxel will represent in each dimension. Controls the resolution of the volume.
    float voxel_scale { 0.004f };

    // bool use_weight { true };
    // float weight_factor { 4.f };

    // Step 4 surface prediction configuration
    // choose version of step4
    bool step4_3d = true;


    // Output configuration
    bool output_middle_model = true;

    int output_frequency = 50;


    int mesh_buffer_size { 3 * 2000000 };

    int pointcloud_buffer_size { 3 * 2000000 };

    bool verbose = false;

    // if this parameter is true, we generate mesh; else we generate pointcloud
    bool mesh_output = true;

    bool gpu_output = false;
};

#endif //INC_3D_PROJ_CONFIGURATION_H


#include <iostream>
#include <fstream>
#include "include/Eigen.h"
#include "include/data_types.h"
#include "include/VirtualSensor.h"
#include "include/SimpleMesh.h"
#include "include/KinectFusion.h"
#include "include/MarchingCube_cpu.h"
#include <chrono>


#define RUN_SEQUENCE_ICP	1


int reconstructRoom() {
    std::string filenameIn = std::string("../data/rgbd_dataset_freiburg1_xyz/");
    std::string filenameBaseOut = std::string("mesh_");

    // Load video
    std::cout << "Initialize virtual sensor..." << std::endl;
    VirtualSensor sensor;
    if (!sensor.init(filenameIn)) {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    }

    // Initialize the KinectFusion Algorithm
    //第一帧相机坐标为世界坐标
    Matrix4f initial_pose = Matrix4f::Identity();
    Configuration configuration;
    configuration.verbose = true;

    //原答案中第一帧的相机位姿设置在 Volume 的中心, 然后在z轴上拉远一点，暂时没有定义configuration所以没有调整初始位置
    initial_pose(0, 3) = configuration.volume_size.x / 2 * configuration.voxel_scale;
    initial_pose(1, 3) = configuration.volume_size.y / 2 * configuration.voxel_scale;
    initial_pose(2, 3) = configuration.volume_size.z / 2 * configuration.voxel_scale - configuration.init_depth;

    CameraParameters camera_parameters{sensor.getDepthIntrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight()};
    KinectFusion kinectfusion(configuration, camera_parameters, initial_pose);
    std::cout << "Initial camera pose: " << std::endl << initial_pose << std::endl;

    int i = 1;
    const int iMax = configuration.iMax; //处理50帧
    double time_accumulation = 0;
    double time_info[5] = {0, 0, 0, 0, 0};
    while (sensor.processNextFrame() && i <= iMax) {

        cv::Mat depth_map(sensor.getColorImageHeight(), sensor.getColorImageWidth(), CV_32FC1, sensor.getDepth(), cv::Mat::AUTO_STEP);

        /*
        float sum_depth = 0.f;
        float sum_num = 0.f;
        float* dep_ = sensor.getDepth();
        for(int i = 0; i < sensor.getColorImageHeight() * sensor.getColorImageWidth(); i++) {
            if (dep_[i] != 0) {
                sum_depth += dep_[i];
                sum_num += 1;
            }
        }
        std::cout << "Mean DEPTH:" << sum_depth / sum_num << std::endl;
        std::cout << "Sum DEPTH:" << sum_depth << std::endl;
         */
        if(configuration.verbose) {
            auto t_start = std::chrono::high_resolution_clock::now();
            kinectfusion.process(depth_map, sensor.getColormap(),time_info);
            auto t_end = std::chrono::high_resolution_clock::now();
            time_accumulation += std::chrono::duration<double, std::milli>(t_end-t_start).count();
        } else
            kinectfusion.process(depth_map, sensor.getColormap(),time_info);

        if (kinectfusion.get_if_output()) {

            if (configuration.verbose) {
                time_accumulation /= configuration.output_frequency;
                std::cout << "Time for each iteration process:" << time_accumulation << std::endl;
                time_accumulation = 0;

                for(int i = 0; i <= 5; i++) {
                    time_info[i] /= configuration.output_frequency;
                }

                std::cout << "Upload Data Time:" << time_info[0] << std::endl;
                std::cout << "Step 1 Time:" << time_info[1] << std::endl;
                std::cout << "Step 2 Time:" << time_info[2] << std::endl;
                std::cout << "Step 3 Time:" << time_info[3] << std::endl;
                std::cout << "Step 4 Time:" << time_info[4] << std::endl;

                for(int i = 0; i <= 5; i++) {
                    time_info[i] = 0;
                }

                std::string str_volume_size = std::to_string(configuration.volume_size.x);
                std::string str_voxel_scale = std::to_string(static_cast<int>(configuration.voxel_scale*1000));
                std::string str_td = std::to_string(static_cast<int>(configuration.truncation_distance*1000));
                std::string str_iso = std::to_string(static_cast<int>(configuration.iso_level*1000));
                std::string str_s4 = std::to_string(static_cast<int>(static_cast<int>(configuration.step4_3d)));
                std::string middle = "_" + str_volume_size + "_" + str_voxel_scale + "_" + str_td + "_" +
                        str_iso  + "_" + str_s4  + "_";

                if (configuration.mesh_output) {

                    if(configuration.gpu_output) {

                        //kinectfusion.extract_mesh_export("Mesh_new_" + middle);

                        auto t1 = std::chrono::high_resolution_clock::now();
                        Mesh mesh_output = kinectfusion.extract_mesh_cuda();
                        auto t2 = std::chrono::high_resolution_clock::now();
                        export_mesh_ply("Mesh" + middle + std::to_string(i), mesh_output);
                        auto t3 = std::chrono::high_resolution_clock::now();
                        std::cout << "Generate Mesh:"<< std::chrono::duration<double, std::milli>(t2-t1).count() << std::endl;
                        std::cout << "Write Mesh:"<< std::chrono::duration<double, std::milli>(t3-t2).count() << std::endl;

                    }
                    else {
                        auto t1 = std::chrono::high_resolution_clock::now();

                        cv::Mat tsdf;
                        cv::Mat color;


                        kinectfusion.get_volume()->tsdf_volume.download(tsdf);
                        kinectfusion.get_volume()->color_volume.download(color);
                        float scale = kinectfusion.get_volume()->voxel_scale;
                        int3 vol_size = kinectfusion.get_volume()->volume_size;
                        SimpleMesh mesh;


                        for(int x = 0; x < vol_size.x - 2; x++)
                            for(int y = 0; y < vol_size.y - 2; y++)
                                for(int z = 0; z < vol_size.z - 2; z++)
                                    ProcessVolumeCell(tsdf, color, scale, vol_size.y, x, y, z, &mesh);


                        auto t2 = std::chrono::high_resolution_clock::now();
                        std::stringstream ss;
                        ss << "Mesh_cpu_" << middle << std::to_string(i) << ".OFF";
                        if (!mesh.writeMesh(ss.str()))
                        {
                            std::cout << "ERROR: unable to write output file!" << std::endl;
                            return -1;
                        }
                        auto t3 = std::chrono::high_resolution_clock::now();
                        std::cout << "Generate Mesh:"<< std::chrono::duration<double, std::milli>(t2-t1).count() << std::endl;
                        std::cout << "Write Mesh:"<< std::chrono::duration<double, std::milli>(t3-t2).count() << std::endl;
                    }
                } else {


                    auto t1 = std::chrono::high_resolution_clock::now();
                    PointCloud cloud_output = kinectfusion.extract_pointcloud_cuda();
                    auto t2 = std::chrono::high_resolution_clock::now();
                    export_pointcloud_ply("PointCloud" + middle + std::to_string(i), cloud_output);
                    auto t3 = std::chrono::high_resolution_clock::now();
                    std::cout << "Generate PointCloud:"<< std::chrono::duration<double, std::milli>(t2-t1).count() << std::endl;
                    std::cout << "Write PointCloud:"<< std::chrono::duration<double, std::milli>(t3-t2).count() << std::endl;

                }
            }
            else {

                std::string str_volume_size = std::to_string(configuration.volume_size.x);
                std::string str_voxel_scale = std::to_string(configuration.voxel_scale*1000);
                std::string str_td = std::to_string(configuration.truncation_distance*1000);
                std::string str_iso = std::to_string(configuration.iso_level*1000);
                std::string str_s4 = std::to_string(static_cast<int>(configuration.step4_3d));
                std::string middle = "_" + str_volume_size + "_" + str_voxel_scale + "_" + str_td + "_" +
                                     str_iso  + "_" + str_s4  + "_";

                if (configuration.mesh_output) {

                    // Mesh mesh_output = kinectfusion.extract_mesh_cpu();
                    if(configuration.gpu_output) {
                        // kinectfusion.extract_mesh_export("Mesh_new_" + middle);
                        Mesh mesh_output = kinectfusion.extract_mesh_cuda();
                        export_mesh_ply("Mesh" + middle + std::to_string(i), mesh_output);
                    }
                    else {
                        cv::Mat tsdf;
                        cv::Mat color;

                        kinectfusion.get_volume()->tsdf_volume.download(tsdf);
                        kinectfusion.get_volume()->color_volume.download(color);
                        float scale = kinectfusion.get_volume()->voxel_scale;
                        int3 vol_size = kinectfusion.get_volume()->volume_size;
                        SimpleMesh mesh;

                        for(int x = 0; x < vol_size.x - 2; x++)
                            for(int y = 0; y < vol_size.y - 2; y++)
                                for(int z = 0; x < vol_size.x - 2; z++)
                                    ProcessVolumeCell(tsdf, color, scale, vol_size.y, x, y, z, &mesh);

                        std::stringstream ss;
                        ss << "Mesh_cpu_" << middle << std::to_string(i) << ".OFF";
                        if (!mesh.writeMesh(ss.str()))
                        {
                            std::cout << "ERROR: unable to write output file!" << std::endl;
                            return -1;
                        }
                    }

                } else {

                    PointCloud cloud_output = kinectfusion.extract_pointcloud_cuda();
                    export_pointcloud_ply("PointCloud" + middle + std::to_string(i), cloud_output);

                }

            }
        }

        i++;
    }

    return 0;
}

int main() {
    int result = 0;

    if (RUN_SEQUENCE_ICP)
        result += reconstructRoom();

    return result;
}

//
// Created by slt on 1/20/2021.
//
#include "data_types.h"
#include "KinectFusion.h"
#include <chrono>
using namespace std;
using namespace Eigen;

KinectFusion::KinectFusion (const Configuration& _con, const CameraParameters& _camera_parameters, const Matrix4f& _init_pos) :
        model(_con.used_pyramid_level, _camera_parameters),
        volume(_con.volume_size, _con.voxel_scale),
        pyramid_level(_con.used_pyramid_level),
        configuration(_con),
        current_pose(_init_pos),
        output(_camera_parameters.height, _camera_parameters.width, CV_8UC3),
        frame(0)
        {
            for (int i = 0; i < pyramid_level; i++) {
                CameraParameters cam_level{_camera_parameters, i+1};
                camera_parameters.push_back(cam_level);
            }
        }

void KinectFusion::process(const cv::Mat_<float>& depth_map, const cv::Mat_<cv::Vec3b>& color_map, double* time_info)
 {
    // cv::cuda::GpuMat color_map_ = cv::cuda::createContinuous(height, width, CV_8UC3);
    // color_map_.upload(color_map);

     auto t_start = std::chrono::high_resolution_clock::now();

     frame++;
     // std::cout << "++++++++++++++++" << std::endl;
     if ((configuration.output_middle_model &&  frame % configuration.output_frequency == 0) ||
         frame == configuration.iMax) {
         output_update = true;
         // model.color_pyramid[0].download(output);
     } else
         output_update = false;


     InputData input{pyramid_level, camera_parameters[0]};
     input.color_pyramid[0].upload(color_map);

     if(configuration.verbose) {
         time_info[0] += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-t_start).count();
         t_start = std::chrono::high_resolution_clock::now();
     }



     const float maxDistanceHalved = configuration.maxDistance / 2.f;
    const float maxDepthHalved = configuration.maxDepth;

    // Step 1
    surface_measurement(depth_map, input, camera_parameters,
                        configuration.used_pyramid_level,
                        maxDepthHalved,
                        maxDistanceHalved,
                        configuration.kernel_size,
                        configuration.color_sigma,
                        configuration.spatial_sigma);

    //visual_input(input);

     if(configuration.verbose) {
         time_info[1] += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-t_start).count();
         t_start = std::chrono::high_resolution_clock::now();
     }

    // Step 2
    if (frame > 1) {
        bool icp_success = pose_estimation(
                current_pose,                                   // 输入: 上一帧的相机位姿; 输出: 当前帧得到的相机位姿
                input.vertex_pyramid,                           // 当前帧的深度图/顶点图/法向图数据
                input.normal_pyramid,
                model.vertex_pyramid,                            // 上一帧图像输入后, 推理出的平面模型，使用顶点图、法向图来表示
                model.normal_pyramid,                            //！！！！ATTENTION由于后三部分为循环，此处本应使用上一帧推理出的平面模型，现暂时用上一帧直接得到的顶点图、法向图来表示
                camera_parameters,                              // 相机内参
                pyramid_level,                                // 金字塔层数

                //originally defined in configuration
                configuration.distance_threshold,                // icp 匹配过程中视为 outlier 的距离差
                configuration.angle_threshold,                  // icp 匹配过程中视为 outlier 的角度差 (deg)
                configuration.det_threshold,
                configuration.icp_iterations);                  // icp 过程的迭代次数

        // 如果 icp 过程不成功, 那么就说明当前失败了
        if (!icp_success) {
            // icp失败之后本次处理退出,但是上一帧推理的得到的平面将会一直保持, 每次新来一帧都会重新icp后一直都在尝试重新icp, 尝试重定位回去
            std::cout << "pose estimation failed for this frame\n";
            return;
        }
            // 记录当前帧的位姿
        else {
            // std::cout << "Current camera pose: " << std::endl << current_pose << std::endl;
            poses.push_back(current_pose);
        }

    }

     if(configuration.verbose) {
         time_info[2] += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-t_start).count();
         t_start = std::chrono::high_resolution_clock::now();
     }

    // Step 3

    if(configuration.step3_3d)
        cuda::surface_reconstruction_3d(input.depth_pyramid[0], input.color_pyramid[0],
                                    volume, camera_parameters[0], configuration.truncation_distance,
                                    configuration.iso_level, current_pose.inverse());
    else
        cuda::surface_reconstruction_2d(input.depth_pyramid[0], input.color_pyramid[0],
                                        volume, camera_parameters[0], configuration.truncation_distance,
                                        configuration.iso_level, current_pose.inverse());


    // visual_volume(false);
     if(configuration.verbose) {
         time_info[3] += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-t_start).count();
         t_start = std::chrono::high_resolution_clock::now();
     }
    
    // Step 4
    if(configuration.step4_3d) {
       for (int level = 0; level < pyramid_level; ++level)
           cuda::surface_prediction_3d(volume, model.vertex_pyramid[level],
                                       model.normal_pyramid[level],
                                       model.color_pyramid[level],
                                       camera_parameters[level], configuration.iso_level,
                                       current_pose);
    } else {
        for (int level = 0; level < pyramid_level; ++level)
            cuda::surface_prediction_2d(volume, model.vertex_pyramid[level],
                                        model.normal_pyramid[level],
                                        model.color_pyramid[level],
                                        camera_parameters[level], configuration.iso_level,
                                        current_pose);
    }


    // visual_model(false);
     if(configuration.verbose) {
         time_info[4] += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-t_start).count();
         t_start = std::chrono::high_resolution_clock::now();
     }
}


PointCloud KinectFusion::extract_pointcloud_cuda() const
{
    PointCloud cloud_data = cuda::extract_pointcloud(volume, configuration.pointcloud_buffer_size);
    return cloud_data;
}

void export_pointcloud_ply(const std::string& filename, const PointCloud& point_cloud)
{
    std::ofstream file_out { filename + ".ply" };
    if (!file_out.is_open())
        return;

    file_out << "ply" << std::endl;
    file_out << "format ascii 1.0" << std::endl;
    file_out << "element vertex " << point_cloud.num_point << std::endl;
    file_out << "property float x" << std::endl;
    file_out << "property float y" << std::endl;
    file_out << "property float z" << std::endl;
    file_out << "property float nx" << std::endl;
    file_out << "property float ny" << std::endl;
    file_out << "property float nz" << std::endl;
    file_out << "property uchar red" << std::endl;
    file_out << "property uchar green" << std::endl;
    file_out << "property uchar blue" << std::endl;
    file_out << "end_header" << std::endl;

    for (int i = 0; i < point_cloud.num_point ; ++i) {
        float3 vertex = point_cloud.vertex.ptr<float3>(0)[i];
        float3 normal = point_cloud.normal.ptr<float3>(0)[i];
        uchar3 color = point_cloud.color.ptr<uchar3>(0)[i];
        file_out << vertex.x << " " << vertex.y << " " << vertex.z << " " << normal.x << " " << normal.y << " "
                 << normal.z << " ";
        file_out << static_cast<int>(color.x) << " " << static_cast<int>(color.y) << " "
                 << static_cast<int>(color.z) << std::endl;
    }
}

Mesh KinectFusion::extract_mesh_cuda() const
{
    Mesh mesh = cuda::extract_mesh(volume, configuration.mesh_buffer_size);
    return mesh;
}

void export_mesh_ply(const std::string& filename, const Mesh& surface_mesh)
{
    std::ofstream file_out { filename + ".ply" };
    if (!file_out.is_open())
        return;

    file_out << "ply" << std::endl;
    file_out << "format ascii 1.0" << std::endl;
    file_out << "element vertex " << surface_mesh.num_vertices << std::endl;
    file_out << "property float x" << std::endl;
    file_out << "property float y" << std::endl;
    file_out << "property float z" << std::endl;
    file_out << "property uchar red" << std::endl;
    file_out << "property uchar green" << std::endl;
    file_out << "property uchar blue" << std::endl;
    file_out << "element face " << surface_mesh.num_triangles << std::endl;
    file_out << "property list uchar int vertex_index" << std::endl;
    file_out << "end_header" << std::endl;

    for (int v_idx = 0; v_idx < surface_mesh.num_vertices; ++v_idx) {
        float3 vertex = surface_mesh.triangles.ptr<float3>(0)[v_idx];
        uchar3 color = surface_mesh.colors.ptr<uchar3>(0)[v_idx];
        file_out << vertex.x << " " << vertex.y << " " << vertex.z << " ";
        file_out << static_cast<int>(color.x) << " " << static_cast<int>(color.y) << " "
                 << static_cast<int>(color.z) << std::endl;
    }

    for (int t_idx = 0; t_idx < surface_mesh.num_vertices; t_idx += 3) {
        file_out << 3 << " " << t_idx + 1 << " " << t_idx << " " << t_idx + 2 << std::endl;
    }
}


void KinectFusion::extract_mesh_export(const std::string& filename) {

    auto t1 = std::chrono::high_resolution_clock::now();
    cv::cuda::GpuMat vertex = cv::cuda::createContinuous(volume.volume_size.y * volume.volume_size.z, volume.volume_size.x * 15, CV_32FC3);
    cv::cuda::GpuMat color = cv::cuda::createContinuous(volume.volume_size.y * volume.volume_size.z, volume.volume_size.x * 15, CV_8UC3);
    cv::cuda::GpuMat num = cv::cuda::createContinuous(volume.volume_size.y * volume.volume_size.z, volume.volume_size.x, CV_32FC1);

    vertex.setTo(0);
    color.setTo(0);
    num.setTo(0);

    cv::Mat vertex_output;
    cv::Mat color_output;
    cv::Mat num_output;

    int count = cuda::extract_mesh_new(volume, vertex, color, num);
    vertex.download(vertex_output);
    color.download(color_output);
    num.download(num_output);

    auto t2 = std::chrono::high_resolution_clock::now();

    std::ofstream outFile(filename + ".OFF");
    if (!outFile.is_open()) return;

    // Write header.
    outFile << "COFF" << std::endl;
    outFile << count * 3 << " " << count << " 0" << std::endl;

    // Save vertices.
    int size_x = volume.volume_size.x;
    int size_y = volume.volume_size.y;
    int size_z = volume.volume_size.z;
    for (int x = 0; x < size_x; x++) {
        for (int y = 0; y < size_y; y++) {
            for (int z = 0; z < size_z; z++) {
                 int count_idx = num.ptr<int>(z * size_y  + y)[x];
                 for(int i = 0; i < count_idx; i++) {
                     float3 pos = vertex.ptr<float3>(z * size_y  + y)[x * 15 + i];
                     uchar3 col = color.ptr<uchar3>(z * size_y  + y)[x * 15 + i];
                     outFile << pos.x << " "
                             << pos.y << " "
                             << pos.z << " "
                             << int(col.x) << " "
                             << int(col.y) << " "
                             << int(col.z) << " "
                             << 255 << std::endl;
                 }
            }
        }
    }


        // Save faces.
    for (unsigned int i = 0; i < count; i++) {
        outFile << "3 " << i * 3 << " " << i * 3 + 1 << " " << i * 3 + 2<< std::endl;
    }

    // Close file.
    outFile.close();

    auto t3 = std::chrono::high_resolution_clock::now();
    if (configuration.verbose) {
        std::cout << "Generate Mesh:"<< std::chrono::duration<double, std::milli>(t2-t1).count() << std::endl;
        std::cout << "Write Mesh:"<< std::chrono::duration<double, std::milli>(t3-t2).count() << std::endl;
    }

}


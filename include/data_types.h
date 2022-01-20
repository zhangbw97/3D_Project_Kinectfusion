// This header contains globally used data types

#ifndef KINECTFUSION_DATA_TYPES_H
#define KINECTFUSION_DATA_TYPES_H
// HACK
// #define EIGEN_NO_CUDA

// #if (defined __GNUC__) && (__GNUC__>4 || __GNUC_MINOR__>=7)
//   #undef _GLIBCXX_ATOMIC_BUILTINS
//   #undef _GLIBCXX_USE_INT128
// #endif

// #define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Weffc++"
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <Eigen/Eigen>
#pragma GCC diagnostic pop
#else
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <Eigen/Eigen>
#endif
#include "Configuration.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
using namespace Eigen;
using cv::cuda::GpuMat;

struct CameraParameters {
        int width, height;
        float fovX, fovY;
        float cX, cY;
        CameraParameters () = default;
        CameraParameters(int _width, int _height,
                float _fovX, float _fovY, float _cX, float _cY) : width(_width), height(_height),
                fovX(_fovX), fovY(_fovY), cX(_cX), cY(_cY){};

        CameraParameters(const CameraParameters& old_cam, int level) : width(old_cam.width), height(old_cam.height),
            fovX(old_cam.fovX), fovY(old_cam.fovY), cX(old_cam.cX), cY(old_cam.cY){
            for (int i = 1; i < level; i++) {
                fovX = fovX * 0.5f;
                fovY = fovY * 0.5f;
                cX = (cX + 0.5f) * 0.5 - 0.5f;
                cY = (cY + 0.5f) * 0.5 - 0.5f;
                width = (width + 1) / 2;
                height = (height + 1) / 2;
            }
        }

        CameraParameters(const Matrix3f& Intrinsics, unsigned _width, unsigned _height) : width(_width), height(_height){
            fovX = Intrinsics(0, 0);
            fovY = Intrinsics(1, 1);
            cX = Intrinsics(0, 2);
            cY = Intrinsics(1, 2);
        }
};

struct InputData {

    std::vector<cv::cuda::GpuMat> depth_pyramid;                              // 原始深度图的金字塔
    std::vector<cv::cuda::GpuMat> filtered_depth_pyramid;                     // 经过过滤后的深度图的金字塔
    std::vector<cv::cuda::GpuMat> color_pyramid;                              // 彩色图像的金字塔

    std::vector<cv::cuda::GpuMat> vertex_pyramid;                             // 3D点金字塔
    std::vector<cv::cuda::GpuMat> normal_pyramid;                             // 法向量金字塔

    InputData () = default;
    InputData (const int level, const CameraParameters& camera)/*:
        depth_pyramid(level),
        filtered_depth_pyramid(level),
        color_pyramid(level),

        vertex_pyramid(level),
        normal_pyramid(level)*/
    {
        int width = camera.width;
        int height = camera.height;
        for (int i = 0; i < level; i++) {
            depth_pyramid.push_back(cv::cuda::createContinuous(height,width,CV_32FC1));
            filtered_depth_pyramid.push_back(cv::cuda::createContinuous(height,width,CV_32FC1));

            color_pyramid.push_back(cv::cuda::createContinuous(height,width,CV_8UC3));

            vertex_pyramid.push_back(cv::cuda::createContinuous(height,width,CV_32FC3));
            normal_pyramid.push_back(cv::cuda::createContinuous(height,width,CV_32FC3));

            vertex_pyramid[i].setTo(0);
            normal_pyramid[i].setTo(0);

            width = (width + 1)/ 2;
            height = (height + 1) / 2;
        }
    };

    InputData(const InputData& data) = delete;
    InputData& operator=(const InputData& data) = delete;

    InputData(InputData&& data){
        int level = data.depth_pyramid.size();
        for (int i = 0; i < level; i++) {
            depth_pyramid.push_back(std::move(data.depth_pyramid[i]));
            filtered_depth_pyramid.push_back(std::move(filtered_depth_pyramid[i]));
            color_pyramid.push_back(std::move(data.color_pyramid[i]));
            vertex_pyramid.push_back(std::move(data.vertex_pyramid[i]));
            normal_pyramid.push_back(std::move(data.normal_pyramid[i]));
        }
    }

    InputData& operator=(InputData&& data)
    {
        int level = data.depth_pyramid.size();
        for (int i = 0; i < level; i++) {
            depth_pyramid.push_back(std::move(data.depth_pyramid[i]));
            filtered_depth_pyramid.push_back(std::move(filtered_depth_pyramid[i]));
            color_pyramid.push_back(std::move(data.color_pyramid[i]));
            vertex_pyramid.push_back(std::move(data.vertex_pyramid[i]));
            normal_pyramid.push_back(std::move(data.normal_pyramid[i]));
        }
        return *this;
    }

};

struct ModelData {

    std::vector<cv::cuda::GpuMat> color_pyramid;                              // 彩色图像的金字塔
    std::vector<cv::cuda::GpuMat> vertex_pyramid;                             // 3D点金字塔
    std::vector<cv::cuda::GpuMat> normal_pyramid;                             // 法向量金字塔

    ModelData () = default;
    ModelData (const int level, const CameraParameters& camera):
        color_pyramid(level),
        vertex_pyramid(level),
        normal_pyramid(level)
    {
        int width = camera.width;
        int height = camera.height;
        for (int i = 0; i < level; i++) {

            color_pyramid[i]=(cv::cuda::createContinuous(height,width,CV_8UC3));
            vertex_pyramid[i]=(cv::cuda::createContinuous(height,width,CV_32FC3));
            normal_pyramid[i]=(cv::cuda::createContinuous(height,width,CV_32FC3));

            vertex_pyramid[i].setTo(0);
            normal_pyramid[i].setTo(0);

            width = (width + 1 ) / 2;
            height = (height +1 ) / 2;
        }
    };

    ModelData(const ModelData& data) = delete;
    ModelData& operator=(const ModelData& data) = delete;

    ModelData(ModelData&& data){

        int level = data.color_pyramid.size();
        for (int i = 0; i < level; i++) {
            color_pyramid[i]=(std::move(data.color_pyramid[i]));
            vertex_pyramid[i]=(std::move(data.vertex_pyramid[i]));
            normal_pyramid[i]=(std::move(data.normal_pyramid[i]));
        }
    }

    ModelData& operator=(ModelData&& data)
    {

        int level = data.color_pyramid.size();
        for (int i = 0; i < level; i++) {
            color_pyramid.push_back(std::move(data.color_pyramid[i]));
            vertex_pyramid.push_back(std::move(data.vertex_pyramid[i]));
            normal_pyramid.push_back(std::move(data.normal_pyramid[i]));
        }

        return *this;
    }

};

struct VolumeData {

    cv::cuda::GpuMat tsdf_volume;  //float2
    cv::cuda::GpuMat color_volume; //uchar3
    int3 volume_size;
    float voxel_scale;
    VolumeData () = default;
    VolumeData(const int3 _volume_size, const float _voxel_scale) :
            volume_size(_volume_size), voxel_scale(_voxel_scale)
    {
        tsdf_volume = cv::cuda::createContinuous(_volume_size.y * _volume_size.z, _volume_size.x, CV_32FC2);
        color_volume = cv::cuda::createContinuous(_volume_size.y * _volume_size.z, _volume_size.x, CV_8UC3);

        tsdf_volume.setTo(-1.f);
        color_volume.setTo(0);
    }
};

struct CloudData {
    GpuMat vertex;
    GpuMat normal;
    GpuMat color;

    int* num_point;

    explicit CloudData(const int max_number) :
            vertex{cv::cuda::createContinuous(1, max_number, CV_32FC3)},
            normal{cv::cuda::createContinuous(1, max_number, CV_32FC3)},
            color{cv::cuda::createContinuous(1, max_number, CV_8UC3)},
            num_point{nullptr}
    {
        vertex.setTo(0.f);
        normal.setTo(0.f);
        color.setTo(0.f);

        cudaMalloc(&num_point, sizeof(int));
        cudaMemset(num_point, 0, sizeof(int));
    }

    CloudData(const CloudData&) = delete;
    CloudData& operator=(const CloudData& data) = delete;

};

struct PointCloud {
    // World coordinates of all vertices
    cv::Mat vertex;
    // Normal directions
    cv::Mat normal;
    // RGB color values
    cv::Mat color;

    // Total number of valid points
    int num_point;
};


struct MeshData {
    GpuMat occupied_voxel_ids_buffer;
    GpuMat number_vertices_buffer;
    GpuMat vertex_offsets_buffer;
    GpuMat triangle_buffer;
    GpuMat occupied_voxel_ids;
    GpuMat number_vertices;
    GpuMat vertex_offsets;
    explicit MeshData(const int buffer_size):
            occupied_voxel_ids_buffer{cv::cuda::createContinuous(1, buffer_size, CV_32SC1)},
            number_vertices_buffer{cv::cuda::createContinuous(1, buffer_size, CV_32SC1)},
            vertex_offsets_buffer{cv::cuda::createContinuous(1, buffer_size, CV_32SC1)},
            triangle_buffer{cv::cuda::createContinuous(1, buffer_size * 3, CV_32FC3)},
            occupied_voxel_ids{}, number_vertices{}, vertex_offsets{}
    { }
    void create_view(const int length)
    {
        occupied_voxel_ids = GpuMat(1, length, CV_32SC1, occupied_voxel_ids_buffer.ptr<int>(0),
                                    occupied_voxel_ids_buffer.step);
        number_vertices = GpuMat(1, length, CV_32SC1, number_vertices_buffer.ptr<int>(0),
                                 number_vertices_buffer.step);
        vertex_offsets = GpuMat(1, length, CV_32SC1, vertex_offsets_buffer.ptr<int>(0),
                                vertex_offsets_buffer.step);
    }
};

struct Mesh {
    // Triangular faces
    cv::Mat triangles;
    // Colors of the vertices
    cv::Mat colors;

    // Total number of vertices
    int num_vertices;
    // Total number of triangles
    int num_triangles;
};

using Matf31da = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;
using Matrix3frm = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;

#endif //KINECTFUSION_DATA_TYPES_H

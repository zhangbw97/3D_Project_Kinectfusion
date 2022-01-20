
#include <limits>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <cmath>

const float blah = INFINITY;
using namespace std;
// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"
// #include "main.h"

// #define MINF -std::numeric_limits<float>::infinity()
// global const float blah = INFINITY;

namespace cuda {

    // CUDA kernel for vector addition
    // No change when using CUDA unified memory
    __global__ void kernel_ComputeVertex(float *depth_, Eigen::Vector3f* pointsTmp_,  size_t N, int width,float fovX,float fovY,float cX,float cY)
    {


      // Calculate global thread thread ID
      // 固定写法，索引值，下面的函数只要写对单个索引的操作就可以 tid
      // int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
      // int v = idx / width;
      // int u = idx - v * width;

      const int u = blockIdx.x * blockDim.x + threadIdx.x;
      const int v = blockIdx.y * blockDim.y + threadIdx.y;
      const int idx = v * width + u;
      // Boundary check
      if (idx < N) {
        if (depth_[idx] == blah) {
            pointsTmp_[idx].x() = blah;
            pointsTmp_[idx].y() = blah;
            pointsTmp_[idx].z() = blah;
            }
          else {
            // Back-projection to world space.
            pointsTmp_[idx] = Eigen::Vector3f((u - cX) / fovX * depth_[idx], (v - cY) / fovY * depth_[idx], depth_[idx]);
            }
      }
    }

    void ComputeVertex(float* depthMap, std::vector<Eigen::Vector3f>& pointsTmp, float fovX, float fovY, float cX, float cY , int width, int height)
     {
      // Array size of 2^16 (65536 elements)
      size_t N = width * height;
      size_t bytes = N * sizeof(float);
      // Declare unified memory pointers
      float *depthMap_;
      Eigen::Vector3f *pointsTmp_;

      // Allocation memory for these pointers
      // 给GPU数组分配内存（固定内存）
      cudaMallocManaged(&depthMap_, bytes);
      cudaMallocManaged(&pointsTmp_, sizeof(Eigen::Vector3f)*N);

      // 把普通数组的内容拷贝到GPU数组
      cudaMemcpy(depthMap_, depthMap, bytes, cudaMemcpyHostToDevice);
      // cudaMemcpy(pointsTmp_, pointsTmp, 3 * bytes, cudaMemcpyHostToDevice);

      dim3 threads(32, 32);
      dim3 blocks((width + threads.x - 1) / threads.x,
                  (height + threads.y - 1) / threads.y);
      // Call CUDA kernel
      // 启动核函数计算
      // kernel_ComputeVertex<<<GRID_SIZE, BLOCK_SIZE>>>(depthMap_, pointsTmp_, N, width,fovX,fovY,cX,cY) ;
      kernel_ComputeVertex<<<blocks, threads>>>(depthMap_, pointsTmp_, N, width,fovX,fovY,cX,cY) ;


      // 固定写法，等待全部计算完（没细研究了）
      cudaDeviceSynchronize();



      // 把计算完的GPU数组的内容在赋值给普通数组 vector<>（算出来的结果）
      cudaMemcpy(pointsTmp.data(), pointsTmp_, N * sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost);

      // // Free unified memory (same as memory allocated with cudaMalloc)
      // 释放GPU数组内存
      cudaFree(depthMap_);
      cudaFree(pointsTmp_);


      cout << "成功计算当前层点云!\n";
    }

}
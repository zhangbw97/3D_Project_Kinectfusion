#include "common.h"
using namespace std;

namespace cuda {

    __global__ void kernel_ComputeVertex(const PtrStepSz<float> depth_map,       // 滤波之后的深度图像对象;PtrStepSz也是一个矩阵，核函数里面GPUMat要写成cv::cuda::PtrStepSz，但是在调用这个函数的输入数据只要是GPUMat就可以
      PtrStepSz<float3> vertex_map,
      const int width,const int height,
      const float fovX,const float fovY,
      const float cX,const float cY,
      const float depth_cutoff)
    {
      // Calculate global thread thread ID
      // 固定写法，索引值，下面的函数只要写对单个索引的操作就可以
      const int x = blockIdx.x * blockDim.x + threadIdx.x;
      const int y = blockIdx.y * blockDim.y + threadIdx.y;

      // Boundary check
      if (x < width && y < height && x >= 0 && y >= 0) {
        if (depth_map.ptr(y)[x] == 0.f || depth_map.ptr(y)[x] >= depth_cutoff) // 大于 depth_cutoff 的舍弃
        {
          vertex_map.ptr(y)[x] = make_float3(0.f,0.f,0.f);
        }
        else
        {
          float depth_value = depth_map.ptr(y)[x];
          // float depth_value = 1.f;
          // Back-projection to world space.
          Eigen::Vector3f vertex(
          (x - cX) * depth_value / fovX,
          (y - cY) * depth_value / fovY,
          depth_value);

      // 保存计算结果
            vertex_map.ptr(y)[x] = make_float3(vertex.x(), vertex.y(), vertex.z());
         }
      }
    }

    __host__
    void ComputeVertex(const GpuMat& depth_map,                // 输入的一层滤波之后的深度图像
      GpuMat& vertex_map,
       const float fovX, const float fovY,
       const float cX, const float cY,
       const int width,const int height,
       const float depth_cutoff)
     {

      dim3 threads(32, 32);
      dim3 blocks((width + threads.x - 1) / threads.x,
                  (height + threads.y - 1) / threads.y);
      // Call CUDA kernel
      // 启动核函数计算
      // kernel_ComputeVertex<<<GRID_SIZE, BLOCK_SIZE>>>(depthMap_, pointsTmp_, N, width,fovX,fovY,cX,cY,depth_cutoff);
      kernel_ComputeVertex<<<blocks, threads>>>(depth_map, vertex_map, width,height,fovX,fovY,cX,cY,depth_cutoff);

      // 固定写法，等待全部计算完
      cudaDeviceSynchronize();

      // cout << "Computed Vertex Successfully!\n";
    }

}
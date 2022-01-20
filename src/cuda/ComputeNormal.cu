#include "common.h"

namespace cuda {

    __global__ void kernel_Normals(                // 输入, 顶点图
      PtrStepSz<float3> normals_map,
      const PtrStepSz<float3> vertex_map,
        const float maxDistanceHalved)
       {
        const float blah = INFINITY;
        // step 1 根据当前线程id得到要进行处理的像素, 并且进行区域有效性判断
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if ( x < 1 || x >= vertex_map.cols - 1 || y < 1 || y >= vertex_map.rows - 1) // 无效边界点
            return;
        // step 2 获取以当前顶点为中心, 上下左右四个方向的顶点数据, 都是在当前帧相机坐标系下的坐标表示
        const Vector3f left(&vertex_map.ptr(y)[x - 1].x);
        const Vector3f right(&vertex_map.ptr(y)[x + 1].x);
        const Vector3f upper(&vertex_map.ptr(y - 1)[x].x);
        const Vector3f lower(&vertex_map.ptr(y + 1)[x].x);


        const float du = 0.5f * (right.z() - left.z());
        const float dv = 0.5f * (lower.z() - upper.z());

        Vector3f normal;
        if (left.z() == 0 || right.z() == 0 || upper.z() == 0 || lower.z() == 0 || // 前面把深度大于2m的设为0了
            abs(du) > maxDistanceHalved || abs(dv) > maxDistanceHalved) // 无效深度值相差太大的点
        {
            normal = Eigen::Vector3f(0.f, 0.f, 0.f);
        }
        else
        {
            normal = (left-right).cross(lower-upper);
            normal.normalize();
            //保持法向量指向同一侧
            if (normal.z() > 0)
                normal *= -1;
        }
        // 保存计算的法向量结果
        normals_map.ptr(y)[x] = make_float3(normal.x(), normal.y(), normal.z());
    }

    __host__
    void ComputeNormal(GpuMat& normals_map, const GpuMat& vertex_map,const int width,const int height,const float maxDistanceHalved)
     {


      dim3 threads(32, 32);
      dim3 blocks((width + threads.x - 1) / threads.x,
                  (height + threads.y - 1) / threads.y);

      kernel_Normals<<<blocks, threads>>>(normals_map, vertex_map, maxDistanceHalved) ;



      // 固定写法，等待全部计算完（没细研究了）
      cudaDeviceSynchronize();


      // std::cout << "Computed Normals Successfully!!\n";
    }

}
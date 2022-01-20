#include "common.h"
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32
namespace cuda {
    /*
    template<int SIZE>
    static __device__ __forceinline__
    void  warpReduce(volatile double* sdata, const int tid)
    {
         if (SIZE >= 64) sdata[tid] += sdata[tid + 32];
         if (SIZE >= 32) sdata[tid] += sdata[tid + 16];
         if (SIZE >= 16) sdata[tid] += sdata[tid + 8];
         if (SIZE >= 8) sdata[tid] += sdata[tid + 4];
         if (SIZE >= 4) sdata[tid] += sdata[tid + 2];
         if (SIZE >= 2) sdata[tid] += sdata[tid + 1];
    }
    */

    template<int SIZE>
    static __device__ __forceinline__
    void  Reduce(volatile double* sdata, const int tid)
    {
         if (SIZE >= 1024) {
             if (tid < 512) sdata[tid] += sdata[tid + 512];
             __syncthreads();
         }
         if (SIZE >= 512) {
             if (tid < 256) sdata[tid] += sdata[tid + 256];
             __syncthreads();
         }
         if (SIZE >= 256) {
             if (tid < 128) sdata[tid] += sdata[tid + 128];
             __syncthreads();
         }
         if (SIZE >= 128) {
             if (tid < 64) sdata[tid] += sdata[tid + 64];
             __syncthreads();
         }
         if (tid < 32) {
             if (SIZE >= 64) sdata[tid] += sdata[tid + 32];
             if (SIZE >= 32) sdata[tid] += sdata[tid + 16];
             if (SIZE >= 16) sdata[tid] += sdata[tid + 8];
             if (SIZE >= 8) sdata[tid] += sdata[tid + 4];
             if (SIZE >= 4) sdata[tid] += sdata[tid + 2];
             if (SIZE >= 2) sdata[tid] += sdata[tid + 1];
         }
    }

    __global__
    void estimate_kernel(const Matf33da rotation,
                            const Matf31da translation,
                            const CameraParameters camera_parameters,
                            const PtrStep<float3> vertex_map_current, const PtrStep<float3> normal_map_current,
                            const PtrStep<float3> vertex_map_previous, const PtrStep<float3> normal_map_previous,
                            const float distance_threshold, const float angle_threshold, const int cols,
                            const int rows,
                            PtrStep<double> output)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        Matf31da n, d, s;
        bool correspondence_found = false;

        if (x < cols && y < rows) {

            if (vertex_map_current.ptr(y)[x].x != 0.f) {
                Matf31da vertex_camera(vertex_map_current.ptr(y)[x].x,
                                        vertex_map_current.ptr(y)[x].y,
                                        vertex_map_current.ptr(y)[x].z);

                Matf31da vertex_current_global = rotation * vertex_camera + translation;


                Eigen::Vector2i v(__float2int_rd(
                                  vertex_camera.x() * camera_parameters.fovX / vertex_camera.z() +
                                  camera_parameters.cX + 0.5f),
                                  __float2int_rd(
                                  vertex_camera.y() * camera_parameters.fovY / vertex_camera.z() +
                                  camera_parameters.cY + 0.5f));

                if (v.x() >= 0 && v.x() < cols && v.y() >= 0 && v.y() < rows &&
                    vertex_camera.z() >= 0) {

                    if ( vertex_map_previous.ptr(v.y())[v.x()].x != 0.f ) {
                        Matf31da vertex_previous_global(
                                                     vertex_map_previous.ptr(v.y())[v.x()].x,
                                                     vertex_map_previous.ptr(v.y())[v.x()].y,
                                                     vertex_map_previous.ptr(v.y())[v.x()].z);

                        const float distance = (vertex_previous_global - vertex_current_global).norm();
                        if (distance <= distance_threshold) {
                            Matf31da normal_camera(normal_map_current.ptr(y)[x].x,
                                                           normal_map_current.ptr(y)[x].y,
                                                           normal_map_current.ptr(y)[x].z);

                            Matf31da normal_current_global = rotation * normal_camera;

                            Matf31da normal_previous_global(normal_map_previous.ptr(v.y())[v.x()].x,
                                                            normal_map_previous.ptr(v.y())[v.x()].y,
                                                            normal_map_previous.ptr(v.y())[v.x()].z);

                            const float angle_distance = normal_current_global.cross(normal_previous_global).norm();

                            if (angle_distance <= angle_threshold) {

                                n = normal_previous_global;
                                d = vertex_previous_global;
                                s = vertex_current_global;

                                correspondence_found = true;
                            }
                        }
                    }
                }
            }
        }

        float row[7];

        if (correspondence_found) {
            Matf31da temp = s.cross(n);
            row[0] = temp[0];
            row[1] = temp[1];
            row[2] = temp[2];
            row[3] = n[0];
            row[4] = n[1];
            row[5] = n[2];
            row[6] = n.dot(d - s);
        } else {
            for (int i = 0; i < 7; i++)
                row[i] = 0;
        }


        __shared__ double sdata[1024];
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;

        int num = 0;
        for (int i = 0; i < 6; ++i) { // Rows
            for (int j = i; j < 7; ++j) { // Columns and B
                __syncthreads();
                sdata[tid] = row[i] * row[j];
                __syncthreads();

                Reduce<1024>(sdata,tid);

                if (tid == 0)
                    output.ptr(num++)[gridDim.x * blockIdx.y + blockIdx.x] = sdata[0];
            }
        }
    }

    __global__
    void reduction_kernel(PtrStep<double> input, const int length, PtrStep<double> output)
    {
        double sum = 0.0;
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        for (int t = tid; t < length; t += 512)
            sum += *(input.ptr(bid) + t);

        __shared__ double sdata[512];

        sdata[tid] = sum;
        __syncthreads();

        Reduce<512>(sdata,tid);

        if (tid== 0)
            output.ptr(bid)[0] = sdata[0];
    };

    void icp_step(const Eigen::Matrix3f& rotation, const Matf31da& translation,
                       const CameraParameters& camera_parameters,
                        const cv::cuda::GpuMat& vertex_map_current, const cv::cuda::GpuMat& normal_map_current,
                        const cv::cuda::GpuMat& vertex_map_prev, const cv::cuda::GpuMat& normal_map_prev,
                        float distance_threshold, float angle_threshold,
                        Eigen::Matrix<double, 6, 6, Eigen::RowMajor>& A, Eigen::Matrix<double, 6, 1>& b)
    {
        const int cols = vertex_map_current.cols;
        const int rows = vertex_map_current.rows;

        dim3 threads(32, 32);
        dim3 blocks(std::ceil(cols / threads.x),
                  std::ceil(rows / threads.y));

        cv::cuda::GpuMat matrix_buffer { cv::cuda::createContinuous(27, 1, CV_64FC1) };
        cv::cuda::GpuMat frame_buffer { cv::cuda::createContinuous(27, blocks.x * blocks.y, CV_64FC1) };

        estimate_kernel<<<blocks, threads>>>(rotation, translation,
                camera_parameters,
                vertex_map_current, normal_map_current,
                vertex_map_prev, normal_map_prev,
                distance_threshold, angle_threshold,
                cols, rows,
                frame_buffer);

        //cudaDeviceSynchronize();

        reduction_kernel<<<27, 512>>>(frame_buffer, blocks.x * blocks.y, matrix_buffer);

        //cudaDeviceSynchronize();

        cv::Mat temp { 27, 1, CV_64FC1 };
        matrix_buffer.download(temp);

        int num = 0;
        for (int i = 0; i < 6; ++i) {
            for (int j = i; j < 7; ++j) {
                double value = temp.ptr<double>(num++)[0];
                if (j == 6)
                    b(i) = value;
                else {
                    A(i,j) = value;
                    if (i != j)
                        A(j,i) = value;
                }
            }
        }
    }
}
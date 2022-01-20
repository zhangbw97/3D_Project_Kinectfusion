#include "common.h"

namespace cuda {

    __device__ __forceinline__
    float tri_interpolate_2d(const float& x, const float& y, const float& z, const PtrStep<float2>& tsdf_volume,
                                                              const int3& volume_size)
    {
        const float x_grid = floorf(x) + 0.5f;
        const float y_grid = floorf(y) + 0.5f;
        const float z_grid = floorf(z) + 0.5f;

        const int x_grid_int = (x < x_grid) ?
            (static_cast<int>(x) - 1) : static_cast<int>(x);
        const int y_grid_int = (y < y_grid) ?
            (static_cast<int>(y) - 1): static_cast<int>(y);
        const int z_grid_int = (z < z_grid) ?
            (static_cast<int>(z) - 1): static_cast<int>(z);

        const float u = x - (static_cast<float>(x_grid_int) + 0.5f);
        const float v = y - (static_cast<float>(y_grid_int) + 0.5f);
        const float w = z - (static_cast<float>(z_grid_int) + 0.5f);

        return tsdf_volume.ptr((z_grid_int) * volume_size.y + y_grid_int)[x_grid_int].x  * (1 - u) * (1 - v) * (1 - w) +
               tsdf_volume.ptr((z_grid_int + 1) * volume_size.y + y_grid_int)[x_grid_int].x  * (1 - u) * (1 - v) * w +
               tsdf_volume.ptr((z_grid_int) * volume_size.y + y_grid_int)[x_grid_int + 1].x  * u * (1 - v) * (1 - w) +
               tsdf_volume.ptr((z_grid_int) * volume_size.y + y_grid_int + 1)[x_grid_int].x  * (1 - u) * v * (1 - w) +
               tsdf_volume.ptr((z_grid_int + 1) * volume_size.y + y_grid_int)[x_grid_int + 1].x  * u * (1 - v) * w +
               tsdf_volume.ptr((z_grid_int) * volume_size.y + y_grid_int + 1)[x_grid_int + 1].x  * u * v * (1 - w) +
               tsdf_volume.ptr((z_grid_int + 1) * volume_size.y + y_grid_int + 1)[x_grid_int].x  * (1 - u) * v * w +
               tsdf_volume.ptr((z_grid_int + 1) * volume_size.y + y_grid_int + 1)[x_grid_int + 1].x  * u * v * w;

    }


    __device__ __forceinline__
    float get_min_time_2d(const float3& volume_boundary, const Vec3fda& origin, const Vec3fda& direction)
    {
        float txmin = ((direction.x() > 0 ? 0.f : volume_boundary.x) - origin.x()) / direction.x();
        float tymin = ((direction.y() > 0 ? 0.f : volume_boundary.y) - origin.y()) / direction.y();
        float tzmin = ((direction.z() > 0 ? 0.f : volume_boundary.z) - origin.z()) / direction.z();

        return fmaxf(fmaxf(txmin, tymin), tzmin);
    }

    __device__ __forceinline__
    float get_max_time_2d(const float3& volume_boundary, const Vec3fda& origin, const Vec3fda& direction)
    {
        float txmax = ((direction.x() > 0 ? volume_boundary.x : 0.f) - origin.x()) / direction.x();
        float tymax = ((direction.y() > 0 ? volume_boundary.y : 0.f) - origin.y()) / direction.y();
        float tzmax = ((direction.z() > 0 ? volume_boundary.z : 0.f) - origin.z()) / direction.z();

        return fminf(fminf(txmax, tymax), tzmax);
    }

    __global__
    void raycast_tsdf_2d_kernel(const PtrStepSz<float2> tsdf_volume, const PtrStepSz<uchar3> color_volume,
                             PtrStepSz<float3> model_vertex, PtrStepSz<float3> model_normal,
                             PtrStepSz<uchar3> model_color,
                             const int3 volume_size, const float voxel_scale,
                             const CameraParameters cam_parameters,
                             const float step, const float max_step_grid,
                             const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> rotation,
                             const Vec3fda translation)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        // const int z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= model_vertex.cols || y >= model_vertex.rows)
            return;

        // if (z == 0)
        //    return;

        const float3 volume_boundary = make_float3(volume_size.x * voxel_scale,
                                                volume_size.y * voxel_scale,
                                                volume_size.z * voxel_scale);

        const Vec3fda pixel_pos(
                (x - cam_parameters.cX) / cam_parameters.fovX,
                (y - cam_parameters.cY) / cam_parameters.fovY,
                1.f);

        Vec3fda ray_direction = (rotation * pixel_pos);
        ray_direction.normalize();

        float ray_length_back = fmaxf(get_min_time_2d(volume_boundary, translation, ray_direction), 0.f);
        float ray_length_front = get_max_time_2d(volume_boundary, translation, ray_direction);

        // check whether have too long distance from camera center which leads the vertex outside the volume
        if (ray_length_back >= ray_length_front)
            return;

        // ray_length_back += voxel_scale;
        float ray_pos = ray_length_back + voxel_scale;
        Vec3fda grid = (translation + (ray_direction * ray_pos)) / voxel_scale;
        // Vec3fda grid_prev = (translation + (ray_direction * (ray_length_back + (z - 1) * step))) / voxel_scale;


        float tsdf = tsdf_volume.ptr(
                __float2int_rd(grid.z()) * volume_size.y + __float2int_rd(grid.y()))[__float2int_rd(grid.x())].x;



        // Zero Crossing
        const float max_search_length = ray_pos + max_step_grid;
        // const float max_search_length = ray_length_front + step;
       for (; ray_pos < max_search_length; ray_pos += step) {
           grid = ((translation + (ray_direction * (ray_pos + step))) / voxel_scale);

           if (grid.x() < 1 || grid.x() >= volume_size.x - 1 || grid.y() < 1 ||
               grid.y() >= volume_size.y - 1 ||
               grid.z() < 1 || grid.z() >= volume_size.z - 1) {
                   continue;
               }

           const float tsdf_prev = tsdf;
           tsdf = tsdf_volume.ptr(
                   __float2int_rd(grid(2)) * volume_size.y + __float2int_rd(grid(1)))[__float2int_rd(
                   grid(0))].x;

           if (tsdf_prev< 0.f && tsdf > 0.f) //Zero crossing from behind
               return;

           if (tsdf_prev > 0.f && tsdf < 0.f) {

                const float t_star =
                        ray_pos - step - step * tsdf_prev / (tsdf - tsdf_prev);

               const Vec3fda vertex = translation + ray_direction * t_star;
               const Vec3fda vertex_in_grid =  vertex / voxel_scale;
               const float vx = vertex_in_grid.x();
               const float vy = vertex_in_grid.y();
               const float vz = vertex_in_grid.z();


               // To confirm that the grid after shift is also in the volume

               if (vx < 2 || vx >= volume_size.x - 2 ||
                   vy < 2 || vy >= volume_size.x - 2 ||
                   vz < 2 || vz >= volume_size.z - 2)
                   return;

               // Vec3fda delta_x(1,0,0), delta_y(0,1,0), delta_z(0,0,1);

               const float Fx1 = tri_interpolate_2d(vx + 1, vy, vz, tsdf_volume, volume_size);
               const float Fx2 = tri_interpolate_2d(vx - 1, vy, vz, tsdf_volume, volume_size);

               const float Fy1 = tri_interpolate_2d(vx, vy + 1, vz, tsdf_volume, volume_size);
               const float Fy2 = tri_interpolate_2d(vx, vy - 1, vz, tsdf_volume, volume_size);

               const float Fz1 = tri_interpolate_2d(vx, vy, vz + 1, tsdf_volume, volume_size);
               const float Fz2 = tri_interpolate_2d(vx, vy, vz - 1, tsdf_volume, volume_size);

                Vec3fda normal(Fx1 - Fx2, Fy1 - Fy2, Fz1 - Fz2);

                if (normal.norm() == 0)
                    return;

                normal.normalize();

                model_vertex.ptr(y)[x] = make_float3(vertex.x(), vertex.y(), vertex.z());
                model_normal.ptr(y)[x] = make_float3(normal.x(), normal.y(), normal.z());

                Vec3ida vertex_in_grid_int = vertex_in_grid.cast<int>();
                model_color.ptr(y)[x] = color_volume.ptr(
                        vertex_in_grid_int.z() * volume_size.y +
                        vertex_in_grid_int.y())[vertex_in_grid_int.x()];

                return;
            }

         }
    }

    void surface_prediction_2d(const VolumeData& volume,
                            GpuMat& model_vertex, GpuMat& model_normal, GpuMat& model_color,
                            const CameraParameters& cam_parameters,
                            const float step,
                            const Eigen::Matrix4f& pose)
    {

        model_vertex.setTo(0);
        model_normal.setTo(0);
        model_color.setTo(0);

        const float max_step_grid = sqrt(
                              static_cast<float>(volume.volume_size.x * volume.volume_size.x) +
                              static_cast<float>(volume.volume_size.y * volume.volume_size.y) +
                              static_cast<float>(volume.volume_size.z * volume.volume_size.z));

        // std::cout << max_step_grid << std::endl;
        dim3 threads(32, 32);
        dim3 blocks((model_vertex.cols + threads.x - 1) / threads.x,
                    (model_vertex.rows + threads.y - 1) / threads.y);

        // We use parallel algorithm instead of iteration when calculating raycast

        raycast_tsdf_2d_kernel<<<blocks, threads>>>(volume.tsdf_volume, volume.color_volume,
                model_vertex, model_normal, model_color,
                volume.volume_size, volume.voxel_scale,
                cam_parameters,
                step, max_step_grid,
                pose.block(0, 0, 3, 3), pose.block(0, 3, 3, 1));

        cudaDeviceSynchronize();
    }
}
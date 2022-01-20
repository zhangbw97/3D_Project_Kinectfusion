#include "common.h"

namespace cuda {

    __global__
    void extract_pointcloud_kernel(const PtrStep<float2> tsdf_volume, const PtrStep<uchar3> color_volume,
                               const int3 volume_size, const float voxel_scale,
                               PtrStep<float3> vertices, PtrStep<float3> normals, PtrStep<uchar3> color,
                               int *num_point)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= volume_size.x - 1 || y >= volume_size.y - 1 || z >= volume_size.z - 1)
            return;

        const float2 value = tsdf_volume.ptr(z * volume_size.y + y)[x];

        const float tsdf = value.x;
        if (tsdf == 0 || tsdf <= -0.99f || tsdf >= 0.99f)
            return;

        float2 vx = tsdf_volume.ptr((z) * volume_size.y + y)[x + 1];
        float2 vy = tsdf_volume.ptr((z) * volume_size.y + y + 1)[x];
        float2 vz = tsdf_volume.ptr((z + 1) * volume_size.y + y)[x];

        if (vx.y <= 0 || vy.y <= 0 || vz.y <= 0)
            return;

        const float tsdf_x = vx.x;
        const float tsdf_y = vy.x;
        const float tsdf_z = vz.x;

        const bool is_surface_x = ((tsdf > 0) && (tsdf_x < 0)) || ((tsdf < 0) && (tsdf_x > 0));
        const bool is_surface_y = ((tsdf > 0) && (tsdf_y < 0)) || ((tsdf < 0) && (tsdf_y > 0));
        const bool is_surface_z = ((tsdf > 0) && (tsdf_z < 0)) || ((tsdf < 0) && (tsdf_z > 0));

        if (is_surface_x || is_surface_y || is_surface_z) {
            Eigen::Vector3f normal;
            normal.x() = (tsdf_x - tsdf);
            normal.y() = (tsdf_y - tsdf);
            normal.z() = (tsdf_z - tsdf);
            if (normal.norm() == 0)
                return;
            normal.normalize();

            int count = 0;
            if (is_surface_x) count++;
            if (is_surface_y) count++;
            if (is_surface_z) count++;
            int index = atomicAdd(num_point, count);

            Vec3fda position((static_cast<float>(x) + 0.5f) * voxel_scale,
                             (static_cast<float>(y) + 0.5f) * voxel_scale,
                             (static_cast<float>(z) + 0.5f) * voxel_scale);
            if (is_surface_x) {
                position.x() = position.x() - (tsdf / (tsdf_x - tsdf)) * voxel_scale;

                vertices.ptr(0)[index] = float3{position(0), position(1), position(2)};
                normals.ptr(0)[index] = float3{normal(0), normal(1), normal(2)};
                color.ptr(0)[index] = color_volume.ptr(z * volume_size.y + y)[x];
                index++;
            }
            if (is_surface_y) {
                position.y() -= (tsdf / (tsdf_y - tsdf)) * voxel_scale;

                vertices.ptr(0)[index] = float3{position(0), position(1), position(2)};;
                normals.ptr(0)[index] = float3{normal(0), normal(1), normal(2)};
                color.ptr(0)[index] = color_volume.ptr(z * volume_size.y + y)[x];
                index++;
            }
            if (is_surface_z) {
                position.z() -= (tsdf / (tsdf_z - tsdf)) * voxel_scale;

                vertices.ptr(0)[index] = float3{position(0), position(1), position(2)};;
                normals.ptr(0)[index] = float3{normal(0), normal(1), normal(2)};
                color.ptr(0)[index] = color_volume.ptr(z * volume_size.y + y)[x];
                index++;
            }
        }
    }

    PointCloud extract_pointcloud(const VolumeData& volume, const int buffer_size)
    {
        CloudData cloud_data { buffer_size };
        dim3 threads(8, 8, 16);
        dim3 blocks((volume.volume_size.x + threads.x - 1) / threads.x,
                    (volume.volume_size.y + threads.y - 1) / threads.y,
                    (volume.volume_size.z + threads.z - 1) / threads.z);

        extract_pointcloud_kernel<<<blocks, threads>>>(volume.tsdf_volume, volume.color_volume,
                volume.volume_size, volume.voxel_scale,
                cloud_data.vertex, cloud_data.normal, cloud_data.color,
                cloud_data.num_point);

        cudaDeviceSynchronize();

        PointCloud cloud;

        cloud_data.vertex.download(cloud.vertex);
        cloud_data.normal.download(cloud.normal);
        cloud_data.color.download(cloud.color);

        cudaMemcpy(&cloud.num_point, cloud_data.num_point, sizeof(int), cudaMemcpyDeviceToHost);

        return cloud;

    }
}
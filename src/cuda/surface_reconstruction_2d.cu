#include "common.h"
using namespace std;

namespace cuda {

    __global__
    void update_tsdf_2d_kernel(const PtrStepSz<float> depth_image,
                            const PtrStepSz<uchar3> color_image,
                            PtrStep<float2> tsdf_volume, PtrStep<uchar3> color_volume,
                            int3 volume_size, float voxel_scale,
                            CameraParameters cam_params,
                            const float truncation_distance,
                            const float iso_level,
                            Eigen::Matrix<float, 3, 3, Eigen::DontAlign> rotation, Vec3fda translation)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        // const int z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= volume_size.x || y >= volume_size.y)
            return;

        for (int z = 0; z <  volume_size.z; z++) {

            const Vec3fda global_p((static_cast<float>(x) + 0.5f) * voxel_scale,
                                   (static_cast<float>(y) + 0.5f) * voxel_scale,
                                   (static_cast<float>(z) + 0.5f) * voxel_scale);

            //convert position in global coordinate to camera coordinate
            const Vec3fda camera_p = rotation * global_p + translation;

            if (camera_p.z() <= 0)
                continue;


            const Vec2ida uv(
                    __float2int_rn(camera_p.x() / camera_p.z() * cam_params.fovX + cam_params.cX),
                    __float2int_rn(camera_p.y() / camera_p.z() * cam_params.fovY + cam_params.cY));


            if (uv.x() < 0 || uv.x() >= depth_image.cols || uv.y() < 0 || uv.y() >= depth_image.rows)
                continue;

            const float depth = depth_image.ptr(uv.y())[uv.x()];

            if (depth <= 0.f)
                continue;

            const Vec3fda qlambda(
                    (uv.x() - cam_params.cX) / cam_params.fovX,
                    (uv.y() - cam_params.cY) / cam_params.fovY,
                    1.f);
            const float lambda = qlambda.norm();

            // camera_p = p - t_{g,k} since t_{g,k} is 0 in camera coordinate
            const float sdf = depth - camera_p.norm() / lambda;

            if (sdf >= (-1.f) * truncation_distance) {

                const float new_tsdf = fminf(1.f, sdf / truncation_distance);
                float2 current_tsdf_tuple = tsdf_volume.ptr(z * volume_size.y + y)[x];

                const float current_tsdf = current_tsdf_tuple.x;
                const float current_weight = fmaxf(current_tsdf_tuple.y, 0.f);

                // TODO: add weight
                const float add_weight = 1.f;
                const float new_value = (current_weight * current_tsdf + add_weight * new_tsdf) /
                                           (current_weight + add_weight);
                const float new_weight = fminf(current_weight + add_weight, MAX_WEIGHT);

                tsdf_volume.ptr(z * volume_size.y + y)[x] = make_float2(new_value, new_weight);

                //update color in space
                if (sdf <= iso_level && sdf >= (-1.f) * iso_level ) {
                    uchar3& space_color = color_volume.ptr(z * volume_size.y + y)[x];
                    const uchar3 image_color = color_image.ptr(uv.y())[uv.x()];

                    space_color.x = static_cast<uchar>(
                            (current_weight * space_color.x + add_weight * image_color.x) / (current_weight + add_weight));
                    space_color.y = static_cast<uchar>(
                            (current_weight * space_color.y + add_weight * image_color.y) / (current_weight + add_weight));
                    space_color.z = static_cast<uchar>(
                            (current_weight * space_color.z + add_weight * image_color.z) / (current_weight + add_weight));
                }
            } /*else {
                tsdf_volume.ptr(z * volume_size.y + y)[x] = make_float2(0.f, 0.f);
                color_volume.ptr(z * volume_size.y + y)[x] = make_uchar3(0, 0, 0);
            }*/
        }
    }


    void surface_reconstruction_2d(const cv::cuda::GpuMat& depth_image,
                                const cv::cuda::GpuMat& color_image,
                                VolumeData& volume,
                                const CameraParameters& cam_params, const float truncation_distance,
                                const float iso_level,
                                const Eigen::Matrix4f& extrinsic)
    {
        const dim3 threads(32, 32);
        const dim3 blocks((volume.volume_size.x + threads.x - 1) / threads.x,
                          (volume.volume_size.y + threads.y - 1) / threads.y);

        update_tsdf_2d_kernel<<<blocks, threads>>>(depth_image, color_image,
                volume.tsdf_volume, volume.color_volume,
                volume.volume_size, volume.voxel_scale,
                cam_params, truncation_distance, iso_level,
                extrinsic.block(0, 0, 3, 3), extrinsic.block(0, 3, 3, 1));

        cudaDeviceSynchronize();

        // cout << "Computed TSDF!\n";
    }
}
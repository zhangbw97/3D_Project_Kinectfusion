#include "data_types.h"
#include "Eigen.h"

namespace cuda {
    void icp_step(const Eigen::Matrix3f& rotation_current, const Matf31da& translation_current,
                       const CameraParameters& camera_parameters,
                       const cv::cuda::GpuMat& vertex_map_current, const cv::cuda::GpuMat& normal_map_current,
                       const cv::cuda::GpuMat& vertex_map_prev, const cv::cuda::GpuMat& normal_map_prev,
                       float distance_threshold, float angle_threshold,
                       Eigen::Matrix<double, 6, 6, Eigen::RowMajor>& A, Eigen::Matrix<double, 6, 1>& b);
}

bool pose_estimation(Eigen::Matrix4f& pose,
                        const std::vector<cv::cuda::GpuMat>& frame_vertex_pyramid,
                        const std::vector<cv::cuda::GpuMat>& frame_normal_pyramid,
                        const std::vector<cv::cuda::GpuMat>& model_vertex_pyramid,
                        const std::vector<cv::cuda::GpuMat>& model_normal_pyramid,
                        const std::vector<CameraParameters>& camera_parameters,
                        const int pyramid_height,
                        const float distance_threshold, const float angle_threshold,
                        const float det_threshold,
                        const std::vector<int>& iterations)
{
    // Get initial rotation and translation
    Eigen::Matrix3f rotation = pose.block(0, 0, 3, 3);
    Eigen::Vector3f translation = pose.block(0, 3, 3, 1);

    // ICP loop, from coarse to fine
    for (int level = pyramid_height - 1; level >= 0; --level) {
        for (int iter = 0; iter < iterations[level]; ++iter) {
            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A {};
            Eigen::Matrix<double, 6, 1> b {};

            // Estimate one step on the GPU
            cuda::icp_step(rotation, translation,
                                camera_parameters[level],
                                frame_vertex_pyramid[level], frame_normal_pyramid[level],
                                model_vertex_pyramid[level], model_normal_pyramid[level],
                                distance_threshold, sinf(angle_threshold * 3.1415926f / 180.f),
                                A, b);

            // Solve equation by derivative
            double det = A.determinant();

            // std::cout<<det<<std::endl;
            if (fabs(det) < det_threshold || std::isnan(det))
                return false;
            Eigen::Matrix<float, 6, 1> result { A.fullPivLu().solve(b).cast<float>() };
            // JacobiSVD<MatrixXf> svd(A, ComputeFullU | ComputeFullV);
            // Eigen::Matrix<float, 6, 1> result = svd.solve(b) ;
            float alpha = result(0);
            float beta = result(1);
            float gamma = result(2);

            // Update rotation
            auto camera_rotation_incremental(
                    Eigen::AngleAxisf(gamma, Eigen::Vector3f::UnitZ()) *
                    Eigen::AngleAxisf(beta, Eigen::Vector3f::UnitY()) *
                    Eigen::AngleAxisf(alpha, Eigen::Vector3f::UnitX()));
            auto camera_translation_incremental = result.tail<3>();

            // Update translation
            translation =
                    camera_rotation_incremental * translation + camera_translation_incremental;
            rotation = camera_rotation_incremental * rotation;
        }
    }

    // Find the new pose
    pose.block(0, 0, 3, 3) = rotation;
    pose.block(0, 3, 3, 1) = translation;

    return true;
}

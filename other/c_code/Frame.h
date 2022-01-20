#pragma once
#include "../../include/SimpleMesh.h"
#include "../../include/Eigen.h"
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include "../../include/data_types.h"


void bilateral(float* depthMapSrc, float* depthMapDst,int width, int height);
void pyrdown(float* depthMapSrc, float* depthMapDst,int width, int height);



class KinectFusion {
public:
	KinectFusion() {}

	KinectFusion(const SimpleMesh& mesh) {
		const auto& vertices = mesh.getVertices();
		const auto& triangles = mesh.getTriangles();
		const unsigned nVertices = vertices.size();
		const unsigned nTriangles = triangles.size();

		// Copy vertices.
		m_points.reserve(nVertices);
		for (const auto& vertex : vertices) {
			m_points.push_back(Vector3f{ vertex.position.x(), vertex.position.y(), vertex.position.z() });
		}

		// Compute normals (as an average of triangle normals).
		m_normals = std::vector<Vector3f>(nVertices, Vector3f::Zero());
		for (size_t i = 0; i < nTriangles; i++) {
			const auto& triangle = triangles[i];
			Vector3f faceNormal = (m_points[triangle.idx1] - m_points[triangle.idx0]).cross(m_points[triangle.idx2] - m_points[triangle.idx0]);

			m_normals[triangle.idx0] += faceNormal;
			m_normals[triangle.idx1] += faceNormal;
			m_normals[triangle.idx2] += faceNormal;
		}
		for (size_t i = 0; i < nVertices; i++) {
			m_normals[i].normalize();
		}
	}

	KinectFusion(float* depthMap, const Matrix3f& depthIntrinsics, const Matrix4f& depthExtrinsics, unsigned width, unsigned height,  const cv::Mat_<cv::Vec3b>& color_map, unsigned downsampleFactor = 1, float maxDistance = 0.1f) {
	
		// 彩色图只有第一层
		cv::cuda::GpuMat color_map_ = cv::cuda::createContinuous(height, width, CV_8UC3);
		color_map_.upload(color_map);
		color_pyramid.push_back(color_map_);

        CameraParameters camera{depthIntrinsics, width, height};
		const float maxDistanceHalved = maxDistance / 2.f;

		// 生成三层点云和法向量金字塔
		for(int level = 0; level<3; level++)
		{
			// 第二层开始才需要改变相机参数
			/*if( level  > 0)
			{
                fovX = fovX * 0.5f;
                fovY = fovY * 0.5f;
                cX = (cX + 0.5f) * 0.5 - 0.5f;
                cY = (cY + 0.5f) * 0.5 - 0.5f;
                width = (width + 1 )/ 2;
                height = (height +1 ) / 2;
			} */

			CameraParameters camera_level{camera, level};
			camera_parameters.push_back(camera_level);

			// 金字塔下一层
			if (level != 0) {

			}
			float depthMap1[int((width +1 ) / 2)  * int((height+1)/2)];
			pyrdown(depthMap, depthMap1, width, height);

			// 金字塔当前层双边过滤
			float depthMap_bi[width * height];
			bilateral(depthMap, depthMap_bi, width, height);

			// 把当前层数组全部改成 CV矩阵， 再转成GPU矩阵
			cv::Mat depth_map(height, width, CV_32FC1, depthMap_bi);
			cv::cuda::GpuMat gpu_depth = cv::cuda::createContinuous(height, width, CV_32FC1);
			gpu_depth.upload(depth_map);
			depth_pyramid.push_back(gpu_depth);


			// 创建点云数组，再转成CV矩阵，再转GPU矩阵计算，计算完转std::vector<Vector3f>主要是为了能用老师的接口
			Vector3f* pointsTmp_ =  new Vector3f[width * height];
			cv::Mat vertex_map(height, width, CV_32FC3, pointsTmp_);
			cv::cuda::GpuMat gpu_vertex = cv::cuda::createContinuous(height, width, CV_32FC3);
			gpu_vertex.upload(vertex_map);
			cuda::ComputeVertex(gpu_depth, gpu_vertex, fovX, fovY, cX, cY , width, height);
			vertex_pyramid.push_back(gpu_vertex);

			gpu_vertex.download(vertex_map);
			/*
			std::vector<Vector3f> pointsTmp;
			for(int i = 0; i < (width * height); i++) 
			{	
				pointsTmp.push_back(pointsTmp_[i]);
			}
			delete[] pointsTmp_;
			 */

			// 同上， 计算法向量
			// Vector3f* normalsTmp_ = new Vector3f[width * height];
			cv::Mat normals_map(height, width, CV_32FC3, normalsTmp_);
			cv::cuda::GpuMat gpu_normals = cv::cuda::createContinuous(height, width, CV_32FC3);
			gpu_normals.upload(normals_map);
		
			cuda::ComputeNormal(gpu_normals, gpu_vertex, width, height, maxDistanceHalved);
			normal_pyramid.push_back(gpu_normals);
			gpu_normals.download(normals_map);
			/*
			std::vector<Vector3f> normalsTmp;
			for(int i = 0; i < (width * height); i++) 
			{	
				normalsTmp.push_back(normalsTmp_[i]);
			}
			delete[] normalsTmp_;
			*/
			

			m_points_level.push_back(pointsTmp);
			m_normals_level.push_back(normalsTmp);

			depthMap = depthMap1; // 以金字塔下一层的深度图作为计算的当前图
		}
		
		// 因为exercise05的代码框架只取第一层的点云，为了保证一贯性，仍然把第一层图单独拿出来，赋值给成员变量
		m_points = m_points_level[0]; 
		m_normals = m_normals_level[0];
	}

	std::vector<Vector3f>& getPoints() {
		return m_points;
	}

	const std::vector<Vector3f>& getPoints() const {
		return m_points;
	}

	std::vector<Vector3f>& getNormals() {
		return m_normals;
	}
	

	const std::vector<Vector3f>& getNormals() const {
		return m_normals;
	}

//private: 我把这里的private注释掉，就可以直接读取三层金字塔数据了
	// 因为上面的各种getClosestPoint等各种函数只会处理第一层，大家根据自己的需要修改一下，或者直接拿成员变量
	std::vector<Vector3f> m_points;  // 第一层的数据 (for visualization)
	std::vector<Vector3f> m_normals; // 第一层的数据 (for visualization)

	std::vector<std::vector<Vector3f>> m_points_level;//点云金字塔 (for visualization)
	std::vector<std::vector<Vector3f>> m_normals_level;//法向量金字塔 (for visualization)

	std::vector<CameraParameters> camera_parameters;
	std::vector<cv::cuda::GpuMat> depth_pyramid;                              // 原始深度图的金字塔
    std::vector<cv::cuda::GpuMat> filtered_depth_pyramid;                     // 经过过滤后的深度图的金字塔
	std::vector<cv::cuda::GpuMat> color_pyramid;                              // 彩色图像的金字塔

	std::vector<cv::cuda::GpuMat> vertex_pyramid;                             // 3D点金字塔
	std::vector<cv::cuda::GpuMat> normal_pyramid;                             // 法向量金字塔
};


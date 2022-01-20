#pragma once
#include "SimpleMesh.h"
#include "Eigen.h"
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
void ComputeVertex(float* depthMap, std::vector<Eigen::Vector3f>& pointsTmp, float fovX, float fovY, float cX, float cY , int width, int height);
void bilateral(float* depthMapSrc, float* depthMapDst,int width, int height);
void pyrdown(float* depthMapSrc, float* depthMapDst,int width, int height);
void ComputeNormal(std::vector<Eigen::Vector3f>& normalsTmp, std::vector<Eigen::Vector3f>& pointsTmp, int width, int height, float maxDistanceHalved);


class PointCloud {
public:
	PointCloud() {}

	PointCloud(const SimpleMesh& mesh) {
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

	PointCloud(float* depthMap, const Matrix3f& depthIntrinsics, const Matrix4f& depthExtrinsics, unsigned width, unsigned height, unsigned downsampleFactor = 1, float maxDistance = 0.1f) {
		// Get depth intrinsics.
		float fovX = depthIntrinsics(0, 0);
		float fovY = depthIntrinsics(1, 1);
		float cX = depthIntrinsics(0, 2);
		float cY = depthIntrinsics(1, 2);
		const float maxDistanceHalved = maxDistance / 2.f;

		// 生成三层点云和法向量金字塔
		for(int level = 0; level<3; level++)
		{
			// 第二层开始才需要改变相机参数
			if( level  > 0)
			{
			fovX = fovX * 0.5f;
			fovY = fovY * 0.5f;
			cX = (cX + 0.5f) * 0.5 - 0.5f;
			cY = (cY + 0.5f) * 0.5 - 0.5f;
			width = (width + 1 )/ 2;
			height = (height +1 ) / 2;
			}
			// 金字塔下一层
			float depthMap1[int((width +1 ) / 2)  * int((height+1)/2)];
			pyrdown(depthMap, depthMap1, width, height);

			// 金字塔当前层双边过滤
			float depthMap_bi[width * height];
			bilateral(depthMap, depthMap_bi, width, height);

			// 金字塔当前层计算3D点云
			std::vector<Vector3f> pointsTmp(width * height);
			ComputeVertex(depthMap_bi, pointsTmp, fovX, fovY, cX, cY , width, height);
			

			// 金字塔当前层计算法向量
			std::vector<Vector3f> normalsTmp(width * height);
			// 下面这行代码用cuda写的计算向量和上面GpuMat(pyr和bilateral)冲突了
			// 舍弃，直接用普通CPU计算, 后期我在问问老师为啥
			// ComputeNormal(normalsTmp, pointsTmp, width, height, maxDistanceHalved);
		
			#pragma omp parallel for
			for (int v = 1; v < height - 1; ++v) {
				for (int u = 1; u < width - 1; ++u) {
					unsigned int idx = v*width + u; // linearized index

					const float du = 0.5f * (depthMap[idx + 1] - depthMap[idx - 1]);
					const float dv = 0.5f * (depthMap[idx + width] - depthMap[idx - width]);
					if (!std::isfinite(du) || !std::isfinite(dv) || abs(du) > maxDistanceHalved || abs(dv) > maxDistanceHalved) {
						normalsTmp[idx] = Vector3f(blah, blah, blah);
						continue;
					}

					// TODO: Compute the normals using central differences. 
					// normalsTmp[idx] = Vector3f(du, dv, -1.0); // Needs to be replaced.
					normalsTmp[idx] = (pointsTmp[idx-1]-pointsTmp[idx+1]).cross(pointsTmp[idx+width]-pointsTmp[idx-width]);
					normalsTmp[idx].normalize();
				}
			}

			
			// We set invalid normals for border regions.
			// 把深度图边界点的设为无效
			for (int u = 0; u < width; ++u) {
				normalsTmp[u] = Vector3f(blah, blah, blah);
				normalsTmp[u + (height - 1) * width] = Vector3f(blah, blah, blah);
			}
			for (int v = 0; v < height; ++v) {
				normalsTmp[v * width] = Vector3f(blah, blah, blah);
				normalsTmp[(width - 1) + v * width] = Vector3f(blah, blah, blah);
			}

			// We filter out measurements where either point or normal is invalid.
			// 过滤无效的点，然后存入m_points_level/点云金字塔的当前层,
			// m_points_temp是当前层的点云
			// m_points_level里面存放所有层
			const unsigned nPoints = pointsTmp.size();
			std::vector<Vector3f> m_points_temp;
			std::vector<Vector3f> m_normals_temp;
		
			m_points_temp.reserve(std::floor(float(nPoints) / downsampleFactor));
			m_normals_temp.reserve(std::floor(float(nPoints) / downsampleFactor));

			for (int i = 0; i < nPoints; i = i + downsampleFactor) {
				const auto& point = pointsTmp[i];
				const auto& normal = normalsTmp[i];

				if (point.allFinite() && normal.allFinite()) {
					m_points_temp.push_back(point);
					m_normals_temp.push_back(normal);
				}
			}
	
			m_points_level.push_back(m_points_temp);
			m_normals_level.push_back(m_normals_temp);
			depthMap = depthMap1; // 以金字塔下一层的深度图作为计算的当前图
		}
		
		// 因为exercise05的代码框架只取第一层的点云，为了保证一贯性，仍然把第一层图单独拿出来，赋值给成员变量
		m_points = m_points_level[0]; 
		m_normals = m_normals_level[0];
	}

	bool readFromFile(const std::string& filename) {
		std::ifstream is(filename, std::ios::in | std::ios::binary);
		if (!is.is_open()) {
			std::cout << "ERROR: unable to read input file!" << std::endl;
			return false;
		}

		char nBytes;
		is.read(&nBytes, sizeof(char));

		unsigned int n;
		is.read((char*)&n, sizeof(unsigned int));

		if (nBytes == sizeof(float)) {
			float* ps = new float[3 * n];

			is.read((char*)ps, 3 * sizeof(float) * n);

			for (unsigned int i = 0; i < n; i++) {
				Eigen::Vector3f p(ps[3 * i + 0], ps[3 * i + 1], ps[3 * i + 2]);
				m_points.push_back(p);
			}

			is.read((char*)ps, 3 * sizeof(float) * n);
			for (unsigned int i = 0; i < n; i++) {
				Eigen::Vector3f p(ps[3 * i + 0], ps[3 * i + 1], ps[3 * i + 2]);
				m_normals.push_back(p);
			}

			delete ps;
		}
		else {
			double* ps = new double[3 * n];

			is.read((char*)ps, 3 * sizeof(double) * n);

			for (unsigned int i = 0; i < n; i++) {
				Eigen::Vector3f p((float)ps[3 * i + 0], (float)ps[3 * i + 1], (float)ps[3 * i + 2]);
				m_points.push_back(p);
			}

			is.read((char*)ps, 3 * sizeof(double) * n);

			for (unsigned int i = 0; i < n; i++) {
				Eigen::Vector3f p((float)ps[3 * i + 0], (float)ps[3 * i + 1], (float)ps[3 * i + 2]);
				m_normals.push_back(p);
			}

			delete ps;
		}


		//std::ofstream file("pointcloud.off");
		//file << "OFF" << std::endl;
		//file << m_points.size() << " 0 0" << std::endl;
		//for(unsigned int i=0; i<m_points.size(); ++i)
		//	file << m_points[i].x() << " " << m_points[i].y() << " " << m_points[i].z() << std::endl;
		//file.close();

		return true;
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

	unsigned int getClosestPoint(Vector3f& p) {
		unsigned int idx = 0;

		float min_dist = std::numeric_limits<float>::max();
		for (unsigned int i = 0; i < m_points.size(); ++i) {
			float dist = (p - m_points[i]).norm();
			if (min_dist > dist) {
				idx = i;
				min_dist = dist;
			}
		}

		return idx;
	}

//private:  我把这里的private注释掉，就可以直接读取三层金字塔数据了
	// 因为上面的各种getClosestPoint等各种函数只会处理第一层，大家根据自己的需要修改一下，或者直接拿成员变量
	std::vector<Vector3f> m_points; // 第一层的数据
	std::vector<Vector3f> m_normals; // 第一层的数据
	std::vector<std::vector<Vector3f>> m_points_level;//点云金字塔
	std::vector<std::vector<Vector3f>> m_normals_level;//法向量金字塔

};
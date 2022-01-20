#include <opencv2/core/cuda.hpp> //GPU矩阵的头文件
// #include <opencv2/cudaimgproc.hpp>
// #include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp> //创建矩阵的头文件
#include <opencv2/cudawarping.hpp> // 金字塔
#include "KinectFusion.h"

void pyrdown(cv::cuda::GpuMat& depthMapSrc, cv::cuda::GpuMat& depthMapDst)
{

    // // 开始计算金字塔下一层数据;
    cv::cuda::Stream stream;
    cv::cuda::pyrDown(depthMapSrc, depthMapDst, stream);
    stream.waitForCompletion();

}
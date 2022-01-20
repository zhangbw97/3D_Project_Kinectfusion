#include <opencv2/core/cuda.hpp> //GPU矩阵的头文件
// #include <opencv2/cudaimgproc.hpp>
// #include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp> //创建矩阵的头文件
#include <opencv2/cudaimgproc.hpp> //双边过滤
#include "KinectFusion.h"

void bilateral(cv::cuda::GpuMat& depthMapSrc, cv::cuda::GpuMat& depthMapDst, const int kernel_size,
               const float color_sigma, const float spatial_sigma )
{


    // 开始计算金字塔下一层数据
    cv::cuda::Stream stream;
    cv::cuda::bilateralFilter(depthMapSrc,            // source
                              depthMapDst,            // destination
                              kernel_size,                          // 这个是双边滤波器滤波的核大小, 不是GPU核函数的那个核
                              color_sigma,
                              spatial_sigma,
                              cv::BORDER_DEFAULT,                   // 默认边缘的补充生成方案 gfedcb|abcdefgh|gfedcba
                              stream);                              // 加入到指定的流中
    stream.waitForCompletion();

}
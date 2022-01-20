# 3D motion project
This is the repo for the collaboration on the 3D motion project.

## 1.先下载最新版（我的是3.3.8版）的EIGEN3
然后编译安装到计算机上（默认位置就可以）
* cd eigen-3.3.8
* mkdir build
* cd build
* cmake ..
* sudo make install 

## 2.按照exercise05/04什么的把ceres Flann 什么东西安装好
## 3.安装最新版的Opencv
配置时请注意勾选cuda。
参考以下博客，配置环境那一步可以跳过，我的是3.4.0版本的。
[https://blog.csdn.net/zhaoxr233/article/details/90036824?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~first_rank_v2~rank_v25-1-90036824.nonecase]



## 4.把上课的数据集按下面格式存放

./3D-proj/data/rgbd_dataset_freiburg1_xyz

## 5. 编译该项目
* cd 3D-proj
* mkdir build
* cd build
* cmake ../src
* make  #编译的时候CUDA会有很多WARNING，只要不是error就先不用管
* ./3D-proj #执行后效果和exercise05一样，生成点云

## 志雄
我主要更改了PointCloud.h的文件，新增了双边过滤和生成金字塔功能（利用OpenCV的GPU矩阵），生成3D点云坐标（自己写的cuda，ComputeVertex.cu），本来生成法向量也要用cuda的，但是一直和OpenCV冲突，debug两天了先利用CPU算一下吧，这样你们才能继续。最后成员变量新增了两个金字塔std::vector。但是由于整个框架是只取第一层计算，所以后期大家改框架的时候根据需要取用。

## 博文
主要更改了pose_estimation.cu,pose_estimation.cpp,以及main.cpp里面更新了pose estimation的流程。其中需要第四部分的模型得出的vertex map和 normals map，暂时用上一帧的point cloud得出的vertex map和 normals map代替了。
Pointcloud.h里面一维的点阵和法向量阵另外展开了一个二维的m_points_2D和m_normals_2D，以及金字塔，需要的话可以取用。
现在还有个gpu到cpu内存读取故障的问题，不过整体编译是通过的，我这部分的框架也不会有大改动，为了避免拖大家的进度所以先放上来供大家参考，在这基础上进行后面的编程。

## 乐天
梳理了函数的整体框架，将内容整理为KinectFusion方法. 完成surface_reconstruction函数. 修改了ComputeNormal.cu加了调整noraml方向的语句.
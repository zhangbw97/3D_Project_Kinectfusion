cmake_minimum_required(VERSION 3.6)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

project(3D-proj LANGUAGES CXX CUDA)

# Options
set(LIBRARY_DIR libs CACHE PATH "Path to lib folder")
set(Eigen3_INCLUDE_DIR ${LIBRARY_DIR}/Eigen/ CACHE PATH "Path to Eigen source folder")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${Ceres_DIR})


# Use pre-compiled libraries on Windows
if(WIN32)
    set(FreeImage_INCLUDE_DIR ${LIBRARY_DIR}/FreeImage/x64/ CACHE PATH "Path to FreeImage header file")
    set(FreeImage_LIBRARY_DIR ${LIBRARY_DIR}/FreeImage/x64/ CACHE PATH "Path to FreeImage .lib/.dll folder")
endif(WIN32)

# Set C++ flags
set(CMAKE_CXX_STANDARD 14)


find_package(CUDA 8.0 REQUIRED)
IF (CUDA_FOUND)
    include_directories("${CUDA_INCLUDE_DIRS}")
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++11;-Wcpp -gencode arch=compute_52,code=sm_52)
ENDIF ()

find_package(OpenCV 3.0 REQUIRED)
if (OpenCV_INCLUDE_DIRS)
    include_directories("${OpenCV_INCLUDE_DIRS}")
    message("-- OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
endif (OpenCV_INCLUDE_DIRS)

find_package(Eigen3 REQUIRED)
if (EIGEN3_INCLUDE_DIR)
    include_directories("${EIGEN3_INCLUDE_DIR}")
    message("-- EIGEN3_INCLUDE_DIR: " ${EIGEN3_INCLUDE_DIR})
endif (EIGEN3_INCLUDE_DIR)



add_definitions("-D_DISABLE_EXTENDED_ALIGNED_STORAGE")

if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj")
endif()





# Set files to be compiled
set(HEADER_FILES
        include/Eigen.h
        include/SimpleMesh.h
        include/VirtualSensor.h
        include/FreeImageHelper.h
        include/data_types.h
        include/Configuration.h
        include/KinectFusion.h
        include/common.h
        )

set(SOURCE_FILES
        src/FreeImageHelper.cpp
        src/bilateral.cpp
        src/pyrdown.cpp
        src/pose_estimation.cpp
        src/cuda/ComputeVertex.cu
        src/cuda/ComputeNormal.cu
        src/cuda/pose_estimation.cu
        src/cuda/surface_reconstruction_2d.cu
        src/cuda/surface_reconstruction_3d.cu
        src/cuda/MarchingCube_old.cu
        src/cuda/MarchingCube.cu
        src/cuda/extract_pointcloud.cu
        src/cuda/surface_prediction_2d.cu
        src/cuda/surface_prediction_3d.cu
        src/KinectFusion.cpp
        src/surface_measurement.cpp
        src/visual.cpp)



link_directories(${FreeImage_LIBRARY_DIR} ${OpenCV_LIBS})

add_executable(3D-proj main.cpp ${SOURCE_FILES})
target_link_libraries(3D-proj freeimage  ${OpenCV_LIBS})
target_include_directories(3D-proj PUBLIC include ${Eigen3_INCLUDE_DIR}  ${FreeImage_INCLUDE_DIR})


# add_executable(3D-proj main.cpp ${SOURCE_FILES} ${HEADER_FILES} )
# target_link_libraries(3D-proj ceres freeimage  ${OpenCV_LIBS})
# target_include_directories(3D-proj PUBLIC  ${Eigen3_INCLUDE_DIR}  ${Ceres_INCLUDE_DIR} ${glog_INCLUDE_DIR} ${FreeImage_INCLUDE_DIR} ${Flann_INCLUDE_DIR})

# # Visual Studio properties
# set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT exercise_05)
# set_property(TARGET exercise_05 PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)


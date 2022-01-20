//
// Created by slt on 1/25/2021.
//
#include "data_types.h"
#include "KinectFusion.h"
using namespace std;

void KinectFusion::visual_volume(bool output_file) {
    cv::Mat print_vol_tsdf(configuration.volume_size.y * configuration.volume_size.z, configuration.volume_size.x,
                           CV_32FC2);
    cv::Mat print_vol_color(configuration.volume_size.y * configuration.volume_size.z, configuration.volume_size.x,
                            CV_8UC3);

    volume.tsdf_volume.download(print_vol_tsdf);
    volume.color_volume.download(print_vol_color);

    //cout << "Max TSDF:" << cv::max(print_vol_tsdf)[0] << endl;
    //cout << "Min TSDF:" << cv::min(print_vol_tsdf)[0] << endl;
    cout << "Mean TSDF:" << cv::mean(print_vol_tsdf)[0] << endl;
    cout << "Sum TSDF:" << cv::sum(print_vol_tsdf)[0] << endl;
    cout << "Mean abs TSDF:" << cv::mean(print_vol_tsdf)[1] << endl;
    cout << "Sum abs TSDF:" << cv::sum(print_vol_tsdf)[1] << endl;
    //cout << "Max color:" << cv::max(print_vol_color)[0] << endl;
    //cout << "Min color:" << cv::min(print_vol_color)[0] << endl;
    cout << "Mean volume color:" << cv::mean(print_vol_color)[0] << endl;
    cout << "Sum volume color:" << cv::sum(print_vol_color)[0] << endl;

    if(output_file) {
        string filename{"volume_info_" + to_string(frame) + ".txt"};
        ofstream file_out{filename};

        for (int i = 0; i < configuration.volume_size.x; i++) {
            for (int j = 0; j < configuration.volume_size.y; j++) {
                for (int k = 0; k < configuration.volume_size.z; k++) {
                    float2 tsdf = print_vol_tsdf.at<float2>(k * configuration.volume_size.y + j,i);
                    uchar3 color = print_vol_color.at<uchar3>(k * configuration.volume_size.y + j,i);

                    int x = log10(i), y = log10(j), z = log10(k);
                    file_out << "(" << setw(3) << i;
                    file_out << "," << setw(3) << j;
                    file_out << "," << setw(3) << k;
                    file_out << "):";

                    file_out << "TSDF-" << fixed << setprecision(5) << tsdf.x << ",";
                    file_out << "color-(";

                    file_out << setw(3) << color.x << ",";
                    file_out << setw(3) << color.y << ",";
                    file_out << setw(3) << color.z << ")" << endl;
                }
            }
        }
    }
}

void KinectFusion::visual_model(bool output_file) {

    cv::Mat print_model_color(camera_parameters[0].height,camera_parameters[0].width,CV_8UC3);
    cv::Mat print_model_vertex(camera_parameters[0].height,camera_parameters[0].width,CV_32FC3);
    cv::Mat print_model_normal(camera_parameters[0].height,camera_parameters[0].width,CV_32FC3);
    model.color_pyramid[0].download(print_model_color);
    model.vertex_pyramid[0].download(print_model_vertex);
    model.normal_pyramid[0].download(print_model_normal);

    cout << "Mean Model Vertex x:" << cv::mean(print_model_vertex)[0] << endl;
    cout << "Sum Model Vertex x:" << cv::sum(print_model_vertex)[0] << endl;
    cout << "Mean Model Vertex y:" << cv::mean(print_model_vertex)[1] << endl;
    cout << "Sum Model Vertex y:" << cv::sum(print_model_vertex)[1] << endl;
    cout << "Mean Model Vertex z:" << cv::mean(print_model_vertex)[2] << endl;
    cout << "Sum Model Vertex z:" << cv::sum(print_model_vertex)[2] << endl;

    cout << "Mean Model Normal x:" << cv::mean(print_model_normal)[0] << endl;
    cout << "Sum Model Normal x:" << cv::sum(print_model_normal)[0] << endl;
    cout << "Mean Model Normal y:" << cv::mean(print_model_normal)[1] << endl;
    cout << "Sum Model Normal y:" << cv::sum(print_model_normal)[1] << endl;
    cout << "Mean Model Normal z:" << cv::mean(print_model_normal)[2] << endl;
    cout << "Sum Model Normal z:" << cv::sum(print_model_normal)[2] << endl;

    if(output_file) {

    }
}

void KinectFusion::visual_input(InputData& input) {

    int level0 = 0;
    cout << camera_parameters[level0].fovX << ",";
    cout << camera_parameters[level0].fovY << ",";
    cout << camera_parameters[level0].cX << ",";
    cout << camera_parameters[level0].cY << ",";
    cout << camera_parameters[level0].width << ",";
    cout << camera_parameters[level0].height << endl;

    cv::Mat print_input_vertex(camera_parameters[0].height,camera_parameters[0].width,CV_32FC3);
    cv::Mat print_input_normal(camera_parameters[0].height,camera_parameters[0].width,CV_32FC3);
    cv::Mat print_input_color(camera_parameters[0].height,camera_parameters[0].width,CV_8UC3);
    cv::Mat print_input_depth(camera_parameters[0].height,camera_parameters[0].width,CV_32FC1);
    cv::Mat print_input_filtered_depth(camera_parameters[0].height,camera_parameters[0].width,CV_32FC1);

    input.vertex_pyramid[0].download(print_input_vertex);
    input.normal_pyramid[0].download(print_input_normal);
    input.color_pyramid[0].download(print_input_color);
    input.depth_pyramid[0].download(print_input_depth);
    input.filtered_depth_pyramid[0].download(print_input_filtered_depth);

    cout << "Mean vertex x:" << cv::mean(print_input_vertex)[0] << endl;
    cout << "Mean vertex y:" << cv::mean(print_input_vertex)[1] << endl;
    cout << "Mean vertex z:" << cv::mean(print_input_vertex)[2] << endl;
    cout << "Sum vertex x:" << cv::sum(print_input_vertex)[0] << endl;
    cout << "Sum vertex y:" << cv::sum(print_input_vertex)[1] << endl;
    cout << "Sum vertex z:" << cv::sum(print_input_vertex)[2] << endl;

    cout << "Mean normal x:" << cv::mean(print_input_normal)[0] << endl;
    cout << "Mean normal y:" << cv::mean(print_input_normal)[1] << endl;
    cout << "Mean normal z:" << cv::mean(print_input_normal)[2] << endl;
    cout << "Sum normal x:" << cv::sum(print_input_normal)[0] << endl;
    cout << "Sum normal y:" << cv::sum(print_input_normal)[1] << endl;
    cout << "Sum normal z:" << cv::sum(print_input_normal)[2] << endl;

    cout << "Mean color:" << cv::mean(print_input_color)[0] << endl;
    cout << "Sum color:" << cv::sum(print_input_color)[0] << endl;

    cout << "Mean depth:" << cv::mean(print_input_depth)[0] << endl;
    cout << "Sum depth:" << cv::sum(print_input_depth)[0] << endl;

    cout << "Mean filtered depth:" << cv::mean(print_input_filtered_depth)[0] << endl;
    cout << "Sum filtered depth:" << cv::sum(print_input_filtered_depth)[0] << endl;

}
#ifndef YOLO_H
#define YOLO_H

#include <fstream>

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <chrono>

using namespace cv;

class Yolo{
    public :
        struct Detection
        {
            int class_id;
            float confidence;
            cv::Rect box;
        };
        Yolo();
        std::vector<Detection> exec(Mat frame);
        std::chrono::time_point<std::chrono::high_resolution_clock> start;
        
    
    private :
        int frame_count = 0;
        float fps = -1;
        cv::dnn::Net net;
        std::vector<std::string> class_list;

        const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};
        bool is_cuda = true;

        const float INPUT_WIDTH = 640.0;
        const float INPUT_HEIGHT = 640.0;
        const float SCORE_THRESHOLD = 0.2;
        const float NMS_THRESHOLD = 0.4;
        const float CONFIDENCE_THRESHOLD = 0.4;

        std::vector<int> nms_result;

        std::vector<std::string> load_class_list();
        void load_net(cv::dnn::Net &net, bool is_cuda);
        cv::Mat format_yolov5(const cv::Mat &source);
        void detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className);
};

#endif

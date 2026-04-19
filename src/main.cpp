#include "Yolo.h"
#include "OpticalFlow.h"

#include "opencv2/opencv.hpp"
using namespace cv;

struct ObjectDetected
{
    int class_id;
    float confidence;
    cv::Rect box;
    OpticalFlow::vectDepl vect;
    Point centreGravity;
};

std::vector<ObjectDetected> mapping(Mat deplacement, std::vector<Yolo::Detection> yoloDetection)
{
    std::vector<ObjectDetected> objects;

    for (Yolo::Detection detect : yoloDetection)
    {
        float sum_u = 0, sum_v = 0;
        int count = 0;

        for (int j = detect.box.y; j < detect.box.y + detect.box.height; j++)
        {
            for (int i = detect.box.x; i < detect.box.x + detect.box.width; i++)
            {           
                OpticalFlow::vectDepl& p = deplacement.at<OpticalFlow::vectDepl>(j, i);
                sum_u += p.u;
                sum_v += p.v;
                count++;
            }
        }

        float u = sum_u / count;
        float v = sum_v / count;

        float norm = sqrt(u*u + v*v);
        if (norm < 1) continue;

        int xGravity = detect.box.x + (detect.box.width) / 2;
        int yGravity = detect.box.y + (detect.box.height) / 2;

        ObjectDetected object{detect.class_id, detect.confidence, detect.box, {u, v}, Point(xGravity, yGravity)};

        objects.push_back(object);
    }

    return objects;
}

Mat drawMeanFlow(Mat frame, std::vector<ObjectDetected> objects)
{
    cv::Mat meanFlowDraw(frame.rows, frame.cols, CV_8UC3, cv::Scalar(255, 255, 255));

    float scale = 20;

    for (ObjectDetected object : objects)
    {
        cv::Point p1(object.centreGravity.x, object.centreGravity.y);
        cv::Point p2(object.centreGravity.x + object.vect.u * scale, object.centreGravity.y + object.vect.v * scale);

        cv::arrowedLine(meanFlowDraw, p1, p2, cv::Scalar(0, 0, 255), 2);
    }

    return meanFlowDraw;
}

int main(int argc, char** argv)
{

    VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    OpticalFlow opticalFlow;
    Yolo yolo;
    Mat frameOld;
    Mat frame;


    cap >> frameOld;
    Size newSize(200, 200);
        
    resize(frameOld, frameOld, newSize, 0, 0, INTER_LINEAR);

    for(;;){

        cap >> frame;
        resize(frame, frame, newSize, 0, 0, INTER_LINEAR);

        Mat deplacement = opticalFlow.exec(frame, frameOld);
        std::vector<Yolo::Detection> yoloDetection = yolo.exec(frame);

        std::vector<ObjectDetected> objects = mapping(deplacement, yoloDetection);

        Mat meanFlowDraw = drawMeanFlow(frame, objects);

        cv::imshow("capture", frame);

        cv::imshow("deplacement", meanFlowDraw);

        frame.copyTo(frameOld);

        if(waitKey(33) == 27) break;
    }

}
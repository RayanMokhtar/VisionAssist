#include "Yolo.h"
#include "OpticalFlow.h"

#include "opencv2/opencv.hpp"
using namespace cv;

struct ObjectDetected
{
    int class_id;
    float confidence;
    cv::Rect box;
    Vec2f vect;
    Point centreGravity;
};


#define USE_WEBCAM_FALLBACK 1  // 1 = activé, 0 = désactivé


#define FALLBACK_VIDEO "../../data/videos/video_test_camera_fonctionne_pas.mp4"  


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
                // TODO: Yolo out of box
                if (j >= 0 && j < deplacement.rows && i >= 0 && i < deplacement.cols)
                {
                    cv::Vec2f& p = deplacement.at<cv::Vec2f>(j, i);

                    sum_u += p[0];
                    sum_v += p[1];
                    count++;
                }
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
        cv::Point p2(object.centreGravity.x + object.vect[0] * scale, object.centreGravity.y + object.vect[1] * scale);

        cv::arrowedLine(meanFlowDraw, p1, p2, cv::Scalar(0, 0, 255), 2);
    }

    return meanFlowDraw;
}

int main(int argc, char** argv)
{
    OpticalFlow opticalFlow;
    Yolo yolo;
    Mat frameOld;
    Mat frame;

    VideoCapture cap;   

    // TODO : à déplacer peut être dans un meilleur endroit ? où à injecter directement dnas le cap selon choix config
    std::string pipeline =
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=620, height=480, framerate=30/1 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "appsink drop=true max-buffers=1 sync=false";

    cap.open(pipeline, CAP_GSTREAMER);

    // si problème récupérationl video ? on teste avec webcam classqieu (marche avec linux , sinon test video dans le capture ... )
    if (!cap.isOpened()) {
        std::cerr << "Pipeline GStreamer nvidia failed test avec webcam...\n";

        if (USE_WEBCAM_FALLBACK) {
            cap.open(0); 
            if (!cap.isOpened()) {
                std::cerr << "Webcam fallback erreur, test avec vidéo\n";
                cap.open(FALLBACK_VIDEO);
                if (!cap.isOpened()) {
                    std::cerr << "Video fallback erreur\n";
                    return -1;
                }
            }
        }
        else {
            std::cerr << "pas de fallback actif , vérifier chemin vers video ou potentiels problemes avec l'environnement.\n";
            return -1;
        }
    }

    cap >> frameOld;
    cv::cvtColor(frameOld, frameOld, cv::COLOR_BGRA2BGR);

    for(;;){

        cap >> frame;
        cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);

        Mat deplacement = opticalFlow.exec(frame, frameOld);
        std::vector<Yolo::Detection> yoloDetection = yolo.exec(frame);

        std::vector<ObjectDetected> objects = mapping(deplacement, yoloDetection);

        Mat meanFlowDraw = drawMeanFlow(frame, objects);

        cv::imshow("capture", frame);

        cv::imshow("deplacement", meanFlowDraw);

        frame.copyTo(frameOld);

        if(waitKey(1) == 27) break;
    }
}

#include "Yolo.h"
#include "OpticalFlow.h"

#include "opencv2/opencv.hpp"
using namespace cv;

#define JETSON 0
#define runOnGPU JETSON==1
#define USE_WEBCAM_FALLBACK 1  // 1 = activé, 0 = désactivé
#define FALLBACK_VIDEO "../../data/videos/video_test_camera_fonctionne_pas.mp4"
#define MAXDISTANCE 50

struct ObjectDetected
{
    int id;
    int classYolo;
    float confidence;
    cv::Rect box;
    Vec2f vect;
    Point centreGravity;
    cv::Scalar color{};
    std::string className{};
};

int globalId = 0;

void tracker(std::vector<ObjectDetected>& objectsNew, const std::vector<ObjectDetected>& objectsCopy)
{
    if (objectsCopy.empty())
        return;

    std::set<int> usedIds;

    for (ObjectDetected& objectNew : objectsNew)
    {
        float minDist = 1000;
        const ObjectDetected* objectOldRef = nullptr;

        for (const ObjectDetected& objectOld : objectsCopy)
        {
            if(usedIds.count(objectOld.id))
                continue;

            if (objectOld.classYolo != objectNew.classYolo) {
               continue;
            }

            Point pOld(objectOld.centreGravity.x + objectOld.vect[0], objectOld.centreGravity.y + objectOld.vect[1]);
            Point pNew = objectNew.centreGravity;

            float dist = sqrt(((pNew.x  - pOld.x) * (pNew.x  - pOld.x)) + ((pNew.y  - pOld.y) * (pNew.y  - pOld.y)));
            if (dist > MAXDISTANCE)
                continue;

            if (dist < minDist) {
                minDist = dist;
                objectOldRef = &objectOld;
            }
        }

        if (objectOldRef != nullptr) {
            objectNew.id = objectOldRef->id;
            usedIds.insert(objectOldRef->id);
        }
        else {
            objectNew.id = globalId++;       
        }
    }
}

std::vector<ObjectDetected> mapping(Mat deplacement,  const std::vector<YOLO::Detection> yoloDetection)
{
    std::vector<ObjectDetected> objectsNew;

    for (YOLO::Detection detect : yoloDetection)
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
        if (norm < 0.1) continue;

        int xGravity = detect.box.x + (detect.box.width) / 2;
        int yGravity = detect.box.y + (detect.box.height) / 2;

        globalId = globalId % 100000;

        ObjectDetected object{-1, detect.class_id, detect.confidence, detect.box, {u, v}, Point(xGravity, yGravity), detect.color, detect.className};

        objectsNew.push_back(object);
    }

    return objectsNew;
}

Mat drawMeanFlow(Mat frame, const std::vector<ObjectDetected> objects)
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

void drawTracking(Mat frame, const std::vector<ObjectDetected> objects)
{
    int size = objects.size();
    for (int i = 0; i < size; ++i)
    {
        ObjectDetected object = objects[i];

        cv::Rect box = object.box;
        cv::Scalar color = object.color;

        // Detection box
        cv::rectangle(frame, box, color, 2);

        // Detection box text
        std::string classString = object.className + ' ' + std::to_string(object.confidence).substr(0, 4) + " id:" + std::to_string(object.id);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

        cv::rectangle(frame, textBox, color, cv::FILLED);
        cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }
}

int main(int argc, char** argv)
{
    OpticalFlow opticalFlow;
    YOLO yolo("../YoloUtils/yolov8n.onnx", cv::Size(640, 640), "classes.txt", runOnGPU);
    Mat frameOld;
    Mat frame;

#if JETSON

    // TODO : à déplacer peut être dans un meilleur endroit ? où à injecter directement dnas le cap selon choix config
    std::string pipeline =
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=620, height=480, framerate=30/1 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "appsink drop=true max-buffers=1 sync=false";

    VideoCapture cap(pipeline, CAP_GSTREAMER);
#else
    VideoCapture cap(0);    
#endif

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
    std::vector<ObjectDetected> objects;
    std::vector<ObjectDetected> objectsNew;


    for(;;){

        cap >> frame;
        cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);

        Mat deplacement = opticalFlow.exec(frame, frameOld);
        std::vector<YOLO::Detection> yoloDetection = yolo.exec(frame);

        if(!objectsNew.empty()) 
            objects = objectsNew;

        objectsNew = mapping(deplacement, yoloDetection);
        tracker(objectsNew, objects);

        Mat meanFlowDraw = drawMeanFlow(frame, objectsNew);
        drawTracking(frame, objectsNew);

        cv::imshow("capture", frame);

        cv::imshow("deplacement", meanFlowDraw);

        frame.copyTo(frameOld);

        if(waitKey(1) == 27) break;
    }
}

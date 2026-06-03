#include "Yolo.h"
#include "OpticalFlow.h"

#include "opencv2/opencv.hpp"
#include <gpiod.h>
#include <chrono>
#include <thread>

using namespace cv;

#define JETSON 1
#define runOnGPU JETSON==1
#define USE_WEBCAM_FALLBACK 1  // 1 = activé, 0 = désactivé
#define FALLBACK_AVEC_CHEMIN_VIDEO "../../data/vid3.mp4"
#define MAXDISTANCE 100
#define MAX_PERSISTANCE 10
#define RAYON_DETECTION 150

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
    int counterDetectionFailed = 0;
    std::vector<cv::Point> trajectory;
    std::vector<cv::Vec2f> velocity;
};

int globalId = 0;

void tracker(std::vector<ObjectDetected>& objectsNew, const std::vector<ObjectDetected>& objectsCopy)
{
    //std::cout << "Début tracker ---------------------------------------------------------------------- "<< std::endl;

    // for (const ObjectDetected& objectOld : objectsCopy)
    // {
    //     std::cout << "objectOld id: " << objectOld.id << " class: " << objectOld.className << std::endl;
    // }

    // for (ObjectDetected& objectNew : objectsNew)
    // {
    //     std::cout << "objectNew id: " << objectNew.id << " class: " << objectNew.className << std::endl;
    // }

    if (objectsCopy.empty())
        return;

    std::set<int> usedIds;

    for (ObjectDetected& objectNew : objectsNew)
    {
        float minDist = 1000;
        const ObjectDetected* objectOldRef = nullptr;

        for (const ObjectDetected& objectOld : objectsCopy)
        {
            if(usedIds.count(objectOld.id)) {
                //std::cout << "Used id: " << objectOld.id << " class: " << objectOld.className << std::endl;
                continue;
            }

            if (objectOld.classYolo != objectNew.classYolo) {
                //std::cout << "Class mismatch: " << objectOld.className << " vs " << objectNew.className << " id: " << objectNew.id << std::endl;
               continue;
            }

            Point pOld(objectOld.centreGravity.x + objectOld.vect[0], objectOld.centreGravity.y + objectOld.vect[1]);
            Point pNew = objectNew.centreGravity;

            //std::cout << "pOld: " << pOld << " pNew: " << pNew << " id new: " << objectNew.id << " class: " << objectNew.className << " id old: " << objectOld.id << " class old: " << objectOld.className << std::endl;

            float dist = sqrt(((pNew.x  - pOld.x) * (pNew.x  - pOld.x)) + ((pNew.y  - pOld.y) * (pNew.y  - pOld.y)));
            //std::cout << "id new: " << objectNew.id << " class new: " << objectNew.className << " id old: " << objectOld.id << "class old: " << objectOld.className << " dist: " << dist << " > " << MAXDISTANCE << std::endl;
            if (dist > MAXDISTANCE) {
                continue;
            }

            if (dist < minDist) {
                minDist = dist;
                objectOldRef = &objectOld;
                
            }
        }

        if (objectOldRef != nullptr) {
            objectNew.id = objectOldRef->id;
            usedIds.insert(objectOldRef->id);
            objectNew.trajectory = objectOldRef->trajectory;
            objectNew.trajectory.push_back(objectNew.centreGravity);
            objectNew.velocity = objectOldRef->velocity;
            objectNew.velocity.push_back(objectNew.vect);

            // if (objectNew.id == 0) {
            //     cv::Mat vis(620, 620, CV_8UC3, cv::Scalar(255, 255, 255));
            //     for(const Point& p : objectNew.trajectory) {
            //         //std::cout << "object id: " << objectNew.id << " class: " << objectNew.className << " trajectory point: " << p << std::endl;
            //         cv::circle(vis, p, 10, cv::Scalar(0, 0, 255), -1);
            //     }
            //     cv::imshow("pointo", vis);
            // }
            

            //std::cout << "SET object id: " << objectNew.id << " class: " << objectNew.className << std::endl;   
        }
        // else {
        //     objectNew.id = globalId++;    
        //     std::cout << "new object id: " << objectNew.id << " class: " << objectNew.className << std::endl;   
        // }
    }

        //std::cout << "Fin tracker ---------------------------------------------------------------------- "<< std::endl;

}

std::vector<ObjectDetected> mapping(const cv::Mat& deplacement, const std::vector<YOLO::Detection>& yoloDetection)
{
    std::vector<ObjectDetected> objectsNew;

    for (const YOLO::Detection& detect : yoloDetection)
    {
        // if (detect.className == "person")
        //     continue;

        float sum_u = 0, sum_v = 0;
        int count = 0;

        // std::vector<float> us;
        // std::vector<float> vs;

        for (int j = detect.box.y; j < detect.box.y + detect.box.height; j++)
        {
            for (int i = detect.box.x; i < detect.box.x + detect.box.width; i++)
            {
                // TODO: Yolo out of box
                if (j >= 0 && j < deplacement.rows && i >= 0 && i < deplacement.cols)
                {
                    const cv::Vec2f& p = deplacement.at<cv::Vec2f>(j, i);

                    sum_u += p[0];
                    sum_v += p[1];
                    count++;

                    // us.push_back(p[0]);
                    // vs.push_back(p[1]);
                }
            }
        }

        // std::sort(us.begin(), us.end());
        // std::sort(vs.begin(), vs.end());

        // float u = us[us.size()/2];
        // float v = vs[vs.size()/2];

        float u = sum_u / count;
        float v = sum_v / count;

        // float norm = sqrt(u*u + v*v);
        // if (norm < 0.1) continue;

        int xGravity = detect.box.x + (detect.box.width) / 2;
        int yGravity = detect.box.y + (detect.box.height) / 2;

        globalId = globalId % 100000;

        ObjectDetected object{globalId++, detect.class_id, detect.confidence, detect.box, {u, v}, Point(xGravity, yGravity), detect.color, detect.className, 0, {Point(xGravity, yGravity)}, {{u, v}}};

        objectsNew.push_back(object);
    }

    return objectsNew;
}

Mat drawOpticalFlow(const Mat& matDepl)
{
    cv::Mat matOpticalFlow(matDepl.rows, matDepl.cols, CV_8UC3, cv::Scalar(255, 255, 255));

    int step = 3;
    float scale = 1;

    for (int y = step; y < matDepl.rows - step; y += step)
    {
        for (int x = step; x < matDepl.cols - step; x += step)
        {
            float sum_u = 0, sum_v = 0;
            int count = 0;

            for (int j = y - step/2; j <= y + step/2; j++)
            {
                for (int i = x - step/2; i <= x + step/2; i++)
                {
                    const cv::Vec2f& p = matDepl.at<cv::Vec2f>(j, i);
                    sum_u += p[0];
                    sum_v += p[1];
                    count++;
                }
            }

            float u = sum_u / count;
            float v = sum_v / count;

            // cv::Vec2f& p = matDepl.at<cv::Vec2f>(y, x);
            // float u = p[0];
            // float v = p[1];

            //float norm = sqrt(u*u + v*v);
            //if (norm < 1) continue;

            cv::Point p1(x, y);
            cv::Point p2(x + u * scale, y + v * scale);

            cv::arrowedLine(matOpticalFlow, p1, p2, cv::Scalar(0, 0, 255), 1);
        }
    }

    return matOpticalFlow;
}

Mat drawSparseFlow(const Mat& deplacement, const std::vector<ObjectDetected>& objects)
{
    cv::Mat sparseFlowDraw(deplacement.rows, deplacement.cols, CV_8UC3, cv::Scalar(255, 255, 255));

    int step = 3;
    float scale = 1;

    for (const ObjectDetected& object : objects)
    {
        float total_u = 0.0;
        float total_v = 0.0;
        int total_count = 00;

        for (int y = (object.box.y + step); y < (object.box.y + object.box.height - step); y += step)
        {
            for (int x = (object.box.x + step); x < (object.box.x + object.box.width - step); x += step)
            {
                float sum_u = 0, sum_v = 0;
                int count = 0;

                for (int j = y - step/2; j < y + step/2; j++)
                {
                    for (int i = x - step/2; i < x + step/2; i++)
                    {
                        const cv::Vec2f& p = deplacement.at<cv::Vec2f>(j, i);
                        sum_u += p[0];
                        sum_v += p[1];
                        count++;

                        total_u += p[0];
                        total_v += p[1];
                        total_count++;
                    }
                }

                float u = sum_u / count;
                float v = sum_v / count;

                cv::Point p1(x, y);
                cv::Point p2(x + u * scale, y + v * scale);

                cv::arrowedLine(sparseFlowDraw, p1, p2, cv::Scalar(0, 0, 255), 1);
            }
        }

        float mean_u = total_u / total_count;
        float mean_v = total_v / total_count;

        float normeFlow = sqrt((mean_u * mean_u) + (mean_v * mean_v));

        //std::cout << "Norme du flux optique: " << normeFlow << " moyenne u: " << mean_u << " moyenne v: " << mean_v << std::endl;
    }

    return sparseFlowDraw;
}

Mat drawMeanFlow(const Mat& frame, const std::vector<ObjectDetected>& objects)
{
    cv::Mat meanFlowDraw(frame.rows, frame.cols, CV_8UC3, cv::Scalar(255, 255, 255));

    float scale = 20;

    for (const ObjectDetected& object : objects)
    {
        cv::Point p1(object.centreGravity.x, object.centreGravity.y);
        cv::Point p2(object.centreGravity.x + object.vect[0] * scale, object.centreGravity.y + object.vect[1] * scale);

        cv::arrowedLine(meanFlowDraw, p1, p2, cv::Scalar(0, 0, 255), 2);
    }

    return meanFlowDraw;
}


Mat drawTrackingYolo(const Mat& frame, const std::vector<ObjectDetected>& objects, double fps)
{
    Mat yoloDraw;
    frame.copyTo(yoloDraw);
    float scale = 20;

    int size = objects.size();
    for (int i = 0; i < size; ++i)
    {
        ObjectDetected object = objects[i];

        cv::Rect box = object.box;
        cv::Scalar color = object.color;

        // Detection box
        cv::rectangle(yoloDraw, box, color, 2);

        // Detection box text
        std::string classString = object.className + ' ' + std::to_string(object.confidence).substr(0, 4) + " id:" + std::to_string(object.id);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

        cv::rectangle(yoloDraw, textBox, color, cv::FILLED);
        cv::putText(yoloDraw, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        cv::circle(yoloDraw, object.centreGravity, 5, cv::Scalar(0, 0, 255), -1);

        cv::Point p1(object.centreGravity.x, object.centreGravity.y);
        cv::Point p2(object.centreGravity.x + object.vect[0] * scale, object.centreGravity.y + object.vect[1] * scale);

        cv::arrowedLine(yoloDraw, p1, p2, cv::Scalar(0, 0, 255), 2);

        std::string fpsText = "FPS: " + std::to_string(fps).substr(0, 5);

        cv::putText(yoloDraw, fpsText, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    }

    int x1 = yoloDraw.cols / 3;
    int x2 = x1*2;
    int y1 = yoloDraw.rows / 3;
    int y2 = y1*2;

    cv::Point p1(0, x1);
    cv::Point p2(yoloDraw.rows, x1);
    cv::line(yoloDraw, p1, p2, cv::Scalar(0, 255, 0), 1);

    p1 = {0, x2};
    p2 = {yoloDraw.rows, x2};
    cv::line(yoloDraw, p1, p2, cv::Scalar(0, 255, 0), 1);

    p1 = {y1, 0};
    p2 = {y1, yoloDraw.cols};
    cv::line(yoloDraw, p1, p2, cv::Scalar(0, 255, 0), 1);

    p1 = {y2, 0};
    p2 = {y2, yoloDraw.cols};
    cv::line(yoloDraw, p1, p2, cv::Scalar(0, 255, 0), 1);

    return yoloDraw;
}

std::vector<ObjectDetected> persistanceBetweenFrame(const std::vector<ObjectDetected> objectsNew, std::vector<ObjectDetected> objectsCopy)
{
    std::vector<ObjectDetected> objectsPersistant;
    objectsPersistant = objectsNew;

    for (ObjectDetected& objectCopy : objectsCopy)
    {
        bool found = false;
        for (const ObjectDetected& objectNew : objectsNew)
        {
            if (objectCopy.id == objectNew.id) {
                found = true;
                break;
            }
        }

        if (!found && objectCopy.counterDetectionFailed < MAX_PERSISTANCE) {
            objectCopy.counterDetectionFailed++;
            // std::cout << "object id: " << objectCopy.id << " class: " << objectCopy.className << " counterDetectionFailed: " << objectCopy.counterDetectionFailed << std::endl;
            // objectCopy.centreGravity.x += objectCopy.vect[0];
            // objectCopy.centreGravity.y += objectCopy.vect[1];

            // objectCopy.box.x += objectCopy.vect[0];
            // objectCopy.box.y += objectCopy.vect[1];

            objectsPersistant.push_back(objectCopy);
        }        
    }

    return objectsPersistant;
}

void decisionMaking(Mat& frame, const Point& centre, const std::vector<ObjectDetected>& objectsNew)
{
    for (const ObjectDetected& object : objectsNew)
    {   
        cv::arrowedLine(frame, object.centreGravity, centre, cv::Scalar(0, 0, 255), 2);

        float dx = object.centreGravity.x - centre.x;
        float dy = object.centreGravity.y - centre.y;

        float dist = std::sqrt(dx * dx + dy * dy);

        Vec2f versCentre(centre.x - object.centreGravity.x, centre.y - object.centreGravity.y);

        float dot = object.vect[0] * versCentre[0] + object.vect[1] * versCentre[1];
        float normVect = std::sqrt(object.vect[0] * object.vect[0] + object.vect[1] * object.vect[1]);
        float normVersCentre = std::sqrt(versCentre[0] * versCentre[0] + versCentre[1] * versCentre[1]);
        float cosTetha = dot / (normVect * normVersCentre);

        // std::cout << "object id: " << object.id << " x : " << object.centreGravity.x  << " y : " << object.centreGravity.y << " class: " << object.className << " distance from center: " << dist << " cos: " << cosTetha << "angle: " << std::acos(cosTetha) * 180 / CV_PI << std::endl;

        // if(dist < RAYON_DETECTION && cosTetha > 0.8f) {
        //     std::cout << "ALEEEEEEEEEEEEEEERRRRRRRRRRRTTTTTTTTTT: object id: " << object.id << " class: " << object.className << " distance: " << dist << std::endl;
        // }
        // else if(dist < RAYON_DETECTION && cosTetha < 0.8f && cosTetha > 0.5f) {
        //     std::cout << "WARNING QUAND MEME: object id: " << object.id << " class: " << object.className << " distance: " << dist << std::endl;
        // }
        // else if(dist < RAYON_DETECTION) {
        //     std::cout << "OBJET AU CENTRE MAIS ANGLE PAS DETECTE" << " object id: " << object.id << " class: " << object.className << " distance: " << dist << std::endl;
        // }
        // else {
        //     std::cout << "Pas de danger immédiat: object id: " << object.id << " class: " << object.className << " distance: " << dist << std::endl;
        // }
    }
}

void beep(int secondes)
{
    gpiod_chip* chip = gpiod_chip_open_by_name("gpiochip0");
    gpiod_line* line = gpiod_chip_get_line(chip, 144);

    gpiod_line_request_output(line, "buzzer", 0);

    auto fin = std::chrono::steady_clock::now() + std::chrono::seconds(secondes);

    gpiod_line_set_value(line, 1);

    while (std::chrono::steady_clock::now() < fin)
    {
        // gpiod_line_set_value(line, 1);
        // std::this_thread::sleep_for(std::chrono::microseconds(250));

        // gpiod_line_set_value(line, 0);
        // std::this_thread::sleep_for(std::chrono::microseconds(250));
    }

    gpiod_line_set_value(line, 0);

    gpiod_line_release(line);
    gpiod_chip_close(chip);
}

int diff_ms(timeval t1, timeval t2)
{
    return (((t1.tv_sec - t2.tv_sec) * 1000000) +
            (t1.tv_usec - t2.tv_usec))/1000;
}

using Clock = std::chrono::high_resolution_clock;

static double elapsedMs(
    const Clock::time_point& start,
    const Clock::time_point& end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main(int argc, char** argv)
{
    OpticalFlow opticalFlow;
    YOLO yolo("../YoloUtils/yolov8n.onnx", cv::Size(640, 640), "classes.txt", true);
    Mat frameOld;
    Mat frame;
    timeval start, end;

    VideoCapture cap;   

    beep(1);

#if USE_WEBCAM_FALLBACK
    // TODO : à déplacer peut être dans un meilleur endroit ? où à injecter directement dnas le cap selon choix config
    std::string pipeline =
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
        "nvvidconv ! "
        "video/x-raw, width=640, height=640, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false";

    cap.open(pipeline, CAP_GSTREAMER);
#endif

    // si problème récupérationl video ? on teste avec webcam classqieu (marche avec linux , sinon test video dans le capture ... )
    if (!cap.isOpened()) {
        std::cerr << "Pipeline GStreamer nvidia failed test avec webcam...\n";

        if (USE_WEBCAM_FALLBACK) {
            cap.open(0);
            if (!cap.isOpened()) {
                std::cerr << "Webcam fallback erreur, test avec vidéo\n";
            }        
        }
        else if (FALLBACK_AVEC_CHEMIN_VIDEO) {
            std::cerr << "Webcam fallback erreur, test avec vidéo\n";
            cap.open(FALLBACK_AVEC_CHEMIN_VIDEO);
            if (!cap.isOpened()) {
                std::cerr << "Video fallback erreur\n";
                return -1;
            }
        }
        else {
            std::cerr << "pas de fallback actif , vérifier chemin vers video ou potentiels problemes avec l'environnement.\n";
            return -1;
        }
    }

    cap >> frameOld;

    if(frameOld.empty())
    {
        std::cerr << "Impossible de lire la premiere frame" << std::endl;
        return -1;
    }

    cv::cvtColor(frameOld, frameOld, cv::COLOR_BGRA2BGR);
    std::vector<ObjectDetected> objects;
    std::vector<ObjectDetected> objectsNew;


    for(;;)
    {
        auto tGlobalStart = Clock::now();

        cap >> frame;

        if(frame.empty())
        {
            std::cerr << "Frame vide" << std::endl;
            break;
        }

        cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);

        Mat deplacement;
        Mat matOpticalFlow;

        auto tFlowStart = Clock::now();
        std::thread threadFlow([&]() {
            deplacement = opticalFlow.exec(frame, frameOld); 
            matOpticalFlow = drawOpticalFlow(deplacement);
        });
        
        std::vector<YOLO::Detection> yoloDetection;
        
        std::thread threadYolo([&]() {
            yoloDetection = yolo.exec(frame);
            objects = persistanceBetweenFrame(objectsNew, objects);
        });

        threadFlow.join();
        threadYolo.join();
        auto tFlowEnd = Clock::now();

        auto tMappingStart = Clock::now();
        objectsNew = mapping(deplacement, yoloDetection);
        auto tMappingEnd = Clock::now();

        auto tTrackerStart = Clock::now();
        tracker(objectsNew, objects);
        auto tTrackerEnd = Clock::now();

        auto tDrawStart = Clock::now();
        Mat meanFlowDraw = drawMeanFlow(frame, objectsNew);
        //Mat sparseFlowDraw = drawSparseFlow(deplacement, objectsNew);
        auto tBeforeYoloDraw = Clock::now();

        double globalMsTemp = elapsedMs(tGlobalStart, tBeforeYoloDraw);
        double fps = 1000.0 / std::max(globalMsTemp, 1.0);

        Mat yoloDraw = drawTrackingYolo(frame, objectsNew, fps);
        auto tDrawEnd = Clock::now();

        auto tImshowStart = Clock::now();
        cv::imshow("yolo", yoloDraw);
        cv::imshow("flux optique", matOpticalFlow);
        //cv::imshow("sparse", sparseFlowDraw);
        cv::imshow("deplacement", meanFlowDraw);
        auto tImshowEnd = Clock::now();

        auto tCopyStart = Clock::now();
        frame.copyTo(frameOld);
        auto tCopyEnd = Clock::now();

        auto tGlobalEnd = Clock::now();

        double flowMs = elapsedMs(tFlowStart, tFlowEnd);
        double mappingMs = elapsedMs(tMappingStart, tMappingEnd);
        double trackerMs = elapsedMs(tTrackerStart, tTrackerEnd);
        double drawMs = elapsedMs(tDrawStart, tDrawEnd);
        double imshowMs = elapsedMs(tImshowStart, tImshowEnd);
        double copyMs = elapsedMs(tCopyStart, tCopyEnd);
        double globalMs = elapsedMs(tGlobalStart, tGlobalEnd);

        std::cout << "\n========== PROFILING ==========\n";
        std::cout << "GLOBAL        : " << globalMs << " ms | FPS: " << 1000.0 / globalMs << "\n";
        std::cout << "Optical flow et YOLO et draw flow : " << flowMs << " ms\n";
        std::cout << "Mapping       : " << mappingMs << " ms\n";
        std::cout << "Tracker       : " << trackerMs << " ms\n";
        std::cout << "Draw all      : " << drawMs << " ms\n";
        std::cout << "Imshow        : " << imshowMs << " ms\n";
        std::cout << "Copy frame    : " << copyMs << " ms\n";
        std::cout << "===============================\n";

        //sleep(0.067);

        if(waitKey(1) == 27)
            break;
    }
}

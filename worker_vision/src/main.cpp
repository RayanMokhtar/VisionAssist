#include "Yolo.h"
#include "OpticalFlow.h"
#include "imu.h"

#include "opencv2/opencv.hpp"
#include <gpiod.h>
#include <chrono>
#include <thread>
#include <atomic>
#include <cmath>
#include <numeric>

using namespace cv;

#define JETSON 1
#define runOnGPU JETSON==1
#define USE_WEBCAM_FALLBACK 1  // 1 = activé, 0 = désactivé
#define FALLBACK_AVEC_CHEMIN_VIDEO "../../data/vid3.mp4"
#define MAXDISTANCE 100
#define MAX_PERSISTANCE 10
#define RAYON_DETECTION 150
#define PI 3.14159265

#define SEUIL_FILTRAGE 1.2
#define SEUIL_DECISION_BRUIT 15
#define SEUIL_DECISION_AVANT_ARRIERE 30
#define MAX_FRAMES_DECISION 5
#define SEUIL_DANGER 120

enum Decisions {
    RIEN,
    DEVANT,
    AVANT, 
    ARRIERE,
    GAUCHE,
    DROITE,
    NB_DECISIONS
};

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
    std::array<int,360> histo;
    std::vector<Decisions> decisions;
    float ttc = 0;
    Decisions decision = RIEN;
    float normeHisto = 0.0;
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
            objectNew.normeHisto = objectOldRef->normeHisto;

            objectNew.decisions = objectOldRef->decisions;
            objectNew.ttc = objectOldRef->ttc;

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
        int compteurVecteursFiltresParSeuil = 0 ; 
        std::array<int,360> histo{};

        // std::vector<float> us;
        // std::vector<float> vs;

        //std::vector<cv::Vec2f> flows;

        for (int j = detect.box.y; j < detect.box.y + detect.box.height; j++)
        {
            for (int i = detect.box.x; i < detect.box.x + detect.box.width; i++)
            {
                // TODO: Yolo out of box
                if (j >= 0 && j < deplacement.rows && i >= 0 && i < deplacement.cols)
                {
                    const cv::Vec2f& p = deplacement.at<cv::Vec2f>(j, i);

                    float norme = sqrt(p[0]*p[0] + p[1]*p[1]);
                    
                    if (norme < SEUIL_FILTRAGE){
                        compteurVecteursFiltresParSeuil++;
                        continue;
                    }
                    
                    sum_u += p[0];
                    sum_v += p[1];
                    count++;

                    int angle = (int)std::round(std::atan2(p[1], p[0]) * 180.0 / PI);

                    if (angle < 0)
                        angle += 360;

                    if (angle >= 360)
                        std::cout << "angle: " << angle << std::endl;

                    histo[angle]++;

                    // us.push_back(p[0]);
                    // vs.push_back(p[1]);

                    //flows.push_back(p);
                }
            }
        }
        //std::cout << "nombre de vecteurs filtres par la norme : => " << compteurVecteursFiltresParSeuil << " => ratio ==> " << (float)((float)(100*compteurVecteursFiltresParSeuil) / (float)((detect.box.height* detect.box.width)))<< std::endl;

        // std::sort(flows.begin(), flows.end(),
        //     [](const cv::Vec2f& a, const cv::Vec2f& b)
        //     {
        //         float angleA = atan((float)a[1]/a[0])*180.0/PI;
        //         float angleB = atan((float)b[1]/b[0])*180.0/PI;
        //         //std::cout << "a0: " << a[0] << " a1: " << a[1] << "angle : " << atan((float)a[1]/a[0])*180.0/PI <<std::endl;
        //         return angleA < angleB;
        //     });

        // cv::Vec2f medianFlow = flows[flows.size() / 2];

        // float u = medianFlow[0];
        // float v = medianFlow[1];

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

        ObjectDetected object{globalId++, detect.class_id, detect.confidence, detect.box, {u, v}, Point(xGravity, yGravity), detect.color, detect.className, 0, {Point(xGravity, yGravity)}, {{u, v}}, histo};

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

Mat drawOpticalFlowFiltered(const Mat& matDepl)
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

            float norme = sqrt(u*u + v*v);

            if (norme < SEUIL_FILTRAGE){
                continue;
            }

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

                for (int j = y - step/2; j <= y + step/2; j++)
                {
                    for (int i = x - step/2; i <= x + step/2; i++)
                    {
                        if ((j < 0 || j >= deplacement.rows) || (i < 0 || i >= deplacement.cols))
                            continue;

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

                float norme = sqrt(u*u + v*v);

                if (norme < SEUIL_FILTRAGE){
                    continue;
                }

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

    return yoloDraw;
}

void drawHistogram(const std::vector<ObjectDetected>& objects)
{
    int width = 1800;
    int height = 640;
    int margin = 50;

    for (const ObjectDetected& object : objects)
    {
        if (object.className != "person")
            continue;
        // for (int i = 0; i < 360; i++)
        // {
        //     std::cout << "[" << i << "]=" << object.histo[i] << " ";
        // }
        // std::cout << std::endl;

        cv::Mat img(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

        int maxVal = 400;
        // for (int i = 0; i < 360; i++)
        //     maxVal = std::max(maxVal, object.histo[i]);

        float binWidth = (float)(width - 2 * margin) / 360.0f;

        // Grille Y + labels
        for (int k = 0; k <= 4; k++)
        {
            int value = (maxVal * k) / 4;

            int y = height - margin -
                    (int)((float)value / maxVal * (height - 2 * margin));

            cv::line(img,
                     cv::Point(margin, y),
                     cv::Point(width - margin, y),
                     cv::Scalar(220, 220, 220),
                     1);

            cv::putText(img,
                        std::to_string(value),
                        cv::Point(5, y + 5),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.5,
                        cv::Scalar(0, 0, 0),
                        1);
        }

        float norme = 0.0; 
        float norme_haut_bas = 0.0;
        float norme_droite = 0.0;
        float norme_gauche = 0.0;
        // float taux_haut_bas = 0.0;
        // float taux_droite = 0.0;
        // float taux_gauche = 0.0;
        float taux_null_droite = 0.0;
        float taux_null_gauche = 0.0;

        // Histogramme
        for (int i = 0; i < 360; i++)
        {
            int x1 = margin + (int)(i * binWidth);
            int x2 = margin + (int)((i + 1) * binWidth);

            int barHeight =
                (int)((float)object.histo[i] / maxVal *
                (height - 2 * margin));

            cv::rectangle(img,
                          cv::Point(x1, height - margin),
                          cv::Point(x2, height - margin - barHeight),
                          cv::Scalar(0, 0, 0),
                          cv::FILLED);

            norme += object.histo[i];
            if ((i > 45 && i <= 135) || (i > 225 && i <= 315)) {
                //taux_haut_bas += object.histo[i];
                norme_haut_bas += object.histo[i];
            } else if (i > 135 && i <= 225) {
                //taux_gauche += object.histo[i];
                norme_gauche += object.histo[i];
            } else {
                //taux_droite += object.histo[i];
                norme_droite += object.histo[i];
            }

            if (i < 90 || i > 270) {
                taux_null_droite += object.histo[i];
            } else {
                taux_null_gauche += object.histo[i];
            }
        }

        norme_haut_bas /= 180;
        norme_droite /= 90;
        norme_gauche /= 90;
        // taux_haut_bas /= norme;
        // taux_droite /= norme;
        // taux_gauche /= norme;
        taux_null_droite /= norme;
        taux_null_gauche /= norme;
        norme /= 360.0;

        // taux_haut_bas *= 100;
        // taux_droite *= 100;
        // taux_gauche *= 100;
        taux_null_droite *= 100;
        taux_null_gauche *= 100;
        // taux_haut_bas /= 2;

        // Axe X + graduations
        for (int angle = 0; angle <= 360; angle += 30)
        {
            int x = margin + (int)(angle * binWidth);

            cv::line(img,
                     cv::Point(x, height - margin),
                     cv::Point(x, height - margin + 8),
                     cv::Scalar(0, 0, 0),
                     1);

            cv::putText(img,
                        std::to_string(angle),
                        cv::Point(x - 12, height - margin + 25),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.45,
                        cv::Scalar(0, 0, 0),
                        1);
        }

        // Lignes principales 0° 90° 180° 270°
        for (int angle : {0, 90, 180, 270})
        {
            int x = margin + (int)(angle * binWidth);

            cv::line(img,
                     cv::Point(x, margin),
                     cv::Point(x, height - margin),
                     cv::Scalar(0, 0, 255),
                     1);
        }

        // Axes
        cv::line(img,
                 cv::Point(margin, height - margin),
                 cv::Point(width - margin, height - margin),
                 cv::Scalar(0, 0, 0),
                 2);

        cv::line(img,
                 cv::Point(margin, margin),
                 cv::Point(margin, height - margin),
                 cv::Scalar(0, 0, 0),
                 2);

        // Titre
        cv::putText(img,
                    "Histogramme - " + object.className + " Norme: " + std::to_string(norme)  + " Norme haut bas: " + std::to_string(norme_haut_bas) 
                    + " Norme droite: " + std::to_string(norme_droite) + " Norme gauche: " + std::to_string(norme_gauche),
                    cv::Point(20, 30),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    cv::Scalar(0, 100, 0),
                    2);

        // Label axe X
        cv::putText(img,
                    "Angle (degres)",
                    cv::Point(width / 2 - 80, height - 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(255, 0, 0),
                    2);

        // Label axe Y
        cv::putText(img,
                    "Occurrences",
                    cv::Point(10, margin - 10),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(255, 0, 0),
                    2);

        cv::imshow("histogramme", img);
    }
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

void beep(int secondes)
{
    gpiod_chip* chip = gpiod_chip_open_by_name("gpiochip0");
    gpiod_line* line = gpiod_chip_get_line(chip, 144);

    gpiod_line_request_output(line, "buzzer", 0);

    auto fin = std::chrono::steady_clock::now() + std::chrono::seconds(secondes);

    gpiod_line_set_value(line, 1);

    std::this_thread::sleep_for(std::chrono::seconds(secondes));

    gpiod_line_set_value(line, 0);

    gpiod_line_release(line);
    gpiod_chip_close(chip);
}

std::atomic<bool> beepRunning(false);

void startBeepAsync(int nbBeep)
{
    if (beepRunning.exchange(true))
        return;

    std::thread([nbBeep]() {
        for (int i = 0; i < nbBeep; i++)
        {
            beep(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
        }

        beepRunning = false;
    }).detach();
}

void decisionMaking(std::vector<ObjectDetected>& objectsNew, Point pointRef)
{
    std::vector<ObjectDetected> decisions;

    for (ObjectDetected& object : objectsNew)
    {   
        if (object.className != "person")
            continue;

        float dx = object.centreGravity.x - pointRef.x;
        float dy = object.centreGravity.y - pointRef.y;
 
        float r = sqrt(dx*dx + dy*dy);
        float v = sqrt((object.vect[0]*object.vect[0]) + (object.vect[1]*object.vect[1]));

        //std::cout << "Norme flux optique ===> " << v << std::endl;

        //float vr = (dx*object.vect[0] + dy*object.vect[1]) / r;

        float ttc = -r / v;

        object.ttc += ttc;

        // if (ttc > - 500){
        //     std::cout<< "DAAAAAAAAANNNNNNNNNNNGEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEER"<< std::endl;
        // }

        // if (ttc < 500)
        // {
        //     std::cout << "LOOOOOOOOOOOOOOOOOL1" << std::endl;
        //     startBeepAsync(3);
        // } else if (ttc < 0)
        // {
        //     std::cout << "LOOOOOOOOOOOOOOOOOL2" << std::endl;
        //     startBeepAsync(6);
        // } else if (ttc < -500)
        // {
        //     std::cout << "LOOOOOOOOOOOOOOOOOL3" << std::endl;
        //     startBeepAsync(9);
        // }

        //std::cout << "class: " << object.className << " class id: " << object.id << " r: " << r << std::endl;

        // std::cout << "class: " << object.className << " class id: " << object.id << " ttc: " << ttc << " r: " << r << " v: " << v << std::endl;

        float somme =  0.0;

        // float taux_haut_bas = 0.0;
        // float taux_droite = 0.0;
        // float taux_gauche = 0.0;
        float norme_haut_bas = 0.0;
        float norme_droite = 0.0;
        float norme_gauche = 0.0;
        float taux_null_droite = 0.0;
        float taux_null_gauche = 0.0;


        // Histogramme
        for (int i = 0; i < 360; i++)
        {
            somme += object.histo[i];

            if ((i > 45 && i <= 135) || (i > 225 && i <= 315)) {
                //taux_haut_bas += object.histo[i];
                norme_haut_bas += object.histo[i];
            } else if (i > 135 && i <= 225) {
                //taux_gauche += object.histo[i];
                norme_gauche += object.histo[i];
            } else {
                //taux_droite += object.histo[i];
                norme_droite += object.histo[i];
            }

            if (i < 90 || i > 270) {
                taux_null_droite += object.histo[i];
            } else {
                taux_null_gauche += object.histo[i];
            }
        }

        // taux_haut_bas /= somme;
        // taux_droite /= somme;
        // taux_gauche /= somme;
        norme_haut_bas /= 180;
        norme_droite /= 90;
        norme_gauche /= 90;
        taux_null_droite /= somme;
        taux_null_gauche /= somme;
        float norme = somme / 360.0;

        // norme_haut_bas /= 2;

        // taux_haut_bas *= 100;
        // taux_droite *= 100;
        // taux_gauche *= 100;
        taux_null_droite *= 100;
        taux_null_gauche *= 100;

        // taux_haut_bas /= 2;

        //std::cout << "norme " << norme << std::endl;

        //float maxTaux = taux_haut_bas;
        float maxTaux = norme_haut_bas;
        Decisions directionMax;

        if ((norme_gauche <= SEUIL_DECISION_BRUIT) && (norme_droite <= SEUIL_DECISION_BRUIT) && ((norme_haut_bas/2) <= SEUIL_DECISION_BRUIT)) {
            directionMax = RIEN;
        } else {
            directionMax = DEVANT;

            if (norme_haut_bas > maxTaux) {
                maxTaux = norme_haut_bas;
                directionMax = DEVANT;
            }

            if (norme_droite > maxTaux) {
                maxTaux = norme_droite;
                directionMax = DROITE;
            }

            if (norme_gauche > maxTaux) {
                maxTaux = norme_gauche;
                directionMax = GAUCHE;
            }

            if (directionMax == DEVANT) {
                float diff = norme - object.normeHisto;

                if (diff > 0)
                    directionMax = AVANT;
                else
                    directionMax = ARRIERE;
            }
         }

        object.decisions.push_back(directionMax);

        object.normeHisto = norme;
        // if (abs(taux_gauche - taux_droite) <= SEUIL_DECISION_BRUIT) {
        //     //std::cout << "walou" << std::endl;
        //     object.decisions.push_back(RIEN);
        // }
        // else if (abs(taux_gauche - taux_droite) <= SEUIL_DECISION_AVANT_ARRIERE) {
        //     //std::cout << "ça avance ou ça recule" << std::endl;
        //     if (norme_histo > 100) {
        //         //std::cout << "ça avance" << std::endl;
        //         object.decisions.push_back(AVANT);
        //     } else {
        //         //std::cout << "ça recule" << std::endl;
        //         object.decisions.push_back(ARRIERE);
        //     }
        // }
        // else if (taux_gauche > taux_droite) {
        //     //std::cout << "ça part vers la gauche" << std::endl;
        //     object.decisions.push_back(GAUCHE);
        // } else {
        //     //std::cout << "ça part vers la droite" << std::endl;
        //     object.decisions.push_back(DROITE);
        // }

        if (object.decisions.size() == MAX_FRAMES_DECISION) {
            std::array<int,NB_DECISIONS> votes{};

            int max = -1;
            Decisions finalDecision = RIEN;

            for (Decisions& decision : object.decisions) {
                votes[decision]++;

                if (votes[decision] > max) {
                    max = votes[decision];
                    finalDecision = decision;
                }
            }

            if (finalDecision == RIEN)
                std::cout << "DECISION FINAL ==> walou" << std::endl;
            else if (finalDecision == ARRIERE)
                std::cout << "DECISION FINAL ==> ARRIERE" << std::endl;
            else if (finalDecision == AVANT)
                std::cout << "DECISION FINAL ==> AVANT " << std::endl;
            else if (finalDecision == GAUCHE)
                std::cout << "DECISION FINAL ==> ça part vers la gauche" << std::endl;
            else if (finalDecision == DROITE)
                std::cout << "DECISION FINAL ==> ça part vers la droite" << std::endl;

            object.decision = finalDecision;

            object.decisions.clear();
            //object.decisions.erase(object.decisions.begin());

            float ttcFinal = object.ttc / (float) MAX_FRAMES_DECISION;

            // std::cout<< "norme ==> " << norme << std::endl;

            if ((norme_gauche > SEUIL_DANGER) && (norme_droite > SEUIL_DANGER) && (norme_haut_bas > SEUIL_DANGER) && (finalDecision != RIEN)) {
                std::cout<< "DAAAAAAAAANNNNNNNNNNNGEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;

            //     if (finalDecision == AVANT) {
            //         startBeepAsync(1);
            //     } else if (finalDecision == GAUCHE) {
            //         startBeepAsync(2);
            //     } else if (finalDecision == DROITE) {
            //         startBeepAsync(3);
            //     }
            }

            object.ttc = 0.0;
        }


        std::cout << "----------------------" << std::endl;
    }
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

    Point pointRef(frameOld.cols/2, frameOld.rows);

    std::cout << "version opencv " << CV_VERSION << std::endl;

    cv::VideoWriter virtualCam;

    std::string pipelineOut =
        "appsrc is-live=true block=true format=time ! "
        "video/x-raw,format=BGR,width=640,height=640,framerate=30/1 ! "
        "videoconvert ! "
        "video/x-raw,format=YUY2,width=640,height=640,framerate=30/1 ! "
        "v4l2sink device=/dev/video1 sync=false";

    virtualCam.open(pipelineOut, cv::CAP_GSTREAMER, 0, 30, cv::Size(640, 640), true);

    if (!virtualCam.isOpened())
    {
        std::cerr << "Impossible d'ouvrir la camera virtuelle /dev/video1" << std::endl;
        return -1;
    }

    int id = 0;

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

        virtualCam.write(frame);
        
        Mat deplacement;
        Mat matOpticalFlow;
        Mat filtredFlow;

        auto tFlowStart = Clock::now();
        std::thread threadFlow([&]() {
            deplacement = opticalFlow.exec(frame, frameOld); 
            matOpticalFlow = drawOpticalFlow(deplacement);
            filtredFlow = drawOpticalFlowFiltered(deplacement);
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

        decisionMaking(objectsNew, pointRef);

        auto tDrawStart = Clock::now();
        Mat meanFlowDraw = drawMeanFlow(frame, objectsNew);
        Mat sparseFlowDraw = drawSparseFlow(deplacement, objectsNew);
        auto tBeforeYoloDraw = Clock::now();

        double globalMsTemp = elapsedMs(tGlobalStart, tBeforeYoloDraw);
        double fps = 1000.0 / std::max(globalMsTemp, 1.0);

        Mat yoloDraw = drawTrackingYolo(frame, objectsNew, fps);

        drawHistogram(objectsNew);
        auto tDrawEnd = Clock::now();
        
        auto tImshowStart = Clock::now();
        cv::imshow("yolo", yoloDraw);
        cv::imshow("flux optique", matOpticalFlow);
        cv::imshow("flux optique filtrés", filtredFlow);
        cv::imshow("sparse", sparseFlowDraw);
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

        // std::cout << "\n========== PROFILING ==========\n";
        // std::cout << "GLOBAL        : " << globalMs << " ms | FPS: " << 1000.0 / globalMs << "\n";
        // std::cout << "Optical flow et YOLO et draw flow : " << flowMs << " ms\n";
        // std::cout << "Mapping       : " << mappingMs << " ms\n";
        // std::cout << "Tracker       : " << trackerMs << " ms\n";
        // std::cout << "Draw all      : " << drawMs << " ms\n";
        // std::cout << "Imshow        : " << imshowMs << " ms\n";
        // std::cout << "Copy frame    : " << copyMs << " ms\n";
        // std::cout << "===============================\n";

        id++;
        // cv::imwrite("opticalFlow/yolo_" + std::to_string(id) + ".png", yoloDraw);
        // cv::imwrite("opticalFlow/frame_" + std::to_string(id) + ".png", frame);
        // cv::imwrite("opticalFlow/opticalFlow_" + std::to_string(id) + ".png", sparseFlowDraw);

        //sleep(0.067);

        if(waitKey(1) == 27)
            break;
    }
}

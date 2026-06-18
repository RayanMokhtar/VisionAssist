#include "Yolo.h"
#include "OpticalFlow.h"

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
#define MAX_PERSISTANCE 20
#define PI 3.14159265

#define SEUIL_FILTRAGE 1.2
#define SEUIL_DECISION_BRUIT 15
#define MAX_FRAMES_DECISION 5
#define SEUIL_DANGER 120


#define TAILLE_WINDOW 450

#define RESET   "\033[0m"
#define GRIS    "\033[90m"
#define ROUGE   "\033[91m"
#define VERT    "\033[92m"
#define JAUNE   "\033[93m"
#define BLEU    "\033[94m"
#define MAGENTA "\033[95m"
#define CYAN    "\033[96m"
#define BG_ROUGE "\033[41m\033[97m"

enum Decisions {
    RIEN,
    DEVANT,
    AVANT, 
    ARRIERE,
    GAUCHE,
    DROITE,
    NB_DECISIONS
};

struct ObjectDetected {
    int id;
    int classYolo;
    float confidence;
    cv::Rect box;
    Vec2f vect;
    Point centreGravity;
    cv::Scalar color{};
    std::string className{};
    int counterDetectionFailed = 0;
    std::array<int,360> histo;
    std::vector<Decisions> decisions;
    Decisions decision = RIEN;
    float normeHisto = 0.0;
};

int globalId = 0;

bool isDynamicClass(const std::string& className) {
    static const std::set<std::string> dynamicClasses = {
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "sports ball",
        "kite",
        "skateboard",
        "surfboard"
    };

    return dynamicClasses.count(className) > 0;
}

std::string decisionColor(Decisions decision)
{
    switch (decision)
    {
        case RIEN:
            return RESET;

        case AVANT:
            return ROUGE;

        case ARRIERE:
            return VERT;

        case GAUCHE:
            return JAUNE;

        case DROITE:
            return BLEU;

        default:
            return RESET;
    }
}

std::string decisionText(Decisions decision)
{
    switch (decision)
    {
        case RIEN:
            return "[ ---------- ] Rien";

        case AVANT:
            return "[ ^^^^^^^^^^ ] Rapprochement";

        case ARRIERE:
            return "[ vvvvvvvvvv ] Eloignement";

        case GAUCHE:
            return "[ <<<<<<<<<< ] Gauche";

        case DROITE:
            return "[ >>>>>>>>>> ] Droite";

        default:
            return "[ ?????????? ] INCONNU";
    }
}

void tracker(std::vector<ObjectDetected>& objectsNew, const std::vector<ObjectDetected>& objectsCopy) {
    if (objectsCopy.empty())
        return;

    std::set<int> usedIds;

    for (ObjectDetected& objectNew : objectsNew) {
        float minDist = 1000;
        const ObjectDetected* objectOldRef = nullptr;

        for (const ObjectDetected& objectOld : objectsCopy) {
            if(usedIds.count(objectOld.id)) {
                continue;
            }

            if (objectOld.classYolo != objectNew.classYolo) {
               continue;
            }

            Point pOld(objectOld.centreGravity.x + objectOld.vect[0], objectOld.centreGravity.y + objectOld.vect[1]);
            Point pNew = objectNew.centreGravity;

            float dist = sqrt(((pNew.x  - pOld.x) * (pNew.x  - pOld.x)) + ((pNew.y  - pOld.y) * (pNew.y  - pOld.y)));
            
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
            objectNew.normeHisto = objectOldRef->normeHisto;
            objectNew.decisions = objectOldRef->decisions;
        }
    }
}

std::vector<ObjectDetected> mapping(const cv::Mat& deplacement, const std::vector<YOLO::Detection>& yoloDetection) {
    std::vector<ObjectDetected> objectsNew;

    for (const YOLO::Detection& detect : yoloDetection) {

        float sum_u = 0, sum_v = 0;
        int count = 0;
        int compteurVecteursFiltresParSeuil = 0; 
        std::array<int,360> histo{};

        for (int j = detect.box.y; j < detect.box.y + detect.box.height; j++) {
            for (int i = detect.box.x; i < detect.box.x + detect.box.width; i++) {
                // TODO: Yolo out of box
                if (j >= 0 && j < deplacement.rows && i >= 0 && i < deplacement.cols) {
                    const cv::Vec2f& p = deplacement.at<cv::Vec2f>(j, i);

                    float norme = sqrt(p[0]*p[0] + p[1]*p[1]);
                    
                    if (norme < SEUIL_FILTRAGE) {
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
                }
            }
        }

        float u = sum_u / count;
        float v = sum_v / count;

        int xGravity = detect.box.x + (detect.box.width) / 2;
        int yGravity = detect.box.y + (detect.box.height) / 2;

        globalId = globalId % 100000;

        ObjectDetected object{globalId++, detect.class_id, detect.confidence, detect.box, Vec2f(u, v), Point(xGravity, yGravity), detect.color, detect.className, 0, histo};

        objectsNew.push_back(object);
    }

    return objectsNew;
}

Mat drawOpticalFlow(const Mat& matDepl) {
    cv::Mat matOpticalFlow(matDepl.rows, matDepl.cols, CV_8UC3, cv::Scalar(255, 255, 255));

    int step = 3;
    float scale = 1;

    for (int y = step; y < matDepl.rows - step; y += step) {
        for (int x = step; x < matDepl.cols - step; x += step) {
            float sum_u = 0, sum_v = 0;
            int count = 0;

            for (int j = y - step/2; j <= y + step/2; j++) {
                for (int i = x - step/2; i <= x + step/2; i++) {
                    const cv::Vec2f& p = matDepl.at<cv::Vec2f>(j, i);
                    sum_u += p[0];
                    sum_v += p[1];
                    count++;
                }
            }

            float u = sum_u / count;
            float v = sum_v / count;

            cv::Point p1(x, y);
            cv::Point p2(x + u * scale, y + v * scale);

            cv::arrowedLine(matOpticalFlow, p1, p2, cv::Scalar(0, 0, 255), 1);
        }
    }

    return matOpticalFlow;
}

Mat drawOpticalFlowFiltered(const Mat& matDepl) {
    cv::Mat matOpticalFlow(matDepl.rows, matDepl.cols, CV_8UC3, cv::Scalar(255, 255, 255));

    int step = 3;
    float scale = 1;

    for (int y = step; y < matDepl.rows - step; y += step) {
        for (int x = step; x < matDepl.cols - step; x += step) {
            float sum_u = 0, sum_v = 0;
            int count = 0;

            for (int j = y - step/2; j <= y + step/2; j++) {
                for (int i = x - step/2; i <= x + step/2; i++) {
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

            cv::Point p1(x, y);
            cv::Point p2(x + u * scale, y + v * scale);

            cv::arrowedLine(matOpticalFlow, p1, p2, cv::Scalar(0, 0, 255), 1);
        }
    }

    return matOpticalFlow;
}

Mat drawSparseFlow(const Mat& deplacement, const std::vector<ObjectDetected>& objects) {
    cv::Mat sparseFlowDraw(deplacement.rows, deplacement.cols, CV_8UC3, cv::Scalar(255, 255, 255));

    int step = 3;
    float scale = 1;

    for (const ObjectDetected& object : objects) {
        float total_u = 0.0;
        float total_v = 0.0;
        int total_count = 00;

        for (int y = (object.box.y + step); y < (object.box.y + object.box.height - step); y += step) {
            for (int x = (object.box.x + step); x < (object.box.x + object.box.width - step); x += step) {
                float sum_u = 0, sum_v = 0;
                int count = 0;

                for (int j = y - step/2; j <= y + step/2; j++) {
                    for (int i = x - step/2; i <= x + step/2; i++) {
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

                if (norme < SEUIL_FILTRAGE) {
                    continue;
                }

                cv::Point p1(x, y);
                cv::Point p2(x + u * scale, y + v * scale);

                cv::arrowedLine(sparseFlowDraw, p1, p2, cv::Scalar(0, 0, 255), 1);
            }
        }
    }

    return sparseFlowDraw;
}

Mat drawMeanFlow(const Mat& frame, const std::vector<ObjectDetected>& objects) {
    cv::Mat meanFlowDraw(frame.rows, frame.cols, CV_8UC3, cv::Scalar(255, 255, 255));

    float scale = 20;

    for (const ObjectDetected& object : objects) {
        cv::Point p1(object.centreGravity.x, object.centreGravity.y);
        cv::Point p2(object.centreGravity.x + object.vect[0] * scale, object.centreGravity.y + object.vect[1] * scale);

        cv::arrowedLine(meanFlowDraw, p1, p2, cv::Scalar(0, 0, 255), 2);
    }

    return meanFlowDraw;
}

Mat drawTrackingYolo(const Mat& frame, const std::vector<ObjectDetected>& objects, double fps) {
    Mat yoloDraw;
    frame.copyTo(yoloDraw);
    float scale = 20;

    int size = objects.size();
    for (int i = 0; i < size; ++i) {
        ObjectDetected object = objects[i];

        cv::Rect box = object.box;
        cv::Scalar color = object.color;

        cv::rectangle(yoloDraw, box, color, 2);

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

void drawHistogram(const std::vector<ObjectDetected>& objects) {
    int width = TAILLE_WINDOW*2;
    int height = TAILLE_WINDOW;
    int margin = 50;

    for (const ObjectDetected& object : objects) {
        if (!isDynamicClass(object.className))
            continue;

        cv::Mat img(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

        int maxVal = 400;

        float binWidth = (float)(width - 2 * margin) / 360.0f;

        for (int k = 0; k <= 4; k++) {
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
        float taux_null_droite = 0.0;
        float taux_null_gauche = 0.0;

        for (int i = 0; i < 360; i++) {
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
                norme_haut_bas += object.histo[i];
            } else if (i > 135 && i <= 225) {
                norme_gauche += object.histo[i];
            } else {
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
        taux_null_droite /= norme;
        taux_null_gauche /= norme;
        norme /= 360.0;

        taux_null_droite *= 100;
        taux_null_gauche *= 100;

        for (int angle = 0; angle <= 360; angle += 30) {
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

        for (int angle : {0, 90, 180, 270}) {
            int x = margin + (int)(angle * binWidth);

            cv::line(img,
                     cv::Point(x, margin),
                     cv::Point(x, height - margin),
                     cv::Scalar(0, 0, 255),
                     1);
        }

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

        cv::putText(img,
                    "Histogramme - " + object.className + " Norme: " + std::to_string(norme)  + " Norme haut bas: " + std::to_string(norme_haut_bas) 
                    + " Norme droite: " + std::to_string(norme_droite) + " Norme gauche: " + std::to_string(norme_gauche),
                    cv::Point(20, 30),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    cv::Scalar(0, 100, 0),
                    2);

        cv::putText(img,
                    "Angle (degres)",
                    cv::Point(width / 2 - 80, height - 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(255, 0, 0),
                    2);

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

std::vector<ObjectDetected> persistanceBetweenFrame(const std::vector<ObjectDetected> objectsNew, std::vector<ObjectDetected> objectsCopy) {
    std::vector<ObjectDetected> objectsPersistant;
    objectsPersistant = objectsNew;

    for (ObjectDetected& objectCopy : objectsCopy) {
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
            objectsPersistant.push_back(objectCopy);
        }        
    }

    return objectsPersistant;
}

void beep(int secondes) {
    gpiod_chip* chip = gpiod_chip_open_by_name("gpiochip0");
    gpiod_line* line = gpiod_chip_get_line(chip, 144);

    gpiod_line_request_output(line, "buzzer", 0);

    gpiod_line_set_value(line, 1);

    std::this_thread::sleep_for(std::chrono::seconds(secondes));

    gpiod_line_set_value(line, 0);

    gpiod_line_release(line);
    gpiod_chip_close(chip);
}

std::atomic<bool> beepRunning(false);

void startBeepAsync(int nbBeep) {
    if (beepRunning.exchange(true))
        return;

    std::thread([nbBeep]() {
        for (int i = 0; i < nbBeep; i++) {
            beep(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
        }

        beepRunning = false;
    }).detach();
}

void decisionMaking(std::vector<ObjectDetected>& objectsNew) {
    std::vector<ObjectDetected> decisions;

    for (ObjectDetected& object : objectsNew) {   
        if (!isDynamicClass(object.className))
            continue;

        float somme =  0.0;
        float norme_haut_bas = 0.0;
        float norme_droite = 0.0;
        float norme_gauche = 0.0;
        float taux_null_droite = 0.0;
        float taux_null_gauche = 0.0;

        for (int i = 0; i < 360; i++) {
            somme += object.histo[i];

            if ((i > 45 && i <= 135) || (i > 225 && i <= 315)) {
                norme_haut_bas += object.histo[i];
            } else if (i > 135 && i <= 225) {
                norme_gauche += object.histo[i];
            } else {
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
        taux_null_droite /= somme;
        taux_null_gauche /= somme;
        float norme = somme / 360.0;

        taux_null_droite *= 100;
        taux_null_gauche *= 100;

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

            auto now = std::chrono::system_clock::now();
            std::time_t t = std::chrono::system_clock::to_time_t(now);
            auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()) % 100; 
            std::cout << std::put_time(std::localtime(&t), "%H:%M:%S") << ':' << std::setfill('0') << std::setw(3) << nanoseconds.count() << decisionColor(finalDecision) << " | DECISION FINALE ==> " << "classe: " << object.className << " | id: " << object.id << " | " << decisionText(finalDecision) << RESET << std::endl;

            object.decision = finalDecision;

            object.decisions.clear();

            if ((norme_gauche > SEUIL_DANGER) && (norme_droite > SEUIL_DANGER) && (norme_haut_bas > SEUIL_DANGER) && (finalDecision != RIEN)) {

                now = std::chrono::system_clock::now();
                t = std::chrono::system_clock::to_time_t(now);
                nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()) % 100; 
                std::cout << std::put_time(std::localtime(&t), "%H:%M:%S") << ':' << std::setfill('0') << std::setw(3) << nanoseconds.count() << BG_ROUGE << " | !!! DAAAAAAAAAAAANNNNNNNNNNNGGGGGGGGGGGGEEEEEEEEEEEERRRRRRRRRRRR !!! " << RESET << std::endl;

                if (finalDecision == AVANT) {
                    startBeepAsync(1);
                } else if (finalDecision == GAUCHE) {
                    startBeepAsync(2);
                } else if (finalDecision == DROITE) {
                    startBeepAsync(3);
                }
            }
        }
    }
}

using Clock = std::chrono::high_resolution_clock;

static double elapsedMs(
    const Clock::time_point& start,
    const Clock::time_point& end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main(int argc, char** argv) {
    OpticalFlow opticalFlow;
    YOLO yolo("../YoloUtils/yolov8n.onnx", cv::Size(640, 640), "classes.txt", true);
    Mat frameOld;
    Mat frame;

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
        } else if (FALLBACK_AVEC_CHEMIN_VIDEO) {
            std::cerr << "Webcam fallback erreur, test avec vidéo\n";
            cap.open(FALLBACK_AVEC_CHEMIN_VIDEO);
            if (!cap.isOpened()) {
                std::cerr << "Video fallback erreur\n";
                return -1;
            }
        } else {
            std::cerr << "pas de fallback actif , vérifier chemin vers video ou potentiels problemes avec l'environnement.\n";
            return -1;
        }
    }

    cv::VideoWriter virtualCam;

    std::string pipelineOut =
        "appsrc is-live=true block=true format=time ! "
        "video/x-raw,format=BGR,width=640,height=640,framerate=30/1 ! "
        "videoconvert ! "
        "video/x-raw,format=YUY2,width=640,height=640,framerate=30/1 ! "
        "v4l2sink device=/dev/video1 sync=false";

    virtualCam.open(pipelineOut, cv::CAP_GSTREAMER, 0, 30, cv::Size(640, 640), true);

    if (!virtualCam.isOpened()) {
        std::cerr << "Impossible d'ouvrir la camera virtuelle /dev/video1" << std::endl;
        return -1;
    }

    cap >> frameOld;

    if(frameOld.empty()) {
        std::cerr << "Impossible de lire la premiere frame" << std::endl;
        return -1;
    }

    cv::cvtColor(frameOld, frameOld, cv::COLOR_BGRA2BGR);

    std::vector<ObjectDetected> objects;
    std::vector<ObjectDetected> objectsNew;

    std::cout << "version opencv " << CV_VERSION << std::endl;

    cv::namedWindow("yolo");
    cv::namedWindow("flux optique global");
    cv::namedWindow("flux optique filtrés");
    cv::namedWindow("flux optique boite englobante");
    cv::namedWindow("histogramme");

    cv::moveWindow("yolo", 0, 0);
    cv::moveWindow("flux optique global", 525, 0);
    cv::moveWindow("flux optique filtrés", 980, 0);
    cv::moveWindow("flux optique boite englobante", 1435, 0);
    cv::moveWindow("histogramme", 980, 520);

    int frameId = 0;

    for(;;)
    {
        auto tGlobalStart = Clock::now();
        
        cap >> frame;
        
        if(frame.empty()) {
            std::cerr << "Frame vide" << std::endl;
            break;
        }
        
        cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);

        virtualCam.write(frame);
        
        Mat deplacement;
        Mat matOpticalFlow;
        Mat filtredFlow;

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

        objectsNew = mapping(deplacement, yoloDetection);

        tracker(objectsNew, objects);

        decisionMaking(objectsNew);

        // Mat meanFlowDraw = drawMeanFlow(frame, objectsNew);
        Mat sparseFlowDraw = drawSparseFlow(deplacement, objectsNew);
        auto tBeforeYoloDraw = Clock::now();

        drawHistogram(objectsNew);

        double globalMsTemp = elapsedMs(tGlobalStart, tBeforeYoloDraw);
        double fps = 1000.0 / std::max(globalMsTemp, 1.0);

        Mat yoloDraw = drawTrackingYolo(frame, objectsNew, fps);

        std::cout << "Frame: " << frameId << std::endl;

        frameId++;

        Mat yoloDisplay;
        Mat opticalFlowDisplay;
        Mat filtredFlowDisplay;
        Mat sparseFlowDisplay;
        
        cv::resize(yoloDraw, yoloDisplay, cv::Size(TAILLE_WINDOW, TAILLE_WINDOW));
        cv::resize(matOpticalFlow, opticalFlowDisplay, cv::Size(TAILLE_WINDOW, TAILLE_WINDOW));
        cv::resize(filtredFlow, filtredFlowDisplay, cv::Size(TAILLE_WINDOW, TAILLE_WINDOW));
        cv::resize(sparseFlowDraw, sparseFlowDisplay, cv::Size(TAILLE_WINDOW, TAILLE_WINDOW));

        cv::imshow("yolo", yoloDisplay);
        cv::imshow("flux optique global", opticalFlowDisplay);
        cv::imshow("flux optique filtrés", filtredFlowDisplay);
        cv::imshow("flux optique boite englobante", sparseFlowDisplay);
        // cv::imshow("moyenne du flux optique sur la bbox", meanFlowDraw);

        frame.copyTo(frameOld);

        if(waitKey(1) == 27)
            break;
    }
}

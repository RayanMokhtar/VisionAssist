#include "main.cpp"
#include "opencv2/opencv.hpp"

#include <sys/stat.h>

using namespace cv;

#define MAX_U 640
#define MAX_V 640
#define STEP 1.5
#define ACCEL_MAX 6.0
#define ACCEL_STEP 0.1
#define NB_FRAME 200

struct VoteEntry
{
    int idxU;
    int idxV;
    float u;
    float v;
    float count;
};

Vec2f maxVectorFlow(const Mat& matDepl)
{
    float max_u = -1000;
    float max_v = -1000;

    for (int y = 0; y < matDepl.rows; y += 1)
    {
        for (int x = 0; x < matDepl.cols; x += 1)
        {
                const cv::Vec2f& p = matDepl.at<cv::Vec2f>(y, x);
                float u = p[0];
                float v = p[1];
            
                if (u > max_u)
                    max_u = u;
                
                if (v > max_v)
                    max_v = v;
        }
    }

    return Vec2f(max_u, max_v);
}

bool saveOneVoteMatToImage(const std::string& filename,
                           const cv::Mat& voteMat,
                           int accelIndex)
{
    std::vector<float> counts;

    // 1) Récupérer tous les votes non nuls, ligne par ligne
    for (int y = 0; y < voteMat.rows; y++)
    {
        for (int x = 0; x < voteMat.cols; x++)
        {
            float count = voteMat.at<float>(y, x);

            if (count <= 0.0f)
                continue;

            counts.push_back(count);
        }
    }

    if (counts.empty())
    {
        std::cerr << "Aucun vote pour accel index " << accelIndex << std::endl;
        return false;
    }

    double maxCount = *std::max_element(counts.begin(), counts.end());

    if (maxCount <= 0.0)
        return false;

    int nbBars = (int)counts.size();

    int barWidth = 2;
    int gap = 1;
    int chartHeight = 500;
    int marginLeft = 90;
    int marginRight = 40;
    int marginTop = 80;
    int marginBottom = 90;

    int chartWidth = nbBars * (barWidth + gap);

    // éviter une image énorme si beaucoup de votes
    if (chartWidth < 1000)
        chartWidth = 1000;

    int canvasWidth = marginLeft + chartWidth + marginRight;
    int canvasHeight = marginTop + chartHeight + marginBottom;

    cv::Mat canvas(canvasHeight,
                   canvasWidth,
                   CV_8UC3,
                   cv::Scalar(255, 255, 255));

    double accelNorm = accelIndex * ACCEL_STEP;

    std::ostringstream title;
    title << "Histogramme count par index - acceleration = "
          << std::fixed << std::setprecision(2)
          << accelNorm << " m/s2";

    cv::putText(canvas,
                title.str(),
                cv::Point(40, 40),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(0, 0, 0),
                2);

    cv::Point origin(marginLeft, marginTop + chartHeight);

    // Axe X
    cv::line(canvas,
             origin,
             cv::Point(marginLeft + chartWidth, marginTop + chartHeight),
             cv::Scalar(0, 0, 0),
             2);

    // Axe Y
    cv::line(canvas,
             origin,
             cv::Point(marginLeft, marginTop),
             cv::Scalar(0, 0, 0),
             2);

    // Graduations Y
    int nbTicksY = 5;

    for (int i = 0; i <= nbTicksY; i++)
    {
        float ratio = (float)i / nbTicksY;
        int y = marginTop + chartHeight - (int)(ratio * chartHeight);
        float tickValue = ratio * 200;

        cv::line(canvas,
                 cv::Point(marginLeft - 5, y),
                 cv::Point(marginLeft, y),
                 cv::Scalar(0, 0, 0),
                 1);

        std::ostringstream tickText;
        tickText << std::fixed << std::setprecision(0) << tickValue;

        cv::putText(canvas,
                    tickText.str(),
                    cv::Point(15, y + 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.45,
                    cv::Scalar(0, 0, 0),
                    1);
    }

    cv::putText(canvas,
                "count",
                cv::Point(20, marginTop - 15),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(0, 0, 0),
                2);

    // 2) Dessiner les barres : X = index 1..N, Y = count
    for (int i = 0; i < nbBars; i++)
    {
        float count = counts[i];

        int x0 = marginLeft + i * (barWidth + gap);
        int barH = (int)((count / maxCount) * chartHeight);
        int y0 = marginTop + chartHeight - barH;

        cv::rectangle(canvas,
                      cv::Point(x0, y0),
                      cv::Point(x0 + barWidth, marginTop + chartHeight),
                      cv::Scalar(180, 80, 30),
                      cv::FILLED);
    }

    // Graduations X
    int nbTicksX = 10;

    for (int i = 0; i <= nbTicksX; i++)
    {
        float ratio = (float)i / nbTicksX;
        int index = 1 + (int)(ratio * (nbBars - 1));
        int x = marginLeft + (int)(ratio * chartWidth);

        cv::line(canvas,
                 cv::Point(x, marginTop + chartHeight),
                 cv::Point(x, marginTop + chartHeight + 5),
                 cv::Scalar(0, 0, 0),
                 1);

        std::ostringstream indexText;
        indexText << index;

        cv::putText(canvas,
                    indexText.str(),
                    cv::Point(x - 15, marginTop + chartHeight + 25),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.45,
                    cv::Scalar(0, 0, 0),
                    1);
    }

    cv::putText(canvas,
                "index du vote",
                cv::Point(marginLeft + chartWidth / 2 - 80,
                          canvasHeight - 30),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::Scalar(0, 0, 0),
                2);

    cv::putText(canvas,
                "Axe X : index 1..N    Axe Y : count",
                cv::Point(marginLeft, canvasHeight - 60),
                cv::FONT_HERSHEY_SIMPLEX,
                0.55,
                cv::Scalar(0, 0, 0),
                1);

    return cv::imwrite(filename, canvas);
}

bool saveAllVoteMatsToImages(const std::string& folder, const std::vector<cv::Mat>& matVotesAccel)
{
    mkdir(folder.c_str(), 0777);

    bool allOk = true;

    for (size_t i = 0; i < matVotesAccel.size(); i++)
    {
        std::ostringstream filename;

        filename << folder << "/vote_accel_"
                 << std::setw(3) << std::setfill('0') << i
                 << ".png";

        bool ok = saveOneVoteMatToImage(filename.str(),
                                        matVotesAccel[i],
                                        (int)i);

        if (!ok)
        {
            allOk = false;
        }
    }

    return allOk;
}

bool saveLUTToJSON(const std::string& filename, const std::vector<cv::Vec2f>& lut)
{
    std::ofstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Impossible d'ouvrir le fichier JSON: " << filename << std::endl;
        return false;
    }

    file << std::fixed << std::setprecision(4);

    file << "{\n";
    file << "  \"metadata\": {\n";
    file << "    \"max_u\": " << MAX_U << ",\n";
    file << "    \"max_v\": " << MAX_V << ",\n";
    file << "    \"step\": " << STEP << ",\n";
    file << "    \"accel_max\": " << ACCEL_MAX << ",\n";
    file << "    \"accel_step\": " << ACCEL_STEP << ",\n";
    file << "    \"size\": " << lut.size() << "\n";
    file << "  },\n";

    file << "  \"lut\": [\n";

    for (size_t i = 0; i < lut.size(); i++)
    {
        double accelNorm = i * ACCEL_STEP;

        file << "    {\n";
        file << "      \"index\": " << i << ",\n";
        file << "      \"accel_norm\": " << accelNorm << ",\n";
        file << "      \"u\": " << lut[i][0] << ",\n";
        file << "      \"v\": " << lut[i][1] << "\n";
        file << "    }";

        if (i + 1 < lut.size())
            file << ",";

        file << "\n";
    }

    file << "  ]\n";
    file << "}\n";

    file.close();
    return true;
}

bool saveOneVoteMatToJSON(const std::string& filename, const cv::Mat& voteMat, int accelIndex)
{
    std::ofstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Impossible d'ouvrir le fichier JSON: " << filename << std::endl;
        return false;
    }

    double accelNorm = accelIndex * ACCEL_STEP;

    file << std::fixed << std::setprecision(4);

    file << "{\n";

    file << "  \"metadata\": {\n";
    file << "    \"accel_index\": " << accelIndex << ",\n";
    file << "    \"accel_norm\": " << accelNorm << ",\n";
    file << "    \"u_min\": " << -MAX_U << ",\n";
    file << "    \"u_max\": " <<  MAX_U << ",\n";
    file << "    \"v_min\": " << -MAX_V << ",\n";
    file << "    \"v_max\": " <<  MAX_V << ",\n";
    file << "    \"step\": " << STEP << ",\n";
    file << "    \"rows\": " << voteMat.rows << ",\n";
    file << "    \"cols\": " << voteMat.cols << "\n";
    file << "  },\n";

    file << "  \"votes\": [\n";

    bool firstVote = true;

    for (int y = 0; y < voteMat.rows; y++)
    {
        for (int x = 0; x < voteMat.cols; x++)
        {
            float count = voteMat.at<float>(y, x);

            if (count <= 0.0f)
                continue;

            float u = (x - (int)(MAX_U / STEP)) * STEP;
            float v = (y - (int)(MAX_V / STEP)) * STEP;

            if (!firstVote)
                file << ",\n";

            file << "    {";
            file << "\"idx_u\": " << x << ", ";
            file << "\"idx_v\": " << y << ", ";
            file << "\"u\": " << u << ", ";
            file << "\"v\": " << v << ", ";
            file << "\"count\": " << count;
            file << "}";

            firstVote = false;
        }
    }

    file << "\n";
    file << "  ]\n";
    file << "}\n";

    file.close();
    return true;
}

bool saveAllVoteMatsToJSON(const std::string& folder, const std::vector<cv::Mat>& matVotesAccel)
{
    mkdir(folder.c_str(), 0777);

    bool allOk = true;

    for (size_t i = 0; i < matVotesAccel.size(); i++)
    {
        std::ostringstream filename;

        filename << folder << "/vote_accel_"
                 << std::setw(3) << std::setfill('0') << i
                 << ".json";

        bool ok = saveOneVoteMatToJSON(filename.str(), matVotesAccel[i], (int)i);

        if (!ok)
        {
            allOk = false;
        }
    }

    return allOk;
}
cv::Vec2f getMedianVote(const cv::Mat& voteMat)
{
    std::vector<VoteEntry> votes;

    for (int y = 0; y < voteMat.rows; y++)
    {
        for (int x = 0; x < voteMat.cols; x++)
        {
            float count = voteMat.at<float>(y, x);

            if (count <= 0.0f)
                continue;

            float u = (x - (int)(MAX_U / STEP)) * STEP;
            float v = (y - (int)(MAX_V / STEP)) * STEP;

            VoteEntry entry;
            entry.idxU = x;
            entry.idxV = y;
            entry.u = u;
            entry.v = v;
            entry.count = count;

            votes.push_back(entry);
        }
    }

    if (votes.empty())
    {
        return cv::Vec2f(0.0f, 0.0f);
    }

    std::sort(votes.begin(), votes.end(),
              [](const VoteEntry& a, const VoteEntry& b)
              {
                  return a.count < b.count;
              });

    size_t middle = votes.size() / 2;

    return cv::Vec2f(votes[middle].u, votes[middle].v);
}

void vote(std::vector<cv::Mat>& matVotesAccel, const cv::Mat& matDepl, double accelNorm)
{
    int accelIdx = (int)std::round(accelNorm / ACCEL_STEP); 

    if (accelIdx >= (int)matVotesAccel.size())
        return;

    cv::Mat& voteMat = matVotesAccel[accelIdx];

    int step = 3;

    for (int y = step; y < matDepl.rows - step; y += step)
    {
        for (int x = step; x < matDepl.cols - step; x += step)
        {
            float sum_u = 0, sum_v = 0;
            int count = 0;

            for (int j = y - step/2; j <= y + step/2; j++)
            {
fi
                cv::Vec2f p = matDepl.at<cv::Vec2f>(y, x);

                // float u = p[0];
                // float v = p[1];

                sum_u += p[0];
                sum_v += p[1];
                count++;
            }
            
            float u = sum_u / count;
            float v = sum_v / count;

            if (abs(u) < 100.0 || abs(v) < 100.0)
                 continue;    

            // std::cout << "u : " << u << "v : " << v << std::endl;
    
            int idxU = (int)std::round(u / STEP) + (int)(MAX_U / STEP);
            int idxV = (int)std::round(v / STEP) + (int)(MAX_V / STEP);
    
            if (idxU < 0 || idxU >= voteMat.cols || idxV < 0 || idxV >= voteMat.rows)
                continue;
            
            voteMat.at<float>(idxV, idxU) += 1.0f;
        }
    }
}

void buildLUT(const std::vector<cv::Mat>& matVotesAccel, std::vector<Vec2f>& lut)
{
    for (size_t i = 0; i < matVotesAccel.size(); i++)
    {
        const cv::Mat& voteMat = matVotesAccel[i];

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;

        cv::minMaxLoc(voteMat, &minVal, &maxVal, &minLoc, &maxLoc);

        if (maxVal <= 0.0)
        {
            lut[i] = cv::Vec2f(0.0f, 0.0f);
            continue;
        }

        int idxU = maxLoc.x;
        int idxV = maxLoc.y;

        float u = (idxU - (int)(MAX_U / STEP)) * STEP;
        float v = (idxV - (int)(MAX_V / STEP)) * STEP;

        lut[i] = cv::Vec2f(u, v);

        // cv::Vec2f medianFlow = getMedianVote(voteMat);

        // lut[i] = medianFlow;
    }
}

Mat yoloMat(const Mat& deplacement, const std::vector<YOLO::Detection>& yoloDetection)
{
    cv::Mat yoloFlow(deplacement.size(), CV_32FC2, cv::Scalar(0, 0));

    for (const YOLO::Detection& detect : yoloDetection)
    {
        for (int j = detect.box.y; j < detect.box.y + detect.box.height; j++)
        {
            for (int i = detect.box.x; i < detect.box.x + detect.box.width; i++)
            {
                if (j >= 0 && j < deplacement.rows && i >= 0 && i < deplacement.cols)
                {
                    const cv::Vec2f& p = deplacement.at<cv::Vec2f>(j, i);

                    yoloFlow.at<cv::Vec2f>(j, i) = p;
                }
            }
        }
    }

    return yoloFlow;
}

Mat drawYolo(const Mat& frame, const std::vector<YOLO::Detection>& yoloDetection)
{
    Mat yoloDraw;
    frame.copyTo(yoloDraw);
    float scale = 20;

    int size = yoloDetection.size();
    for (int i = 0; i < size; ++i)
    {
        YOLO::Detection object = yoloDetection[i];

        cv::Rect box = object.box;
        cv::Scalar color = object.color;

        // Detection box
        cv::rectangle(yoloDraw, box, color, 2);

        // Detection box text
        std::string classString = object.className + ' ' + std::to_string(object.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

        cv::rectangle(yoloDraw, textBox, color, cv::FILLED);
        cv::putText(yoloDraw, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }

    return yoloDraw;
}

double fakeAccel()
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dist(0.0f, 6.0f);

    // return dist(gen);
    return 0.0;
}

int main(int argc, char** argv)
{
    OpticalFlow opticalFlow;
    YOLO yolo("../YoloUtils/yolov8n.onnx", cv::Size(640, 640), "classes.txt", true);

    Mat frameOld;
    Mat frame;

    VideoCapture cap;

    std::string pipeline =
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
        "nvvidconv ! "
        "video/x-raw, width=640, height=640, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false";

    cap.open(pipeline, CAP_GSTREAMER);

    if (!cap.isOpened()) {
        std::cerr << "Pipeline GStreamer nvidia failed test avec webcam...\n";

        return -1;
    }

    cap >> frameOld;

    if(frameOld.empty())
    {
        std::cerr << "Impossible de lire la premiere frame" << std::endl;
        return -1;
    }

    cv::cvtColor(frameOld, frameOld, cv::COLOR_BGRA2BGR);

    // bool initImu;
    // IMU imu(initImu);
    // if (!initImu) {
    //     std::cerr << "Impossible de lancer l'imu\n";
    //     return -1;
    // }

    std::vector<cv::Mat> matVotesAccel((int)(ACCEL_MAX / ACCEL_STEP) + 1);

    for (int i = 0; i < matVotesAccel.size(); i++)
    {
        matVotesAccel[i] = cv::Mat((int)(MAX_V / STEP)*2 + 1, (int)(MAX_U / STEP)*2 + 1, CV_32FC1, cv::Scalar(0));
    }

    std::vector<Vec2f> lut((int)(ACCEL_MAX / ACCEL_STEP) + 1);

    int count = 0;

    for(;;)
    {
        cap >> frame;
        if(frame.empty())
        {
            std::cerr << "Frame vide" << std::endl;
            break;
        }

        cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);

        Mat deplacement = opticalFlow.exec(frame, frameOld); 
        //Vec2f vect = maxVectorFlow(deplacement);
        //std::cout << "max u: " << vect[0] << " max v: " << vect[1] << std::endl;
        //std::vector<YOLO::Detection> yoloDetection = yolo.exec(frame);
        
        //Mat yoloFlow = yoloMat(deplacement, yoloDetection);

        Mat matOpticalFlow = drawOpticalFlow(deplacement);

        //Mat yoloDraw = drawYolo(frame, yoloDetection);

        //IMU::Vec3 accel = imu.readAccel();
        //double accelNorm = std::sqrt(accel.x * accel.x + accel.y * accel.y + accel.z * accel.z);
        double accelNorm = fakeAccel();
        std::cout << "Acceleration = " << accelNorm << " m/s2" << std::endl;
        std::cout << count << std::endl;

        vote(matVotesAccel, deplacement, accelNorm);
        
        //cv::imshow("frame", frame);
        cv::imshow("frame flux optique", matOpticalFlow);
        //cv::imshow("yolo", yoloDraw);

        frame.copyTo(frameOld);

        count++;

        if (count == NB_FRAME)
        {
            count = 0;

            buildLUT(matVotesAccel, lut);
        
            if (saveAllVoteMatsToJSON("votes_matrices", matVotesAccel))
            {
                std::cout << "Toutes les matrices de vote sont sauvegardees." << std::endl;
            }

            if (saveAllVoteMatsToImages("votes_matrices", matVotesAccel))
            {
                std::cout << "Toutes les images des votes sont sauvegardees." << std::endl;
            }


            if (saveLUTToJSON("lut.json", lut))
            {
                std::cout << "LUT sauvegardee dans lut.json" << std::endl;
                break;
            }


            lut.clear();
            lut.reserve(matVotesAccel.size());

            for (size_t i = 0; i < matVotesAccel.size(); i++)
            {
                matVotesAccel[i].setTo(cv::Scalar(0));
            }
        }

        if(waitKey(1) == 27)
            break;
    }
}

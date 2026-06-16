#include "main.cpp"
#include "opencv2/opencv.hpp"

#include <sys/stat.h>

using namespace cv;

#define MAX_U 640
#define MAX_V 640
#define STEP 0.5
#define ACCEL_MAX 6.0
#define ACCEL_STEP 0.1
#define NB_FRAME 100

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

bool drawFlowVsAcceleration(const std::vector<cv::Vec2f>& lut,
                            const std::string& filename)
{
    if (lut.empty())
    {
        std::cerr << "LUT vide, impossible de dessiner la courbe." << std::endl;
        return false;
    }

    const int width = 1000;
    const int height = 700;

    const int leftMargin   = 100;
    const int rightMargin  = 40;
    const int topMargin    = 60;
    const int bottomMargin = 100;

    cv::Mat graph(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    // Zone utile
    int plotWidth  = width - leftMargin - rightMargin;
    int plotHeight = height - topMargin - bottomMargin;

    // Calcul des valeurs Y = norme du flux
    std::vector<double> flowNorms;
    flowNorms.reserve(lut.size());

    double maxFlow = 0.0;
    for (size_t i = 0; i < lut.size(); i++)
    {
        double u = lut[i][0];
        double v = lut[i][1];
        double norm = std::sqrt(u * u + v * v);

        flowNorms.push_back(norm);

        if (norm > maxFlow)
            maxFlow = norm;
    }

    if (maxFlow <= 0.0)
        maxFlow = 1.0;

    double maxAccel = (lut.size() - 1) * ACCEL_STEP;
    if (maxAccel <= 0.0)
        maxAccel = 1.0;

    // Axes
    cv::line(graph,
             cv::Point(leftMargin, height - bottomMargin),
             cv::Point(width - rightMargin, height - bottomMargin),
             cv::Scalar(0, 0, 0), 2);

    cv::line(graph,
             cv::Point(leftMargin, topMargin),
             cv::Point(leftMargin, height - bottomMargin),
             cv::Scalar(0, 0, 0), 2);

    // Titre
    cv::putText(graph,
                "Flux optique en fonction de l'acceleration",
                cv::Point(120, 30),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(0, 0, 0),
                2);

    // Labels axes
    cv::putText(graph,
                "Acceleration (m/s^2)",
                cv::Point(width / 2 - 100, height - 30),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(0, 0, 0),
                2);

    cv::putText(graph,
                "Norme flux optique",
                cv::Point(20, topMargin - 20),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(0, 0, 0),
                2);

    // Graduations X
    int nbTicksX = 6;
    for (int i = 0; i <= nbTicksX; i++)
    {
        double accel = i * maxAccel / nbTicksX;

        int x = leftMargin + static_cast<int>((accel / maxAccel) * plotWidth);
        int y = height - bottomMargin;

        cv::line(graph, cv::Point(x, y), cv::Point(x, y + 8), cv::Scalar(0, 0, 0), 2);

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << accel;

        cv::putText(graph,
                    oss.str(),
                    cv::Point(x - 15, y + 30),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0, 0, 0),
                    1);
    }

    // Graduations Y
    int nbTicksY = 6;
    for (int i = 0; i <= nbTicksY; i++)
    {
        double flow = i * maxFlow / nbTicksY;

        int x = leftMargin;
        int y = height - bottomMargin - static_cast<int>((flow / maxFlow) * plotHeight);

        cv::line(graph, cv::Point(x - 8, y), cv::Point(x, y), cv::Scalar(0, 0, 0), 2);

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << flow;

        cv::putText(graph,
                    oss.str(),
                    cv::Point(20, y + 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0, 0, 0),
                    1);
    }

    // Tracé de la courbe
    for (size_t i = 1; i < flowNorms.size(); i++)
    {
        double accel1 = (i - 1) * ACCEL_STEP;
        double accel2 = i * ACCEL_STEP;

        double flow1 = flowNorms[i - 1];
        double flow2 = flowNorms[i];

        int x1 = leftMargin + static_cast<int>((accel1 / maxAccel) * plotWidth);
        int y1 = height - bottomMargin - static_cast<int>((flow1 / maxFlow) * plotHeight);

        int x2 = leftMargin + static_cast<int>((accel2 / maxAccel) * plotWidth);
        int y2 = height - bottomMargin - static_cast<int>((flow2 / maxFlow) * plotHeight);

        cv::line(graph, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 2);
        cv::circle(graph, cv::Point(x1, y1), 3, cv::Scalar(0, 0, 255), -1);
    }

    // Dernier point
    {
        size_t i = flowNorms.size() - 1;
        double accel = i * ACCEL_STEP;
        double flow  = flowNorms[i];

        int x = leftMargin + static_cast<int>((accel / maxAccel) * plotWidth);
        int y = height - bottomMargin - static_cast<int>((flow / maxFlow) * plotHeight);

        cv::circle(graph, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), -1);
    }

    bool ok = cv::imwrite(filename, graph);

    if (!ok)
    {
        std::cerr << "Erreur lors de l'enregistrement de l'image: " << filename << std::endl;
        return false;
    }

    std::cout << "Courbe enregistree dans : " << filename << std::endl;
    return true;
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

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;

    cv::minMaxLoc(voteMat, &minVal, &maxVal, &minLoc, &maxLoc);
    Mat voteNomaliser((int)(MAX_V / STEP)*2 + 1, (int)(MAX_U / STEP)*2 + 1, CV_32FC1, cv::Scalar(0));

    for (int y = 0; y < voteMat.rows; y++)
    {
        for (int x = 0; x < voteMat.cols; x++)
        {
            float count = voteMat.at<float>(y, x);

            float countNormaliser = ((count - minVal) / (maxVal - minVal)) * 255.0;
            voteNomaliser.at<float>(y, x) = countNormaliser;

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

    cv::imwrite("votes/vote_" + std::to_string(accelIndex) + ".png", voteNomaliser);

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

void vote(std::vector<cv::Mat>& matVotesAccel, const cv::Mat& matDepl, double accelNorm)
{
    int accelIdx = (int)std::round(accelNorm / ACCEL_STEP); 

    if (accelIdx >= (int)matVotesAccel.size())
        return;

    cv::Mat& voteMat = matVotesAccel[accelIdx];

    // int step = 3;

    for (int y = 0; y < matDepl.rows; y += 1)
    {
        for (int x = 0; x < matDepl.cols; x += 1)
        {
            cv::Vec2f p = matDepl.at<cv::Vec2f>(y, x);

            float u = p[0];
            float v = p[1];

            // float norm = sqrt(u*u + v*v);
            // if (abs(u) < 100 || abs(v) < 100) 
            //     continue;

            int idxU = (int)std::round(u / STEP) + (int)(MAX_U / STEP);
            int idxV = (int)std::round(v / STEP) + (int)(MAX_V / STEP);

            if (idxU < 0 || idxU >= voteMat.cols || idxV < 0 || idxV >= voteMat.rows)
                continue;
            
            voteMat.at<float>(idxV, idxU) += 1.0f;
        }
    }


    // for (int y = step; y < matDepl.rows - step; y += step)
    // {
    //     for (int x = step; x < matDepl.cols - step; x += step)
    //     {
    //         float sum_u = 0, sum_v = 0;
    //         int count = 0;

    //         for (int j = y - step/2; j <= y + step/2; j++)
    //         {

    //             cv::Vec2f p = matDepl.at<cv::Vec2f>(y, x);

    //             // float u = p[0];
    //             // float v = p[1];

    //             sum_u += p[0];
    //             sum_v += p[1];
    //             count++;
    //         }
            
    //         float u = sum_u / count;
    //         float v = sum_v / count;

    //         if (abs(u) < 100.0 || abs(v) < 100.0)
    //              continue;    

    //         // std::cout << "u : " << u << "v : " << v << std::endl;
    
    //         int idxU = (int)std::round(u / STEP) + (int)(MAX_U / STEP);
    //         int idxV = (int)std::round(v / STEP) + (int)(MAX_V / STEP);
    
    //         if (idxU < 0 || idxU >= voteMat.cols || idxV < 0 || idxV >= voteMat.rows)
    //             continue;
            
    //         voteMat.at<float>(idxV, idxU) += 1.0f;
    //     }
    // }
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

    bool initImu;
    IMU imu(initImu);
    if (!initImu) {
        std::cerr << "Impossible de lancer l'imu\n";
        return -1;
    }

    sleep(1);

    std::vector<cv::Mat> matVotesAccel((int)(ACCEL_MAX / ACCEL_STEP) + 1);

    for (int i = 0; i < matVotesAccel.size(); i++)
    {
        matVotesAccel[i] = cv::Mat((int)(MAX_V / STEP)*2 + 1, (int)(MAX_U / STEP)*2 + 1, CV_32FC1, cv::Scalar(0));
    }

    std::vector<Vec2f> lut((int)(ACCEL_MAX / ACCEL_STEP) + 1);

    int count = 0;

    int sequenceId = 0;

    for(;;)
    {
        cap >> frame;
        if(frame.empty())
        {
            std::cerr << "Frame vide" << std::endl;
            break;
        }
        sequenceId++;
        cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);

        Mat deplacement = opticalFlow.exec(frame, frameOld); 
        //Vec2f vect = maxVectorFlow(deplacement);
        //std::cout << "max u: " << vect[0] << " max v: " << vect[1] << std::endl;
        //std::vector<YOLO::Detection> yoloDetection = yolo.exec(frame);

        //Mat yoloFlow = yoloMat(deplacement, yoloDetection);

        Mat matOpticalFlow = drawOpticalFlow(deplacement);

        //Mat yoloDraw = drawYolo(frame, yoloDetection);

        IMU::Vec3 accel = imu.readAccel();
        double accelNorm = std::sqrt(accel.x * accel.x + accel.y * accel.y + accel.z * accel.z);
        // double accelNorm = fakeAccel();
        std::cout << "Acceleration = " << accelNorm << " m/s2" << std::endl;
        std::cout << count << std::endl;

        vote(matVotesAccel, deplacement, accelNorm);
        
        //cv::imshow("frame", frame);
        cv::imshow("frame flux optique", matOpticalFlow);
        //cv::imwrite("sequence/frame_" + std::to_string(sequenceId) + ".png", frame);
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

            if (saveLUTToJSON("lut.json", lut))
            {
                std::cout << "LUT sauvegardee dans lut.json" << std::endl;
                drawFlowVsAcceleration(lut, "courbe_flux_acceleration.png");
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

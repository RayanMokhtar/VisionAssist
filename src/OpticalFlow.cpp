#include <iostream>
#include <vector>
#include <sys/time.h>
#include <cmath>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include "opencv2/opencv.hpp"

using namespace cv;

struct vectDepl
{
    uchar u;
    uchar v;
};

void niveauGris(Mat frame, Mat *frameGris)
{
    *frameGris = cv::Mat(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));

    for (int y = 0; y < frame.rows; y++)
    {
        for (int x = 0; x < frame.cols; x++)
        {
            cv::Vec3b pixel = frame.at<cv::Vec3b>(y, x);

            (*frameGris).at<uchar>(y, x) = (pixel[0] + pixel[1] + pixel[2]) / 3;
        }
    }
}

void sobelX(Mat frame, Mat *frameSobelX)
{
    *frameSobelX = cv::Mat(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));
    float somme;

    for (int y = 1; y < frame.rows-1; y++)
    {
        for (int x = 1; x < frame.cols-1; x++)
        {
            somme = frame.at<uchar>(y-1, x-1)*-1 + frame.at<uchar>(y-1, x+1) + frame.at<uchar>(y, x-1)*-2 +
                    frame.at<uchar>(y, x+1)*2 + frame.at<uchar>(y+1, x-1)*-1 + frame.at<uchar>(y+1, x+1);
            (*frameSobelX).at<uchar>(y, x) = somme / 4;
        }
    }
}

void sobelY(Mat frame, Mat *frameSobelY)
{
    *frameSobelY = cv::Mat(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));
    float somme;

    for (int y = 1; y < frame.rows-1; y++)
    {
        for (int x = 1; x < frame.cols-1; x++)
        {
            somme = frame.at<uchar>(y-1, x-1)*-1 + frame.at<uchar>(y-1, x)*-2 + frame.at<uchar>(y-1, x+1)*-1 +
                    frame.at<uchar>(y+1, x-1) + frame.at<uchar>(y+1, x)*2 + frame.at<uchar>(y+1, x+1);
            (*frameSobelY).at<uchar>(y, x) = somme / 4;
        }
    }
}

void diffIntensite(Mat frame1, Mat frame2, Mat *frameDiffIntensite)
{
    *frameDiffIntensite = cv::Mat(frame1.rows, frame2.cols, CV_8UC1, cv::Scalar(0));

    for (int y = 0; y < frame1.rows; y++)
    {
        for (int x = 0; x < frame2.cols; x++)
        {
            (*frameDiffIntensite).at<uchar>(y, x) = frame1.at<uchar>(y, x) - frame2.at<uchar>(y, x);
        }
    }
}

int main(int argc, char** argv)
{
    Mat frame;
    Mat frameGris, frameSobelX, frameSobelY, frameDiffIntensite, frameOld, frameGrisOld;

    VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    cap >> frameOld;
    niveauGris(frameOld, &frameGrisOld);

    for(;;)
    {
        cap >> frame;

        niveauGris(frame, &frameGris);
        sobelX(frameGris, &frameSobelX);
        sobelY(frameGris, &frameSobelY);
        diffIntensite(frameGris, frameGrisOld, &frameDiffIntensite);

        imshow("capture", frame);
        imshow("gris", frameGris);
        imshow("sobel x", frameSobelX);
        imshow("sobel y", frameSobelY);
        imshow("difference", frameDiffIntensite);

        frameGris.copyTo(frameGrisOld);

        if(waitKey(33) == 27) break;
    }

    return 0;
}

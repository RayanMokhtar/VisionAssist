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
    *frameGris = cv::Mat(frame.rows, frame.cols, CV_32FC1, cv::Scalar(0));

    for (int y = 0; y < frame.rows; y++)
    {
        for (int x = 0; x < frame.cols; x++)
        {
            cv::Vec3b pixel = frame.at<cv::Vec3b>(y, x);

            (*frameGris).at<float>(y, x) = (pixel[0] + pixel[1] + pixel[2]) / 3;
        }
    }
}

void sobelX(Mat *frame, Mat *frameSobelX)
{
    *frameSobelX = cv::Mat(frame->rows, frame->cols, CV_32FC1, cv::Scalar(0));
    float somme;

    for (int y = 1; y < frame->rows-1; y++)
    {
        for (int x = 1; x < frame->cols-1; x++)
        {
            somme = frame->at<float>(y-1, x-1)*-1 + frame->at<float>(y-1, x+1) + frame->at<float>(y, x-1)*-2 +
                    frame->at<float>(y, x+1)*2 + frame->at<float>(y+1, x-1)*-1 + frame->at<float>(y+1, x+1);
            (*frameSobelX).at<float>(y, x) = somme / 4;
        }
    }
}

void sobelY(Mat *frame, Mat *frameSobelY)
{
    *frameSobelY = cv::Mat(frame->rows, frame->cols, CV_32FC1, cv::Scalar(0));
    float somme;

    for (int y = 1; y < frame->rows-1; y++)
    {
        for (int x = 1; x < frame->cols-1; x++)
        {
            somme = frame->at<float>(y-1, x-1)*-1 + frame->at<float>(y-1, x)*-2 + frame->at<float>(y-1, x+1)*-1 +
                    frame->at<float>(y+1, x-1) + frame->at<float>(y+1, x)*2 + frame->at<float>(y+1, x+1);
            (*frameSobelY).at<float>(y, x) = somme / 4;
        }
    }
}

void diffIntensite(Mat frame1, Mat frame2, Mat *frameDiffIntensite)
{
    *frameDiffIntensite = cv::Mat(frame1.rows, frame2.cols, CV_32FC1, cv::Scalar(0));

    for (int y = 0; y < frame1.rows; y++)
    {
        for (int x = 0; x < frame2.cols; x++)
        {
            (*frameDiffIntensite).at<float>(y, x) = frame1.at<float>(y, x) - frame2.at<float>(y, x);
        }
    }
}

void carre(Mat *frame, Mat *frameCarre)
{
    *frameCarre = cv::Mat(frame->rows, frame->cols, CV_32FC1, cv::Scalar(0));

    for (int y = 0; y < frame->rows; y++)
    {
        for (int x = 0; x < frame->cols; x++)
        {
            (*frameCarre).at<float>(y, x) = frame->at<float>(y, x) * frame->at<float>(y, x);
        }
    }
}

void masqueGaussien(Mat *masque, float sigma, int taille)
{
    *masque = cv::Mat(taille, taille, CV_32FC1, cv::Scalar(0));
    int idx = taille / 2;
   
    for (int y = -idx; y <= idx; y++)
    {
        for (int x = -idx; x <= idx; x++)
        {
            (*masque).at<float>(y + idx, x + idx) = (1 / (2*M_PI*(sigma*sigma))) * std::exp(-((y*y + x*x) / (2*(sigma*sigma))));
        }
    }
}

void filtreGaussien(Mat frame, Mat *frameGaussien, Mat masque)
{
    *frameGaussien = cv::Mat(frame.rows, frame.cols, CV_32FC1, cv::Scalar(0));
    float somme;

    for (int y = 1; y < frame.rows-1; y++)
    {
        for (int x = 1; x < frame.cols-1; x++)
        {
            somme = 0;
            for (int i = 0; i < masque.rows; i++)
            {
                for (int j = 0; j < masque.cols; j++)
                {
                    somme += masque.at<float>(i, j)*frame.at<float>(y, x);
                }
            }
            (*frameGaussien).at<float>(y, x) = somme ;
        }
    }
}

void somme(Mat frame1, Mat frame2, Mat *frameSomme)
{
    *frameSomme = cv::Mat(frame1.rows, frame1.cols, CV_32FC1, cv::Scalar(0));

    for (int y = 0; y < frame2.rows; y++)
    {
        for (int x = 0; x < frame2.cols; x++)
        {
            (*frameSomme).at<float>(y, x) = frame1.at<float>(y, x) + frame2.at<float>(y, x);
        }
    }
}

void produit(Mat *frame1, Mat *frame2, Mat *frameProduit)
{
    *frameProduit = cv::Mat(frame1->rows, frame1->cols, CV_32FC1, cv::Scalar(0));

    for (int y = 0; y < frame2->rows; y++)
    {
        for (int x = 0; x < frame2->cols; x++)
        {
            (*frameProduit).at<float>(y, x) = frame1->at<float>(y, x) * frame2->at<float>(y, x);
        }
    }
}

void formuleHarris(Mat frame1, Mat frame2, Mat frame3, Mat *frameHarris, float lambda)
{
    *frameHarris = cv::Mat(frame1.rows, frame1.cols, CV_32FC1, cv::Scalar(0));

    for (int y = 0; y < frame2.rows; y++)
    {
        for (int x = 0; x < frame2.cols; x++)
        {
            (*frameHarris).at<float>(y, x) = frame1.at<float>(y, x) - frame2.at<float>(y, x) - lambda*frame3.at<float>(y, x);
        }
    }
}

void harris(Mat frame, Mat *frameGris, Mat *frameSobelX, Mat *frameSobelY, Mat *frameHarris, Mat masque, float lambda)
{
    Mat frameSobelCarreX;
    Mat frameSobelCarreY;
    Mat frameSobelCarreGaussienX;
    Mat frameSobelCarreGaussienY;
    Mat frameSobelProduitXY;
    Mat frameSobelCarreGaussienProduitXY;
    Mat frameSobelProduitGaussienXY;
    Mat frameSobelProduitGaussienXYCarre;
    Mat frameSobelCarreGaussienSommeXY;
    Mat frameSobelCarreGaussienSommeCarreXY;

    niveauGris(frame, frameGris);

    sobelX(frameGris, frameSobelX);
    sobelY(frameGris, frameSobelY);

    carre(frameSobelX, &frameSobelCarreX);
    carre(frameSobelY, &frameSobelCarreY);

    filtreGaussien(frameSobelCarreX, &frameSobelCarreGaussienX, masque);
    filtreGaussien(frameSobelCarreY, &frameSobelCarreGaussienY, masque);

    produit(frameSobelX, frameSobelY, &frameSobelProduitXY);
    produit(&frameSobelCarreGaussienX, &frameSobelCarreGaussienY, &frameSobelCarreGaussienProduitXY);

    filtreGaussien(frameSobelProduitXY, &frameSobelProduitGaussienXY, masque);

    carre(&frameSobelProduitGaussienXY, &frameSobelProduitGaussienXYCarre);

    somme(frameSobelCarreGaussienX, frameSobelCarreGaussienY, &frameSobelCarreGaussienSommeXY);

    carre(&frameSobelCarreGaussienSommeXY, &frameSobelCarreGaussienSommeCarreXY);

    formuleHarris(frameSobelCarreGaussienProduitXY, frameSobelProduitGaussienXYCarre, frameSobelCarreGaussienSommeCarreXY, frameHarris, lambda);
}

std::vector<cv::Point> trouverPointsInteret(Mat frameHarris, Mat &frameCouleur)
{
    std::vector<cv::Point> pointInterets;

    float seuil = 200;

    for (int y = 1; y < frameHarris.rows - 1; y++)
    {
        for (int x = 1; x < frameHarris.cols - 1; x++)
        {
            float val = frameHarris.at<float>(y, x);

            if (val < seuil)
            {
                circle(frameCouleur, Point(x, y), 3, Scalar(0, 0, 255), 1);
                pointInterets.push_back(Point(x, y));
            }
        }
    }

    return pointInterets;
}

void normalisation(Mat frame, Mat *frameNormaliser)
{
    double min, max;
   
    cv::minMaxLoc(frame, &min, &max);
   
    for (int y = 0; y < frame.rows; y++)
    {
        for (int x = 0; x < frame.cols; x++)
        {
            (*frameNormaliser).at<float>(y, x) = (float)((float)(frame.at<float>(y, x) - min) / (float)(max - min)) * 255;
        }
    }
}

int main(int argc, char** argv)
{
    Mat frame;
    Mat frameGris, frameSobelX, frameSobelY, frameDiffIntensite, frameOld, frameGrisOld, frameHarris, masque, frameGrisFloat;

    VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    cap >> frameOld;

    masqueGaussien(&masque, 0.04, 3);

    /* for(;;)
    {
        cap >> frame;

        Size newSize(620, 480);
        
        resize(frame, frame, newSize, 0, 0, INTER_LINEAR);

        niveauGris(frame, &frameGris);
        sobelX(frameGris, &frameSobelX);
        sobelY(frameGris, &frameSobelY);
        niveauGris(frameOld, &frameGrisOld);
        diffIntensite(frameGris, frameGrisOld, &frameDiffIntensite);

        harris(frameGris, &frameHarris, masque, 1.0);
//cornerHarris(frameGris, frameHarris, 1, 3, 0.04);
//normalize(frameHarris, frameHarris, 0, 255, NORM_MINMAX, CV_8UC1);
        normalisation(frameHarris, &frameHarris);
        //trouverPointsInteret(frameHarris, frame);

        //frameGris.convertTo(frameGris, CV_8UC1);
        //frameSobelX.convertTo(frameSobelX, CV_8UC1);
        //frameSobelY.convertTo(frameSobelY, CV_8UC1);
        //frameDiffIntensite.convertTo(frameDiffIntensite, CV_8UC1);
        frameHarris.convertTo(frameHarris, CV_8UC1);

        imshow("capture", frame);
        //imshow("gris", frameGris);
        //imshow("sobel x", frameSobelX);
        //imshow("sobel y", frameSobelY);
        //imshow("difference", frameDiffIntensite);
        imshow("harris", frameHarris);

        frame.copyTo(frameOld);

        if(waitKey(33) == 27) break;
    } */

    frame = imread("Harris_Detector_Original_Image.jpg", IMREAD_COLOR);
//niveauGris(frame, &frameGris);
//cornerHarris(frameGris, frameHarris, 1, 3, 0.04);
    harris(frame, &frameGris, &frameSobelX, &frameSobelY, &frameHarris, masque, 0.04);
    normalisation(frameHarris, &frameHarris);
    trouverPointsInteret(frameHarris, frame);
    imwrite("Harris_Detector_Original_Image_Output.jpg", frame);

    for(;;)
    {
        cap >> frame;

        Size newSize(620, 480);
        
        resize(frame, frame, newSize, 0, 0, INTER_LINEAR);

        niveauGris(frameOld, &frameGrisOld);

        harris(frame, &frameGris, &frameSobelX, &frameSobelY, &frameHarris, masque, 0.04);
        normalisation(frameHarris, &frameHarris);

        diffIntensite(frameGris, frameGrisOld, &frameDiffIntensite);

        std::vector<cv::Point> pointInterets = trouverPointsInteret(frameHarris, frame);
        cv::Point pointInteret = pointInterets.back();

        int cols = 20;
        int rows = 20;
        Mat intensiteBlock = cv::Mat(rows, cols, CV_32FC1, cv::Scalar(0));
        Mat sobelXBlock = cv::Mat(rows, cols, CV_32FC1, cv::Scalar(0));
        Mat sobelYBlock = cv::Mat(rows, cols, CV_32FC1, cv::Scalar(0));

        for (int j = pointInteret.y-rows/2; j < pointInteret.y+rows/2; j++)
        {
            for (int i = pointInteret.x-cols/2; i < pointInteret.x+cols/2; i++)
            {
                if (j < 0 || j >= frame.rows)
                    continue;

                if (i < 0 || i >= frame.cols)
                    continue;

                int bj = j - (pointInteret.y - rows / 2);
                int bi = i - (pointInteret.x - cols / 2);

                intensiteBlock.at<float>(bj, bi) = frameDiffIntensite.at<float>(j, i);
                sobelXBlock.at<float>(bj, bi) = frameSobelX.at<float>(j, i);
                sobelYBlock.at<float>(bj, bi) = frameSobelY.at<float>(j, i);

                circle(frame, Point(i, j), 3, Scalar(255, 0, 0), 1);
            }
        }

        Mat carreX = cv::Mat(rows, cols, CV_32FC1, cv::Scalar(0));
        Mat carreY = cv::Mat(rows, cols, CV_32FC1, cv::Scalar(0));
        Mat produitXY = cv::Mat(rows, cols, CV_32FC1, cv::Scalar(0));
        Mat produitXT = cv::Mat(rows, cols, CV_32FC1, cv::Scalar(0));
        Mat produitYT = cv::Mat(rows, cols, CV_32FC1, cv::Scalar(0));

        carre(&sobelXBlock, &carreX);
        carre(&sobelYBlock, &carreY);
        produit(&sobelXBlock, &sobelYBlock, &produitXY);

        produit(&sobelXBlock, &intensiteBlock, &produitXT);
        produit(&sobelYBlock, &intensiteBlock, &produitYT);

        Mat A = cv::Mat(2, 2, CV_32FC1, cv::Scalar(0));
        Mat b = cv::Mat(2, 1, CV_32FC1, cv::Scalar(0));

        float x2, y2, xy, xt, yt;

        for (int j = 0; j < rows; j++) {
            for (int i = 0; i < cols; i++)
            {
                x2 += carreX.at<float>(j, i);
                y2 += carreY.at<float>(j, i);
                xy += produitXY.at<float>(j, i);

                xt += produitXT.at<float>(j, i);
                yt += produitYT.at<float>(j, i);
            }
        }

        A.at<float>(0, 0) = x2;
        A.at<float>(0, 1) = xy;
        A.at<float>(1, 0) = xy;
        A.at<float>(1, 1) = y2;

        b.at<float>(0, 0) = xt;
        b.at<float>(1, 0) = yt;

        float det = x2*y2 - xy*xy;

        Mat inverseA = cv::Mat(2, 2, CV_32FC1, cv::Scalar(0));

        float u, v;

        if ((det - 0.1) != 0) {
            inverseA.at<float>(0, 0) = y2 * 1/det;
            inverseA.at<float>(0, 1) = -xy * 1/det;
            inverseA.at<float>(1, 0) = -xy * 1/det;
            inverseA.at<float>(1, 1) = x2 * 1/det;

            u = inverseA.at<float>(0, 0) * b.at<float>(0, 0) + inverseA.at<float>(0, 1) * b.at<float>(1, 0);
            v = inverseA.at<float>(1, 0) * b.at<float>(0, 0) + inverseA.at<float>(1, 1) * b.at<float>(1, 0);
        }

        std::cout << u << " " << v << std::endl;

        //frameGris.convertTo(frameGris, CV_8UC1);
        //frameSobelX.convertTo(frameSobelX, CV_8UC1);
        //frameSobelY.convertTo(frameSobelY, CV_8UC1);
        //frameDiffIntensite.convertTo(frameDiffIntensite, CV_8UC1);
        frameHarris.convertTo(frameHarris, CV_8UC1);

        imshow("capture", frame);
        //imshow("gris", frameGris);
        //imshow("sobel x", frameSobelX);
        //imshow("sobel y", frameSobelY);
        //imshow("difference", frameDiffIntensite);
        imshow("harris", frameHarris);

        frame.copyTo(frameOld);

        if(waitKey(33) == 27) break;
    }

    return 0;
}

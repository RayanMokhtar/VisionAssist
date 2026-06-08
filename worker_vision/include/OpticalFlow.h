#ifndef OPTICALFLOW_H
#define OPTICALFLOW_H

#include <iostream>
#include <vector>
#include <sys/time.h>
#include <cmath>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include "opencv2/opencv.hpp"
#include <omp.h>

#ifndef _OPENMP
#define omp_get_thread_num() 0
#endif

using namespace cv;

class OpticalFlow{


    public : 
        OpticalFlow();
        Mat exec(const Mat& frame, const Mat& frameOld);
    
    private :

        void niveauGris(const Mat& frame, Mat& frameGris);
        void sobelX(const Mat& frame, Mat& frameSobelX);
        void sobelY(const Mat& frame, Mat& frameSobelY);
        void diffIntensite(const Mat& frame1, const Mat& frame2, Mat& frameDiffIntensite);
        void carre(const Mat& frame, Mat& frameCarre);
        void somme(const Mat& frame1, const Mat& frame2, Mat& frameSomme);
        void produit(const Mat& frame1, const Mat& frame2, Mat& frameProduit);      

        Mat frameGris;
        Mat frameGrisOld;
        Mat frameSobelX;
        Mat frameSobelY;
        Mat frameDiffIntensite;
        Mat matDepl;
};

#endif

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
        Mat exec(Mat frame, Mat frameOld);
    
    private :

        void niveauGris(Mat frame, Mat *frameGris);
        void sobelX(Mat *frame, Mat *frameSobelX);
        void sobelY(Mat *frame, Mat *frameSobelY);
        void diffIntensite(Mat frame1, Mat frame2, Mat *frameDiffIntensite);
        void carre(Mat *frame, Mat *frameCarre);
        void somme(Mat frame1, Mat frame2, Mat *frameSomme);
        void produit(Mat *frame1, Mat *frame2, Mat *frameProduit);      
};

#endif

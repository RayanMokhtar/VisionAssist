#include "OpticalFlow.h"

void OpticalFlow::niveauGris(Mat frame, Mat *frameGris)
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

void OpticalFlow::sobelX(Mat *frame, Mat *frameSobelX)
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

void OpticalFlow::sobelY(Mat *frame, Mat *frameSobelY)
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

void OpticalFlow::diffIntensite(Mat frame1, Mat frame2, Mat *frameDiffIntensite)
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

void OpticalFlow::carre(Mat *frame, Mat *frameCarre)
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

void OpticalFlow::somme(Mat frame1, Mat frame2, Mat *frameSomme)
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

void OpticalFlow::produit(Mat *frame1, Mat *frame2, Mat *frameProduit)
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

Mat OpticalFlow::exec(Mat frame, Mat frameOld)
{
    Mat frameGris, frameSobelX, frameSobelY, frameDiffIntensite, frameGrisOld;

    niveauGris(frame, &frameGris);
    niveauGris(frameOld, &frameGrisOld);

    sobelX(&frameGris, &frameSobelX);
    sobelY(&frameGris, &frameSobelY);

    diffIntensite(frameGris, frameGrisOld, &frameDiffIntensite);

    Mat matDepl = cv::Mat(frame.rows, frame.cols, CV_32FC2, cv::Scalar(0,0));

    #pragma omp parallel default(none) firstprivate(frame, frameSobelX, frameSobelY, frameDiffIntensite) shared(matDepl, std::cout)
    {

        /*#pragma omp single
        {
            std::cout << "Nb threads actifs : " << omp_get_num_threads() << std::endl;
        }*/

        #pragma omp for
        for (int y = 1; y < frame.rows-1; y++)
        {
            for (int x = 1; x < frame.cols-1; x++)
            {
                int cols = 9;
                int rows = 9;
                Mat intensiteBlock = cv::Mat(rows, cols, CV_32FC1, cv::Scalar(0));
                Mat sobelXBlock = cv::Mat(rows, cols, CV_32FC1, cv::Scalar(0));
                Mat sobelYBlock = cv::Mat(rows, cols, CV_32FC1, cv::Scalar(0));

                for (int j = y-rows/2; j < y+rows/2; j++)
                {
                    for (int i = x-cols/2; i < x+cols/2; i++)
                    {
                        //std::cout << "X " << x << "Y " << y << std::endl;

                        if (j < 0 || j >= frame.rows)
                            continue;

                        if (i < 0 || i >= frame.cols)
                            continue;

                        int bj = j - (y - rows / 2);
                        int bi = i - (x - cols / 2);

                        intensiteBlock.at<float>(bj, bi) = frameDiffIntensite.at<float>(j, i);
                        sobelXBlock.at<float>(bj, bi) = frameSobelX.at<float>(j, i);
                        sobelYBlock.at<float>(bj, bi) = frameSobelY.at<float>(j, i);

                        //circle(frame, Point(i, j), 3, Scalar(255, 0, 0), 1);
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

                float x2 = 0, y2 = 0, xy = 0, xt = 0, yt = 0;

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

                b.at<float>(0, 0) = -xt;
                b.at<float>(1, 0) = -yt;

                float det = x2*y2 - xy*xy;

                Mat inverseA = cv::Mat(2, 2, CV_32FC1, cv::Scalar(0));

                float u = 0, v = 0;

                if (std::fabs(det) > 0.01) {
                    inverseA.at<float>(0, 0) = y2 * 1/det;
                    inverseA.at<float>(0, 1) = -xy * 1/det;
                    inverseA.at<float>(1, 0) = -xy * 1/det;
                    inverseA.at<float>(1, 1) = x2 * 1/det;

                    u = inverseA.at<float>(0, 0) * b.at<float>(0, 0) + inverseA.at<float>(0, 1) * b.at<float>(1, 0);
                    v = inverseA.at<float>(1, 0) * b.at<float>(0, 0) + inverseA.at<float>(1, 1) * b.at<float>(1, 0);
                }

                matDepl.at<vectDepl>(y, x) = {u, v};
            }
        }
    }
    
    cv::Mat vis(frame.rows, frame.cols, CV_8UC3, cv::Scalar(255, 255, 255));

    int step = 20;
    float scale = 50;

    for (int y = step; y < matDepl.rows - step; y += step)
    {
        for (int x = step; x < matDepl.cols - step; x += step)
        {
            float sum_u = 0, sum_v = 0;
            int count = 0;

            for (int j = y - step/2; j < y + step/2; j++)
            {
                for (int i = x - step/2; i < x + step/2; i++)
                {
                    vectDepl& p = matDepl.at<vectDepl>(j, i);
                    sum_u += p.u;
                    sum_v += p.v;
                    count++;
                }
            }

            float u = sum_u / count;
            float v = sum_v / count;

            float norm = sqrt(u*u + v*v);
            if (norm < 1) continue;

            cv::Point p1(x, y);
            cv::Point p2(x + u * scale, y + v * scale);

            cv::arrowedLine(vis, p1, p2, cv::Scalar(0, 0, 255), 2);
        }
    }

    cv::imshow("vis", vis);

    return matDepl;
}

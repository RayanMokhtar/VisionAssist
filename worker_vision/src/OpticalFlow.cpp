#include "OpticalFlow.h"

void OpticalFlow::niveauGris(Mat frame, Mat *frameGris)
{
    *frameGris = cv::Mat(frame.rows, frame.cols, CV_32FC1);

    #pragma omp parallel default(none) shared(frame, frameGris)
    {
        #pragma omp for
        for (int y = 0; y < frame.rows; y++)
        {
            cv::Vec3b *src = frame.ptr<cv::Vec3b>(y);
            float *dst = frameGris->ptr<float>(y);

            for (int x = 0; x < frame.cols; x++)
            {
                const cv::Vec3b& pixel = src[x];

                dst[x] = (pixel[0] + pixel[1] + pixel[2]) / 3.0f;
            }
        }
    }
}

void OpticalFlow::sobelX(Mat *frame, Mat *frameSobelX)
{
    *frameSobelX = cv::Mat(frame->rows, frame->cols, CV_32FC1);

    for (int y = 1; y < frame->rows - 1; y++)
    {
        float *prev = frame->ptr<float>(y - 1);
        float *curr = frame->ptr<float>(y);
        float *next = frame->ptr<float>(y + 1);

        float *out = frameSobelX->ptr<float>(y);

        for (int x = 1; x < frame->cols - 1; x++)
        {
            float v = (-1 * prev[x - 1]) + (1 * prev[x + 1]) +
                        (-2 * curr[x - 1]) + (2 * curr[x + 1]) +
                        (-1 * next[x - 1]) + (1 * next[x + 1]);

            out[x] = v / 4.0f;
        }
    }
}

void OpticalFlow::sobelY(Mat *frame, Mat *frameSobelY)
{
    *frameSobelY = cv::Mat(frame->rows, frame->cols, CV_32FC1);

    for (int y = 1; y < frame->rows - 1; y++)
    {
        float *prev = frame->ptr<float>(y - 1);
        float *next = frame->ptr<float>(y + 1);

        float *out = frameSobelY->ptr<float>(y);

        for (int x = 1; x < frame->cols - 1; x++)
        {
            float v = (-1 * prev[x - 1]) + (-2 * prev[x]) + (-1 * prev[x + 1]) +
                        (1 * next[x - 1]) + ( 2 * next[x]) + ( 1 * next[x + 1]);

            out[x] = v / 4.0f;
        }
    }
}

void OpticalFlow::diffIntensite(Mat frame1, Mat frame2, Mat *frameDiffIntensite)
{
    *frameDiffIntensite = cv::Mat(frame1.rows, frame1.cols, CV_32FC1);

    #pragma omp parallel default(none) shared(frame1, frame2, frameDiffIntensite)
    {
        #pragma omp for
        for (int y = 0; y < frame1.rows; y++)
        {
            float *p1 = frame1.ptr<float>(y);
            float *p2 = frame2.ptr<float>(y);
            float *pd = frameDiffIntensite->ptr<float>(y);

            for (int x = 0; x < frame1.cols; x++)
            {
                pd[x] = p1[x] - p2[x];
            }
        }
    }
}

void OpticalFlow::carre(Mat *frame, Mat *frameCarre)
{
    *frameCarre = cv::Mat(frame->rows, frame->cols, CV_32FC1);

    #pragma omp parallel default(none) shared(frame, frameCarre)
    {
        #pragma omp for
        for (int y = 0; y < frame->rows; y++)
        {
            float *src = frame->ptr<float>(y);
            float *dst = frameCarre->ptr<float>(y);

            for (int x = 0; x < frame->cols; x++)
            {
                float v = src[x];
                dst[x] = v * v;
            }
        }

    }
}

void OpticalFlow::somme(Mat frame1, Mat frame2, Mat *frameSomme)
{
    *frameSomme = cv::Mat(frame1.rows, frame1.cols, CV_32FC1);

    #pragma omp parallel default(none) shared(frame1, frame2, frameSomme)
    {
        #pragma omp for
        for (int y = 0; y < frame1.rows; y++)
        {
            float *p1 = frame1.ptr<float>(y);
            float *p2 = frame2.ptr<float>(y);
            float *pd = frameSomme->ptr<float>(y);

            for (int x = 0; x < frame1.cols; x++)
            {
                pd[x] = p1[x] + p2[x];
            }
        }
    }
}

void OpticalFlow::produit(Mat *frame1, Mat *frame2, Mat *frameProduit)
{
    *frameProduit = cv::Mat(frame1->rows, frame1->cols, CV_32FC1);

    #pragma omp parallel default(none) shared(frame1, frame2, frameProduit)
    {
        #pragma omp for
        for (int y = 0; y < frame1->rows; y++)
        {
            float *p1 = frame1->ptr<float>(y);
            float *p2 = frame2->ptr<float>(y);
            float *pd = frameProduit->ptr<float>(y);

            for (int x = 0; x < frame1->cols; x++)
            {
                pd[x] = p1[x] * p2[x];
            }
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

        #pragma omp for collapse(2) schedule(dynamic)
        for (int y = 1; y < frame.rows-1; y++)
        {
            for (int x = 1; x < frame.cols-1; x++)
            {
                int cols = 3;
                int rows = 3;

                float x2 = 0.0, y2 = 0.0, xy = 0.0, xt = 0.0, yt = 0.0;

                for (int j = y-rows/2; j < y+rows/2; j++)
                {
                    for (int i = x-cols/2; i < x+cols/2; i++)
                    {
                        float Ix = frameDiffIntensite.ptr<float>(j)[i];
                        float Sx = frameSobelX.ptr<float>(j)[i];
                        float Sy = frameSobelY.ptr<float>(j)[i];

                        x2 += Sx * Sx;
                        y2 += Sy * Sy;
                        xy += Sx * Sy;

                        xt += Sx * Ix;
                        yt += Sy * Ix;
                    }
                }

                float det = x2 * y2 - xy * xy;

                float u = 0.0, v = 0.0;

                if (fabs(det) > 0.01)
                {
                    u = (y2 * (-xt) - xy * (-yt)) / det;
                    v = (-xy * (-xt) + x2 * (-yt)) / det;
                }

                matDepl.ptr<cv::Vec2f>(y)[x] = cv::Vec2f(u, v);
            }
        }
    }

    cv::Mat matOpticalFlow(frame.rows, frame.cols, CV_8UC3, cv::Scalar(255, 255, 255));

    int step = 3;
    float scale = 1;

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
                    cv::Vec2f& p = matDepl.at<cv::Vec2f>(j, i);
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

    cv::imshow("flux optique", matOpticalFlow);

    return matDepl;
}

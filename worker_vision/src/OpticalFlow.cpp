#include "OpticalFlow.h"

void OpticalFlow::niveauGris(const Mat& frame, Mat& frameGris)
{
    #pragma omp parallel default(none) shared(frame, frameGris)
    {
        #pragma omp for
        for (int y = 0; y < frame.rows; y++)
        {
            const cv::Vec3b *src = frame.ptr<cv::Vec3b>(y);
            float *dst = frameGris.ptr<float>(y);

            for (int x = 0; x < frame.cols; x++)
            {
                const cv::Vec3b& pixel = src[x];

                dst[x] = (pixel[0] + pixel[1] + pixel[2]) / 3.0f;
            }
        }
    }
}

void OpticalFlow::sobelX(const Mat& frame, Mat& frameSobelX)
{
    for (int y = 1; y < frame.rows - 1; y++)
    {
        const float *prev = frame.ptr<float>(y - 1);
        const float *curr = frame.ptr<float>(y);
        const float *next = frame.ptr<float>(y + 1);

        float *out = frameSobelX.ptr<float>(y);

        for (int x = 1; x < frame.cols - 1; x++)
        {
            float v = (-1 * prev[x - 1]) + (1 * prev[x + 1]) +
                        (-2 * curr[x - 1]) + (2 * curr[x + 1]) +
                        (-1 * next[x - 1]) + (1 * next[x + 1]);

            out[x] = v / 4.0f;
        }
    }
}

void OpticalFlow::sobelY(const Mat& frame, Mat& frameSobelY)
{
    for (int y = 1; y < frame.rows - 1; y++)
    {
        const float *prev = frame.ptr<float>(y - 1);
        const float *next = frame.ptr<float>(y + 1);

        float *out = frameSobelY.ptr<float>(y);

        for (int x = 1; x < frame.cols - 1; x++)
        {
            float v = (-1 * prev[x - 1]) + (-2 * prev[x]) + (-1 * prev[x + 1]) +
                        (1 * next[x - 1]) + ( 2 * next[x]) + ( 1 * next[x + 1]);

            out[x] = v / 4.0f;
        }
    }
}

void OpticalFlow::diffIntensite(const Mat& frame1, const Mat& frame2, Mat& frameDiffIntensite)
{
    #pragma omp parallel default(none) shared(frame1, frame2, frameDiffIntensite)
    {
        #pragma omp for
        for (int y = 0; y < frame1.rows; y++)
        {
            const float *p1 = frame1.ptr<float>(y);
            const float *p2 = frame2.ptr<float>(y);
            float *pd = frameDiffIntensite.ptr<float>(y);

            for (int x = 0; x < frame1.cols; x++)
            {
                pd[x] = p1[x] - p2[x];
            }
        }
    }
}

void OpticalFlow::carre(const Mat& frame, Mat& frameCarre)
{
    #pragma omp parallel default(none) shared(frame, frameCarre)
    {
        #pragma omp for
        for (int y = 0; y < frame.rows; y++)
        {
            const float *src = frame.ptr<float>(y);
            float *dst = frameCarre.ptr<float>(y);

            for (int x = 0; x < frame.cols; x++)
            {
                float v = src[x];
                dst[x] = v * v;
            }
        }

    }
}

void OpticalFlow::somme(const Mat& frame1, const Mat& frame2, Mat& frameSomme)
{
    #pragma omp parallel default(none) shared(frame1, frame2, frameSomme)
    {
        #pragma omp for
        for (int y = 0; y < frame1.rows; y++)
        {
            const float *p1 = frame1.ptr<float>(y);
            const float *p2 = frame2.ptr<float>(y);
            float *pd = frameSomme.ptr<float>(y);

            for (int x = 0; x < frame1.cols; x++)
            {
                pd[x] = p1[x] + p2[x];
            }
        }
    }
}

void OpticalFlow::produit(const Mat& frame1, const Mat& frame2, Mat& frameProduit)
{
    #pragma omp parallel default(none) shared(frame1, frame2, frameProduit)
    {
        #pragma omp for
        for (int y = 0; y < frame1.rows; y++)
        {
            const float *p1 = frame1.ptr<float>(y);
            const float *p2 = frame2.ptr<float>(y);
            float *pd = frameProduit.ptr<float>(y);

            for (int x = 0; x < frame1.cols; x++)
            {
                pd[x] = p1[x] * p2[x];
            }
        }
    }
}

OpticalFlow::OpticalFlow()
{
    frameGris.create(640,640,CV_32FC1);
    frameGrisOld.create(640,640,CV_32FC1);
    frameSobelX.create(640,640,CV_32FC1);
    frameSobelY.create(640,640,CV_32FC1);
    frameDiffIntensite.create(640,640,CV_32FC1);
    matDepl.create(640,640,CV_32FC2);
}

Mat OpticalFlow::exec(const Mat& frame, const Mat& frameOld)
{
    niveauGris(frame, frameGris);
    niveauGris(frameOld, frameGrisOld);

    sobelX(frameGris, frameSobelX);
    sobelY(frameGris, frameSobelY);

    diffIntensite(frameGris, frameGrisOld, frameDiffIntensite);

    matDepl.setTo(cv::Scalar(0,0));

    #pragma omp parallel default(none) shared(matDepl, frame, frameSobelX, frameSobelY, frameDiffIntensite, std::cout)
    {
        /*#pragma omp single
        {
            std::cout << "Nb threads actifs : " << omp_get_num_threads() << std::endl;
        }*/

        #pragma omp for schedule(static)
        for (int y = 1; y < frame.rows-1; y++)
        {
            for (int x = 1; x < frame.cols-1; x++)
            {
                int cols = 3;
                int rows = 3;

                float x2 = 0.0, y2 = 0.0, xy = 0.0, xt = 0.0, yt = 0.0;

                for (int j = y-rows/2; j <= y+rows/2; j++)
                {
                    for (int i = x-cols/2; i <= x+cols/2; i++)
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

    return matDepl;
}

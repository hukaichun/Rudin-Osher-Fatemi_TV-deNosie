#include <opencv2/opencv.hpp>
#include <cmath> 

#include <iostream>
using namespace std;

void deltaX_p(const cv::Mat& u, cv::Mat& result)
{
    cv::Size s = u.size();
    int type   = u.type(); 

    result.create(s, type);
    float* data_u = (float*)u.data;
    float* data_r = (float*)result.data;

    for(int i=0; i<s.height; ++i)
    {
        for(int j=0; j<s.width-1; ++j)
        {
            data_r[i*s.width+j] = -data_u[i*s.width+j]  + data_u[i*s.width+j+1];
        }
        data_r[i*s.width+s.width-1] = 0;
    }
}

void deltaX_n(const cv::Mat& u, cv::Mat& result)
{
    cv::Size s = u.size();
    int type   = u.type(); 

    result.create(s, type);
    float* data_u = (float*)u.data;
    float* data_r = (float*)result.data;

    for(int i=0; i<s.height; ++i)
    {
        data_r[i*s.width] = 0;
        for(int j=1; j<s.width; ++j)
        {
            data_r[i*s.width+j] = - data_u[i*s.width + j-1] + data_u[i*s.width+j];
        }
    }
}

void deltaY_p(const cv::Mat& u, cv::Mat& result)
{
    cv::Size s = u.size();
    int type = u.type();
    
    result.create(s, type);
    float* data_u = (float*)u.data;
    float* data_r = (float*)result.data;

    for(int j = 0; j<s.width; ++j)
    {
        for(int i = 0; i<s.height-1; ++i)
        {
            data_r[i*s.width+j] = data_u[(i+1)*s.width+j] - data_u[i*s.width+j];
        }
        data_r[(s.height-1)*s.width+j] = 0;
    }

}

void deltaY_n(const cv::Mat& u, cv::Mat& result)
{
    cv::Size s = u.size();
    int type = u.type();
    
    result.create(s, type);
    float* data_u = (float*)u.data;
    float* data_r = (float*)result.data;

    for(int j = 0; j<s.width; ++j)
    {
        for(int i = 1; i<s.height; ++i)
        {
            data_r[i*s.width+j] = - data_u[(i-1)*s.width+j] + data_u[i*s.width+j];
        }
        data_r[j] = 0;
    }

}

float minmod(const float a, const float b)
{
    int sgn_a = (a>0) - (a<0),
        sgn_b = (b>0) - (b<0);
    
    return 0.5*static_cast<float>(sgn_a+sgn_b)*std::min(abs(a),abs(b));    
}

float lambda(const cv::Mat& u0x, const cv::Mat& u0y, 
             const cv::Mat& ux,  const cv::Mat& uy, 
             float coeff)
{
    int index;
    float lam = 0, temp, elem;
    cv::Size s = u0x.size();

    float *data_u0x = (float*) u0x.data,
          *data_u0y = (float*) u0y.data,
          *data_ux  = (float*) ux.data,
          *data_uy  = (float*) uy.data;

    for(int i=0,j; i<s.height; ++i)
    {
        for(j=0; j<s.width; ++j)
        {
            index = i*s.width+j;
            temp = data_ux[index]*data_ux[index] 
                  +data_uy[index]*data_uy[index];
            
            elem =  temp 
                  - data_ux[index]*data_u0x[index]
                  - data_uy[index]*data_u0y[index];

            elem /= (sqrt(temp) + 0.0001);

            lam += elem;
        }
    }

    lam *= coeff;
    return lam;    
}






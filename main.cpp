#include <iostream>
#include <utility>
#include <vector>
#include "finiteDifference.hpp"
#include <ctime>
#include <cstdlib>

using namespace std;
using namespace cv;

Mat ROFtv(const Mat& u0,
                   int  N = 300,
                   float sigma = 0.002,
                   float deltaT = 1e-6,
                   bool show = false)
{
    char ch[512];
    int index = 0;
    float l=-0, normValue;
    Size s = u0.size();
    float h = 1./s.width;
    Mat U0x(s, CV_32F),
        U0y(s, CV_32F),
        Ux(s, CV_32F), 
        Uy(s, CV_32F),
        xU(s, CV_32F),
        yU(s, CV_32F),
        pool(s, CV_32F),
        X(s, CV_32F),
        Y(s, CV_32F),
        u(s, CV_32F),
        temp,view;
    
    
    float *data_Ux = (float*) Ux.data,
          *data_Uy = (float*) Uy.data,
          *data_xU = (float*) xU.data,
          *data_yU = (float*) yU.data,
          *data_X  = (float*) X.data,
          *data_Y  = (float*) Y.data,
          *data_u  = (float*) u.data,
          *data_temp;

    u0.copyTo(u);

    for(int k=0;k<N ; ++k)
    {
        deltaX_p(u, Ux);
        deltaX_n(u, xU);
        deltaY_p(u, Uy);
        deltaY_n(u, yU);

        if(k == 0 )
        {
            Ux.copyTo(U0x);
            Uy.copyTo(U0y);
        }

        if(k>0)
        {
            l = lambda(U0x, U0y, 
                        Ux,  Uy,
                        -0.5*h/sigma);

        }
                    
        for(int i=0,j; i<s.height; ++i)
            for(j=0; j<s.width; ++j)
            {
                index = i*s.width+j;
                data_X[index] = data_Ux[index]
                                /(sqrt(data_Ux[index]*data_Ux[index]  
                                      +pow(minmod(data_yU[index], data_Uy[index]),2))+0.00001)/h;

                data_Y[index] = data_Uy[index]
                                /(sqrt(data_Uy[index]*data_Uy[index]
                                      +pow(minmod(data_xU[index], data_Ux[index]),2))+0.00001)/h;
            }

        pool = u-u0;
        temp = -l*pool;
        deltaX_n(X, pool);
        temp += pool;
        deltaY_n(Y, pool);
        temp += pool;
        temp *= deltaT;
        normValue=0;
        data_temp = (float*)temp.data;
        for(int i=1, j; i<s.height-1; ++i)
            for(j=1; j<s.width; ++j)
                normValue += abs(data_temp[i*s.width+j]);
        normValue*=(h*h);

//        if(normValue<0.0001 && k>2)
//            break;
//        else
            u += temp;


        for(int i=0; i<s.width; ++i)
            data_u[i] = data_u[i+s.width];
        for(int i=0; i<s.width; ++i)
            data_u[(s.height-1)*s.width+i] = data_u[(s.height-2)*s.width+i];
        for(int i=0; i<s.height; ++i)
        {
            data_u[i*s.width] = data_u[(i+1)*s.width-2];
            data_u[(i+1)*s.width-1] = data_u[(i+1)*s.width-2];
        }
        
        if(show)
        {
            sprintf(ch,"k=%d    |u-u0|=%f", k, normValue);
            u.convertTo(view, CV_8U, 255);
            putText(view, ch, Point(20,40), FONT_HERSHEY_COMPLEX, 0.7, {0,0,255});
            imshow("ing", view);
            waitKey(1);
        }

    }
    return u;
}


int main(int argc, char** argv)
{
    
    const string keys = 
    "{help h ? |      | print this message    }"
    "{@imag    |      | imput image           }"
    "{N steps  | 100  | total Time step       integer}"
    "{s sigma2 | 1e-3 | value of sigma square float}"
    "{d deltaT | 1e-6 | time step             float}"
    "{show     | false| show process          bool}";

    CommandLineParser parser(argc, argv, keys);
    if(parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    const string fileName = parser.get<string>("@imag");
    int N        = parser.get<int>("N");
    float sigma2 = parser.get<float>("s");
    float deltaT = parser.get<float>("d");
    bool show    = parser.get<bool>("show");
    Mat u = imread(fileName, CV_LOAD_IMAGE_GRAYSCALE );
    if(!parser.check() || u.dims==0)
    {
        parser.printErrors();
        parser.printMessage();
        return -1;
    }

    cout << "file  \t" << fileName   << endl 
         << "N     \t" << N          << endl
         << "sigma2\t" << sigma2     << endl
         << "deltaT\t" << deltaT     << endl
         << "show  \t" << show       << endl;




    vector<int> para;
    Mat temp, view;
    para.push_back(9);

    Mat result(u.size(), CV_32F);
    u.convertTo(temp, CV_32F);
    temp.copyTo(u);
    u/=255;

    cout << "start" << endl;
    clock_t start = clock();
    result = move(ROFtv(u, N, sigma2, deltaT, show));
    cout << (clock()-start)/CLOCKS_PER_SEC << endl;

    result.convertTo(temp, CV_8U, 255);
    imwrite("output.png",temp, para);

    waitKey(0);

    return 0;
}



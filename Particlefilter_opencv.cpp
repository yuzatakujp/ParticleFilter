//
//  Particlefilter_opencv.cpp
//  test00
//
//  Created by yuzatakujp on 2015/06/27.
//  Copyright (c) 2015年 yuzatakujp. All rights reserved.
//

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <ctype.h>
#include <math.h>
#include <opencv2/legacy/legacy.hpp>


//肌色の平均値
double skin_R = 120.0, skin_G = 63.0, skin_B = 45.0;
//分散
double sigma = 10.0;

/*尤度推定*/
double Likelihood(IplImage* image, int x, int y){
    double R, G, B;
    double color_dist = 0.0;

    //色をとってくる
    B = image -> imageData[image -> widthStep*y+x*3];
    G = image -> imageData[image -> widthStep*y+x*3+1];
    R = image -> imageData[image -> widthStep*y+x*3+2];
    
    //肌色との色空間内の距離を計算
    color_dist = sqrt((R-skin_R)*(R-skin_R)
                      +(G-skin_G)*(G-skin_G)
                      +(B-skin_B)*(B-skin_B));
    
    //尤度を計算
    return 1.0 / (sqrt(2.0*M_PI)*sigma) * exp(-color_dist*color_dist / (2.0*sigma*sigma));
}

int main(){
    
    int i, x, y;
    CvCapture* capture = 0;
    IplImage* image = NULL;
    
    //状態ベクトルの次元数
    int dimention = 4;
    //サンプル数
    int num_of_particles = 1000;
    
    CvConDensation* particle_filter = NULL;
    CvMat* lowerBound = NULL;
    CvMat* upperBound = NULL;
    
    //カメラキャプチャの準備
    capture = cvCreateCameraCapture(CV_CAP_ANY);
    
    //ウインドウの生成
    cvNamedWindow("Particle Filter", CV_WINDOW_AUTOSIZE);
    
    image = cvQueryFrame(capture);
    
    //Condensation構造体を作成
    particle_filter = cvCreateConDensation(dimention, 0, num_of_particles);
    
    //状態ベクトルの範囲の指定
    lowerBound = cvCreateMat(4, 1, CV_32FC1); //上限値
    upperBound = cvCreateMat(4, 1, CV_32FC1); //下限値
    //上限値の設定
    cvmSet(lowerBound, 0, 0, 0.0); //x座標
    cvmSet(lowerBound, 1, 0, 0.0); //y座標
    cvmSet(lowerBound, 2, 0, -3.0); //x方向の速度
    cvmSet(lowerBound, 3, 0, -3.0); //y方向の速度
    //下限値の設定
    cvmSet(upperBound, 0, 0, image->width); //x座標
    cvmSet(upperBound, 1, 0, image->height); //y座標
    cvmSet(upperBound, 2, 0, 3.0); //x方向の速度
    cvmSet(upperBound, 3, 0, 3.0); //y方向の速度
    
    //Condensation構造体の初期化(初期サンプルを生成)
    cvConDensInitSampleSet(particle_filter, lowerBound, upperBound);
    
    //状態遷移行列の設定 4*4行列
    //1行目
    particle_filter->DynamMatr[0] = 1.0;
    particle_filter->DynamMatr[1] = 0.0;
    particle_filter->DynamMatr[2] = 1.0;
    particle_filter->DynamMatr[3] = 0.0;
    //2行目
    particle_filter->DynamMatr[4] = 0.0;
    particle_filter->DynamMatr[5] = 1.0;
    particle_filter->DynamMatr[6] = 0.0;
    particle_filter->DynamMatr[7] = 1.0;
    //3行目
    particle_filter->DynamMatr[8] = 0.0;
    particle_filter->DynamMatr[9] = 0.0;
    particle_filter->DynamMatr[10] = 1.0;
    particle_filter->DynamMatr[11] = 0.0;
    //4行目
    particle_filter->DynamMatr[12] = 0.0;
    particle_filter->DynamMatr[13] = 0.0;
    particle_filter->DynamMatr[14] = 0.0;
    particle_filter->DynamMatr[15] = 1.0;
    
    //システムノイズの分布モデルを設定
    //平均0, 分散1の正規分布
    cvRandInit(&(particle_filter->RandS[0]), 0, 1, 1, CV_RAND_NORMAL);
    cvRandInit(&(particle_filter->RandS[1]), 0, 1, 2, CV_RAND_NORMAL);
    cvRandInit(&(particle_filter->RandS[2]), 0, 1, 3, CV_RAND_NORMAL);
    cvRandInit(&(particle_filter->RandS[3]), 0, 1, 4, CV_RAND_NORMAL);
    
    //追跡開始
    while (cvWaitKey(1) != 'q') {
        //画像取得
        image = cvQueryFrame(capture);
        
        //各サンプルの尤度を計算
        for (i = 0; i < num_of_particles; i++) {
            x = (int)(particle_filter->flSamples[i][0]);
            y = (int)(particle_filter->flSamples[i][1]);
            if (x<0 || x>=image->width || y<0 || y>=image->height) {
                //画像の範囲を超えたものは尤度を0に
                particle_filter->flConfidence[i] = 0.0;
            }else{
                //尤度の計算
                particle_filter->flConfidence[i] = Likelihood(image, x, y);
                
                //サンプル位置を描画
                cvCircle(image, cvPoint(x, y), 2, CV_RGB(0, 0, 255), 1, 8, 0);
            }
        }
        
        //結果を表示
        cvShowImage("Particle filter", image);
        
        //状態を更新
        cvConDensUpdateByTime(particle_filter);
    }
    
    cvReleaseImage(&image);
    cvReleaseConDensation(&particle_filter);
    
    return 0;
}

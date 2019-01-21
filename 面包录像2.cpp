//This program detects breads from the video and applies tracking while counting
#include<opencv2/opencv.hpp>
#include<opencv2/tracking.hpp>
#include<vector>
#include<math.h>
#include<iostream>
#define pi 3.1415
#define c_ratio 2
#define h_ratio 1.6

using namespace cv;
using namespace std;

//记录面包位置的类，用于面包的跟踪
class Bread
{
public:
	void display(Mat dst){
		circle(dst,center,radius+10,Scalar(0,255,255),2,8);
	}
	int compare(Point2f point, float r);
	void set(Point2f center, float radius){
		this->center = center;
		this->radius = radius;
	}
	Point2f getCenter(){return this->center;}
	float getRadius(){return this->radius+10.0;}
private:
	Point2f center;
	float radius;
};

//这个函数判断新检测出的轮廓是否有效
int Bread::compare(Point2f point, float r)
{
	if(radius/r > 4)
		return -1;
	float dx = center.x-point.x;
	float dy = center.y-point.y;
	double distance = sqrt(dx*dx+dy*dy);
	if(distance>1.2*(radius+r))                       //如果两圆心相距足够大，认为是新加入的面包
		return 0;
	else if(distance<10 && r>radius)
		return 1;                     //用于对已检测出的面包大小的修正
	else
		return -1;                    //其他情况，数据无效
}

void findBread(Mat&);
void workHorse(Mat&);
int bestParaProcess(Mat&, Mat&);

int breadNum = 0;                  //面包个数
int trackObject = 0;               //是否有被跟踪物体
vector<Bread> breads;              //储存全部面包信息
int bestLowV = -1;
int bestCloseSize, bestErodeSize;//最佳阈值
int maxScore = -10000;

int main(int argc, char** argv)
{
	int VideoWidth, VideoHeight;
	Mat frame;
	MultiTracker myTracker("KCF");  //利用"tracking.hpp"里的多目标跟踪类

	VideoCapture capture(argv[1]);
	VideoWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	VideoHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	VideoWriter writer(argv[2],(int)capture.get(CV_CAP_PROP_FOURCC), 20.0, Size(VideoWidth, VideoHeight));

	if(!capture.isOpened())
		return -1;

	bool stop = false;
	while(!stop){
		double time0 = getTickCount();
		capture>>frame;
		if(frame.rows==0 || frame.cols==0)  //视频结束退出
			break;

		if(trackObject){  //若有被跟踪物体，首先在新的一帧中更新breads中的面包信息
			vector<Rect2d> r;
			myTracker.update(frame,r);
			for(size_t i=0; i<r.size(); i++){
				Point2f newCenter = Point(r[i].x + r[i].width*0.5, r[i].y + r[i].height*0.5);
				breads[i].set(newCenter,(r[i].width+r[i].height)*0.25);
			}
		}
		int n=breadNum;
		findBread(frame);  //新图像上重新检测
		if(trackObject<0){ //添加第一个面包进入跟踪
			for(int i=n; i<breadNum;i++){
				Point2f CM = breads[i].getCenter();
				float sideLenth = breads[i].getRadius()*2;
				Rect selection(CM.x-sideLenth*0.5, CM.y-sideLenth*0.5, sideLenth, sideLenth);
				myTracker.add(frame, selection);
				trackObject = 1;
			}
		}
		for(size_t i=0; i<breads.size(); i++){
			breads[i].display(frame);
		}
		
		//显示帧率和面包数量
		char str1[20], str2[20];
		double FPS = getTickFrequency()/(getTickCount()-time0);  //帧率
		sprintf(str1, "FPS:%lf", FPS);
		sprintf(str2, "%d", breadNum);
		putText(frame, str1, Point(25,20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255), 1.5, CV_AA);
		putText(frame, str2, Point(VideoWidth-40,20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255), 1.5, CV_AA);
		imshow("tracker", frame);
		writer << frame;
		
		if(waitKey(1) == 27)
			break;
	}
	return 0;
}

//该函数用于在图像中检测面包并更新面包数据
void findBread(Mat& frame)
{
	vector<Mat> hsvSplit;
	Mat HSVImage, thresholdImage, edge;
	cvtColor(frame, HSVImage, COLOR_BGR2HSV);  //利用HSV色域进行颜色检测
	split(HSVImage, hsvSplit);
	equalizeHist(hsvSplit[2], hsvSplit[2]);
	merge(hsvSplit, HSVImage);

	//电脑在一定范围内自动选择最佳阈值
	workHorse(HSVImage);
	if(bestParaProcess(HSVImage, edge) == -1)
		return;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	int model=CV_RETR_EXTERNAL;
	int method = CV_CHAIN_APPROX_SIMPLE;
	findContours(edge,contours,hierarchy,model,method);  //查找轮廓
	for(size_t i=0; i<contours.size(); i++){
		float radius;
		Point2f center;
		double area = contourArea(contours[i]);
		if(area>314){  //检测轮廓大小是弥补形态学处理后的问题
			minEnclosingCircle(contours[i],center,radius);  //画出有效轮廓的最小包围圆
			if(!breadNum){  //breadNum为零，第一个面包
				breads.resize(1);
				breads[0].set(center,radius);
				breadNum++;
			}
			int flag = 1;
			for(size_t j=0; j<breads.size();j++){
				int s = breads[j].compare(center,radius);  //对画出圆的有效性检测
				if(s>0){
					breads[j].set(center,radius);
					flag = 0;
					break;
				}
				if(s<0){
					flag = 0;
					break;
				}
			}
			if(flag){
				breads.resize(breadNum+1);
				breads[breadNum].set(center,radius);
				breadNum++;
				trackObject = -1;
			}
		}
	}
}

void workHorse(Mat& HSVImage)
{
	int iLowV, closeSize, erodeSize;
	for(iLowV=207;iLowV<=215;iLowV++){
		Mat thresholdImage;
		inRange(HSVImage, Scalar(0, 55, iLowV), Scalar(22, 255, 255), thresholdImage);//化成二值图

		for(closeSize=3; closeSize<=5; closeSize++){
			Mat closedImg;
			Mat element = getStructuringElement(MORPH_ELLIPSE, Size(closeSize*2+1,closeSize*2+1));//闭运算填补漏洞
			morphologyEx(thresholdImage,closedImg,CV_MOP_CLOSE,element);
			
			for(erodeSize=3; erodeSize<=5; erodeSize++){
				Mat erodeImg;
				Mat element2 = getStructuringElement(MORPH_RECT, Size(erodeSize*2+1,erodeSize*2+1));//腐蚀断开相邻面包
				erode(closedImg,erodeImg,element2);
				erode(erodeImg,erodeImg,element2);

				Mat dilateImg;
				Mat element3 = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
				dilate(erodeImg,dilateImg,element3);

				Mat edge = dilateImg-erodeImg;
				vector<vector<Point>> contours;
				vector<Vec4i> hierarchy;
				int model=CV_RETR_EXTERNAL;//改进：放到循环外面
				int method = CV_CHAIN_APPROX_SIMPLE;
				findContours(edge,contours,hierarchy,model,method);
				vector<vector<Point>> hull(contours.size());
				
				int nBig, nSmall, nTwisted;
				int flag = 0;
				nBig = nSmall = nTwisted = 0;
				for(size_t i=0; i<contours.size(); i++){
					double area = contourArea(contours[i]);
					float radius;
					Point2f center;
					minEnclosingCircle(contours[i], center, radius);
					convexHull(Mat(contours[i]), hull[i], false);
					double hullArea = contourArea(hull[i]);
					double circleArea = radius*radius*pi;
					if(area>13*13*pi){//大轮廓
						if(hullArea/area > h_ratio && area > 2500){//连通轮廓
							flag = 1;
							break;
						}
						if(circleArea/area > c_ratio)
							nTwisted++;
						nBig++;
					}
					else if(area<10*10*pi)//小轮廓
						nSmall++;
				}
				int score;
				if(!flag)
					score = 14*nBig - 5*nSmall - 5*nTwisted;
				else
					score = -9999;
				if(score>maxScore){
					maxScore = score;
					bestLowV = iLowV;
					bestCloseSize = closeSize;
					bestErodeSize = erodeSize;
				}
				if(score>20)
					return;
			}
		}
	}
}

int bestParaProcess(Mat& HSVImage, Mat& edge)
{
	if(bestLowV < 0)
		return -1;
	Mat thresholdImage;
	inRange(HSVImage, Scalar(0, 55, bestLowV), Scalar(22, 255, 255), thresholdImage);

	Mat closedImg;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(bestCloseSize*2+1,bestCloseSize*2+1));
	morphologyEx(thresholdImage,closedImg,CV_MOP_CLOSE,element);

	Mat erodeImg;
	Mat element2 = getStructuringElement(MORPH_RECT, Size(bestErodeSize*2+1,bestErodeSize*2+1));
	erode(closedImg,erodeImg,element2);
	erode(erodeImg,erodeImg,element2);

	Mat dilateImg;
	Mat element3 = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
	dilate(erodeImg,dilateImg,element3);

	edge = dilateImg-erodeImg;
	return 0;
}


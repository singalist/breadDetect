//This program detects breads from the video and applies tracking while counting
#include<opencv2/opencv.hpp>
#include<opencv2/tracking.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<vector>
#include<math.h>
#include<iostream>

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
	float dx = center.x-point.x;
	float dy = center.y-point.y;
	double distance = sqrt(dx*dx+dy*dy);
	if(distance>85)                       //如果两圆心相距大于85，认为是新加入的面包
		return 0;
	else if(distance<10 && r>radius)
		return 1;                     //用于对已检测出的面包大小的修正
	else
		return -1;                    //其他情况，数据无效
}

void findBread(Mat);

int breadNum = 0;                  //面包个数
int trackObject = 0;               //是否有被跟踪物体
vector<Bread> breads;              //储存全部面包信息
Mat frame;

int main(int argc, char** argv)
{
	MultiTracker myTracker("KCF");  //利用"tracking.hpp"里的多目标跟踪类

	VideoCapture capture(argv[1]);
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
		

		char str1[10], str2[10];
		double FPS = getTickFrequency()/(getTickCount()-time0);  //帧率
		sprintf(str1, "FPS:%lf", FPS);
		sprintf(str2, "%d", breadNum);
		putText(frame, str1, Point(25,20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255), 1.5, CV_AA);
		putText(frame, str2, Point(315,20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255), 1.5, CV_AA);
		imshow("tracker", frame);
		
		if(waitKey(1) == 27)
			break;
	}
	return 0;
}

//该函数用于在图像中检测面包并更新面包数据
void findBread(Mat frame)
{
	vector<Mat> hsvSplit;
	Mat HSVImage, thresholdImage;
	cvtColor(frame, HSVImage, COLOR_BGR2HSV);  //利用HSV色域进行颜色检测
	split(HSVImage, hsvSplit);
	equalizeHist(hsvSplit[2], hsvSplit[2]);
	merge(hsvSplit, HSVImage);
	inRange(HSVImage, Scalar(0,55,210), Scalar(22,255,255), thresholdImage);  //二值化

	//以下的形态学处理把连在一起的面包分离开来
	Mat closedImg, erodeImg, dilateImg;
	Mat element1 = getStructuringElement(MORPH_ELLIPSE, Size(9,9));
	morphologyEx(thresholdImage,closedImg,CV_MOP_CLOSE,element1);
//	imshow("step1", closedImg);

	Mat element2 = getStructuringElement(MORPH_RECT, Size(9,9));
	erode(closedImg,erodeImg,element2);
	erode(erodeImg,erodeImg,element2);
//	imshow("step2", erodeImg);

	Mat element3 = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
	dilate(erodeImg,dilateImg,element3);
//	imshow("step3", dilateImg);

	Mat edge = dilateImg-erodeImg;
//	imshow("edge", edge);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	int model=CV_RETR_EXTERNAL;
	int method = CV_CHAIN_APPROX_SIMPLE;
	findContours(edge,contours,hierarchy,model,method);  //查找轮廓
	for(int i=0;i>=0;i=hierarchy[i][0]){
		float radius;
		Point2f center;
		double area = contourArea(contours[i]);
		if(area>100){  //检测轮廓大小是弥补形态学处理后的问题
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


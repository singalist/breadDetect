#include<iostream>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

int iLowH = 0;
int iHighH = 22;
int iLowS = 55;
int iHighS = 255;
int iLowV = 209;
int iHighV = 255;
Mat srcImage, HSVImage, thresholdImage;

void on_range(int, void*)
{
	Mat local = srcImage.clone();
	inRange(HSVImage, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), thresholdImage);
	imshow("3", thresholdImage);

/*	Mat blurImg;
	blur(thresholdImage, blurImg, Size(9,9));
	imshow("blur", blurImg);*/
	Mat closedImg;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(9,9));
	morphologyEx(thresholdImage,closedImg,CV_MOP_CLOSE,element);
//	morphologyEx(closedImg,closedImg,CV_MOP_CLOSE,element);
	imshow("step1", closedImg);

	Mat erodeImg;
	Mat element2 = getStructuringElement(MORPH_RECT, Size(9,9));
	erode(closedImg,erodeImg,element2);
	erode(erodeImg,erodeImg,element2);
//	imshow("step2", erodeImg);

	Mat dilateImg;
	Mat element3 = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
	dilate(erodeImg,dilateImg,element3);
//	imshow("step3", dilateImg);

	Mat edge = dilateImg-erodeImg;
	imshow("edge", edge);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	int model=CV_RETR_EXTERNAL;
	int method = CV_CHAIN_APPROX_SIMPLE;
	int counter = 0;
	findContours(edge,contours,hierarchy,model,method);
	for(int i=0;i>=0;i=hierarchy[i][0]){
		float radius;
		Point2f center;
		minEnclosingCircle(contours[i],center,radius);
		circle(local,center,radius+10,Scalar(255,0,0),2,8);
		counter++;
	}
	cout<<counter<<endl;
	imshow("src with fitting-circles",local);
}


int main(int argc, char** argv)
{
	namedWindow("Control", CV_WINDOW_AUTOSIZE);

	createTrackbar("LowH","Control",&iLowH, 179, on_range);
	createTrackbar("HighH","Control",&iHighH, 179, on_range);
	createTrackbar("LowS","Control",&iLowS, 255, on_range);
	createTrackbar("HighS","Control",&iHighS, 255, on_range);
	createTrackbar("LowV","Control",&iLowV, 255, on_range);
	createTrackbar("HighV","Control",&iHighV, 255, on_range);

	vector<Mat> hsvSplit;

	srcImage = imread(argv[1],1);
	cvtColor(srcImage, HSVImage, COLOR_BGR2HSV);
//	imshow("1", HSVImage);

	split(HSVImage, hsvSplit);
//	imshow("red", hsvSplit[2]);

	equalizeHist(hsvSplit[2], hsvSplit[2]);
//	imshow("red2", hsvSplit[2]);

	merge(hsvSplit, HSVImage);
	imshow("2", HSVImage);
	
	on_range(0,0);
	waitKey(0);
}


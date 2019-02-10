#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video.hpp"
#include <iostream>
using namespace std;
using namespace cv;
static void help()
{
	printf("Usage: dis_optflow <video_file>\n");
}
int main(int argc, char **argv)
{
	VideoCapture cap;
	if (argc < 2)
	{
		help();
		exit(1);
	}
	cap.open(argv[1]);
	if (!cap.isOpened())
	{
		printf("ERROR: Cannot open file %s\n", argv[1]);
		return -1;
	}
	Mat prevgray, gray, rgb, frame;
	Mat flow, flow_uv[2];
	Mat flow_Farneback;
	Mat flow_uv_Farneback[2];
	Mat mag, ang;
	Mat mag_Farneback, ang_Farneback;
	Mat hsv_split[3], hsv;
	Mat hsv_split_Farneback[3], hsv_Farneback;
	Mat rgb_Farneback;


	Ptr<DenseOpticalFlow> algorithm = DISOpticalFlow::create(DISOpticalFlow::PRESET_MEDIUM);

	int idx = 0;
	while (true)
	{
		cap >> frame;
		if (frame.empty())
			break;
		cv::resize(frame, frame, cv::Size(0.8*frame.cols, 0.8*frame.rows), 0, 0, cv::INTER_LINEAR);
		idx++;
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		cv::imshow("orig", frame);

		if (!prevgray.empty())
		{
			/*DISOpticalFlow*/
			/*main function of DISOpticalFlow*/
			algorithm->calc(prevgray, gray, flow);
			split(flow, flow_uv);
			multiply(flow_uv[1], -1, flow_uv[1]);
			cartToPolar(flow_uv[0], flow_uv[1], mag, ang, true);
			normalize(mag, mag, 0, 1, NORM_MINMAX);
			hsv_split[0] = ang;
			hsv_split[1] = mag;
			hsv_split[2] = Mat::ones(ang.size(), ang.type());
			merge(hsv_split, 3, hsv);
			cvtColor(hsv, rgb, COLOR_HSV2BGR);
			cv::Mat rgbU;
			rgb.convertTo(rgbU, CV_8UC3, 255, 0);
			cv::imshow("DISOpticalFlow", rgbU);
			Mat rgbU_b = rgbU.clone();
			Mat split_dis[3];
			split(rgbU_b, split_dis);

			split_dis[2] = prevgray;
			Mat merge_dis;
			merge(split_dis, 3, merge_dis);
			cv::imshow("DISOpticalFlow_mask", merge_dis);
			/*Farneback*/
			cv::calcOpticalFlowFarneback(prevgray, gray, flow_Farneback, 0.5, 3, 15, 3, 5, 1.2, 0);

			split(flow_Farneback, flow_uv_Farneback);
			multiply(flow_uv_Farneback[1], -1, flow_uv_Farneback[1]);
			cartToPolar(flow_uv_Farneback[0], flow_uv_Farneback[1], mag_Farneback, ang_Farneback, true);
			normalize(mag_Farneback, mag_Farneback, 0, 1, NORM_MINMAX);
			hsv_split_Farneback[0] = ang_Farneback;
			hsv_split_Farneback[1] = mag_Farneback;
			hsv_split_Farneback[2] = Mat::ones(ang_Farneback.size(), ang_Farneback.type());
			merge(hsv_split_Farneback, 3, hsv_Farneback);
			cvtColor(hsv_Farneback, rgb_Farneback, COLOR_HSV2BGR);
			cv::Mat rgbU_Farneback;
			rgb_Farneback.convertTo(rgbU_Farneback, CV_8UC3, 255, 0);
			cv::imshow("FlowFarneback", rgbU_Farneback);
			Mat rgbU_Farneback_b = rgbU_Farneback.clone();
			Mat split_Fb[3];
			split(rgbU_Farneback_b, split_Fb);
			split_Fb[2] = prevgray;
			Mat merge_Fb;
			merge(split_Fb, 3, merge_Fb);
			cv::imshow("FlowFarneback_mask", merge_Fb);
			cv::waitKey(1);
		}


		std::swap(prevgray, gray);
	}

	return 0;
}
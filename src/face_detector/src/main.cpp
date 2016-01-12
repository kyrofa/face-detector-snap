#include <iostream>
#include <string>
#include <vector>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

namespace Classifiers
{
	cv::CascadeClassifier FACE;
}

namespace Publishers
{
	ros::Publisher FACE_PUBLISHER;
}

void detectFaces(const cv_bridge::CvImageConstPtr &image, std::vector<cv::Rect> &faces)
{
	// Convert to grayscale to do the face detection.
	cv_bridge::CvImagePtr grayscaleImage = cv_bridge::cvtColor(image, sensor_msgs::image_encodings::MONO8);
	if (!grayscaleImage)
	{
		ROS_ERROR("Unable to convert image to grayscale-- cannot detect faces");
		return;
	}

	try
	{
		cv::equalizeHist(grayscaleImage->image, grayscaleImage->image);
	}
	catch(const cv::Exception &exception)
	{
		ROS_ERROR("Unable to equalize histogram: %s. Cannot detect faces", exception.what());
		return;
	}

	// Limit the face size to 1/10th of the image
	int minimumFaceSize = grayscaleImage->image.cols / 10;

	try
	{
		Classifiers::FACE.detectMultiScale(grayscaleImage->image, faces,
		                                   1.1, 2, CV_HAAR_DO_CANNY_PRUNING,
		                                   cv::Size(minimumFaceSize,
		                                            minimumFaceSize));
	}
	catch(const cv::Exception &exception)
	{
		ROS_ERROR("Unable to detect faces: %s", exception.what());
		return;
	}
}

void imageReceivedCallback(const sensor_msgs::Image::ConstPtr &image)
{
	cv_bridge::CvImagePtr cvImage = cv_bridge::toCvCopy(image);

	std::vector<cv::Rect> faces;
	detectFaces(cvImage, faces);

	
	for (const auto &face : faces)
	{
		cv::Point center(face.x + face.width*0.5, face.y + face.height*0.5);
		cv::ellipse(cvImage->image, center, cv::Size(face.width*0.5, face.height*0.5), 0, 0, 360, cv::Scalar(255, 0, 255), 4, 8, 0);
	}

	Publishers::FACE_PUBLISHER.publish(cvImage->toImageMsg());
}

int main(int argc, char *argv[])
{
	ros::init(argc, argv, "face_detector");

	// Obtain parameters
	ros::NodeHandle privateNodeHandle("~");
	std::string faceClassifierPath;
	privateNodeHandle.param("face_classifier_path", faceClassifierPath,
	                        std::string("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"));

	if (!Classifiers::FACE.load(faceClassifierPath))
	{
		ROS_FATAL("Unable to load face classifier.");
	}

	// Setup publisher and subsriber and spin.
	ros::NodeHandle nodeHandle;
	Publishers::FACE_PUBLISHER = nodeHandle.advertise<sensor_msgs::Image>("faces_image", 1);
	ros::Subscriber subscriber = nodeHandle.subscribe("image", 1, imageReceivedCallback);

	ros::spin();

	return 0;
}

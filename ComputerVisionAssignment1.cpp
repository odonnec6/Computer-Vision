#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

using namespace cv;
using namespace std;

Mat src; Mat src_gray;//Mat roi_gray;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);


/// Function header
void thresh_callback(int, void* );

/*function main */
int main( int argc, char** argv )
{
/// Load source image and convert it to gray
	argv[1] = "c:\\Glue1.jpg";				//testing the various bottles of glue images
	//argv[1] = "c:\\Glue2.jpg";
	//argv[1] = "c:\\Glue3.jpg";
	//argv[1] = "c:\\Glue4.jpg";
	//argv[1] = "c:\\Glue5.jpg";
	//argv[1] = "c:\\Glue6.jpg";
	
  src = imread( argv[1], 1 );
  /// Convert image to gray and blur it
  cvtColor(src , src_gray, CV_BGR2GRAY );
  blur( src_gray, src_gray, Size(3,3) );

  /// Create Window
  char* source_window = "Source";
  namedWindow( source_window, CV_WINDOW_AUTOSIZE );
  imshow( source_window, src );

  createTrackbar( " Threshold:", "Source", &thresh, max_thresh, thresh_callback );

  thresh_callback( 0, 0 );

  waitKey(0);
  return(0);
}

//method to place writing onto bottle with a label
void set_label(cv::Mat& im, cv::Rect r, const std::string label)
{
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.5;
    int thickness = 1;
    int baseline = 0;

    cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    cv::Point pt(r.x + (r.width-text.width)/2, r.y + (r.height+text.height)/2);

    cv::rectangle(
        im, 
        pt + cv::Point(0, baseline), 
        pt + cv::Point(text.width, -text.height), 
        CV_RGB(255,0,0), CV_FILLED
    );

    cv::putText(im, label, pt, fontface, scale, CV_RGB(255,255,0), thickness, 8);
}


/*function thresh_callback */
void thresh_callback(int, void* )
{
  Mat threshold_output;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  /// Detect edges using Threshold
  threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY );
  /// Find contours
  findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
  /// Approximate contours to polygons + get bounding rects
  vector<vector<Point> > contours_poly( contours.size() );
  vector<Rect> boundRect( contours.size() );
  
  int largest_area1=0;
  int largest_contour_index1=0;
  int largest_area=0;
  int largest_contour_index=0;
  Rect bounding_rect;
  Rect bounding_rect1;
  
  //int black_pixel_count=0;      attempted to count black pixels within bounding boxes but din't work

  //this finds all bounding rectangles in the image
  for( int i = 0; i < contours.size(); i=hierarchy[i][1])
     { approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
       boundRect[i] = boundingRect( Mat(contours_poly[i]) );
	   double a=contourArea( contours[i],false);
	   if(a>largest_area){
       largest_area=a;
       largest_contour_index=i;                //Store the index of largest contour
       bounding_rect=boundingRect(contours[i]);// Find the bounding rectangle for biggest contour
	   //black_pixel_count = cv::countNonZero(boundRect[i]); couldn't get black pixel count working within the bounding boxes
	   }
	
  }
  //this finds all bounding rectangles in the image
  for( int j = 0; j < contours.size(); j++)
     { approxPolyDP( Mat(contours[j]), contours_poly[j], 3, true );
       boundRect[j] = boundingRect( Mat(contours_poly[j]) );
	   double a1=contourArea( contours[j],false);
	   if(a1>largest_area1){
       largest_area1=a1;
       largest_contour_index1=j;                //Store the index of largest contour
       bounding_rect1=boundingRect(contours[j]);// Find the bounding rectangle for biggest contour
	   //black_pixel_count = cv::countNonZero(boundRect[i]); couldn't get black pixel count working within the bounding boxes
	   }
	
  }
  
  Mat roi = src( bounding_rect );		//used for presenting an image of the current region with a label based on threshold in a new window

  vector<Point> approx;
  
  /// Draw polygonal contour + bounding rectangles
  Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
  Mat dst(src.rows,src.cols,CV_8UC1,Scalar::all(0));


  for( int i = 0; i< contours.size(); i++ )
     {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       drawContours( dst, contours,largest_contour_index, color, CV_FILLED, 8, hierarchy ); // Draw the largest contour in hierarchy using previously stored index.
	   rectangle(drawing, bounding_rect,  Scalar(0,255,0),1, 8,0);
	   drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
       rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
	   set_label(drawing, bounding_rect, "Label");
     }

  /// Show in a window
  imshow("Current Region",roi);
  imshow( "src", drawing );
  imshow( "largest Contour", dst );
  //cout<<black_pixel_count;
}
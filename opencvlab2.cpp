#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "Utilities.h"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

//#include "Binary.cpp"


using namespace cv;
using namespace std;

typedef std::vector<cv::Point> Contour;

/// Global variables

Mat erosion_dst, dilation_dst;

int erosion_elem = 0;
int erosion_size = 0;
int dilation_elem = 0;
int dilation_size = 0;
int const max_elem = 2;
int const max_kernel_size = 21;

Mat src;Mat src1; Mat src2; Mat src_gray;Mat src_gray1; Mat edge; Mat draw1; Mat draw;//Mat roi_gray;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);
/// Global Variables
Mat imgz; Mat templ; Mat result;
char* image_window = "Source Image";
char* result_window = "Result window";

int match_method;
int max_Trackbar = 5;

/// Function Headers
//void MatchingMethod( int, void* );
//int current_threshold = 128;
//int max_threshold = 255;


int threshold_value = 0;
int threshold_type = 3;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;
 
Mat dst;
char* window_name = "Threshold Demo";
 
char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
char* trackbar_value = "Value";
	

/// Global variables
char* source_window = "Source image";
char* warp_window = "Warp";
char* warp_rotate_window = "Warp + Rotate";



void Threshold_Demo1( int, void* )
{
  /* 0: Binary
     1: Binary Inverted
     2: Threshold Truncated
     3: Threshold to Zero
     4: Threshold to Zero Inverted
   */
 
  //threshold( src_gray, dst, threshold_value, max_BINARY_value,threshold_type );
	threshold( src_gray, dst, 115, max_BINARY_value,0 );
 
  //imshow( "threshold1", dst );
   Canny( dst , edge, 50, 150, 3);
 
    edge.convertTo(draw1, CV_8U);
   // namedWindow("threshedges", CV_WINDOW_AUTOSIZE);
   // imshow("threshedges", draw1);




	


 Mat threshchannels;Mat edge1;Mat drawthreshedges;
  Mat hsv;
vector<Mat> channels;
split(src, channels);
threshold( channels[0], threshchannels, 115, max_BINARY_value,0 );

Mat drawingerosion;

  // Create a structuring element
       int erosion_size = 3;  
       Mat element = getStructuringElement(cv::MORPH_CROSS,
              cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
              cv::Point(erosion_size, erosion_size) );
 
       
       erode(threshchannels,threshchannels,element);  


	   imshow("eroded",threshchannels);

Mat drawingdilation;

  // Create a structuring element
       int dilation_size = 10;  
       Mat element1 = getStructuringElement(cv::MORPH_CROSS,
              cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
              cv::Point(dilation_size, dilation_size) );
 
       // Apply dilation or dilation on the image
       dilate(threshchannels,threshchannels,element1);  // dilate(image,dst,element);


	   imshow("dilated",threshchannels);

	  

//imshow("threshch",threshchannels);
//imshow("h channel",channels[1]); 

Canny( threshchannels , edge1, 50, 150, 3);
 
    edge1.convertTo(drawthreshedges, CV_8U);
    namedWindow("threshedges", CV_WINDOW_AUTOSIZE);
    imshow("threshedges", drawthreshedges);


	
Mat drawingdilation2;

  // Create a structuring element
       int dilation_size2 = 3;  
       Mat element2 = getStructuringElement(cv::MORPH_CROSS,
              cv::Size(2 * dilation_size2 + 1, 2 * dilation_size2 + 1),
              cv::Point(dilation_size2, dilation_size2) );
 
       
       dilate(drawthreshedges,drawthreshedges,element2);  


	   imshow("eroded2",drawthreshedges);



  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;


	findContours( draw1, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

  /// Approximate contours to polygons + get bounding rects and circles
  vector<vector<Point> > contours_poly( contours.size() );
  vector<Rect> boundRect( contours.size() );
  vector<Point2f>center( contours.size() );
  vector<float>radius( contours.size() );


  Rect bounding_rectx;
  Mat roi1 = src( bounding_rectx );
   vector<RotatedRect> minRect( contours.size() );
  vector<RotatedRect> minEllipse( contours.size() );

  for( int i = 0; i < contours.size(); i++ )
     { minRect[i] = minAreaRect( Mat(contours[i]) );
       if( contours[i].size() > 5 )
         { minEllipse[i] = fitEllipse( Mat(contours[i]) ); }
     }






   Mat drawing7 = Mat::zeros( draw1.size(), CV_8UC3 );
   Mat drawing6 = Mat::zeros( draw1.size(), CV_8UC3 );

  /// Draw contours + rotated rects + ellipses
  Mat drawing2 = Mat::zeros( draw1.size(), CV_8UC3 );
  Mat dst(src.rows,src.cols,CV_8UC1,Scalar::all(0));
  for( int i = 0; i< contours.size(); i++ )
     {
	   if (contours[i].size() > 350){
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );




	   drawContours( dst, contours,i, color, CV_FILLED, 8, vector<Vec4i>(), 0, Point() ); // Draw the largest contour in hierarchy using previously stored index.
	   //rectangle(drawing2, bounding_rectx,  Scalar(0,255,0),1, 8,0);
	   drawContours( drawing7, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
       rectangle( drawing6, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );

	   //Mat roi = src(  );
       // contour
       drawContours( drawing2, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
       // ellipse
       ellipse( drawing2, minEllipse[i], color, 2, 8 );
       // rotated rectangle
       Point2f rect_points[4]; minRect[i].points( rect_points );
       for( int j = 0; j < 4; j++ )
          line( drawing2, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );
	   }
     }

    namedWindow( "Contours2", CV_WINDOW_AUTOSIZE );
  imshow( "Contours2", drawing2);

  /// Approximate contours to polygons + get bounding rects
  vector<vector<Point> > contours_poly1( contours.size() );
  vector<Rect> boundRect1( contours.size() );
  
  int largest_area1=0;
  int largest_contour_index1=0;
  int largest_area=0;
  int largest_contour_index=0;
  Rect bounding_rect;
  Rect bounding_rect1;
  
  //int black_pixel_count=0;      attempted to count black pixels within bounding boxes but din't work
  
  //this finds all bounding rectangles in the image
  for( int f = 0; f < contours.size(); f++)
     { approxPolyDP( Mat(contours[f]), contours_poly1[f], 3, true );
       boundRect1[f] = boundingRect( Mat(contours_poly1[f]) );
	   double a=contourArea( contours[f],false);
	   if(contourArea(contours[f]) > 1000){
	   if(a>largest_area){
       largest_area=a;
       largest_contour_index=f;                //Store the index of largest contour
       bounding_rect=boundingRect(contours[f]);// Find the bounding rectangle for biggest contour
	   //black_pixel_count = cv::countNonZero(boundRect[i]); couldn't get black pixel count working within the bounding boxes
	   }
	   }
	  // =hierarchy[f][0]
	
  }
  
  //this finds all bounding rectangles in the image
  for( int j = 0; j < contours.size(); j++)
     { approxPolyDP( Mat(contours[j]), contours_poly1[j], 3, true );
       boundRect1[j] = boundingRect( Mat(contours_poly1[j]) );
	   double a1=contourArea( contours[j],false);
	   if(a1>largest_area1){
       largest_area1=a1;
       largest_contour_index1=j;                //Store the index of largest contour
       bounding_rect1=boundingRect(contours[j]);// Find the bounding rectangle for biggest contour
	   //black_pixel_count = cv::countNonZero(boundRect[i]); couldn't get black pixel count working within the bounding boxes
	   }
	
  }
  
  Mat roi = src( bounding_rect );		//used for presenting an image of the current region with a label based on threshold in a new window
  Mat roi2 = src( bounding_rect1 );	
  vector<Point> approx;
  
  /// Draw polygonal contour + bounding rectangles
  Mat drawing9 = Mat::zeros( draw1.size(), CV_8UC3 );
  Mat dst5(src.rows,src.cols,CV_8UC1,Scalar::all(0));


  for( int i = 0; i< contours.size(); i++ )
     {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       drawContours( dst5, contours,largest_contour_index, color, CV_FILLED, 8, hierarchy ); // Draw the largest contour in hierarchy using previously stored index.
	   rectangle(drawing9, bounding_rect,  Scalar(0,255,0),1, 8,0);
	   drawContours( drawing9, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
       rectangle( drawing9, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
	  // set_label(drawing, bounding_rect, "Label");
     }

  /// Show in a window
 // imshow("Current Region",roi);
// imshow("Current Region2",roi2);

 
     
	Mat imgmask;
   // cvtColor(src2, img1, CV_RGB2GRAY);

  //  Canny(threshchannels, imgmask, 100, 200);
	 Canny(drawthreshedges, imgmask, 100, 200);

    // find the contours
    vector< vector<Point> > contoursmask;
    findContours(imgmask, contoursmask, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    Mat mask = Mat::zeros(imgmask.rows, imgmask.cols, CV_8UC1);

    // CV_FILLED fills the connected components found
    drawContours(mask, contoursmask, -1, Scalar(255), CV_FILLED);
    Mat crop(src2.rows, src2.cols, CV_8UC3);

    // set background to green
    crop.setTo(Scalar(255,0,0));

    src2.copyTo(crop, mask);

    // normalize so imwrite(...)/imshow(...) shows the mask correctly!
    normalize(mask.clone(), mask, 0.0, 255.0, CV_MINMAX, CV_8UC1);

    //imshow("original", src2);
    imshow("mask", mask);
  //  imshow("canny", imgmask);
    imshow("cropped", crop);




	Rect biggestcontour;
 vector<vector<Point> > contours_polymask( contours.size() );
  vector<Rect> boundRectmask( contours.size() );

   Mat drawingmask7 = Mat::zeros( imgmask.size(), CV_8UC3 );
   Mat drawingmask6 = Mat::zeros( imgmask.size(), CV_8UC3 );

  /// Draw contours + rotated rects + ellipses
  Mat drawingmask = Mat::zeros( imgmask.size(), CV_8UC3 );
  Mat dstmask(imgmask.rows,imgmask.cols,CV_8UC1,Scalar::all(0));
  Mat dstmaybe(src.rows,src.cols,CV_8UC1,Scalar::all(0));
  for( int i = 0; i< contoursmask.size(); i++ )
     {
	   if (contoursmask[i].size() > 350){
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );




	   drawContours( dstmask, contoursmask,i, color, CV_FILLED, 8, vector<Vec4i>(), 0, Point() ); // Draw the largest contour in hierarchy using previously stored index.
	   //rectangle(drawing2, bounding_rectx,  Scalar(0,255,0),1, 8,0);
	   drawContours( drawingmask7, contours_polymask, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
       rectangle( drawingmask6, boundRectmask[i].tl(), boundRectmask[i].br(), color, 2, 8, 0 );

       // contour
       drawContours( drawingmask, contoursmask, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
       // ellipse
      // ellipse( drawingmask, minEllipse[i], color, 2, 8 );
       // rotated rectangle
       Point2f rect_points[4]; minRect[i].points( rect_points );
       for( int j = 0; j < 4; j++ )
          line( drawingmask, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );
	   }
	    if (contourArea(contoursmask[i])> 500){
	   
		}

	     
     }

    namedWindow( "Contoursmask", CV_WINDOW_AUTOSIZE );
  imshow( "Contoursmask", drawingmask);



  Point2f srcTri[3];
   Point2f dstTri[3];

   Mat rot_mat( 2, 3, CV_32FC1 );
   Mat warp_mat( 2, 3, CV_32FC1 );
   Mat warp_dst, warp_rotate_dst;

   /// Set the dst image the same type and size as src
   warp_dst = Mat::zeros( src2.rows, src2.cols, src2.type() );

   /// Set your 3 points to calculate the  Affine Transform
   srcTri[0] = Point2f( 0,0 );
   srcTri[1] = Point2f( src2.cols - 1, 0 );
   srcTri[2] = Point2f( 0, src2.rows - 1 );

   dstTri[0] = Point2f( src2.cols*0.0, src2.rows*0.0);
   dstTri[1] = Point2f( src2.cols*1-1, src2.rows*0.0 );
   dstTri[2] = Point2f( src2.cols*0.0, src2.rows*1);

   /// Get the Affine Transform
   warp_mat = getAffineTransform( srcTri, dstTri );

   /// Apply the Affine Transform just found to the src image
   warpAffine( src2, warp_dst, warp_mat, warp_dst.size() );

   /** Rotating the image after Warp */

   /// Compute a rotation matrix with respect to the center of the image
   Point center1 = Point( warp_dst.cols/2, warp_dst.rows/2 );
   double angle = 35.0;
   double scale = 1.0;

   /// Get the rotation matrix with the specifications above
   rot_mat = getRotationMatrix2D( center1, angle, scale );

   /// Rotate the warped image
   warpAffine( warp_dst, warp_rotate_dst, rot_mat, warp_dst.size() );

   namedWindow( warp_rotate_window, CV_WINDOW_AUTOSIZE );
   imshow( warp_rotate_window, warp_rotate_dst );


   Point2f inputQuad[4]; 
    Point2f outputQuad[4];
         
    // Lambda Matrix
    Mat lambda( 2, 4, CV_32FC1 );
    //Input and Output Image;
    Mat input, output;
     
    //Load the image
    input = warp_rotate_dst;
    // Set the lambda matrix the same type and size as input
    lambda = Mat::zeros( input.rows, input.cols, input.type() );
 
    // The 4 points that select quadilateral on the input , from top-left in clockwise order
    // These four pts are the sides of the rect box used as input 
    inputQuad[0] = Point2f( 75,120 );
    inputQuad[1] = Point2f( input.cols+50,-50);
    inputQuad[2] = Point2f( input.cols,input.rows);
    inputQuad[3] = Point2f( 100,input.rows+100  );  
    // The 4 points where the mapping is to be done , from top-left in clockwise order
    outputQuad[0] = Point2f( 0,0 );
    outputQuad[1] = Point2f( input.cols-1,0);
    outputQuad[2] = Point2f( input.cols-1,input.rows-1);
    outputQuad[3] = Point2f( 0,input.rows-1  );
 
    // Perspective Transform Matrix i.e. lambda 
    lambda = getPerspectiveTransform( inputQuad, outputQuad );
    // Apply the Perspective Transform just found to the src image
    warpPerspective(input,output,lambda,output.size() );
 
    //Display input and output
    imshow("Input",warp_rotate_dst);
    imshow("Output",output);



	
   /// Source image to display
  Mat img_display;
  output.copyTo( img_display );

  /// result matrix
  int result_cols =  output.cols - templ.cols + 1;
  int result_rows = output.rows - templ.rows + 1;

  result.create( result_rows, result_cols, CV_32FC1 );

  ///Matching and Normalize
  matchTemplate( output, templ, result, match_method );
  normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );

  /// Localizing the best match with minMaxLoc
  double minVal; double maxVal; Point minLoc; Point maxLoc;
  Point matchLoc;

  minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

  /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
  if( match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED )
    { matchLoc = minLoc;
  cout<<"good match:" ;
  cout<< matchLoc;}
  else
    { matchLoc = maxLoc; 
   cout<<"good match:" ;
  cout<< matchLoc;}
  
  rectangle( img_display, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar(255,0,255), 2, 8, 0 );
  rectangle( result, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar(255,0,255), 2, 8, 0 );

  imshow( image_window, img_display );
  imshow( result_window, result );
  imshow( "template", templ );
  

  

}






/**  @function Erosion  *//*
void Erosion( int, void* )
{
  int erosion_type;
  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( erosion_type,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );
  /// Apply the erosion operation
  erode( draw, erosion_dst, element );
  imshow( "Erosion Demo", erosion_dst );
}
*/
/** @function Dilation *//*
void Dilation( int, void* )
{
  int dilation_type;
  if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
  else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
  else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( dilation_type,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
  /// Apply the dilation operation
  dilate( draw, dilation_dst, element );
  imshow( "Dilation Demo", dilation_dst );
}

*/






int main( int argc, char** argv )
{
/// Load source image and convert it to gray
	argv[1] = "c:\\Users/Books/BookView01.jpg";			
	argv[2] = "c:\\Users/Books/BookView02.jpg";
	argv[3] = "c:\\Users/Books/BookView03.jpg";				
	argv[4] = "c:\\Users/Books/BookView04.jpg";
	argv[5] = "c:\\Users/Books/BookView05.jpg";				
	argv[6] = "c:\\Users/Books/BookView06.jpg";
	argv[7] = "c:\\Users/Books/BookView07.jpg";			
	argv[8] = "c:\\Users/Books/BookView08.jpg";
	argv[9] = "c:\\Users/Books/BookView09.jpg";		
	argv[10] = "c:\\Users/Books/BookView10.jpg";
	argv[11] = "c:\\Users/Books/BookView11.jpg";
	argv[12] = "c:\\Users/Books/BookView12.jpg";
	argv[13] = "c:\\Users/Books/BookView13.jpg";
	argv[14] = "c:\\Users/Books/BookView14.jpg";			
	argv[15] = "c:\\Users/Books/BookView15.jpg";
	argv[16] = "c:\\Users/Books/BookView16.jpg";			
	argv[17] = "c:\\Users/Books/BookView17.jpg";
	argv[18] = "c:\\Users/Books/Page01.jpg";		//start of pages	
	argv[19] = "c:\\Users/Books/Page02.jpg";
	argv[20] = "c:\\Users/Books/Page03.jpg";				
	argv[21] = "c:\\Users/Books/Page04.jpg";
	argv[22] = "c:\\Users/Books/Page05.jpg";				
	argv[23] = "c:\\Users/Books/Page06.jpg";
	argv[24] = "c:\\Users/Books/Page07.jpg";			
	argv[25] = "c:\\Users/Books/Page08.jpg";
	argv[26] = "c:\\Users/Books/Page09.jpg";		
	argv[27] = "c:\\Users/Books/Page10.jpg";
	argv[28] = "c:\\Users/Books/Page11.jpg";
	argv[29] = "c:\\Users/Books/Page12.jpg";
	argv[30] = "c:\\Users/Books/Page13.jpg";

  src = imread( argv[1], 1 );
  src1 = imread( argv[2], 1 );
  src2 = src;
  templ = imread( argv[21], 1 );

 
  /// Convert image to gray and blur it
  cvtColor(src , src_gray, CV_BGR2GRAY );
  blur( src_gray, src_gray, Size(3,3) );
  cvtColor(src1 , src_gray1, CV_BGR2GRAY );
  /// Create Window
  char* source_window = "Source";
  namedWindow( source_window, CV_WINDOW_AUTOSIZE );
  imshow( source_window, src );

 
 Threshold_Demo1( 0, 0 );


     /// Create windows
  namedWindow( image_window, CV_WINDOW_AUTOSIZE );
  namedWindow( result_window, CV_WINDOW_AUTOSIZE );

  /// Create Trackbar, Number 2: TM CCORR, Number 3:TM CCORR NORMED and Number 4: TM COEFF seem to work best
  char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
  createTrackbar( trackbar_label, image_window, &match_method, max_Trackbar, Threshold_Demo1 );

  waitKey(0);


  return(0);
}


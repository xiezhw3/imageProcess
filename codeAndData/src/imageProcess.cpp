#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <string.h>

using namespace std;
using namespace cv;

const int YMOVE = 8;

struct  Line {
	Point p1, p2, center;

	Line(Point p1_, Point p2_) :
	p1(p1_), p2(p2_) {
		center = Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
	}
};

inline double cal(int x1, int y1, int x2, int y2,
				  int x3, int y3, int x4, int y4) {
	return (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
}

/***********************************************
 * Description: 计算两条直线的交点
 * @para l1 直线1
 * @para l2 直线2
 * @return 返回代表交点坐标值的点
 ***********************************************/
Point2f getIntersect(const Line& l1, const Line& l2) {
	int x1 = l1.p1.x, y1 = l1.p1.y;
	int x2 = l1.p2.x, y2 = l1.p2.y;
	int x3 = l2.p1.x, y3 = l2.p1.y;
	int x4 = l2.p2.x, y4 = l2.p2.y;
	Point2f point(-1, -1);

	double d = cal(x1, y1, x2, y2, x3, y3, x4, y4);
	if (d) {
		point.x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d;
    	point.y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d;
	}
	return point;
}

/***********************************************
 * Description: 进行图像的缩放
 * @para srcImage 源图像
 * @para dstImage 目标图像
 * @para scale 缩放倍数
 * @para width 源图像宽度
 * @para height 源图像高度
 ***********************************************/
void imageScale(Mat& srcImage, Mat &dstImage, double scale, int width, int height) {
	resize(srcImage, dstImage, Size(width * 1.0 / scale, height * 1.0 / scale));
}

/***********************************************
 * Description: 获取图像的二值化后的图
 * @para srcImage 源图像
 * @para dstImage 保存二值化后的图像的图像
 ***********************************************/
void getGrayImage(Mat& srcImage, Mat &dstImage) {
	cvtColor(srcImage, dstImage, COLOR_BGR2GRAY);
}

/***********************************************
 * Description: 使用 canny 算法获取图像边缘
 * @para srcImage 源图像
 * @para dstImage 保存 canny 处理后图像的图像
 ***********************************************/
void getCannyImage(Mat& srcImage, Mat &dstImage) {
	Mat gray;
	getGrayImage(srcImage, gray);
	Canny(gray, dstImage, 50, 250, 3);
}

/***********************************************
 * Description: 用 HoughLinesP 函数对图像进行
 *				hough 变换			
 * @para canny 使用 canny 算法进行了边缘处理的图像
 * @para lines 保存 hough 变换后的直线的容器
 ***********************************************/
void hough(Mat& canny, vector<Vec4i> &lines) {
	int width = canny.size().width;
	// 85 和 15 是经过测试效果较好的数值
	HoughLinesP(canny, lines, 1, CV_PI / 180, 80, width / 15, 15);
}

/***********************************************
 * Description: 将 hough 识别的直线进行分类，分为
 *				偏竖线和偏横线两类		
 * @para lines 所有直线
 * @para horizontals 保存偏横线的容器
 * @para verticals 保存偏竖线的容器
 ***********************************************/
void classifyLines(vector<Vec4i> &lines, vector<Line> &horizontals, vector<Line> &verticals) {
	for (int i = 0; i < lines.size(); ++i) {
		Vec4i line = lines[i];
		if (fabs(line[0] - line[2]) > fabs(line[1] - line[3])) {
			// 偏横线
			horizontals.push_back(Line(Point(line[0], line[1]), Point(line[2], line[3])));
		} else {
			// 偏竖线
			verticals.push_back(Line(Point(line[0], line[1]), Point(line[2], line[3])));
		}
	}
}

/***********************************************
 * Description: 计算一条直线两端点与图像中心点差值形成
 *				的两个向量的叉积，判断直线方向是否满足
 *				使用右手定则时是向上
 * @para lines 要测试的直线
 * @para width 图像宽度
 * @para height 图像高度
 * @return 叉积的值
 ***********************************************/
int getCrossProduct(Line &line, int width, int height) {
	int centerX = width / 2;
	int centerY = height / 2;

	int lx = line.p1.x - centerX, ly = line.p1.y - centerY;
	int rx = line.p2.x - centerX, ry = line.p2.y - centerY;

	// 因为坐标系是向下的，所以叉积 < 0 代表是正确 方向
	int crossProduct = lx * ry - ly * rx;

	return crossProduct;
}

/***********************************************
 * Description: 交换两个点的值
 ***********************************************/
void swapPoint(Point &a, Point &b) {
	Point temp = a;
	a = b;
	b = temp;
}

/***********************************************
 * Description: 判断直线是否满足右手定则值向上，如果不
 *				是，更改直线的方向
 * @para lines 要测试的直线
 * @para width 图像宽度
 * @para height 图像高度
 ***********************************************/
void setLineDirection(Line &line, int width, int height) {
	int crossProduct = getCrossProduct(line, width, height);
	if (crossProduct > 0) {
		swapPoint(line.p1, line.p2);
	}
}

/***********************************************
 * Description: 计算的表达式中的斜率 k 和y轴偏移 b
 * @para lines 要计算的直线
 * @return 一个保存了 k 和 b 的值的容器
 ***********************************************/
vector<double> getLineKB(Line &line) {
	vector<double> res;
	int lx = line.p1.x, ly = line.p1.y;
	int rx = line.p2.x, ry = line.p2.y;

	// 直线垂直，此时设斜率为一个较大值
	if (abs(lx - rx) < 5) {
		res.push_back(1000.0);
		res.push_back(1000.0);
	} else {
		double k = (ly * 1.0 - ry * 1.0) / (lx * 1.0 - rx * 1.0);
		double b = ly * 1.0 - k * lx;
		res.push_back(k);
		res.push_back(b);
	}
	return res;
}

/***********************************************
 * Description: 沿着线段的边缘进行采样，获取线段两边的
 *				采样点坐标
 * @para lines 要进行采样的直线
 * @return 保存采样点的容器
 ***********************************************/
vector< vector<Point> > getClosePoint(Line &line) {
	int lx = line.p1.x, ly = line.p1.y;
	int rx = line.p2.x, ry = line.p2.y;

	vector<double> kb = getLineKB(line);
	vector<Point> lPoint, rPoint;
	// 偏横线
	if (abs(lx - rx) > abs(ly - ry)) {
		int min = lx < rx ? lx : rx;
		int max = lx > rx ? lx : rx;

		for (int i = min; i <= max; i += 5) {
			// 水平
			if (fabs(kb[0]) < 0.05) {
				if (lx < rx) {
					lPoint.push_back(Point(i, (ly + ry) / 2 - YMOVE));
					rPoint.push_back(Point(i, (ly + ry) / 2 + YMOVE));
				} else {
					lPoint.push_back(Point(i, (ly + ry) / 2 + YMOVE));
					rPoint.push_back(Point(i, (ly + ry) / 2 - YMOVE));
				}
			} else {
				int y = (int)(kb[0] * i + kb[1]);

				// 与原直线垂直的直线
				double k = -1 / kb[0];
				double b = ly - lx * k;
				double kSquare = k * k;

				// 计算 y 轴 和 x 轴的偏移
				double MoveSquare = YMOVE * YMOVE;
				int yMove = (int)(sqrt( MoveSquare * kSquare / (1 + kSquare)));
				int xMove = (int)(sqrt(MoveSquare / (kSquare + 1)));
				if (kb[0] < 0)
					xMove = -xMove;

				if (lx < rx) {
					lPoint.push_back(Point(i + xMove, y - yMove));
					rPoint.push_back(Point(i - xMove, y + yMove));
				} else {
					lPoint.push_back(Point(i - xMove, y + yMove));
					rPoint.push_back(Point(i + xMove, y - yMove));
				}
			}
		}
	} else { // 偏竖线
		int min = ly < ry ? ly : ry;
		int max = ly > ry ? ly : ry;

		for (int i = min; i <= max; i += 5) {
			// 垂直
			if (fabs(kb[0] > 500)) {
				if (ly < ry) {
					lPoint.push_back(Point((lx + rx) / 2 + YMOVE, i));
					rPoint.push_back(Point((lx + rx) / 2 - YMOVE, i));
				} else {
					lPoint.push_back(Point((lx + rx) / 2 - YMOVE, i));
					rPoint.push_back(Point((lx + rx) / 2 + YMOVE, i));
				}
			} else {
				int x = (int)((i - kb[1]) / kb[0]);

				double k = -1 / kb[0];
				double b = ly - lx * k;
				double kSquare = k * k;

				double MoveSquare = YMOVE * YMOVE;
				int yMove = (int)(sqrt( MoveSquare * kSquare / (1 + kSquare)));
				int xMove = (int)(sqrt(MoveSquare / (kSquare + 1)));

				if (kb[0] < 0)
					yMove = -yMove;

				if (ly < ry) {
					lPoint.push_back(Point(x + xMove, i - yMove));
					rPoint.push_back(Point(x - xMove, i + yMove));
				} else {
					lPoint.push_back(Point(x - xMove, i + yMove));
					rPoint.push_back(Point(x + xMove, i - yMove));
				}
			}
		}
	}

	vector< vector<Point> > result;
	result.push_back(lPoint);
	result.push_back(rPoint);

	return result;
}

/***********************************************
 * Description: 判断两个像素点的 RGB 值是否接近
 * @para rgb1 rgb2 两个像素点的 RGB 值
 * @return 两个像素点的 RGB 值是否相似
 ***********************************************/
bool rgbClose(Vec3b &rgb1, Vec3b& rgb2) {
	int num = 0;
	for (int i = 0; i < 3; ++i) {
		if (abs((int)rgb1.val[i] - (int)rgb2.val[i]) > 50) {
			num++;
		}
	}
	return num < 3;
}

/***********************************************
 * Description: 在图像中心附近进行采样并计算这些像素点
 *				的 RGB 的平均值
 * @para centerX 中心点的横坐标
 * @para centerY 中心点的纵坐标
 * @para image 要处理的图片
 * @return 中心范围 RGB 的平均值
 ***********************************************/
Vec3b getCenterRGB(int centerX, int centerY, Mat &image) {
	int num = 0;
	int rgb1 = 0, rgb2 = 0, rgb3 = 0;
	for (int i = -20; i <= 20; i += 5) {
		for (int j = -20; j <= 20; j += 5) {
			Vec3i current = image.at<Vec3b>(Point(centerX + j, centerY + i));
			rgb1 += (int)current.val[0];
			rgb2 += (int)current.val[1];
			rgb3 += (int)current.val[2];
			num++;
		}
	}

	rgb1 /= num;
	rgb2 /= num;
	rgb3 /= num;

	Vec3b res;
	res.val[0] = (uchar)rgb1;
	res.val[1] = (uchar)rgb2;
	res.val[2] = (uchar)rgb3;
	return res;
}

/***********************************************
 * Description: 判断一条直线是否是图像中纸张的边缘
 * @para line 要进行判断的直线
 * @para srcImage 要处理的图片
 * @return 这条直线是否是图片中的纸张边缘
 ***********************************************/
bool isEdge(Line &line, Mat& srcImage) {
	int width = srcImage.size().width;
	int height = srcImage.size().height;
	setLineDirection(line, width, height);

	int centerX = width / 2;
	int centerY = height / 2;
	Vec3b center = getCenterRGB(centerX, centerY, srcImage);

	vector< vector<Point> > closePoint = getClosePoint(line);

	int lPointCloseNum = 0, rPointCloseNum = 0;

	for (int i = 0; i < closePoint[0].size(); ++i) {
		// 去掉超出图片范围的点
		if (closePoint[0][i].y < 0 || closePoint[0][i].y >= height ||
			closePoint[0][i].x < 0 || closePoint[0][i].x >= width ||
			closePoint[1][i].y < 0 || closePoint[1][i].y >= height ||
			closePoint[1][i].x < 0 || closePoint[1][i].x >= width) {
			continue;
		}

		// 获取直线'左右'两边的的采样点的像素值
		Vec3b lPointRgb = srcImage.at<Vec3b>(Point(closePoint[0][i].x, closePoint[0][i].y));
		Vec3b rPointRgb = srcImage.at<Vec3b>(Point(closePoint[1][i].x, closePoint[1][i].y));

		if (rgbClose(lPointRgb, center))
		 	lPointCloseNum++;

		if (rgbClose(rPointRgb, center)) {
			rPointCloseNum++;
		}
	}

	int totalNum = closePoint[0].size();
	double lrate = lPointCloseNum * 1.0 / totalNum * 1.0;
	double rrate = rPointCloseNum * 1.0 / totalNum * 1.0;

	return (lrate > rrate);
}

/***********************************************
 * Description: 删除不是图像中纸张边缘的直线
 * @para lines 检测到的所有直线
 * @para srcImage 要处理的图片
 ***********************************************/
void delLines(vector<Line> &lines,  Mat& srcImage) {
	vector<int> delIndex;
	delIndex.clear();
	for (int i = 0; i < lines.size(); ++i) {
		if (isEdge(lines[i], srcImage) == 0) {
			delIndex.push_back(i);
		}
	}

	for (int i = 0; i < delIndex.size(); ++i) {
		lines.erase(lines.begin() + delIndex[i] - i);
	}
}

/***********************************************
 * Description: 对所有直线进行处理，如果直线不满足纸张
 *				左右和上下边最少各一条直线，那么补全
 * @para horizontals 所有偏横线
 * @para verticals 所有偏竖线
 * @para srcImage 要处理的图片
 ***********************************************/
void linesFill(vector<Line> &horizontals, vector<Line> &verticals, Mat& srcImage) {
	int width = srcImage.size().width;
	int height = srcImage.size().height;

	// 先取出不是纸张边缘的直线
	delLines(verticals, srcImage);
	delLines(horizontals, srcImage);

	if (horizontals.size() < 2) {
		if (horizontals.size() == 0 || horizontals[0].center.y > height / 2) {
			horizontals.push_back(Line(Point(0, 0), Point(width - 1, 0)));
		}
		if (horizontals.size() == 0 || horizontals[0].center.y <= height / 2) {
			horizontals.push_back(Line(Point(0, height - 1), Point(width - 1, height - 1)));
		}
	}

	if (verticals.size() < 2) {
		if (verticals.size() == 0 || verticals[0].center.x > width / 2) {
	      	verticals.push_back(Line(Point(0, 0), Point(0, height - 1)));
	    }
	    if (verticals.size() == 0 || verticals[0].center.x <= width / 2) {
	      	verticals.push_back(Line(Point(width - 1, 0), Point(width - 1, height - 1)));
	    }
	}
}

bool cmpX(const Line& l1, const Line& l2) {
	return l1.center.x < l2.center.x;
}

bool cmpY(const Line& l1, const Line& l2) {
	return l1.center.y < l2.center.y;
}

/***********************************************
 * Description: 根据线段中心的位置对线段进行位置排序，
 * 				对偏横线进行从上到下的排序，偏竖线
 *				进行从左到右的排序
 * @para horizontals 所有偏横线
 * @para verticals 所有偏竖线
 ***********************************************/
void lineSort(vector<Line> &horizontals, vector<Line> &verticals) {
	sort(horizontals.begin(), horizontals.end(), cmpY);
	sort(verticals.begin(), verticals.end(), cmpX);
}

/***********************************************
 * Description: 根据矩形的四个顶点的位置判断矩形的方向
 * @para p1 p2 p3 p4 矩形的四个顶点
 * @para 矩形是否处于偏竖直状态
 ***********************************************/
bool getRectDir(Point2f &p1, Point2f& p2, Point2f& p3, Point2f& p4) {
	if (abs(p1.x - p2.x) > abs(p1.y - p3.y))
		return false;
	return true;
}

/***********************************************
 * Description: 将原图中的纸张映射到 A4 纸的比例和大小
 * @para horizontals 图像中的所有偏横线
 * @para verticals 图像中的所有偏竖线
 * @para image 原图
 * @para dstImage 保存映射后的图片的图片
 * @para scale 原图的缩放比例，用于恢复图片大小
 * @para A4width A4 纸的宽度
 * @para A4height A4 纸的高度
 ***********************************************/
void mapToA4(vector<Line> &horizontals, vector<Line> &verticals, Mat& image, Mat& dstImage,
										double scale, int A4width = 210, int A4height = 297) {
	vector<Point2f> dstPoint, srcPoint;
  	dstPoint.push_back(Point(0, 0));
  	dstPoint.push_back(Point(A4width - 1, 0));
  	dstPoint.push_back(Point(0, A4height - 1));
  	dstPoint.push_back(Point(A4width - 1, A4height - 1));

  	Point2f point0 = getIntersect(horizontals[0], verticals[0]);
  	Point2f point1 = getIntersect(horizontals[0], verticals[verticals.size() - 1]);
  	Point2f point2 = getIntersect(horizontals[horizontals.size() - 1], verticals[0]);
  	Point2f point3 = getIntersect(horizontals[horizontals.size() - 1], verticals[verticals.size() - 1]);

  	if (getRectDir(point0, point1, point2, point3)) {
  		srcPoint.push_back(point0);
  		srcPoint.push_back(point1);
  		srcPoint.push_back(point2);
  		srcPoint.push_back(point3);
  	} else {
  		srcPoint.push_back(point0);
  		srcPoint.push_back(point2);
  		srcPoint.push_back(point1);
  		srcPoint.push_back(point3);
  	}

  	for (int i = 0; i < srcPoint.size(); ++i) {
  		srcPoint[i].x *= scale;
  		srcPoint[i].y *= scale;
  	}

  	Mat transmtx = getPerspectiveTransform(srcPoint, dstPoint);
  	warpPerspective(image, dstImage, transmtx, dstImage.size());
}

/***********************************************
 * Description: 将图片保存到硬盘
 * @para srcPath 原图的路径
 * @para verticals 图像中的所有偏竖线
 * @para image 原图
 ***********************************************/
void writeToFile(const string &srcPath, Mat& image) {
	char res[1024];
	memset(res, 0, sizeof(res));
	strncpy(res, srcPath.c_str(), index(srcPath.c_str(), '.') - srcPath.c_str());
	strtok(res, "/");
	char * resT = res + strlen(res) + 1;
	string imageName = string(string("result/") + string(resT) + "dst.jpg");
	imwrite(imageName, image);
}

/***********************************************
 * Description: 对图像进行处理，获取图像中的纸张并将
 *				其映射到 A4 纸大小
 * @para imagePath 要读取的图像的路径
 ***********************************************/
void imageProcess(const string &imagePath) {
	Mat image = imread(imagePath);

	// 对图像进行缩放，加快处理速度
	Mat scaleImage;
	int width = image.size().width;
	int height = image.size().height;
	double scale = min(10.0, width / 200.0);
	imageScale(image, scaleImage, scale, width, height);

	// canny 边缘处理
	Mat canny;
	getCannyImage(scaleImage, canny);

	// hough 变换获取边缘直线
	vector<Vec4i> lines;
	hough(canny, lines);

	// 将所有直线分为偏竖线和偏横线两组
	vector<Line> horizontals, verticals;
	classifyLines(lines, horizontals, verticals);
	// 如果横竖的直线少于 2 条，那么需要补全
	linesFill(horizontals, verticals, scaleImage);

	// 对直线进行排序
	lineSort(horizontals, verticals);
	int A4width = 210, A4height = 297;
	// 设置目标图片大小
	Mat dstImage = Mat::zeros(A4height, A4width, CV_8UC3);
	mapToA4(horizontals, verticals, image, dstImage, scale, A4width, A4height);
	
	writeToFile(imagePath, dstImage);
}

// 将图像的路径作为参数输入
int main(int argc, char* argv[]) {
	if (argc == 1)
		cerr << "Input with [./image] [path to image]" << endl;

	for (int i = 1; i < argc; ++i) {
		cout << "Process " << " image: " << argv[i] << endl;
		imageProcess(argv[i]);
	}
}
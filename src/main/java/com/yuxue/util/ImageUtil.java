package com.yuxue.util;

import java.util.Arrays;
import java.util.Map;
import java.util.Set;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.opencv_imgproc;

import com.google.common.collect.Maps;


/**
 * 车牌图片处理工具类
 * @author yuxue
 * @date 2020-05-18 12:07
 */
public class ImageUtil {

    private static String DEFAULT_BASE_TEST_PATH = "D:/PlateDetect/temp/";

    // 车牌定位处理步骤，该map用于表示步骤图片的顺序
    private static Map<String, Integer> debugMap = Maps.newLinkedHashMap();
    static {
        // debugMap.put("result", 99);
        debugMap.put("gaussianBlur", 0); // 高斯模糊
        debugMap.put("gray", 1);  // 图像灰度化
        debugMap.put("sobel", 2); // Sobel 算子
        debugMap.put("threshold", 3); //图像二值化
        debugMap.put("morphology", 4); // 图像闭操作
        debugMap.put("contours", 5); // 提取外部轮廓
        debugMap.put("result", 6); // 原图处理结果
        debugMap.put("crop", 7); // 切图
        debugMap.put("resize", 8); // 切图resize
        debugMap.put("char_threshold", 9); // 
        // debugMap.put("char_clearLiuDing", 10); // 去除柳钉
        // debugMap.put("specMat", 11); 
        // debugMap.put("chineseMat", 12);
        // debugMap.put("char_auxRoi", 13);
    }


    public static void main(String[] args) {
        
        String filename = DEFAULT_BASE_TEST_PATH + "test.jpg";
        // String filename = DEFAULT_BASE_TEST_PATH + "test01.jpg";

        String tempPath = DEFAULT_BASE_TEST_PATH + System.currentTimeMillis() + "/";
        FileUtil.createDir(tempPath); // 创建文件夹
        
        FileUtil.renameFile(filename, tempPath + "000_yuantu.jpg");
        
        Mat inMat = opencv_imgcodecs.imread(filename);

        Boolean debug = true;

        Mat gsMat = ImageUtil.gaussianBlur(inMat, debug, tempPath);

        Mat grey = ImageUtil.grey(gsMat, debug, tempPath);

        Mat sobel = ImageUtil.sobel(grey, debug, tempPath);


        // ImageUtil.rgb2Hsv(inMat, debug, tempPath);
    }



    /**
     * 高斯模糊
     * @param inMat
     * @param debug
     * @return
     */
    public static final int DEFAULT_GAUSSIANBLUR_SIZE = 5;
    public static Mat gaussianBlur(Mat inMat, Boolean debug, String tempPath) {
        Mat dst = new Mat();
        opencv_imgproc.GaussianBlur(inMat, dst, new Size(DEFAULT_GAUSSIANBLUR_SIZE, DEFAULT_GAUSSIANBLUR_SIZE), 0, 0, opencv_core.BORDER_DEFAULT);
        if (debug) {
            opencv_imgcodecs.imwrite(tempPath + (debugMap.get("gaussianBlur") + 100) + "_gaussianBlur.jpg", dst);
        }
        return dst;
    }


    /**
     * 将图像进行灰度化
     * @param inMat
     * @param debug
     * @param tempPath
     * @return
     */
    public static Mat grey(Mat inMat, Boolean debug, String tempPath) {
        Mat dst = new Mat();
        opencv_imgproc.cvtColor(inMat, dst, opencv_imgproc.CV_RGB2GRAY);
        if (debug) {
            opencv_imgcodecs.imwrite(tempPath + (debugMap.get("gray") + 100) + "_gray.jpg", dst);
        }
        return dst;
    }


    /**
     * 对图像进行Sobel 运算，得到图像的一阶水平方向导数
     * @param inMat
     * @param debug
     * @param tempPath
     * @return
     */
    public static final int SOBEL_SCALE = 1;
    public static final int SOBEL_DELTA = 0;
    public static final int SOBEL_DDEPTH = opencv_core.CV_16S;
    public static final int SOBEL_X_WEIGHT = 1;
    public static final int SOBEL_Y_WEIGHT = 0;
    public static Mat sobel(Mat inMat, Boolean debug, String tempPath) {

        Mat dst = new Mat();

        Mat grad_x = new Mat();
        Mat grad_y = new Mat();
        Mat abs_grad_x = new Mat();
        Mat abs_grad_y = new Mat();

        opencv_imgproc.Sobel(inMat, grad_x, SOBEL_DDEPTH, 1, 0, 3, SOBEL_SCALE, SOBEL_DELTA, opencv_core.BORDER_DEFAULT);
        opencv_core.convertScaleAbs(grad_x, abs_grad_x);

        opencv_imgproc.Sobel(inMat, grad_y, SOBEL_DDEPTH, 0, 1, 3, SOBEL_SCALE, SOBEL_DELTA, opencv_core.BORDER_DEFAULT);
        opencv_core.convertScaleAbs(grad_y, abs_grad_y);

        // Total Gradient (approximate)
        opencv_core.addWeighted(abs_grad_x, SOBEL_X_WEIGHT, abs_grad_y, SOBEL_Y_WEIGHT, 0, dst);

        if (debug) {
            opencv_imgcodecs.imwrite(tempPath + (debugMap.get("sobel") + 100) + "_sobel.jpg", dst);
        }
        return dst;
    }



    /**
     * 对图像进行二值化。将灰度图像（每个像素点有256 个取值可能）转化为二值图像（每个像素点仅有1 和0 两个取值可能）
     * @param inMat
     * @param debug
     * @param tempPath
     * @return
     */
    public static Mat threshold(Mat inMat, Boolean debug, String tempPath) {
        Mat dst = new Mat();
        opencv_imgproc.threshold(inMat, dst, 0, 255, opencv_imgproc.CV_THRESH_OTSU + opencv_imgproc.CV_THRESH_BINARY);
        if (debug) {
            opencv_imgcodecs.imwrite(tempPath + (debugMap.get("threshold") + 100) + "_threshold.jpg", dst);
        }
        return dst;
    }




   
    /**
     * 使用闭操作。对图像进行闭操作以后，可以看到车牌区域被连接成一个矩形装的区域
     * @param inMat
     * @param debug
     * @param tempPath
     * @return
     */
    public static final int DEFAULT_MORPH_SIZE_WIDTH = 17;
    public static final int DEFAULT_MORPH_SIZE_HEIGHT = 3;
    public static Mat morphology(Mat inMat, Boolean debug, String tempPath) {
        Mat dst = new Mat();
        Size size = new Size(DEFAULT_MORPH_SIZE_WIDTH, DEFAULT_MORPH_SIZE_HEIGHT);
        
        Mat element = opencv_imgproc.getStructuringElement(opencv_imgproc.MORPH_RECT, size);
        opencv_imgproc.morphologyEx(inMat, dst, opencv_imgproc.MORPH_CLOSE, element);

        if (debug) {
            opencv_imgcodecs.imwrite(tempPath + (debugMap.get("morphology") + 100) + "_morphology.jpg", dst);
        }
        return dst;
    }
    
    
    /**
     * Find 轮廓 of possibles plates 求轮廓。求出图中所有的轮廓。
     * 这个算法会把全图的轮廓都计算出来，因此要进行筛选。
     * @param src 原图
     * @param inMat morphology Mat
     * @param debug
     * @param tempPath
     * @return
     */
    public static MatVector contours(Mat src, Mat inMat, Boolean debug, String tempPath) {
        MatVector contours = new MatVector();
        opencv_imgproc.findContours(inMat, contours, // a vector of contours
                opencv_imgproc.CV_RETR_EXTERNAL, // 提取外部轮廓
                opencv_imgproc.CV_CHAIN_APPROX_NONE); // all pixels of each contours
        
        if (debug) {
            // 将轮廓描绘到原图
            opencv_imgproc.drawContours(src, contours, -1, new Scalar(0, 0, 255, 255));
            opencv_imgcodecs.imwrite(tempPath + (debugMap.get("contours") + 100) + "_contours.jpg", src);
        }
        return contours;
    }
    
    
    
    

    /**
     * rgb图像转换为hsv图像
     * @param inMat
     * @param debug
     * @param tempPath
     * @return
     */
    public static Mat rgb2Hsv(Mat inMat, Boolean debug, String tempPath) {
        // 转到HSV空间进行处理
        Mat dst = new Mat();
        opencv_imgproc.cvtColor(inMat, dst, opencv_imgproc.CV_BGR2HSV);
        MatVector hsvSplit = new MatVector();
        opencv_core.split(dst, hsvSplit);
        // 直方图均衡化是一种常见的增强图像对比度的方法，使用该方法可以增强局部图像的对比度，尤其在数据较为相似的图像中作用更加明显
        opencv_imgproc.equalizeHist(hsvSplit.get(2), hsvSplit.get(2));
        opencv_core.merge(hsvSplit, dst);

        if (debug) {
            opencv_imgcodecs.imwrite(tempPath + "hsvMat_"+System.currentTimeMillis()+".jpg", dst);
        }
        return dst;
    }


    /**
     * 获取HSV中各个颜色所对应的H的范围
     * HSV是一种比较直观的颜色模型，所以在许多图像编辑工具中应用比较广泛，这个模型中颜色的参数分别是：色调（H, Hue），饱和度（S,Saturation），明度（V, Value）
     * 1.PS软件时，H取值范围是0-360，S取值范围是（0%-100%），V取值范围是（0%-100%）。         
     * 2.利用openCV中cvSplit函数的在选择图像IPL_DEPTH_32F类型时，H取值范围是0-360，S取值范围是0-1（0%-100%），V取值范围是0-1（0%-100%）。
     * 3.利用openCV中cvSplit函数的在选择图像IPL_DEPTH_8UC类型时，H取值范围是0-180，S取值范围是0-255，V取值范围是0-255
     * @param inMat
     * @param debug
     */
    public static void getHSVValue(Mat inMat, Boolean debug, String tempPath) {

        int channels = inMat.channels();
        int nRows = inMat.rows();
        // 图像数据列需要考虑通道数的影响；
        int nCols = inMat.cols() * channels;

        // 连续存储的数据，按一行处理
        if (inMat.isContinuous()) {
            nCols *= nRows;
            nRows = 1;
        }
        Map<Integer, Integer> map = Maps.newHashMap();
        for (int i = 0; i < nRows; ++i) {
            BytePointer p = inMat.ptr(i);
            for (int j = 0; j < nCols; j += 3) {
                int H = p.get(j) & 0xFF;
                int S = p.get(j + 1) & 0xFF;
                int V = p.get(j + 2) & 0xFF;

                if(map.containsKey(H)) {
                    int count = map.get(H);
                    map.put(H, count+1);
                } else {
                    map.put(H, 1);
                }
            }
        }

        Set set = map.keySet();
        Object[] arr = set.toArray();
        Arrays.sort(arr);
        for (Object key : arr) {
            System.out.println(key + ": " + map.get(key));
        }

        return;
    }


}

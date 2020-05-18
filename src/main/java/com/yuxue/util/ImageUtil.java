package com.yuxue.util;


import java.util.Arrays;
import java.util.Map;
import java.util.Set;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
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

    public static void main(String[] args) {

        String tempPath = DEFAULT_BASE_TEST_PATH ;
        FileUtil.createDir(tempPath); // 创建文件夹

        // String filename = DEFAULT_BASE_TEST_PATH + "test01.jpg";
        String filename = DEFAULT_BASE_TEST_PATH + "test.png";
        Mat inMat = opencv_imgcodecs.imread(filename);

        // ImageUtil.gaussianBlur(inMat, true, tempPath);
        ImageUtil.rgb2Hsv(inMat, true, tempPath);
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
            opencv_imgcodecs.imwrite(tempPath + "gaussianBlur.jpg", dst);
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
            opencv_imgcodecs.imwrite(tempPath + "debugGray.jpg", dst);
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
            opencv_imgcodecs.imwrite(tempPath + "debugSobel.jpg", dst);
        }
        return dst;
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

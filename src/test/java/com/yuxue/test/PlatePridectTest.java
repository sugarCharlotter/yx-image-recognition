package com.yuxue.test;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.SVM;

/**
 * windows下环境配置： 
 * 1、官网下载对应版本的openvp：https://opencv.org/releases/page/2/ 当前使用4.0.1版本 
 * 2、双击exe文件安装，将 安装目录下\build\java\x64\opencv_java401.dll 拷贝到\build\x64\vc14\bin\目录下 3、eclipse添加User Libraries
 * 4、项目右键build path，添加步骤三新增的lib
 * 
 * https://blog.csdn.net/marooon/article/details/80265247
 * 
 * 测试
 * 
 * @author yuxue
 * @date 2020-05-12 21:34
 */
public class PlatePridectTest {

    // 默认的训练操作的根目录
    private static final String DEFAULT_PATH = "D:/PlateDetect/train/plate_detect_svm/";

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    
    public static void main(String[] args) {
        
        String module = DEFAULT_PATH + "svm.xml";
        SVM svm = SVM.load(module); // 加载训练得到的 xml 模型文件
        
        // 136 × 36 像素   需要跟训练的源图像文件保持相同大小
        pridect(svm, DEFAULT_PATH + "test/A01_NMV802_0.jpg");
        pridect(svm, DEFAULT_PATH + "test/debug_resize_1.jpg");
        pridect(svm, DEFAULT_PATH + "test/debug_resize_2.jpg");
        pridect(svm, DEFAULT_PATH + "test/debug_resize_3.jpg");
        pridect(svm, DEFAULT_PATH + "test/S22_KG2187_3.jpg");
        pridect(svm, DEFAULT_PATH + "test/S22_KG2187_5.jpg");
        
    }

    public static void pridect(SVM svm, String imgPath) {
       
        Mat src = Imgcodecs.imread(imgPath);// 图片大小要和样本一致
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
        Mat dst = new Mat();
        Imgproc.Canny(src, dst, 130, 250);

        Mat samples = new Mat(1, dst.cols() * dst.rows(), CvType.CV_32FC1);

        // 转换 src 图像的 cvtype
        // 失败案例：我试图用 dst.convertTo(src, CvType.CV_32FC1); 转换，但是失败了，原因未知。猜测: 内部的数据类型没有转换？
        float[] dataArr = new float[dst.cols() * dst.rows()];
        for (int i = 0, f = 0; i < dst.rows(); i++) {
            for (int j = 0; j < dst.cols(); j++) {
                double pixel = dst.get(i, j)[0];
                dataArr[f] = (float) pixel;
                f++;
            }
        }
        
        samples.put(0, 0, dataArr);

        // 如果训练时使用这个标识，那么符合的图像会返回9.0
        float flag = svm.predict(samples);

        if (flag == 0) {
            System.err.println(imgPath + "： 目标符合");
        }
        if (flag == 1) {
            System.out.println(imgPath + "： 目标不符合");
        }
    }
}
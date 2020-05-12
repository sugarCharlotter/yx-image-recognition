package com.yuxue.test;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.SVM;


/**
 * windows下环境配置：
 * 1、官网下载对应版本的openvp：https://opencv.org/releases/page/2/  当前使用4.0.1版本
 * 2、双击exe文件安装，将 安装目录下\build\java\x64\opencv_java401.dll 拷贝到\build\x64\vc14\bin\目录下
 * 3、eclipse添加User Libraries
 * 4、项目右键build path，添加步骤三新增的lib
 * 
 * https://blog.csdn.net/marooon/article/details/80265247
 * 
 * 未完待续
 * @author yuxue
 * @date 2020-05-12 21:34
 */
public class PridectTest {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        Mat src = Imgcodecs.imread("D:\\xunlian\\a\\0.jpg");//图片大小要和样本一致
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
        Mat dst = new Mat();
        Imgproc.Canny(src, dst, 40, 200);
        test(dst);
    }

    public static void test(Mat src) {
        SVM svm = SVM.load("./Result/a.xml");//加载训练得到的 xml

        Mat samples = new Mat(1,src.cols()*src.rows(),CvType.CV_32FC1);

        //转换 src 图像的 cvtype
        //失败案例：我试图用 src.convertTo(src, CvType.CV_32FC1); 转换，但是失败了，原因未知。猜测: 内部的数据类型没有转换？
        float[] dataArr = new float[src.cols()*src.rows()];
        for(int i =0,f = 0 ;i<src.rows();i++) {
            for(int j = 0;j<src.cols();j++) {
                float pixel = (float)src.get(i, j)[0];
                dataArr[f] = pixel;
                f++;
            }
        }
        samples.put(0, 0, dataArr);

        //预测用的方法，返回定义的标识。
//      int labels[]  = {9,9,9,9,
//                 1,1,1,1,1,1,1,
//                 1,1,1,1,1,1,1};
//      如果训练时使用这个标识，那么符合的图像会返回9.0
        float flag = svm.predict(samples);

        System.out.println("预测结果："+flag);
        if(flag == 0) {
            System.out.println("目标符合");
        }
        if(flag == 1) {
            System.out.println("目标不符合");
        }
    }
}
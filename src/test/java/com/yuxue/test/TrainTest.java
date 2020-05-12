package com.yuxue.test;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.Ml;
import org.opencv.ml.SVM;
import org.opencv.ml.TrainData;


/**
 * windows下环境配置：
 * 1、官网下载对应版本的openvp：https://opencv.org/releases/page/2/  当前使用4.0.1版本
 * 2、双击exe文件安装，将 安装目录下\build\java\x64\opencv_java401.dll 拷贝到\build\x64\vc14\bin\目录下
 * 3、eclipse添加User Libraries
 * 4、项目右键build path，添加步骤三新增的lib
 * 
 * 未完待续
 * @author yuxue
 * @date 2020-05-12 21:34
 */
public class TrainTest {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    /**
     * @author idmin
     * 样本的数量
     */
    private  static final int SAMPLE_NUMBER=18; 

    public static void main(String[] args) {
        //用于存放所有样本矩阵
        Mat trainingDataMat = null;

        //标记：正样本用 0 表示，负样本用 1 表示。
        //图片命名：
        //正样本： 0.jpg  1.jpg  2.jpg  3.jpg  4.jpg  
        //负样本：5.jpg  6.jpg  7.jpg   ...   17.jpg
        int labels[]  = {0,0,0,0, 
                1,1,1,1,1,1,1,1,1,1,1,1,1,1};

        //存放标记的Mat,每个图片都要给一个标记。SAMPLE_NUMBER 是自己定义的图片数量
        Mat labelsMat = new Mat(SAMPLE_NUMBER, 1, CvType.CV_32SC1);
        labelsMat.put(0, 0, labels);

        //这里的意思是，trainingDataMat 存放18张图片的矩阵，trainingDataMat 的每一行存放一张图片的矩阵。
        for(int i = 0;i<SAMPLE_NUMBER;i++) {            
            String path = "D:\\xunlian\\a\\" + i + ".jpg" ;
            Mat src = Imgcodecs.imread(path);

            //创建一个行数为18(正负样本总数量为18),列数为 rows*cols 的矩阵
            if(trainingDataMat == null) {
                trainingDataMat = new Mat(SAMPLE_NUMBER, src.rows()*src.cols(),CvType.CV_32FC1);// CV_32FC1 是规定的训练用的图片格式。
            }

            //转成灰度图并检测边缘
            //这里是为了过滤不需要的特征，减少训练时间。实际处理按情况论。
            Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
            Mat dst = new Mat(src.rows(),src.cols(),src.type());//此时的 dst 是8u1c。
            Imgproc.Canny(src, dst, 130, 250);

            //转成数组再添加。
            //失败案例:这里我试图用 get(row,col,data)方法获取数组，但是结果和这个结果不一样，原因未知。
            float[] arr =new float[dst.rows()*dst.cols()];
            int l=0;
            for (int j=0;j<dst.rows();j++){
                for(int k=0;k<dst.cols();k++) {
                    double[] a=dst.get(j, k);
                    arr[l]=(float)a[0];
                    l++;
                }
            } 
            trainingDataMat.put(i, 0, arr);
        }

        //每次训练的结果得到的 xml 文件都不一样，原因未知。猜测是我的样本数太太太太小
//      MySvm(trainingDataMat, labelsMat, "./Result/a.xml");
//      MySvm(trainingDataMat, labelsMat, "./Result/b.xml");
//      MySvm(trainingDataMat, labelsMat, "./Result/c.xml");
//      MySvm(trainingDataMat, labelsMat, "./Result/d.xml");
//      MySvm(trainingDataMat, labelsMat, "./Result/e.xml");
        MySvm(trainingDataMat, labelsMat, "./Result/f.xml");
    }

    /**
     * SVM 支持向量机
     * @param trainingDataMat 存放样本的矩阵
     * @param labelsMat 存放标识的矩阵
     * @param savePath 保存路径 。例如：d:/svm.xml
     */
    public static void MySvm(Mat trainingDataMat, Mat labelsMat, String savePath) {

        SVM svm = SVM.create();
        // 配置SVM训练器参数
        TermCriteria criteria = new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 1000, 0);
        svm.setTermCriteria(criteria);// 指定
        svm.setKernel(SVM.LINEAR);// 使用预先定义的内核初始化
        svm.setType(SVM.C_SVC); // SVM的类型,默认是：SVM.C_SVC
        svm.setGamma(0.5);// 核函数的参数
        svm.setNu(0.5);// SVM优化问题参数
        svm.setC(1);// SVM优化问题的参数C

        TrainData td = TrainData.create(trainingDataMat, Ml.ROW_SAMPLE, labelsMat);// 类封装的训练数据
        boolean success = svm.train(td.getSamples(), Ml.ROW_SAMPLE, td.getResponses());// 训练统计模型
        System.out.println("Svm training result: " + success);
        svm.save(savePath);// 保存模型
    }

}

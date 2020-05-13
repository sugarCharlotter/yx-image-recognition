package com.yuxue.test;

import java.io.File;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.Ml;
import org.opencv.ml.SVM;
import org.opencv.ml.TrainData;

import com.yuxue.constant.Constant;
import com.yuxue.util.FileUtil;

/**
 * windows下环境配置：
 * 1、官网下载对应版本的openvp：https://opencv.org/releases/page/2/  当前使用4.0.1版本
 * 2、双击exe文件安装，将 安装目录下\build\java\x64\opencv_java401.dll 拷贝到\build\x64\vc14\bin\目录下
 * 3、eclipse添加User Libraries
 * 4、项目右键build path，添加步骤三新增的lib
 * 
 * 图片识别车牌训练
 * 训练出来的库文件，用于判断切图是否包含车牌
 * @author yuxue
 * @date 2020-05-13 10:10
 */
public class PlateDetectTrainTest {

    // 默认的训练操作的根目录
    private static final String DEFAULT_PATH = "D:/PlateDetect/train/plate_detect_svm/";

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] arg) {
        // 用于存放所有样本矩阵
        Mat trainingDataMat = null;
        
        // 正样本  // 136 × 36 像素  训练的源图像文件要相同大小
        List<File> imgList1 = FileUtil.listFile(new File(DEFAULT_PATH + "/learn/HasPlate"), Constant.DEFAULT_TYPE, false);
        
        // 负样本   // 136 × 36 像素 训练的源图像文件要相同大小
        List<File> imgList2 = FileUtil.listFile(new File(DEFAULT_PATH + "/learn/NoPlate"), Constant.DEFAULT_TYPE, false);
 
        // 标记：正样本用 0 表示，负样本用 1 表示。
        int labels[] = createLabelArray(imgList1.size(), imgList2.size());
        
        // 图片数量
        int sample_num = labels.length;
        
        // 存放标记的Mat,每个图片都要给一个标记
        Mat labelsMat = new Mat(sample_num, 1, CvType.CV_32SC1);
        labelsMat.put(0, 0, labels);

        // 这里的意思是，trainingDataMat 存放18张图片的矩阵，trainingDataMat 的每一行存放一张图片的矩阵。
        for (int i = 0; i < sample_num; i++) {

            String path = "";
            if(i < imgList1.size()) {
                path = imgList1.get(i).getAbsolutePath();
            } else {
                path = imgList2.get(i - imgList1.size()).getAbsolutePath(); 
            }
            
            Mat src = Imgcodecs.imread(path);

            // 创建一个行数为18(正负样本总数量为18),列数为 rows*cols 的矩阵
            if (trainingDataMat == null) {
                trainingDataMat = new Mat(sample_num, src.rows() * src.cols(), CvType.CV_32FC1);// CV_32FC1 是规定的训练用的图片格式。
            }

            // 转成灰度图并检测边缘 // 这里是为了过滤不需要的特征，减少训练时间。实际处理按情况论。
            Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY); // 转成灰度图
            Mat dst = new Mat(src.rows(), src.cols(), src.type());// 此时的 dst 是8u1c。
            Imgproc.Canny(src, dst, 130, 250); // 边缘检测

            // 转成数组再添加。
            // 失败案例:这里我试图用 get(row,col,data)方法获取数组，但是结果和这个结果不一样，原因未知。
            float[] arr = new float[dst.rows() * dst.cols()];
            int l = 0;
            for (int j = 0; j < dst.rows(); j++) {
                for (int k = 0; k < dst.cols(); k++) {
                    double[] a = dst.get(j, k);
                    arr[l] = (float) a[0];
                    l++;
                }
            }
            trainingDataMat.put(i, 0, arr);
        }

        String module = DEFAULT_PATH + "svm.xml";
        MySvm(trainingDataMat, labelsMat, module);
    }

    /**
     * SVM 支持向量机
     * 
     * @param trainingDataMat 存放样本的矩阵
     * @param labelsMat 存放标识的矩阵
     */
    public static void MySvm(Mat trainingDataMat, Mat labelsMat, String savePath) {
        
        // 配置SVM训练器参数
        TermCriteria criteria = new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 20000, 0.0001);
        SVM svm = SVM.create();
        svm.setTermCriteria(criteria); // 指定
        svm.setKernel(SVM.RBF); // 使用预先定义的内核初始化
        svm.setType(SVM.C_SVC); // SVM的类型,默认是：SVM.C_SVC
        svm.setGamma(0.1); // 核函数的参数
        svm.setNu(0.1); // SVM优化问题参数
        svm.setC(1); // SVM优化问题的参数C
        svm.setP(0.1);
        svm.setDegree(0.1);
        svm.setCoef0(0.1);

        TrainData td = TrainData.create(trainingDataMat, Ml.ROW_SAMPLE, labelsMat);// 类封装的训练数据
        boolean success = svm.train(td.getSamples(), Ml.ROW_SAMPLE, td.getResponses());// 训练统计模型
        System.err.println("Svm training result: " + success);
        
        svm.save(savePath);// 保存模型
    }
    
    
    public static int[] createLabelArray(Integer i1, Integer i2) {
        int labels[] = new int[i1 + i2];
        
        for (int i = 0; i < labels.length; i++) {
            if(i < i1) {
                labels[i] = 0;
            } else {
                labels[i] = 1;
            }
        }
        return labels;
    }

}

package com.yuxue.easypr.core;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;

import java.util.Vector;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_ml.SVM;


/**
 * 车牌判断
 * @author yuxue
 * @date 2020-04-26 15:21
 */
public class PlateJudge {

    private SVM svm = SVM.create();

    /**
     * EasyPR的getFeatures回调函数, 用于从车牌的image生成svm的训练特征features
     */
    private SVMCallback features = new Features();

    /**
     * 模型存储路径
     */
    private String path = "res/model/svm.xml";


    public PlateJudge() {
        svm.clear();
        // svm=SVM.loadSVM(path, "svm");
        svm=SVM.load(path);
    }

    /**
     * 对单幅图像进行SVM判断
     * @param inMat
     * @return
     */
    public int plateJudge(final Mat inMat) {
        Mat features = this.features.getHistogramFeatures(inMat);
        // 通过直方图均衡化后的彩色图进行预测
        Mat p = features.reshape(1, 1);
        p.convertTo(p, opencv_core.CV_32FC1);
        float ret = svm.predict(features);
        return (int) ret;
        
        /*// 使用com.yuxue.test.PlateDetectTrainTest 生成的训练库文件
        // 在使用的过程中，传入的样本切图要跟训练的时候处理切图的方法一致
        Mat grayImage = new Mat();
        opencv_imgproc.cvtColor(inMat, grayImage, opencv_imgproc.CV_RGB2GRAY);
        Mat dst = new Mat();
        opencv_imgproc.Canny(grayImage, dst, 130, 250);
        Mat samples = dst.reshape(1, 1);
        samples.convertTo(samples, opencv_core.CV_32FC1);

        // 如果训练时使用这个标识，那么符合的图像会返回9.0
        float ret = svm.predict(samples);
        return (int) ret;*/
        
    }

    /**
     * 对多幅图像进行SVM判断
     * @param inVec
     * @param resultVec
     * @return
     */
    public int plateJudge(Vector<Mat> inVec, Vector<Mat> resultVec) {

        for (int j = 0; j < inVec.size(); j++) {
            Mat inMat = inVec.get(j);

            if (1 == plateJudge(inMat)) {
                resultVec.add(inMat);
            } else { // 再取中间部分判断一次
                int w = inMat.cols();
                int h = inMat.rows();

                Mat tmpDes = inMat.clone();
                Mat tmpMat = new Mat(inMat, new Rect((int) (w * 0.05), (int) (h * 0.1), (int) (w * 0.9), (int) (h * 0.8)));
                opencv_imgproc.resize(tmpMat, tmpDes, new Size(inMat.size()));

                if (plateJudge(tmpDes) == 1) {
                    resultVec.add(inMat);
                }
            }
        }
        return 0;
    }


    public void setModelPath(String path) {
        this.path = path;
    }

    public final String getModelPath() {
        return path;
    }

}

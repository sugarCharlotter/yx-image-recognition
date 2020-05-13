package com.yuxue.test;
 
import java.io.File;
import java.util.ArrayList;
 
import org.junit.Test;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.ml.Ml;
import org.opencv.ml.SVM;
import org.opencv.ml.TrainData;
 

/**
 * 网上找来的训练方案
 * @author yuxue
 * @date 2020-05-13 16:53
 */
public class trainsvm {
	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}
	
	// 默认的训练操作的根目录
    private static final String DEFAULT_PATH = "D:/PlateDetect/train/plate_detect_svm/";
    
	//存储训练集
	static ArrayList<Mat> trainingImages = new ArrayList<Mat>();
	//存储标签
	static ArrayList<Integer> trainingLabels = new ArrayList<Integer>();
 
	public static void main(String[] args) {
 
		openFile(1, DEFAULT_PATH + "/learn/HasPlate");
		openFile(0, DEFAULT_PATH + "/learn/NoPlate");
		Mat srcImgs = new Mat();
		Mat flags = new Mat(trainingLabels.size(), 1, CvType.CV_32SC1);
		
		Core.vconcat(trainingImages, srcImgs);    // 样本数量不能太大，trainingImages.size有限制
		
		for (int i = 0; i < trainingLabels.size(); i++) {
			int[] val = { trainingLabels.get(i) };
			flags.put(i, 0, val);
		}
		SVM svm = SVM.create();
		svm.setKernel(SVM.LINEAR);
		svm.setType(SVM.C_SVC);
		svm.setGamma(1);
		svm.setC(1);
		svm.setCoef0(0);
		svm.setNu(0);
		svm.setP(0);
		svm.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER, 20000, 0.0001));
		TrainData trainData = TrainData.create(srcImgs, Ml.ROW_SAMPLE, flags);
		boolean success = svm.train(trainData);
		System.out.println(success);
		svm.save( DEFAULT_PATH + "svm.xml");
 
	}
 
	//读取本地文件
	public static void openFile(int flag, String path) {
		File file = new File(path);
		File[] files = file.listFiles();
		int i = 0;
		for (File file2 : files) {
			Mat input = Imgcodecs.imread(file2.getPath(), 0);
			input.convertTo(input, CvType.CV_32FC1);
			Mat reshape = input.reshape(0, 1);
			// System.out.println(input.dump());
			trainingImages.add(reshape);
			trainingLabels.add(flag);
			i++;
			if(i >100) {
			    break;
			}
		}
	}
 
	@Test
	public void test1() {
 
		// 测试训练的效果
		SVM svm = SVM.load("E:\\svmTrain\\bp.xml");
        Mat responseMat = new Mat(); 
        Mat imread = Imgcodecs.imread("C:\\Users\\98432\\Desktop\\platetest\\NoPlate\\S22_KG2187_3.jpg", 0);
        imread.convertTo(imread, CvType.CV_32FC1);
        Mat reshape = imread.reshape(0, 1);
        svm.predict(reshape, responseMat, 0);  
		System.out.println(responseMat.dump());
        
        
	}
 
}
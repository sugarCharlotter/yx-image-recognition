package com.yuxue.util;

import java.util.Vector;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.ANN_MLP;
import org.opencv.ml.SVM;

import com.yuxue.constant.Constant;
import com.yuxue.enumtype.PlateColor;


/**
 * 车牌处理工具类
 * 车牌切图按字符分割
 * 字符识别
 * @author yuxue
 * @date 2020-05-28 15:11
 */
public class PalteUtil {
    
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }
    
    private static SVM svm = SVM.create();

    private static ANN_MLP ann=ANN_MLP.create();

    public static void loadSvmModel(String path) {
        svm.clear();
        svm=SVM.load(path);
    }

    // 加载ann配置文件  图像转文字的训练库文件
    public static void loadAnnModel(String path) {
        ann.clear();
        ann = ANN_MLP.load(path);
    }
    

    public static void main(String[] args) {
        /*System.err.println(PalteUtil.isPlate("粤AI234K"));
        System.err.println(PalteUtil.isPlate("鄂CD3098"));*/
        
        
        System.err.println("done!!!");
    }
    
    
    /**
     * 根据正则表达式判断字符串是否是车牌
     * @param str
     * @return
     */
    public static Boolean isPlate(String str) {
        Pattern p = Pattern.compile(Constant.plateReg);
        Boolean bl = false;

        //提取车牌
        Matcher m = p.matcher(str);
        while(m.find()) {
            bl = true;
            break;
        }
        return bl;
    }
    
    
    /**
     * 输入车牌切图集合，判断是否包含车牌
     * @param inMat
     * @param dst 包含车牌的图块
     */
    public static final int DEFAULT_WIDTH = 136;
    public static final int DEFAULT_HEIGHT = 36;
    public static void hasPlate(Vector<Mat> inMat, Vector<Mat> dst, String modelPath) {
        loadSvmModel(modelPath);
        
        inMat.stream().forEach(src -> {
            if(src.rows() == DEFAULT_HEIGHT && src.cols() == DEFAULT_WIDTH) {
                Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
                Imgproc.Canny(src, src, 130, 250);
                Mat samples = src.reshape(1, 1);
                samples.convertTo(samples, CvType.CV_32F);
                
                float flag = svm.predict(samples);
                
                if (flag == 0) {
                 // System.out.println("目标符合");
                    dst.add(src);
                } else {
                    // System.out.println("目标不符合");
                }
            }
        });
        
        return;
    }
   
    /**
     * 判断切图车牌颜色
     * @param inMat
     * @return
     */
    public static PlateColor getPlateColor(Mat inMat) {
        
        
        return PlateColor.UNKNOWN;
    }
    
}

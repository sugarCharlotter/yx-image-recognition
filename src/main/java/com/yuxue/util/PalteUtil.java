package com.yuxue.util;

import java.util.List;
import java.util.Map;
import java.util.Vector;
import java.util.Map.Entry;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.ml.ANN_MLP;
import org.opencv.ml.SVM;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.yuxue.constant.Constant;
import com.yuxue.enumtype.PlateColor;
import com.yuxue.train.SVMTrain;


/**
 * 车牌处理工具类
 * 车牌切图按字符分割
 * 字符识别
 * @author yuxue
 * @date 2020-05-28 15:11
 */
public class PalteUtil {
    
    // 车牌定位处理步骤，该map用于表示步骤图片的顺序
    private static Map<String, Integer> debugMap = Maps.newLinkedHashMap();
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        
        debugMap.put("colorMatch", 0); 
        debugMap.put("char_threshold", 0); 
        debugMap.put("char_clearLiuDing", 0); // 去除柳钉
        debugMap.put("specMat", 0); 
        debugMap.put("chineseMat", 0);
        debugMap.put("char_auxRoi", 0);

        // 设置index， 用于debug生成文件时候按名称排序
        Integer index = 200;
        for (Entry<String, Integer> entry : debugMap.entrySet()) {
            entry.setValue(index);
            index ++;
        }
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
    public static void hasPlate(Vector<Mat> inMat, Vector<Mat> dst, String modelPath, 
            Boolean debug, String tempPath) {
        loadSvmModel(modelPath);

        int i = 0;
        for (Mat src : inMat) {
            if(src.rows() == DEFAULT_HEIGHT && src.cols() == DEFAULT_WIDTH) {
                Mat samples = SVMTrain.getFeature(src);

                float flag = svm.predict(samples);

                if (flag == 0) {
                    dst.add(src);
                    if(debug) {
                        System.err.println("目标符合");
                        Imgcodecs.imwrite(tempPath + "199_plate_reco_" + i + ".png", src);
                    }
                    i++;
                } else {
                    System.out.println("目标不符合");
                }
            }
        }
        return;
    }
    
    
    /**
     * 判断切图车牌颜色
     * @param inMat
     * @return
     */
    public static PlateColor getPlateColor(Mat inMat, Boolean adaptive_minsv, Boolean debug, String tempPath) {
        // 判断阈值
        final float thresh = 0.49f;
        
        if(colorMatch(inMat, PlateColor.BLUE, adaptive_minsv, debug, tempPath) > thresh) {
            return PlateColor.BLUE;
        }
        if(colorMatch(inMat, PlateColor.YELLOW, adaptive_minsv, debug, tempPath) > thresh) {
            return PlateColor.YELLOW;
        }
        if(colorMatch(inMat, PlateColor.GREEN, adaptive_minsv, debug, tempPath) > thresh) {
            return PlateColor.GREEN;
        }
        return PlateColor.UNKNOWN;
    }

   
    /**
     * 颜色匹配计算
     * @param inMat
     * @param r
     * @param adaptive_minsv
     * @param debug
     * @param tempPath
     * @return
     */
    public static Float colorMatch(Mat inMat, PlateColor r, Boolean adaptive_minsv, Boolean debug, String tempPath) {
        final float max_sv = 255;
        final float minref_sv = 64;
        final float minabs_sv = 95;

        Mat hsvMat = ImageUtil.rgb2Hsv(inMat, debug, tempPath);

        // 匹配模板基色,切换以查找想要的基色
        int min_h = r.minH;
        int max_h = r.maxH;
        float diff_h = (float) ((max_h - min_h) / 2);
        int avg_h = (int) (min_h + diff_h);

        for (int i = 0; i < hsvMat.rows(); ++i) {
            for (int j = 0; j < hsvMat.cols(); j += 3) {
                int H = (int)hsvMat.get(i, j)[0];
                int S = (int)hsvMat.get(i, j)[1];
                int V = (int)hsvMat.get(i, j)[2];

                boolean colorMatched = false;

                if ( min_h < H && H <= max_h) {
                    int Hdiff = Math.abs(H - avg_h);
                    float Hdiff_p = Hdiff / diff_h;
                    float min_sv = 0;
                    if (adaptive_minsv) {
                        min_sv = minref_sv - minref_sv / 2 * (1 - Hdiff_p);
                    } else {
                        min_sv = minabs_sv;
                    }
                    if ((min_sv < S && S <= max_sv) && (min_sv < V && V <= max_sv)) {
                        colorMatched = true;
                    }
                }

                if (colorMatched == true) {
                    hsvMat.put(i, j, 0, 0, 255);
                } else {
                    hsvMat.put(i, j, 0, 0, 0);
                }
            }
        }
        
        // 获取颜色匹配后的二值灰度图
        List<Mat> hsvSplit = Lists.newArrayList();
        Core.split(hsvMat, hsvSplit);
        Mat grey = hsvSplit.get(2);
        
        float percent = (float) Core.countNonZero(grey) / (grey.rows() * grey.cols());
        if (debug) {
            Imgcodecs.imwrite(tempPath + debugMap.get("colorMatch") + "_colorMatch.jpg", grey);
        }

        return percent;
    }

}

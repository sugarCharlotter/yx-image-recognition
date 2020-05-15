package com.yuxue.train;

import java.util.Vector;


import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.ANN_MLP;
import org.opencv.ml.Ml;
import org.opencv.ml.TrainData;

import com.yuxue.enumtype.Direction;
import com.yuxue.util.FileUtil;


/**
 * 基于org.bytedeco.javacpp包实现的训练
 * 
 * 图片文字识别训练
 * 训练出来的库文件，用于判断切图是否包含车牌
 * 
 * 训练的svm.xml应用：
 * 1、替换res/model/svm.xml文件
 * 2、修改com.yuxue.easypr.core.PlateJudge.plateJudge(Mat) 方法
 *      将样本处理方法切换一下，即将对应被注释掉的模块代码取消注释
 * 
 * @author yuxue
 * @date 2020-05-14 22:16
 */
public class ANNTrain {

    private ANN_MLP ann = ANN_MLP.create();
    
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    // 中国车牌; 34个字符; 没有 字母I、字母O
    private final char strCharacters[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', /* 没有I */ 'J', 'K', 'L', 'M', 'N', /* 没有O */'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' };

    private final int numCharacter = strCharacters.length;

    // 并不全面，有些省份没有训练数据所以没有字符
    // 有些后面加数字2的表示在训练时常看到字符的一种变形，也作为训练数据存储
    private final String strChinese[] = { 
            "zh_cuan",  /*川*/
            "zh_e",     /*鄂*/
            "zh_gan",   /*赣*/
            "zh_gan1",  /*甘*/
            "zh_gui",   /*贵*/
            "zh_gui1",  /*桂*/
            "zh_hei",   /*黑*/
            "zh_hu",    /*沪*/
            "zh_ji",    /*冀*/
            "zh_jin",   /*津*/
            "zh_jing",  /*京*/
            "zh_jl",    /*吉*/
            "zh_liao",  /*辽*/
            "zh_lu",    /*鲁*/
            "zh_meng",  /*蒙*/
            "zh_min",   /*闽*/
            "zh_ning",  /*宁*/
            "zh_qing",  /*青*/
            "zh_qiong", /*琼*/
            "zh_shan",  /*陕*/
            "zh_su",    /*苏*/
            "zh_sx",    /*晋*/
            "zh_wan",   /*皖*/
            "zh_xiang", /*湘*/
            "zh_xin",   /*新*/
            "zh_yu",    /*豫*/
            "zh_yu1",   /*渝*/
            "zh_yue",   /*粤*/
            "zh_yun",   /*云*/
            "zh_zang",  /*藏*/
            "zh_zhe"    /*浙*/
    };

    private final int numAll = strCharacters.length + strChinese.length;
    
    // 默认的训练操作的根目录
    private static final String DEFAULT_PATH = "D:/PlateDetect/train/chars_recognise_ann/";

    // 训练模型文件保存位置
    // private static final String DATA_PATH = DEFAULT_PATH + "ann_data.xml";
    private static final String MODEL_PATH = DEFAULT_PATH + "ann.xml";
    
    
    public static float[] projectedHistogram(final Mat img, Direction direction) {
        int sz = 0;
        switch (direction) {
        case HORIZONTAL:
            sz = img.rows();
            break;

        case VERTICAL:
            sz = img.cols();
            break;

        default:
            break;
        }

        // 统计这一行或一列中，非零元素的个数，并保存到nonZeroMat中
        float[] nonZeroMat = new float[sz];
        Core.extractChannel(img, img, 0);
        for (int j = 0; j < sz; j++) {
            Mat data = (direction == Direction.HORIZONTAL) ? img.row(j) : img.col(j);
            int count = Core.countNonZero(data);
            nonZeroMat[j] = count;
        }

        // Normalize histogram
        float max = 0;
        for (int j = 0; j < nonZeroMat.length; ++j) {
            max = Math.max(max, nonZeroMat[j]);
        }

        if (max > 0) {
            for (int j = 0; j < nonZeroMat.length; ++j) {
                nonZeroMat[j] /= max;
            }
        }

        return nonZeroMat;
    }
    
   
    public Mat features(Mat in, int sizeData) {
        
        float[] vhist = projectedHistogram(in, Direction.VERTICAL);
        float[] hhist = projectedHistogram(in, Direction.HORIZONTAL);

        Mat lowData = new Mat();
        if (sizeData > 0) {
            Imgproc.resize(in, lowData, new Size(sizeData, sizeData));
        }

        int numCols = vhist.length + hhist.length + lowData.cols() * lowData.rows();
        Mat out = new Mat(1, numCols, CvType.CV_32F);

        int j = 0;
        for (int i = 0; i < vhist.length; ++i, ++j) {
            out.put(0, j, vhist[i]);
        }
        for (int i = 0; i < hhist.length; ++i, ++j) {
            out.put(0, j, hhist[i]);
        }
        
        for (int x = 0; x < lowData.cols(); x++) {
            for (int y = 0; y < lowData.rows(); y++, ++j) {
                // float val = lowData.ptr(x, y).get(0) & 0xFF;
                double[] val = lowData.get(x, y);
                out.put(0, j, val[0]);
            }
        }
        return out;
    }

    public void train(int _predictsize, int _neurons) {
        
        // 读取样本文件数据
        /*FileStorage fs = new FileStorage(DATA_PATH, FileStorage.READ);
        Mat samples = new Mat(fs.get("TrainingDataF" + _predictsize));
        Mat classes = new Mat(fs.get("classes"));

        Mat trainClasses = new Mat(samples.rows(), numAll, CV_32F);
        for (int i = 0; i < trainClasses.rows(); i++) {
            for (int k = 0; k < trainClasses.cols(); k++) {
                // If class of data i is same than a k class
                if (k == Convert.toInt(classes.ptr(i))) {
                    trainClasses.ptr(i, k).put(Convert.getBytes(1f));
                    
                } else {
                    trainClasses.ptr(i, k).put(Convert.getBytes(0f));
                }
            }
        }
        samples.convertTo(samples, CV_32F);
        System.out.println(samples.type());*/
        
       
        Mat samples = new Mat(); // 使用push_back，行数列数不能赋初始值

        Vector<Integer> trainingLabels = new Vector<Integer>();
        // 加载数字及字母字符
        for (int i = 0; i < numCharacter; i++) {
            String str = DEFAULT_PATH + strCharacters[i];
            Vector<String> files = new Vector<String>();
            FileUtil.getFiles(str, files);

            int size = (int) files.size();
            for (int j = 0; j < size; j++) {
                Mat img = Imgcodecs.imread(files.get(j), 0);
                // System.err.println(files.get(j)); // 文件名不能包含中文
                Mat f = features(img, _predictsize);
                samples.push_back(f);
                trainingLabels.add(i); // 每一幅字符图片所对应的字符类别索引下标
            }
        }
        
        // 加载汉字字符
        for (int i = 0; i < strChinese.length; i++) {
            String str = DEFAULT_PATH + strChinese[i];
            Vector<String> files = new Vector<String>();
            FileUtil.getFiles(str, files);

            int size = (int) files.size();
            for (int j = 0; j < size; j++) {
                Mat img = Imgcodecs.imread(files.get(j), 0);
                // System.err.println(files.get(j));   // 文件名不能包含中文
                Mat f = features(img, _predictsize);
                samples.push_back(f);
                trainingLabels.add(i + numCharacter);
            }
        }
        
        // CV_32FC1 CV_32SC1 CV_32F
        // samples.convertTo(samples, CvType.CV_32F);
        
        float[] labels = new float[trainingLabels.size()];
        for (int i = 0; i < labels.length; ++i) {
            labels[i] = trainingLabels.get(i).intValue();
        }
        Mat classes = new Mat(labels.length, 440, CvType.CV_32F);
        classes.put(0, 0, labels);

        System.out.println(samples.rows());
        System.out.println(samples.cols());
        System.out.println(samples.type());
        
        System.out.println(classes.rows());
        System.out.println(classes.cols());
        System.out.println(classes.type());
        
        // samples.type() == CV_32F || samples.type() == CV_32S 
        TrainData train_data = TrainData.create(samples, Ml.ROW_SAMPLE, classes);
        
        // //l_count为相量_layer_sizes的维数，即MLP的层数L
        // l_count = _layer_sizes->rows + _layer_sizes->cols - 1;
        ann.clear();
        Mat layers = new Mat(1, 3, CvType.CV_32F);
        layers.put(0, 0, samples.cols());
        layers.put(0, 1, _neurons);
        layers.put(0, 2, classes.cols());
        
        /*layers.ptr(0,0).put(Convert.getBytes(samples.cols()));    //440   vhist.length + hhist.length + lowData.cols() * lowData.rows();
        layers.ptr(0,1).put(Convert.getBytes(_predictsize));
        layers.ptr(0,2).put(Convert.getBytes(numAll));*/
        
        ann.setLayerSizes(layers);
        ann.setActivationFunction(ANN_MLP.SIGMOID_SYM, 1, 1);
        ann.setTrainMethod(ANN_MLP.BACKPROP);
        TermCriteria criteria = new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 30000, 0.0001);
        ann.setTermCriteria(criteria);
        ann.setBackpropWeightScale(0.1);
        ann.setBackpropMomentumScale(0.1);
        ann.train(train_data);
        
        System.err.println("完成 ");
        // FileStorage fsto = new FileStorage(MODEL_PATH, FileStorage.WRITE);
        // ann.write(fsto, "ann");
        ann.save(MODEL_PATH);
    }

    public static void main(String[] args) {

        ANNTrain annT = new ANNTrain();
        // 可根据需要训练不同的predictSize或者neurons的ANN模型
        int _predictsize = 20;
        int _neurons = 40;

        // annT.saveTrainData(_predictsize);

        // 这里演示只训练model文件夹下的ann.xml，此模型是一个predictSize=10,neurons=40的ANN模型。
        // 根据机器的不同，训练时间不一样，但一般需要10分钟左右，所以慢慢等一会吧。
        annT.train(_predictsize, _neurons);

        System.out.println("To be end.");
    }
    
        
}
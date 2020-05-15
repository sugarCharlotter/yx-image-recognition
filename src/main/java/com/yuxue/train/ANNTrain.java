package com.yuxue.train;

import static org.bytedeco.javacpp.opencv_core.CV_32F;
import static org.bytedeco.javacpp.opencv_ml.ROW_SAMPLE;

import java.util.Vector;

import org.bytedeco.javacpp.opencv_core.FileStorage;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.TermCriteria;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.opencv_ml.ANN_MLP;
import org.bytedeco.javacpp.opencv_ml.TrainData;

import com.yuxue.easypr.core.CoreFunc;
import com.yuxue.util.Convert;
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
    private static final String DATA_PATH = DEFAULT_PATH + "ann_data.xml";
    private static final String MODEL_PATH = DEFAULT_PATH + "ann.xml";
    
   
    public int saveTrainData(int _predictsize) {

        Mat classes = new Mat();
        Mat trainingDataf = new Mat();

        Vector<Integer> trainingLabels = new Vector<Integer>();
        
        // 加载数字及字母字符
        for (int i = 0; i < numCharacter; i++) {
            String str = DEFAULT_PATH + strCharacters[i];
            System.err.println(str);
            Vector<String> files = new Vector<String>();
            FileUtil.getFiles(str, files);

            int size = (int) files.size();
            for (int j = 0; j < size; j++) {
                Mat img = opencv_imgcodecs.imread(files.get(j), 0);
                // System.err.println(files.get(j)); // 文件名不能包含中文
                Mat f = CoreFunc.features(img, _predictsize);
                trainingDataf.push_back(f);
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
                Mat img = opencv_imgcodecs.imread(files.get(j), 0);
                // System.err.println(files.get(j));   // 文件名不能包含中文
                Mat f = CoreFunc.features(img, _predictsize);
                trainingDataf.push_back(f);
                trainingLabels.add(i + numCharacter);
            }
        }
        
        // CV_32FC1 CV_32SC1 CV_32F
        trainingDataf.convertTo(trainingDataf, opencv_core.CV_32F);
        
        int[] labels = new int[trainingLabels.size()];
        for (int i = 0; i < labels.length; ++i) {
            labels[i] = trainingLabels.get(i).intValue();
        }
        new Mat(labels).copyTo(classes);

        FileStorage fs = new FileStorage(DATA_PATH, FileStorage.WRITE);
        fs.write("TrainingDataF" + _predictsize, trainingDataf);
        fs.write("classes", classes);
        fs.release();

        System.out.println("End saveTrainData");
        return 0;
    }

    public void train(int _predictsize, int _neurons) {
        
        // 读取样本文件数据
        FileStorage fs = new FileStorage(DATA_PATH, FileStorage.READ);
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
        
        System.out.println(samples.type());

        // samples.type() == CV_32F || samples.type() == CV_32S 
        TrainData train_data = TrainData.create(samples, ROW_SAMPLE, trainClasses);
        
        ann.clear();
        Mat layers = new Mat(1, 3, CV_32F);
        layers.ptr(0).put(Convert.getBytes(samples.cols()));
        layers.ptr(1).put(Convert.getBytes(_neurons));
        layers.ptr(2).put(Convert.getBytes(numAll));
        
        ann.setLayerSizes(layers);
        ann.setActivationFunction(ANN_MLP.SIGMOID_SYM, 1, 1);
        ann.setTrainMethod(ANN_MLP.BACKPROP);
        TermCriteria criteria = new TermCriteria(TermCriteria.MAX_ITER, 30000, 0.0001);
        ann.setTermCriteria(criteria);
        ann.setBackpropWeightScale(0.1);
        ann.setBackpropMomentumScale(0.1);
        ann.train(train_data);
        
        System.err.println("完成 ");
        FileStorage fsto = new FileStorage(MODEL_PATH, FileStorage.WRITE);
        ann.write(fsto, "ann");
    }

    public static void main(String[] args) {

        ANNTrain annT = new ANNTrain();
        // 可根据需要训练不同的predictSize或者neurons的ANN模型
        int _predictsize = 10;
        int _neurons = 40;

        // annT.saveTrainData(_predictsize);

        // 这里演示只训练model文件夹下的ann.xml，此模型是一个predictSize=10,neurons=40的ANN模型。
        // 根据机器的不同，训练时间不一样，但一般需要10分钟左右，所以慢慢等一会吧。
        annT.train(_predictsize, _neurons);

        System.out.println("To be end.");
    }
    
        
}
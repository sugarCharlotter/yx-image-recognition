package com.yuxue.train;

import java.util.Random;
import java.util.Vector;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.ANN_MLP;
import org.opencv.ml.Ml;
import org.opencv.ml.TrainData;

import com.yuxue.constant.Constant;
import com.yuxue.enumtype.Direction;
import com.yuxue.util.FileUtil;


/**
 * 基于org.opencv包实现的训练
 * 
 * 图片文字识别训练
 * 训练出来的库文件，用于识别图片中的文字
 * 
 * 训练的ann.xml应用：
 * 1、替换res/model/ann.xml文件
 * 2、修改com.yuxue.easypr.core.CharsIdentify.charsIdentify(Mat, Boolean, Boolean)方法
 * 
 * @author yuxue
 * @date 2020-05-14 22:16
 */
public class ANNTrain {

    private ANN_MLP ann = ANN_MLP.create();

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    // 默认的训练操作的根目录
    private static final String DEFAULT_PATH = "D:/PlateDetect/train/chars_recognise_ann/";

    // 训练模型文件保存位置
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
                double[] val = lowData.get(x, y);
                out.put(0, j, val[0]);
            }
        }
        return out;
    }


    /**
     * 按随机数，平移或者旋转样本文件
     * @param inMat
     * @return
     */
    public Mat getSyntheticImage(Mat inMat) {
        Random rand = new Random();
        int rand_type = rand.nextInt(10000);
        Mat result = inMat.clone();
        if (rand_type % 2 == 0) {
        int ran_x = rand.nextInt(10000) % 5 - 2; // 控制在-2~3个像素范围内
        int ran_y = rand.nextInt(10000) % 5 - 2;
        result = translateImg(result, ran_x, ran_y);    // 平移

        } else if (rand_type % 2 != 0) {
            float angle = (float) (rand.nextInt(10000) % 15 - 7); // 旋转角度控制在-7~8°范围内
            result = rotateImg(result, angle);  // 旋转
        }
        /*
        //进行膨胀操作
        Mat element1 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1, 1));
        Mat dstImage1;
        Imgproc.dilate(inMat, result, element1);

        //进行腐蚀操作
        Mat element2 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1, 1));
        Mat dstImage2;
        Imgproc.erode(inMat, result, element2);
        */
        return result;
    }


    /**
     * 平移
     * @param img
     * @param offsetx
     * @param offsety
     * @return
     */
    public Mat translateImg(Mat img, int offsetx, int offsety){
        Mat dst = new Mat();
        //定义平移矩阵
        Mat trans_mat = Mat.zeros(2, 3, CvType.CV_32FC1);
        trans_mat.put(0, 0, 1);
        trans_mat.put(0, 2, offsetx);
        trans_mat.put(1, 1, 1);
        trans_mat.put(1, 2, offsety);
        Imgproc.warpAffine(img, dst, trans_mat, img.size());    // 仿射变换
        return dst;
    }

    /**
     * 旋转角度
     * @param source
     * @param angle
     * @return
     */
    public Mat rotateImg(Mat source, float angle){
        Point src_center = new Point(source.cols() / 2.0F, source.rows() / 2.0F);
        Mat rot_mat = Imgproc.getRotationMatrix2D(src_center, angle, 1);
        Mat dst = new Mat();
        // 仿射变换 可以考虑使用投影变换; 这里使用放射变换进行旋转，对于实际效果来说感觉意义不大，反而会干扰结果预测
        Imgproc.warpAffine(source, dst, rot_mat, source.size());    
        return dst;
    }


    public void train(int _predictsize, int _neurons) {
        Mat samples = new Mat(); // 使用push_back，行数列数不能赋初始值
        Vector<Integer> trainingLabels = new Vector<Integer>();
        // 加载数字及字母字符
        for (int i = 0; i < Constant.numCharacter; i++) {
            String str = DEFAULT_PATH + "learn/" + Constant.strCharacters[i];
            Vector<String> files = new Vector<String>();
            FileUtil.getFiles(str, files);  // 文件名不能包含中文

            int count = 100; // 控制每个字符，最多只允许有200个样本文件
            int k = 0;
            // System.out.println("数字+字母：\t" + files.size());
            for (String filePath : files) {
                Mat img = Imgcodecs.imread(filePath, 0);
                Mat f = features(img, _predictsize);
                samples.push_back(f);
                trainingLabels.add(i); // 每一幅字符图片所对应的字符类别索引下标

                // 抽取1/3样本文件，平移或者旋转变换后，加入训练样本
                if (k % 3 == 0) {
                    samples.push_back(features(getSyntheticImage(img), _predictsize));
                    trainingLabels.add(i); // 每一幅字符图片所对应的字符类别索引下标
                }
                k++;
                
                if(count <= 0) {
                    break;
                }
                count--;
            }
        }

        // 加载汉字字符
        for (int i = 0; i < Constant.strChinese.length; i++) {
            String str = DEFAULT_PATH + "learn/" + Constant.strChinese[i];
            Vector<String> files = new Vector<String>();
            FileUtil.getFiles(str, files);

            int count = 50; // 控制每个字符，最多只允许有100个样本文件
            int k = 0;
            // System.out.println("汉字：\t" + files.size());
            for (String filePath : files) {
                Mat img = Imgcodecs.imread(filePath, 0);
                Mat f = features(img, _predictsize);
                samples.push_back(f);
                trainingLabels.add(i + Constant.numCharacter); // 每一幅字符图片所对应的字符类别索引下标

                // 抽取1/3样本文件，平移或者旋转变换后，加入训练样本
                if (k % 3 == 0) {
                    samples.push_back(features(getSyntheticImage(img), _predictsize));
                    trainingLabels.add(i); // 每一幅字符图片所对应的字符类别索引下标
                }
                k++;
                
                if(count <= 0) {
                    break;
                }
                count--;
            }
        }

        samples.convertTo(samples, CvType.CV_32F);

        //440   vhist.length + hhist.length + lowData.cols() * lowData.rows();
        // CV_32FC1 CV_32SC1 CV_32F
        Mat classes = Mat.zeros(trainingLabels.size(), Constant.numAll, CvType.CV_32F);

        float[] labels = new float[trainingLabels.size()];
        for (int i = 0; i < labels.length; ++i) {
            classes.put(i, trainingLabels.get(i), 1.f);
        }

        // samples.type() == CV_32F || samples.type() == CV_32S 
        TrainData train_data = TrainData.create(samples, Ml.ROW_SAMPLE, classes);

        ann.clear();
        Mat layers = new Mat(1, 3, CvType.CV_32F);
        layers.put(0, 0, samples.cols());   // 样本数量
        layers.put(0, 1, _neurons); // 
        layers.put(0, 2, classes.cols());   // 字符数

        ann.setLayerSizes(layers);
        ann.setActivationFunction(ANN_MLP.SIGMOID_SYM, 1, 1);
        ann.setTrainMethod(ANN_MLP.BACKPROP);
        TermCriteria criteria = new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 30000, 0.0001);
        ann.setTermCriteria(criteria);
        ann.setBackpropWeightScale(0.1);
        ann.setBackpropMomentumScale(0.1);
        ann.train(train_data);

        // FileStorage fsto = new FileStorage(MODEL_PATH, FileStorage.WRITE);
        // ann.write(fsto, "ann");
        ann.save(MODEL_PATH);
    }


    public void predict() {
        ann.clear();
        ann = ANN_MLP.load(MODEL_PATH);
        Vector<String> files = new Vector<String>();
        FileUtil.getFiles(DEFAULT_PATH + "test/", files);

        String plate = "";
        for (String string : files) {
            Mat img = Imgcodecs.imread(string, 0);
            Mat f = features(img, Constant.predictSize);

            int index = 0;
            double maxVal = -2;
            Mat output = new Mat(1, Constant.numAll, CvType.CV_32F);
            ann.predict(f, output);  // 预测结果    // 可以考虑将样本进行平移、旋转、腐蚀等算法，进行多次预测，取最大值--未实现
            for (int j = 0; j < Constant.numAll; j++) {
                double val = output.get(0, j)[0];
                if (val > maxVal) {
                    maxVal = val;
                    index = j;
                   
                    // 输出预测可能的值  -- 测试用
                    /*String charValue = "";
                    if (index < Constant.numCharacter) {
                        charValue = String.valueOf(Constant.strCharacters[index]);
                    } else {
                        String s = Constant.strChinese[index - Constant.numCharacter];
                        charValue = Constant.KEY_CHINESE_MAP.get(s);
                    }
                    System.out.println(string + "==>" + j + "\t\t" + charValue + "\t" + val);*/
                }
            }
            
            if (index < Constant.numCharacter) {
                plate += String.valueOf(Constant.strCharacters[index]);
            } else {
                String s = Constant.strChinese[index - Constant.numCharacter];
                plate += Constant.KEY_CHINESE_MAP.get(s);
            }
        }
        System.err.println("===>" + plate);
        return;
    }

    public static void main(String[] args) {

        ANNTrain annT = new ANNTrain();
        // 这里演示只训练model文件夹下的ann.xml，此模型是一个predictSize=10,neurons=40的ANN模型
        // 可根据需要训练不同的predictSize或者neurons的ANN模型
        // 根据机器的不同，训练时间不一样，但一般需要10分钟左右，所以慢慢等一会吧。
        annT.train(Constant.predictSize, Constant.neurons);

        annT.predict();

        System.out.println("The end.");
        return;
    }


}
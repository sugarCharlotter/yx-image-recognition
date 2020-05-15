package com.yuxue.easypr.core;

import static org.bytedeco.javacpp.opencv_core.CV_32FC1;

import java.util.HashMap;
import java.util.Map;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_ml.ANN_MLP;

import com.yuxue.constant.Constant;
import com.yuxue.util.Convert;


/**
 * 字符检测
 * @author yuxue
 * @date 2020-04-24 15:31
 */
public class CharsIdentify {

    private final static int numCharacter = 34; // 没有I和0, 10个数字与24个英文字符之和

    private final static int numAll = 65; /* 34+31=65 34个字符跟31个汉字 */

    private ANN_MLP ann=ANN_MLP.create();

    private String path = "res/model/ann.xml";

    static boolean hasPrint = false;

    private Map<String, String> map = new HashMap<String, String>();

    public CharsIdentify() {
        loadModel(); // 加载ann配置文件

        if (this.map.isEmpty()) {
            map.put("zh_cuan", "川");
            map.put("zh_e", "鄂");
            map.put("zh_gan", "赣");
            map.put("zh_gan1", "甘");
            map.put("zh_gui", "贵");
            map.put("zh_gui1", "桂");
            map.put("zh_hei", "黑");
            map.put("zh_hu", "沪");
            map.put("zh_ji", "冀");
            map.put("zh_jin", "津");
            map.put("zh_jing", "京");
            map.put("zh_jl", "吉");
            map.put("zh_liao", "辽");
            map.put("zh_lu", "鲁");
            map.put("zh_meng", "蒙");
            map.put("zh_min", "闽");
            map.put("zh_ning", "宁");
            map.put("zh_qing", "青");
            map.put("zh_qiong", "琼");
            map.put("zh_shan", "陕");
            map.put("zh_su", "苏");
            map.put("zh_sx", "晋");
            map.put("zh_wan", "皖");
            map.put("zh_xiang", "湘");
            map.put("zh_xin", "新");
            map.put("zh_yu", "豫");
            map.put("zh_yu1", "渝");
            map.put("zh_yue", "粤");
            map.put("zh_yun", "云");
            map.put("zh_zang", "藏");
            map.put("zh_zhe", "浙");
        }
    }
    
    private void loadModel() {
        loadModel(this.path);
    }

    public void loadModel(String s) {
        this.ann.clear();
        //ann=ANN_MLP.loadANN_MLP(s, "ann"); // 加载ann配置文件  图像转文字的训练库文件
        ann = ANN_MLP.load(s);
    }


    /**
     * @param input
     * @param isChinese
     * @return
     */
    public String charsIdentify(final Mat input, final Boolean isChinese, final Boolean isSpeci) {
        String result = "";

        Mat f = CoreFunc.features(input, Constant.predictSize);

        int index = this.classify(f, isChinese, isSpeci);

        if (!isChinese) {
            result = String.valueOf(Constant.strCharacters[index]);
        } else {
            String s = Constant.strChinese[index - numCharacter];
            result = map.get(s);
        }
        return result;
    }

    private int classify(final Mat f, final Boolean isChinses, final Boolean isSpeci) {
        int result = -1;
        Mat output = new Mat(1, numAll, CV_32FC1);

        ann.predict(f, output, 0);  // 预测结果

        int ann_min = (!isChinses) ? ((isSpeci) ? 10 : 0) : numCharacter;
        int ann_max = (!isChinses) ? numCharacter : numAll;

        float maxVal = -2;

        for (int j = ann_min; j < ann_max; j++) {
            float val = Convert.toFloat(output.ptr(0, j));
            if (val > maxVal) {
                maxVal = val;
                result = j;
            }
        }
        return result;
    }

    public final void setModelPath(String path) {
        this.path = path;
    }

    public final String getModelPath() {
        return this.path;
    }



}

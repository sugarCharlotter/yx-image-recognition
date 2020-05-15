package com.yuxue.constant;


/**
 * 系统常量
 * @author yuxue
 * @date 2018-09-07
 */
public class Constant {

	public static final String UTF8 = "UTF-8";

	// 车牌识别， 默认车牌图片保存路径
	// public static String DEFAULT_DIR = "./PlateDetect/"; // 使用项目的相对路径
	public static String DEFAULT_DIR = "D:/PlateDetect/"; // 使用盘符的绝对路径
	
	// 车牌识别， 默认车牌图片处理过程temp路径
	// public static String DEFAULT_TEMP_DIR = "./PlateDetect/temp/"; // 使用项目的相对路径
	public static String DEFAULT_TEMP_DIR = "D:/PlateDetect/temp/"; // 使用盘符的绝对路径
	
	// 车牌识别，默认处理图片类型
	public static String DEFAULT_TYPE = "png,jpg,jpeg";
	
	// 车牌识别，判断是否车牌的正则表达式
	public static String plateReg = "([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领A-Z]{1}[A-Z]{1}(([0-9]{5}[DF])|([DF]([A-HJ-NP-Z0-9])[0-9]{4})))|([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领A-Z]{1}[A-Z]{1}[A-HJ-NP-Z0-9]{4}[A-HJ-NP-Z0-9挂学警港澳]{1})";

	public static int predictSize = 10;
	
	public static int neurons = 40;

	// 中国车牌; 34个字符; 没有 字母I、字母O
	public final static char strCharacters[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', /* 没有I */ 'J', 'K', 'L', 'M', 'N', /* 没有O */'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' };

 // 并不全面，有些省份没有训练数据所以没有字符
    // 有些后面加数字2的表示在训练时常看到字符的一种变形，也作为训练数据存储
	public final static String strChinese[] = { 
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

    

}

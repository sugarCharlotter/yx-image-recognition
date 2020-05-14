package com.yuxue.constant;


/**
 * 系统常量
 * @author yuxue
 * @date 2018-09-07
 */
public class Constant {

	public static final String UTF8 = "UTF-8";

	// 车牌识别， 默认车牌图片保存路径
	public static String DEFAULT_DIR = "./PlateDetect/";
	
	// 车牌识别， 默认车牌图片处理过程temp路径
	public static String DEFAULT_TEMP_DIR = "./PlateDetect/temp/";
	
	// 车牌识别，默认处理图片类型
	public static String DEFAULT_TYPE = "png,jpg,jpeg";
	
	// 车牌识别，判断是否车牌的正则表达式
	public static String plateReg = "([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领A-Z]{1}[A-Z]{1}(([0-9]{5}[DF])|([DF]([A-HJ-NP-Z0-9])[0-9]{4})))|([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领A-Z]{1}[A-Z]{1}[A-HJ-NP-Z0-9]{4}[A-HJ-NP-Z0-9挂学警港澳]{1})";

    

}

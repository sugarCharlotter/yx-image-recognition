package com.yuxue.util;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.yuxue.constant.Constant;

public class PalteUtil {

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
    
    
    public static void main(String[] args) {
        System.err.println(PalteUtil.isPlate("粤AI234K"));
        System.err.println(PalteUtil.isPlate("鄂CD3098"));
    }
    
    
    
}

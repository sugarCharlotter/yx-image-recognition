package com.yuxue.enumtype;


/**
 * 车牌颜色
 * @author yuxue
 * @date 2020-05-08 12:38
 */
public enum PlateColor {

    BLUE("BLUE","蓝牌"), 
    GREEN("GREEN","绿牌"), 
    YELLOW("YELLOW","黄牌"),
    UNKNOWN("UNKNOWN","未知");

    public final String code;
    public final String desc;

    PlateColor(String code, String desc) {
        this.code = code;
        this.desc = desc;
    }

    public static String getDesc(String code) {
        PlateColor[] enums = values();
        for (PlateColor type : enums) {
            if (type.code().equals(code)) {
                return type.desc();
            }
        }
        return null;
    }

    public static String getCode(String desc) {
        PlateColor[] enums = values();
        for (PlateColor type : enums) {
            if (type.desc().equals(desc)) {
                return type.code();
            }
        }
        return null;
    }
    
    
    public String code() {
        return this.code;
    }

    public String desc() {
        return this.desc;
    }
    
}

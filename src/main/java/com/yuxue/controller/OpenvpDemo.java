package com.yuxue.controller;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;


/**
 * opencv 官方demo
 * @author yuxue
 * @date 2020-04-22 14:04
 */
public class OpenvpDemo {

    public static void main(String[] args) {
        System.out.println("Welcome to OpenCV " + Core.VERSION);
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat m  = Mat.eye(3, 3, CvType.CV_8UC1);
        System.out.println("m = " + m.dump());
    }

}

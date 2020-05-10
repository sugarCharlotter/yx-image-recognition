package com.yuxue.controller;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.objdetect.CascadeClassifier;

/**
 * 识别人脸 
 * Detects faces in an image, draws boxes around them, 
 * and writes the results to "faceDetection.png".
 */
public class DetectFaceDemo {
    
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        DetectFaceDemo demo = new DetectFaceDemo();
        demo.run();
    }

    public void run() {

        // Create a face detector from the cascade file in the resources directory.
        // 创建识别器
        CascadeClassifier faceDetector = new CascadeClassifier(getClass().getResource("/haarcascades/lbpcascade_frontalface.xml").getPath());

        String imgPath = "/src/main/resources/DetectFace/huge.jpg";
        Mat image = null; //Highgui.imread(imgPath,3);
        
        String filename = "D:/DetectFace/huge1.png";
        // System.out.println(String.format("Writing %s", filename));

        // Detect faces in the image. MatOfRect is a special container class for Rect.
        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(image, faceDetections);

        System.out.println(String.format("识别出 %s 张人脸", faceDetections.toArray().length));

        // Draw a bounding box around each face.
        for (Rect rect : faceDetections.toArray()) {
            // Core.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0));
        }

        // Save the visualized detection.
        // System.out.println(String.format("Writing %s", filename));
        //Highgui.imwrite(filename, image);
        return;
    }
}

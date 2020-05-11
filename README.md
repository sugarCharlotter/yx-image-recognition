# yx-image-recognition

#### 介绍
- **spring boot + maven实现的车牌识别系统**
- 基于Opencv实现、在EasyPR-Java的基础上优化配置及依赖版本
- 入门级项目

#### 软件版本
- jdk 1.8+
- maven 3.0+
- opencv 4.0.1 ； javacpp1.4.4；opencv-platform 4.0.1-1.4.4
- spring boot 2.1.5.RELEASE
- yx-image-recognition 1.0.0版本

#### 软件架构
- B/S 架构，前端html + requireJS，后端java
- 数据库使用 sqlite3.0
- 接口文档使用swagger 2.0

#### 车牌识别流程
- **操作界面**
![1.png](./res/doc_image/1.png)

- **图片处理过程**
debug_GaussianBlur：
![1.png](./res/doc_image/debug_GaussianBlur.png)

debug_gray：
![1.png](./res/doc_image/debug_gray.png)

debug_Sobel：
![1.png](./res/doc_image/debug_Sobel.png)

debug_threshold：
![1.png](./res/doc_image/debug_threshold.png)

debug_morphology：
![1.png](./res/doc_image/debug_morphology.png)

debug_Contours：
![1.png](./res/doc_image/debug_Contours.png)

debug_result：
![1.png](./res/doc_image/debug_result.png)

debug_crop：
![1.png](./res/doc_image/debug_crop_1.png)
![1.png](./res/doc_image/debug_crop_2.png)
![1.png](./res/doc_image/debug_crop_3.png)
![1.png](./res/doc_image/debug_crop_4.png)

debug_resize：
![1.png](./res/doc_image/debug_resize_1.png)
![1.png](./res/doc_image/debug_resize_2.png)
![1.png](./res/doc_image/debug_resize_3.png)
![1.png](./res/doc_image/debug_resize_4.png)



- **图片车牌文字识别过程**
待补齐

final_result:
![1.png](./res/doc_image/result_0.png)


#### 安装教程

- 将项目拉取到本地，PlateDetect文件夹拷贝到d盘下，默认车牌识别操作均在d:/PlateDetect/目录下处理，可以根据需要自行修改
- lib下依赖包添加到build path；或者修改pom文件的注释内容，将opencv-platform依赖取消注释
- spring boot方式运行项目，浏览器上输入 http://localhost:16666/index 即可打开操作界面
- 浏览器上输入 http://localhost:16666/swagger-ui.html 即可打开接口文档页面
- 

#### 使用说明

- 入门级教程项目，本人目前也正在学习图片识别相关技术；大牛请绕路
- 当前项目仅实现了黄牌、蓝牌车牌识别操作，接下来会继续优化代码架构，并且加上绿牌识别、车牌识别训练等操作
- 后续会逐步加入人脸识别等功能



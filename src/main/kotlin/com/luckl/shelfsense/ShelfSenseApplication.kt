package com.luckl.shelfsense

import org.bytedeco.javacpp.Loader
import org.bytedeco.opencv.global.opencv_core
import org.bytedeco.opencv.global.opencv_imgproc
import org.bytedeco.opencv.opencv_java
import org.opencv.core.Core
import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.runApplication

@SpringBootApplication
class ShelfSenseApplication

fun main(args: Array<String>) {

    System.setProperty("org.bytedeco.javacpp.logger.debug", "true")
    Loader.load(opencv_core::class.java)
    Loader.load(opencv_java::class.java)
    Loader.load(opencv_imgproc::class.java)
    println("OpenCV loaded successfully!")

    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
    println(Core.getBuildInformation())
    System.setProperty("opencv.debug", "true");

    runApplication<ShelfSenseApplication>(*args)


}


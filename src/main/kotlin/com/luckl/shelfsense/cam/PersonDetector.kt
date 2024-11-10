package com.luckl.shelfsense.cam

import jakarta.annotation.PostConstruct
import org.opencv.core.*
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import java.io.File
import kotlin.math.min

@Service
class PersonDetector {

    //logger
    private val logger = LoggerFactory.getLogger(PersonDetector::class.java)
    private val size = Size(300.0, 300.0)

    private val net: Net = Dnn.readNetFromCaffe(
        "C:\\tools\\MobileNetSSD.prototxt",
        "C:\\tools\\MobileNetSSD.caffemodel"
    )

    @PostConstruct
    fun init() {
        logger.info("PersonDetector initialized")

        val layerNames = net.layerNames
        logger.info("Loaded network with layers: $layerNames")

        val inputShape = net.unconnectedOutLayersNames
        logger.info("unconnected out Layers: $inputShape")

        processImage()
    }

    fun processImage() {
        // Load the image from resources
        val imagePath = "src/main/resources/test_image.webp" // Update the path if needed
        val frame: Mat = Imgcodecs.imread(imagePath)

        if (frame.empty()) {
            logger.error("Failed to load image. Make sure the path is correct: $imagePath")
            return
        }

        // Call detectPerson with the loaded frame
        val isPersonDetected = detectPerson(frame)
        logger.info("Person detected: $isPersonDetected")
    }

    fun detectPerson(frame: Mat): Boolean {
        val resizedFrame = Mat()
        Imgproc.resize(frame, resizedFrame, Size(300.0, 300.0))

        logger.info("CvType.CV_8UC3: ${CvType.CV_8UC3}")
        logger.info("resizedFrame rows: ${resizedFrame.rows()}, cols: ${resizedFrame.cols()}, channels: ${resizedFrame.channels()}")
        logger.info("resizedFrame type: ${resizedFrame.type()}")


        // Prepare the frame for the DNN model
        val blob = Dnn.blobFromImage(
            resizedFrame,
            1.0 / 127.5, // Normalize pixel values to [-1, 1]
            Size(300.0, 300.0),
            Scalar(127.5, 127.5, 127.5),
            true,
            false
        )
        logger.info("Blob rows: ${blob.rows()}, cols: ${blob.cols()}, channels: ${blob.channels()}")
        logger.info("Blob dimensions: ${blob.size(0)} x ${blob.size(1)} x ${blob.size(2)} x ${blob.size(3)}")

        net.setInput(blob, "data")
        try {
            val conv11 = net.forward("conv11")
            logger.info("Input blob shape: ${blob.size()}")
            logger.info("conv11 shape: ${conv11.rows()}x${conv11.cols()}")
            val priorbox11 = net.forward("conv11_mbox_priorbox")
            val priorbox13 = net.forward("conv13_mbox_priorbox")
            logger.info("conv11_mbox_priorbox shape: ${priorbox11.rows()}x${priorbox11.cols()}")
            logger.info("conv13_mbox_priorbox shape: ${priorbox13.rows()}x${priorbox13.cols()}")
            val detections = net.forward("detection_out")
            logger.info("Detections size: ${detections.size()}")
            logger.info("Detections type: ${detections.type()}")
            logger.info("Detections empty?: ${detections.empty()}")

            val rows = detections.rows()
            val cols = detections.cols()
            logger.info("Detections rows: $rows")
            logger.info("Detections cols: $cols")

            if (rows > 0 && cols > 6) { // Ensure valid shape
                for (i in 0 until rows) {
                    logger.info("Processing detection #$i")
                    val detection = detections.row(i)
                    val confidence = detection.get(0, 2)[0] // Access confidence
                    if (confidence > 0.5) {
                        val classId = detection.get(0, 1)[1].toInt()
                        logger.info("Class ID: $classId, Confidence: $confidence")
                        if (classId == 15) { // Class ID for 'person'
                            return true
                        }
                    }
                }
            } else {
                logger.warn("Invalid detections shape: rows=$rows, cols=$cols")
            }
        } catch (e: Exception) {
            logger.error("Error processing detections: ${e.message}", e)
        }
        return false
    }

}
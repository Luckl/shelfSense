package com.luckl.shelfsense.cam

import jakarta.annotation.PostConstruct
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.videoio.VideoWriter
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import org.springframework.web.reactive.function.client.WebClient
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.time.Duration
import java.time.Instant
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import javax.imageio.ImageIO

@Service
class CamClient(
    webClientBuilder: WebClient.Builder,
    private val personDetector: PersonDetector
) {
    private var recordingStartTime: Instant? = null // Tracks when recording started
    private val frameBuffer = ByteArrayOutputStream() // Accumulates incoming frame data

    //logger
    private val logger = LoggerFactory.getLogger(CamClient::class.java)

    private val webClient: WebClient = webClientBuilder.build()
    private var writer: VideoWriter? = null
    private var isRecording = false

    @PostConstruct
    fun startStreamProcessing() {
        val streamUrl = "http://192.168.0.149:81/stream" // Replace with your actual URL
        logger.info("Starting video stream processing for $streamUrl")
//        processStream(streamUrl)
    }

    fun processStream(url: String) {
        webClient.get()
            .uri(url)
            .header("Accept", "multipart/x-mixed-replace")
            .retrieve()
            .bodyToFlux(ByteArray::class.java)
            .doOnNext { frameData -> handleFrame(frameData) }
            .blockLast() // Process frames until the stream ends
    }

    private fun handleFrame(frameData: ByteArray) {
        try {
            // Append new frame data to the buffer
            frameBuffer.write(frameData)

            while (true) {
                val bufferBytes = frameBuffer.toByteArray()
                val startIndex = bufferBytes.indexOfStartJPEG()
                val endIndex = bufferBytes.indexOfEndJPEG()

                if (startIndex != -1 && endIndex != -1) {
                    if (endIndex > startIndex) {
                        // Extract the complete JPEG frame
                        val completeFrame = bufferBytes.copyOfRange(startIndex, endIndex + 1)

                        // Remove the processed frame from the buffer
                        frameBuffer.reset()
                        frameBuffer.write(bufferBytes.copyOfRange(endIndex + 1, bufferBytes.size))

                        // Process the complete JPEG frame
                        val inputStream = ByteArrayInputStream(completeFrame)
                        val image = ImageIO.read(inputStream)


                        if (image != null) {
                            val frame = bufferedImageToMat(image)
                            val personDetected = personDetector.detectPerson(frame)
                            manageRecording(frame, personDetected)
                        }
                    } else {
                        // Handle misaligned data: discard bytes before the valid start index
                        frameBuffer.reset()
                        frameBuffer.write(bufferBytes.copyOfRange(startIndex, bufferBytes.size))
                    }
                } else {
                    // No complete frame found, exit the loop
                    break
                }
            }
        } catch (e: Exception) {
            logger.error("Error processing frame: ${e.message}", e)
        }
    }

    // Extension functions to find JPEG markers
    private fun ByteArray.indexOfStartJPEG(): Int {
        for (i in 0 until this.size - 1) {
            if (this[i] == 0xFF.toByte() && this[i + 1] == 0xD8.toByte()) {
                return i
            }
        }
        return -1
    }

    private fun ByteArray.indexOfEndJPEG(): Int {
        for (i in 0 until this.size - 1) {
            if (this[i] == 0xFF.toByte() && this[i + 1] == 0xD9.toByte()) {
                return i + 1
            }
        }
        return -1
    }

    private fun manageRecording(frame: Mat, personDetected: Boolean) {
        if (personDetected && !isRecording) {
            // Start recording
            val outputFileName = "output/${generateTimestamp()}.mp4"
            writer = VideoWriter(
                outputFileName,
                VideoWriter.fourcc('X', 'V', 'I', 'D'),
                20.0, // Assumed FPS
                org.opencv.core.Size(frame.width().toDouble(), frame.height().toDouble())
            )
            isRecording = true
            recordingStartTime = Instant.now()
            logger.info("Recording started: $outputFileName")
        }

        writer?.let {
            if (isRecording) {

                it.write(frame)

                val recordingDuration = Duration.between(recordingStartTime, Instant.now()).seconds
                val shouldStopRecording = !personDetected && recordingDuration >= 5

                if (shouldStopRecording) {
                    // Stop recording
                    it.release()
                    isRecording = false
                    recordingStartTime = null
                    logger.info("Recording stopped.")
                }
            }
        }
    }

    private fun bufferedImageToMat(image: BufferedImage): Mat {
        val pixels = (image.raster.dataBuffer as java.awt.image.DataBufferByte).data

        val mat = Mat(image.height, image.width, CvType.CV_8UC3)
        mat.put(0, 0, pixels)
        return mat
    }

    private fun generateTimestamp(): String {
        return LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
    }
}
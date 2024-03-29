# Multi-Camera Object Detection on Nvidia Jetson Orin Nano

This project integrates the NVIDIA Jetson Orin Nano with two ZED 2i cameras, employing PyTorch and YOLOv8 for real-time object detection. The combination enables efficient processing of stereo images, making it suitable for edge applications such as autonomous navigation and surveillance, where rapid and accurate analysis of visual data is essential.

### Installation and Setup: 

The setup for the project is as follows:

- Download the Jetpack from NVIDIA for Jetson Orin Nano using the SD Card Image method from <code>https://developer.nvidia.com/embedded/jetpack-sdk-512</code>.

- Refer <code>https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#intro</code> for flash and setup instructions.

- Install the ultralytics package of Yolov8 (<code> https://github.com/ultralytics/ultralytics </code>).

      pip install ultralytics


- Download and Install the ZED SDK as per the instructions in <code>https://www.stereolabs.com/docs/installation/linux</code>.

---

Once the necessary installations are done, you can also proceed with exploring the zed tool via the downaloaded and installed SDK package.

The live feed can also be checked through the ZED_Explorer which is present in the path <code>/usr/local/zed/tools/ZED Explorer</code>. For further guide into running applications refer <code>https://www.stereolabs.com/docs/get-started-with-zed</code>






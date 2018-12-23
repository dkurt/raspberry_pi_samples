# Head pose estimation demo of Intel's OpenVINO toolkit for Raspberry Pi and Movidius Neural Compute Stick

To run this demo you'll need:

* Raspberry Pi board (with ARMv7-A CPU architecture). I had Raspberry Pi 2 model B.
* Installed OpenVINO toolkit: https://software.intel.com/articles/OpenVINO-Install-RaspberryPI
* Intel Movidius Neural Compute Stick (VPU)
* USB camera
* One LED
* Pre-trained face detection network:
  * [face-detection-retail-0004.bin](https://download.01.org/openvinotoolkit/2018_R4/open_model_zoo/face-detection-retail-0004/FP16/face-detection-retail-0004.bin)
  * [face-detection-retail-0004.xml](https://download.01.org/openvinotoolkit/2018_R4/open_model_zoo/face-detection-retail-0004/FP16/face-detection-retail-0004.xml)
* Pre-trained head pose estimation network:
  * [head-pose-estimation-adas-0001.bin](https://download.01.org/openvinotoolkit/2018_R4/open_model_zoo/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.bin)
  * [head-pose-estimation-adas-0001.xml](https://download.01.org/openvinotoolkit/2018_R4/open_model_zoo/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml)


Running this demo you can turn the light off looking toward to the camera. By default, script `run.py` assumes that your LED is connected to PIN number 2 (see [specifications](https://www.raspberrypi.org/documentation/usage/gpio/)).

Run a demo by the following command. Do not forget to run `setupvars.sh` script from the OpenVINO toolkit before it.

```
python3 run.py
```

Video: https://youtu.be/1xuSIMQ6Eac

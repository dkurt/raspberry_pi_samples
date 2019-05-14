# This script is based on https://github.com/opencv/open_model_zoo/blob/master/demos/security_barrier_camera_demo
import sys
import argparse
import os
import cv2 as cv
import time
import numpy as np

from threading import Thread

if sys.version[0] == '2':
    import Queue as queue
else:
    import queue as queue

sys.path.append('..')
from common import checkOrDownload_2019R1

parser = argparse.ArgumentParser('This is a demo to demonstrade Chinese License Plate '
                                 'detection and recognition models from Intel OpenVINO Open Model Zoo')
parser.add_argument('-m', default='.', dest='models', help='Path models folder. If not found, download it')
parser.add_argument('--lcd', action='store_true', help='Specify to use LCD for visualization')
parser.add_argument('--show', action='store_true', help='Show detections')
args = parser.parse_args()

if args.lcd:
    import RPi_I2C_driver
    lcd = RPi_I2C_driver.lcd()


# Download necessary files
checkOrDownload_2019R1(os.path.join(args.models, 'vehicle-license-plate-detection-barrier-0106.bin'), 'ba071df336ee5a21fc36795ec094d48c16fcc754')
checkOrDownload_2019R1(os.path.join(args.models, 'vehicle-license-plate-detection-barrier-0106.xml'), 'beef28ca8801d9127ec723e35d04ee40e6e31ca5')
checkOrDownload_2019R1(os.path.join(args.models, 'license-plate-recognition-barrier-0001.bin'), '59a7202c7b3b39fcb88c9226bb868bd3f1102816')
checkOrDownload_2019R1(os.path.join(args.models, 'license-plate-recognition-barrier-0001.xml'), '8c00b4199e956cd399f65c45a35f20e44b655df7')

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

process = True

#
# Camera frames capture thread
#
framesQueue = queue.Queue()
def framesThreadBody():
    global framesQueue, process

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv.CAP_PROP_FPS, 30)
    while process:
        hasFrame, frame = cap.read()
        if not hasFrame:
            process = False
            break
        framesQueue.put(frame)

#
# LCD update thread
#
if args.lcd:
    renderQueue = queue.Queue()
    def lcdRenderThread():
        global renderQueue, process
        lastProvince = ''
        lastNumber = ''
        while process:
            content = renderQueue.get()
            number = content[content.find('>')+1:]
            number = number[:6]  # Just to remove false positives
            province = content[1:content.find('>')]

            if (len(number) != 6) or ('<' in number):
                continue

            if province != lastProvince:
                tmp = province
                province += ' ' * (len(lastProvince) - len(province))
                lastProvince = tmp

                lcd.lcd_display_string(province, 1)

            if number != lastNumber:
                tmp = number
                number += ' ' * (len(lastNumber) - len(number))
                lastNumber = tmp
                lcd.lcd_display_string(number, 2)

    renderThread = Thread(target=lcdRenderThread)
    renderThread.start()


# Initialize networks.
detectionNet = cv.dnn.readNet('vehicle-license-plate-detection-barrier-0106.bin',
                              'vehicle-license-plate-detection-barrier-0106.xml')
recognitionNet = cv.dnn.readNet('license-plate-recognition-barrier-0001.bin',
                                'license-plate-recognition-barrier-0001.xml')
detectionNet.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)
recognitionNet.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

seq_ind = np.ones([88, 1], dtype=np.uint8)
seq_ind[0, 0] = 0

# Warp up
detectionNet.setInput(np.random.standard_normal([1, 3, 300, 300]).astype(np.uint8))
detectionNet.forward()

recognitionNet.setInput(np.random.standard_normal([1, 3, 24, 94]).astype(np.uint8), 'data')
recognitionNet.setInput(seq_ind, 'seq_ind')
recognitionNet.forward()


items = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
         "<Anhui>", "<Beijing>", "<Chongqing>", "<Fujian>",
         "<Gansu>", "<Guangdong>", "<Guangxi>", "<Guizhou>",
         "<Hainan>", "<Hebei>", "<Heilongjiang>", "<Henan>",
         "<HongKong>", "<Hubei>", "<Hunan>", "<InnerMongolia>",
         "<Jiangsu>", "<Jiangxi>", "<Jilin>", "<Liaoning>",
         "<Macau>", "<Ningxia>", "<Qinghai>", "<Shaanxi>",
         "<Shandong>", "<Shanghai>", "<Shanxi>", "<Sichuan>",
         "<Tianjin>", "<Tibet>", "<Xinjiang>", "<Yunnan>",
         "<Zhejiang>", "<police>",
         "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
         "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
         "U", "V", "W", "X", "Y", "Z"]

# Start frames capturing thread
framesThread = Thread(target=framesThreadBody)
framesThread.start()

# Skip the first frame to wait for camera readiness
framesQueue.get()

maxNumRequests = 3
futureDetections = []
framesToRender = []
futureRecognitions = []

# Main processing loop
numProcessedFrames = 0
startTime = time.time()
while process:
    # Get a next frame
    frame = None
    try:
        frame = framesQueue.get_nowait()
        if len(futureDetections) == maxNumRequests:
            frame = None  # Skip the frame
    except queue.Empty:
        pass

    if not frame is None:
        blob = cv.dnn.blobFromImage(frame, size=(300, 300), ddepth=cv.CV_8U)
        detectionNet.setInput(blob)
        futureDetections.append(detectionNet.forwardAsync())
        framesToRender.append(frame)

    # Check for ready detections.
    while futureDetections and futureDetections[0].wait_for(0) == 0:
        frame = framesToRender[0]
        out = futureDetections[0].get()

        # An every detection is a vector [imageId, classId, conf, x, y, X, Y]
        for detection in out.reshape(-1, 7):
            conf = detection[2]
            if conf < 0.4:
                continue

            classId = int(detection[1])
            if classId == 2:
                xmin = int(detection[3] * FRAME_WIDTH)
                ymin = int(detection[4] * FRAME_HEIGHT)
                xmax = int(detection[5] * FRAME_WIDTH)
                ymax = int(detection[6] * FRAME_HEIGHT)

                xmin = max(0, xmin - 5)
                ymin = max(0, ymin - 5)
                xmax = min(xmax + 5, FRAME_WIDTH - 1)
                ymax = min(ymax + 5, FRAME_HEIGHT - 1)

                if xmax - xmin < 93:  # Minimal weight is 94
                    break

                # Crop a license plate. Do some offsets to better fit a plate.
                lpImg = frame[ymin:ymax+1, xmin:xmax+1]
                blob = cv.dnn.blobFromImage(lpImg, size=(94, 24), ddepth=cv.CV_8U)
                recognitionNet.setInput(blob, 'data')
                recognitionNet.setInput(seq_ind, 'seq_ind')
                futureRecognitions.append(recognitionNet.forwardAsync())

                if args.show:
                    cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0))
                break

        if args.show:
            cv.imshow('det', frame)
            cv.waitKey(1)

        numProcessedFrames += 1

        del framesToRender[0]
        del futureDetections[0]

    # Check for ready recognitions.
    while futureRecognitions and futureRecognitions[0].wait_for(0) == 0:
        out = futureRecognitions[0].get()

        content = ''
        for idx in np.int0(out.reshape(-1)):
            if idx == -1:
                break
            content += items[idx]

        if args.lcd:
            renderQueue.put(content)
        else:
            print(content)

        del futureRecognitions[0]

    # print(numProcessedFrames / (time.time() - startTime), framesQueue.qsize())

process = False
framesThread.join()

if args.lcd:
    renderThread.join()

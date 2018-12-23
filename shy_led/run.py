import cv2 as cv
from gpiozero import LED
from math import cos, sin, pi

winName = 'OpenVINO on Raspberry Pi'

cv.namedWindow(winName, cv.WINDOW_NORMAL)

faceDetectionNet = cv.dnn.readNet('face-detection-retail-0004.xml', 'face-detection-retail-0004.bin')
headPoseNet = cv.dnn.readNet('head-pose-estimation-adas-0001.xml', 'head-pose-estimation-adas-0001.bin')
faceDetectionNet.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)
headPoseNet.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

cap = cv.VideoCapture(0)
led = LED(2)
led.on()

while cv.waitKey(1) != 27:
	hasFrame, frame = cap.read()
	if not hasFrame:
		break

	frameHeight, frameWidth = frame.shape[0], frame.shape[1]

	# Detect faces on the image.
	blob = cv.dnn.blobFromImage(frame, size=(300, 300), ddepth=cv.CV_8U)
	faceDetectionNet.setInput(blob)
	detections = faceDetectionNet.forward()

	for detection in detections.reshape(-1, 7):
		confidence = float(detection[2])
		if confidence > 0.5:
			xmin = int(detection[3] * frameWidth)
			ymin = int(detection[4] * frameHeight)
			xmax = int(detection[5] * frameWidth)
			ymax = int(detection[6] * frameHeight)

			xmax = max(1, min(xmax, frameWidth - 1))
			ymax = max(1, min(ymax, frameHeight - 1))
			xmin = max(0, min(xmin, xmax - 1))
			ymin = max(0, min(ymin, ymax - 1))

			# Run head pose estimation network.
			face = frame[ymin:ymax+1, xmin:xmax+1]
			blob = cv.dnn.blobFromImage(face, size=(60, 60), ddepth=cv.CV_8U)

			headPoseNet.setInput(blob)
			headPose = headPoseNet.forward(['angle_p_fc', 'angle_r_fc', 'angle_y_fc'])

			p, r, y = headPose[0][0], headPose[1][0], headPose[2][0]
			cos_r = cos(r * pi / 180)
			sin_r = sin(r * pi / 180)
			sin_y = sin(y * pi / 180)
			cos_y = cos(y * pi / 180)
			sin_p = sin(p * pi / 180)
			cos_p = cos(p * pi / 180)

			x = int((xmin + xmax) / 2)
			y = int((ymin + ymax) / 2)
			# center to right
			cv.line(frame, (x,y), (x+int(50*(cos_r*cos_y+sin_y*sin_p*sin_r)), y+int(50*cos_p*sin_r)), (0, 0, 255), thickness=3)
			# center to top
			cv.line(frame, (x, y), (x+int(50*(cos_r*sin_y*sin_p+cos_y*sin_r)), y-int(50*cos_p*cos_r)), (0, 255, 0), thickness=3)
			# center to forward
			cv.line(frame, (x, y), (x + int(50*sin_y*cos_p), y + int(50*sin_p)), (255, 0, 0), thickness=3)

			cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255))

			if abs(cos_y * cos_p) > 0.9:
				cv.putText(frame, 'FORWARD', (0, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)
				led.off()
			else:
				led.on()

	cv.imshow(winName, frame)

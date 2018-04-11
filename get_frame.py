import cv2

vidcap = cv2.VideoCapture('test1.mov')
start_time = 650000
for i in range(10):
	vidcap.set(cv2.CAP_PROP_POS_MSEC,start_time+(i*1000)) # ms
	success,image = vidcap.read()
	if success:
	    small = cv2.resize(image, (0,0), fx=0.3, fy=0.3)
	    cv2.imwrite("frame%d.jpg" % i, small)          
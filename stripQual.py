import sys
import cv2
import time
import numpy as np

# Record Start time for total FPS
start = time.time()*1000.0

# Define Upper-Lower Bound of threshold
threshval = (20, 30)

# Define Upper-Lower Bound of test strip threshold
#bluethresh = (160,170)
#bluethreshval = ((93,1,1), (98, 255, 255))
bluethreshval = ((90,60,120), (110, 250, 250))

# normaize the grayscale image
# NO LONGER SIGNIFICANT, ONLY USE WHEN CREATING CUSTOM FILTERS
def nm(img):
	normalizedImg = np.zeros(img.shape)
	return cv2.normalize(img, normalizedImg, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# Image Resizing function
def imgresize(timg, scale_percent):
	width = int(timg.shape[1] * scale_percent / 100)
	height = int(timg.shape[0] * scale_percent / 100)
	dim = (width, height)

	# resize image
	return cv2.resize(timg, dim, interpolation = cv2.INTER_AREA)


# Read File name
file = sys.argv[1]


print('Reading from file: ',file)


# Show steps
# 1 is true, 0 is false
disp = 0

# CODE USED FOR READING FILE FROM VIDEO FILE
cap = cv2.VideoCapture(file)

# Set Resolution and framerate
# 320x240-------?fps
# 640x480-------?fps
# 1280x720------?fps
# 1920x1080-----?
cap.set(3, 1280)
cap.set(4, 720)
cap.set(5, 120)




framecount = 0

while cap.isOpened():
	framecount = framecount + 1

	# Read an image
	t, imgorig = cap.read()

	if t == False or (cv2.waitKey(1) & 0xFF == ord('q')):
		break

	# Resize the image
	gray = imgresize(cv2.cvtColor(imgorig, cv2.COLOR_BGR2GRAY), 25)
	imgorig = imgresize(imgorig, 25)
	blue = imgorig.copy()

	# Convert Blue frame for color analysis
	blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)


	# Show both images
	cv2.imshow('Original image',imgorig)
	if disp == 1:
		cv2.imshow('Gray image', gray)

	# Locate Blue test strip
	# Threshold for "blue" values (Blurs with 2x2 kernel)
	bluethresh = cv2.inRange(blue, bluethreshval[0], bluethreshval[1])

	# Morphology to determine contours
	kernel = np.ones((2,2),np.uint8)
	blueret = cv2.dilate(bluethresh,kernel,iterations = 2)
	#blueret = cv2.erode(bluethresh,kernel,iterations = 2)

	# Find Single Contour with greatest area
	bluecontours, hierarchy = cv2.findContours(blueret, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	print(len(bluecontours))

	if len(bluecontours) == 1:

		rect = cv2.minAreaRect(bluecontours[0])
		angle = rect[2]

		if angle < -45:
    			angle = (90 + angle)

		# otherwise, just take the inverse of the angle to make
		# it positive
		else:
			angle = -angle

		# rotate the image to deskew it
		(h, w) = gray.shape[:2]
		center = (w // 2, h // 2)
		M = cv2.getRotationMatrix2D(center, angle, 1.0)
		gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

		if disp == 1:
			cv2.imshow('rotation', gray)


		# Draw Bounding Box
		x, y, w, h = cv2.boundingRect(bluecontours[0])
		cv2.rectangle(imgorig, (x, y), (x+w, y+h), (0,0,255), 2)

		shift = (int(x*1.05), int((y+h)*1.05))

		if disp == 1:
			cv2.imshow('Test Strip Thresh', imgorig)

		# Check Truncated Boundaries
		#if (int((y+h)*1.05 + 5) > int((((x+w)-x)*2+y+h)*.95) and (int(x*1.05) + 5) > int((x+w)*.95):
		if int((y+h)*1.05) + 5 < int((((x+w)-x)*2+y+h)*.95) and int(x*1.05) + 5 < int((x+w)*.95):
			print(y+h+15)
			print((((x+w)-x)*2+y+h-15))
			print(x+10)
			print(x+w-15)

			# Truncate Image
			#gray = gray[(y+h+15):(((x+w)-x)*2+y+h-15), (x+10):(x+w-15)]
			gray = gray[int((y+h)*1.05):int((((x+w)-x)*2+y+h)*.95), int(x*1.05):int((x+w)*.95)]
			if disp == 1:
				cv2.imshow('truncated', gray)

			# Apply a Gaussian filter of size 15x15
			gSize = 5;

			blur = cv2.GaussianBlur(gray,(gSize,gSize),0)

			# Diplay Blur
			if disp == 1:
				cv2.imshow('Gaussian Filter', blur)

			# Sobel Filter
			sobelH = np.array([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=np.float)
			sobelV = np.array([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=np.float)

			# Calculate Gradient Magnitude for image
			dx = cv2.Sobel(blur, cv2.CV_64F,1,0,sobelH)
			dy = cv2.Sobel(blur, cv2.CV_64F,0,1,sobelV)
			axy = cv2.magnitude(dx, dy).astype(np.uint8)


			# Display Stuff
			if disp == 1:
				cv2.imshow('Gradient Magnitude (Sobel Mask)', axy)

			# Threshhold image
			thresh = cv2.inRange(axy, threshval[0], threshval[1], 255)

			if disp == 1:
				cv2.imshow('Threshold', thresh)

			# Morphology to determine contours
			kernel = np.ones((1,1),np.uint8)
			ret = cv2.erode(thresh,kernel,iterations = 2)


			kernel = np.ones((1,1),np.uint8)
			# dilate = cv2.dilate(thresh,kernel,iterations = 1)
			#ret = cv2.erode(ret,kernel,iterations = 2)
			ret = cv2.morphologyEx(ret, cv2.MORPH_CLOSE, kernel)

			kernel = np.ones((5,5),np.uint8)
			ret = cv2.morphologyEx(ret, cv2.MORPH_CLOSE, kernel)

			kernel = np.ones((2,2),np.uint8)
			ret = cv2.dilate(ret,kernel,iterations = 2)
			if disp == 1:
				cv2.imshow('Morphology', ret)

			# Rotate back before determining contours
			(h, w) = ret.shape[:2]
			center = (w // 2, h // 2)
			M = cv2.getRotationMatrix2D(center, -angle, 1.0)
			gray = cv2.warpAffine(ret, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)



			# Calculate Contours
			contours, hierarchy = cv2.findContours(ret, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			cv2.drawContours(imgorig, contours, -1, (0,255,0), 1, offset=shift)
		else:
			print("Truncation Error")
	else:
		print("Blue Detection Error")
	cv2.imshow('Contours', imgorig)

# Print total time for frame
print('total time: ', time.time()*1000.0-start)

print('Average FPS', framecount/((time.time()*1000.0-start)/1000))



cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()

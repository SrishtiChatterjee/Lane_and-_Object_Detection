'''import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import math


#YOLO stuff here
# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO names
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Define classes you want to detect
class_names = ["car", "truck", "motorbike", "bicycle"]

 # Global parameters for lane detection
 # Gaussian smoothing
kernel_size = 3

# Canny Edge Detector
low_threshold = 50
high_threshold = 150

# Region-of-interest vertices
# We want a trapezoid shape, with bottom edge at the bottom of the image
trap_bottom_width = 0.85  # width of bottom edge of trapezoid, expressed as percentage of image width
trap_top_width = 0.07  # ditto for top edge of trapezoid
trap_height = 0.4  # height of the trapezoid expressed as percentage of image height

# Hough Transform
rho = 2 # distance resolution in pixels of the Hough grid
theta = 1 * np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15	 # minimum number of votes (intersections in Hough grid cell)
min_line_length = 10 #minimum number of pixels making up a line
max_line_gap = 20	# maximum gap in pixels between connectable line segments
 


# Helper functions
def grayscale(img):
	
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
def canny(img, low_threshold, high_threshold):
	
	return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
	
	return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
	
	#defining a blank mask to start with
	mask = np.zeros_like(img)   
	
	#defining a 3 channel or 1 channel color to fill the mask with depending on the input image
	if len(img.shape) > 2:
		channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255
		
	#filling pixels inside the polygon defined by "vertices" with the fill color	
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	
	#returning the image only where mask pixels are nonzero
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image



def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
	
	# In case of error, don't draw the line(s)
	if lines is None:
		return
	if len(lines) == 0:
		return
	draw_right = True
	draw_left = True
	
	# Find slopes of all lines
	# But only care about lines where abs(slope) > slope_threshold
	slope_threshold = 0.5
	slopes = []
	new_lines = []
	for line in lines:
		x1, y1, x2, y2 = line[0]  # line = [[x1, y1, x2, y2]]
		
		# Calculate slope
		if x2 - x1 == 0.:  # corner case, avoiding division by 0
			slope = 999.  # practically infinite slope
		else:
			slope = (y2 - y1) / (x2 - x1)
			
		# Filter lines based on slope
		if abs(slope) > slope_threshold:
			slopes.append(slope)
			new_lines.append(line)
		
	lines = new_lines
	
	# Split lines into right_lines and left_lines, representing the right and left lane lines
	# Right/left lane lines must have positive/negative slope, and be on the right/left half of the image
	right_lines = []
	left_lines = []
	for i, line in enumerate(lines):
		x1, y1, x2, y2 = line[0]
		img_x_center = img.shape[1] / 2  # x coordinate of center of image
		if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
			right_lines.append(line)
		elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
			left_lines.append(line)
			
	# Run linear regression to find best fit line for right and left lane lines
	# Right lane lines
	right_lines_x = []
	right_lines_y = []
	
	for line in right_lines:
		x1, y1, x2, y2 = line[0]
		
		right_lines_x.append(x1)
		right_lines_x.append(x2)
		
		right_lines_y.append(y1)
		right_lines_y.append(y2)
		
	if len(right_lines_x) > 0:
		right_m, right_b = np.polyfit(right_lines_x, right_lines_y, 1)  # y = m*x + b
	else:
		right_m, right_b = 1, 1
		draw_right = False
		
	# Left lane lines
	left_lines_x = []
	left_lines_y = []
	
	for line in left_lines:
		x1, y1, x2, y2 = line[0]
		
		left_lines_x.append(x1)
		left_lines_x.append(x2)
		
		left_lines_y.append(y1)
		left_lines_y.append(y2)
		
	if len(left_lines_x) > 0:
		left_m, left_b = np.polyfit(left_lines_x, left_lines_y, 1)  # y = m*x + b
	else:
		left_m, left_b = 1, 1
		draw_left = False
	
	# Find 2 end points for right and left lines, used for drawing the line
	# y = m*x + b --> x = (y - b)/m
	y1 = img.shape[0]
	y2 = img.shape[0] * (1 - trap_height)
	
	right_x1 = (y1 - right_b) / right_m
	right_x2 = (y2 - right_b) / right_m
	
	left_x1 = (y1 - left_b) / left_m
	left_x2 = (y2 - left_b) / left_m
	
	# Convert calculated end points from float to int
	y1 = int(y1)
	y2 = int(y2)
	right_x1 = int(right_x1)
	right_x2 = int(right_x2)
	left_x1 = int(left_x1)
	left_x2 = int(left_x2)
	
	# Draw the right and left lines on image
	if draw_right:
		cv2.line(img, (right_x1, y1), (right_x2, y2), color, thickness)
	if draw_left:
		cv2.line(img, (left_x1, y1), (left_x2, y2), color, thickness)
	
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
	
	lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
	line_img = np.zeros((*img.shape, 3), dtype=np.uint8)  # 3-channel RGB image
	draw_lines(line_img, lines)
	return line_img


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
	
	return cv2.addWeighted(initial_img, α, img, β, λ)

def filter_colors(image):
	
	# Filter white pixels
	white_threshold = 200 #130
	lower_white = np.array([white_threshold, white_threshold, white_threshold])
	upper_white = np.array([255, 255, 255])
	white_mask = cv2.inRange(image, lower_white, upper_white)
	white_image = cv2.bitwise_and(image, image, mask=white_mask)

	# Filter yellow pixels
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower_yellow = np.array([90,100,100])
	upper_yellow = np.array([110,255,255])
	yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
	yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)

	# Combine the two above images
	image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)

	return image2


def annotate_image_array(image_in):
    # Read in and grayscale the image
    gray = grayscale(image_in)

    # Apply Gaussian smoothing
    blur_gray = gaussian_blur(gray, kernel_size)

    # Apply Canny Edge Detector
    edges = canny(blur_gray, low_threshold, high_threshold)

    # Create masked edges using a trapezoid-shaped region-of-interest
    imshape = image_in.shape
    vertices = np.array([[
        ((imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0]),
        ((imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),
        (imshape[1] - (imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),
        (imshape[1] - (imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0])
    ]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # Run Hough on the edge-detected image
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # Draw lane lines on the original image
    annotated_image = weighted_img(line_image, image_in)
    
    return annotated_image


def annotate_video(input_file):
    video = VideoFileClip(input_file)
	video_clip = VideoFileClip(input_file)

 
    
    for frame in video.iter_frames(fps=video.fps):
        annotated_frame = annotate_image_array(frame)
        # Convert from RGB to BGR
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Annotated Video', annotated_frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
		
    
    for frame in video_clip.iter_frames(fps=video_clip.fps):
		blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getUnconnectedOutLayersNames()
        detections = net.forward(layer_names)

    # Process detections
        for detection in detections:
			for obj in detection:
				scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and classes[class_id] in class_names:
		            center_x = int(obj[0] * frame.shape[1])
                    center_y = int(obj[1] * frame.shape[0])
                    width = int(obj[2] * frame.shape[1])
                    height = int(obj[3] * frame.shape[0])

                    # Calculate box coordinates
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    # Draw bounding box and label with color
                    color = (0, 255, 0)  # BGR color for the box (green in this case)
                    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                    label = f"{classes[class_id]}: {confidence:.2f}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame with color
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow("Object Detection", frame)
	
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release OpenCV windows
cv2.destroyAllWindows()
'''


import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import os

# YOLO stuff here
# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO names
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Define classes you want to detect
class_names = ["car", "truck", "motorbike", "bicycle"]

# Global parameters for lane detection
# Gaussian smoothing
kernel_size = 3

# Canny Edge Detector
low_threshold = 50
high_threshold = 150

# Region-of-interest vertices
# We want a trapezoid shape, with a bottom edge at the bottom of the image
trap_bottom_width = 0.85  # width of the bottom edge of the trapezoid, expressed as a percentage of the image width
trap_top_width = 0.07  # ditto for the top edge of the trapezoid
trap_height = 0.4  # height of the trapezoid expressed as a percentage of the image height

# Hough Transform
rho = 2  # distance resolution in pixels of the Hough grid
theta = 1 * np.pi/180  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 10  # minimum number of pixels making up a line
max_line_gap = 20  # maximum gap in pixels between connectable line segments


# Helper functions
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    if lines is None:
        return
    if len(lines) == 0:
        return
    draw_right = True
    draw_left = True
    slope_threshold = 0.5
    slopes = []
    new_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0.:
            slope = 999.
        else:
            slope = (y2 - y1) / (x2 - x1)
        if abs(slope) > slope_threshold:
            slopes.append(slope)
            new_lines.append(line)
    lines = new_lines
    right_lines = []
    left_lines = []
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        img_x_center = img.shape[1] / 2
        if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
            right_lines.append(line)
        elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
            left_lines.append(line)
    right_lines_x = []
    right_lines_y = []
    for line in right_lines:
        x1, y1, x2, y2 = line[0]
        right_lines_x.append(x1)
        right_lines_x.append(x2)
        right_lines_y.append(y1)
        right_lines_y.append(y2)
    if len(right_lines_x) > 0:
        right_m, right_b = np.polyfit(right_lines_x, right_lines_y, 1)
    else:
        right_m, right_b = 1, 1
        draw_right = False
    left_lines_x = []
    left_lines_y = []
    for line in left_lines:
        x1, y1, x2, y2 = line[0]
        left_lines_x.append(x1)
        left_lines_x.append(x2)
        left_lines_y.append(y1)
        left_lines_y.append(y2)
    if len(left_lines_x) > 0:
        left_m, left_b = np.polyfit(left_lines_x, left_lines_y, 1)
    else:
        left_m, left_b = 1, 1
        draw_left = False
    y1 = img.shape[0]
    y2 = img.shape[0] * (1 - trap_height)
    right_x1 = (y1 - right_b) / right_m
    right_x2 = (y2 - right_b) / right_m
    left_x1 = (y1 - left_b) / left_m
    left_x2 = (y2 - left_b) / left_m
    y1 = int(y1)
    y2 = int(y2)
    right_x1 = int(right_x1)
    right_x2 = int(right_x2)
    left_x1 = int(left_x1)
    left_x2 = int(left_x2)
    if draw_right:
        cv2.line(img, (right_x1, y1), (right_x2, y2), color, thickness)
    if draw_left:
        cv2.line(img, (left_x1, y1), (left_x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def filter_colors(image):
    white_threshold = 200
    lower_white = np.array([white_threshold, white_threshold, white_threshold])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(image, lower_white, upper_white)
    white_image = cv2.bitwise_and(image, image, mask=white_mask)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([90, 100, 100])
    upper_yellow = np.array([110, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)
    image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)
    return image2

def annotate_image_array(image_in):
    gray = grayscale(image_in)
    blur_gray = gaussian_blur(gray, kernel_size)
    edges = canny(blur_gray, low_threshold, high_threshold)
    imshape = image_in.shape
    vertices = np.array([[
        ((imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0]),
        ((imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),
        (imshape[1] - (imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),
        (imshape[1] - (imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0])
    ]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    annotated_image = weighted_img(line_image, image_in)
    return annotated_image


def annotate_video(input_file):
    video = VideoFileClip(input_file)
    
    for frame in video.iter_frames(fps=video.fps):
        # Lane detection
        annotated_frame = annotate_image_array(frame)

        # Object detection
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getUnconnectedOutLayersNames()
        detections = net.forward(layer_names)

        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and classes[class_id] in class_names:
                    center_x = int(obj[0] * frame.shape[1])
                    center_y = int(obj[1] * frame.shape[0])
                    width = int(obj[2] * frame.shape[1])
                    height = int(obj[3] * frame.shape[0])

                    # Calculate box coordinates
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    # Draw bounding box and label with color
                    color = (0, 255, 0)  # BGR color for the box (green in this case)
                    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                    label = f"{classes[class_id]}: {confidence:.2f}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Combine lane detection and object detection
        combined_frame = weighted_img(frame, annotated_frame, α=0.8, β=1.0)

        # Convert from RGB to BGR
        combined_frame_bgr = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)

        cv2.imshow('Annotated Video', combined_frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


 
# Main script
if __name__ == '_main_':
    from optparse import OptionParser

    # Configure command line options
    parser = OptionParser()
    parser.add_option("-i", "--input_file", dest="input_file",
                    help="Input video file")

    # Get and parse command line options
    options, args = parser.parse_args()

    input_file = options.input_file

    annotate_video(input_file)

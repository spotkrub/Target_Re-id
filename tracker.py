from __future__ import print_function
import sys
import cv2 as cv
from random import randint


def createTrackerByName(trackerType):
  trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

  # Create a tracker based on tracker name
  if trackerType == trackerTypes[0]:
    tracker = cv.legacy.TrackerBoosting_create()
  elif trackerType == trackerTypes[1]:
    tracker = cv.legacy.TrackerMIL_create()
  elif trackerType == trackerTypes[2]:
    tracker = cv.legacy.TrackerKCF_create()
  elif trackerType == trackerTypes[3]:
    tracker = cv.legacy.TrackerTLD_create()
  elif trackerType == trackerTypes[4]:
    tracker = cv.legacy.TrackerMedianFlow_create()
  elif trackerType == trackerTypes[5]:
    tracker = cv.legacy.TrackerGOTURN_create()
  elif trackerType == trackerTypes[6]:
    tracker = cv.legacy.TrackerMOSSE_create()
  elif trackerType == trackerTypes[7]:
    tracker = cv.legacy.TrackerCSRT_create()
  else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
      print(t)
 
  return tracker


# Set video to load
videoPath = "../data/Blender/cam_move/cam_move_human0.mkv"
 
# Create a video capture object to read videos
cap = cv.VideoCapture(videoPath)
 
# Read first frame
success, frame = cap.read()
# quit if unable to read the video file
if not success:
  print('Failed to read video')
  sys.exit(1)


## Select boxes
bboxes = []
colors = [] 
 
# OpenCV's selectROI function doesn't work for selecting multiple objects in Python
# So we will call this function in a loop till we are done selecting all objects
while True:
  # draw bounding boxes over objects
  # selectROI's default behaviour is to draw box starting from the center
  # when fromCenter is set to false, you can draw box starting from top left corner
  bbox = cv.selectROI('MultiTracker', frame)
  bboxes.append(bbox)
  colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
  print("Press q to quit selecting boxes and start tracking")
  print("Press any other key to select next object")
  k = cv.waitKey(0) & 0xFF
  if (k == 113):  # q is pressed
    break
 
print('Selected bounding boxes {}'.format(bboxes))

# Specify the tracker type
trackerType = "CSRT"

# Create MultiTracker object
multiTracker = cv.legacy.MultiTracker_create()


# Initialize MultiTracker
for bbox in bboxes:
  multiTracker.add(createTrackerByName(trackerType), frame, bbox)


# Process video and track objects
while cap.isOpened():
  success, frame = cap.read()
  if not success:
    break
 
  # get updated location of objects in subsequent frames
  success, boxes = multiTracker.update(frame)

  # draw tracked objects
  for i, newbox in enumerate(boxes):
    p1 = (int(newbox[0]), int(newbox[1]))
    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
    cv.rectangle(frame, p1, p2, colors[i], 2, 1)

  # show frame
  cv.imshow('MultiTracker', frame)
 
  # quit on ESC button
  if cv.waitKey(1) & 0xFF == 27:  # Esc pressed
    break
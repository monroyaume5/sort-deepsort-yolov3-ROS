#!/usr/bin/env python

"""
ROS node to track objects using DEEP_SORT TRACKER and YOLOv4 detector (darknet_ros)
Takes detected bounding boxes from darknet_ros and uses them to calculated tracked bounding boxes
Tracked objects and their ID are published to the sort_track node
For this reason there is a little delay in the publishing of the image that I still didn't solve
"""
import rospy
import numpy as np
from darknet_ros_msgs.msg import BoundingBoxes
from deep_sort.detection import Detection
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
from deep_sort import preprocessing as prep
#from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image
from sort_track.msg import IntList
from _collections import deque
import matplotlib.pyplot as plt
import sys


def imgmsg_to_cv2(img_msg):
    #print('img_msg.encoding',img_msg.encoding)
	# convert rgb8 to bgr8
    if img_msg.encoding != "rgb8":
        rospy.logerr("This Coral detect node has been hardcoded to the 'rgb8' encoding.  Come change the code if you're actually trying to implement a new camera")
    #if img_msg.encoding != "bgr8":
    #    rospy.logerr("This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
    dtype = np.dtype("uint8") # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                    dtype=dtype, buffer=img_msg.data)
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    return image_opencv

def cv2_to_imgmsg(cv_image):
    img_msg = Image()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
    return img_msg


def get_parameters():
    """
    Gets the necessary parameters from .yaml file
    Returns tuple
    """
    camera_topic = rospy.get_param("~camera_topic")
    detection_topic = rospy.get_param("~detection_topic")
    tracker_topic = rospy.get_param('~tracker_topic')
    return (camera_topic, detection_topic, tracker_topic)


def callback_det(data):
    global detections
    global scores
    global classes
    detections = []
    scores = []
    classes = []
    for box in data.bounding_boxes:
        detections.append(np.array([box.xmin, box.ymin, box.xmax-box.xmin, box.ymax-box.ymin]))
        scores.append(float('%.2f' % box.probability))
        classes.append(box.Class)
    #for box in data.bounding_boxes:
    #   if box.Class in allowed_classes:
    #       detections.append(np.array([box.xmin, box.ymin, box.xmax-box.xmin, box.ymax-box.ymin]))
    #       scores.append(float('%.2f' % box.probability))
    #       classes.append(box.Class)   
    detections = np.array(detections)


def callback_image(data):
    #initialize color map and for each bbox using different color
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
    #Display Image
    #bridge = CvBridge()
    #cv_rgb = bridge.imgmsg_to_cv2(data, "bgr8")
    cv_rgb = imgmsg_to_cv2(data)
    cv_rgb = cv2.cvtColor(cv_rgb, cv2.COLOR_BGR2RGB)
    #Features and detections
    features = encoder(cv_rgb, detections)

    detections_new = [Detection(bbox, score,class_name,feature) for bbox,score,class_name,feature in
                        zip(detections,scores,classes, features)]   # transform data format from tlbr to tlwh(top,left,width,height)
    #detections_new = [Detection(bbox, score, feature) for bbox,score, feature in
        #                zip(detections,scores, features)]   # transform data format from tlbr to tlwh(top,left,width,height)
    # Run non-maxima suppression.
    boxes = np.array([d.tlwh for d in detections_new])
    scores_new = np.array([d.confidence for d in detections_new])
    classes_new = np.array([d.class_name for d in detections_new])
    nms_max_overlap = 1.0
    indices = prep.non_max_suppression(boxes, nms_max_overlap, scores_new)
    #indices = prep.non_max_suppression(boxes, 1.0 , scores_new)
    detections_new = [detections_new[i] for i in indices]
    # Call the tracker
    tracker.predict()
    tracker.update(detections_new)
    
    #Detecting bounding boxes
    #for det in detections_new:
    #   bbox = det.to_tlbr()
    #   cv2.rectangle(cv_rgb,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(100,255,50), 1)
    #   cv2.putText(cv_rgb , "person", (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100,255,50), lineType=cv2.LINE_AA)
    
    #Tracking bounding boxes
    for track in tracker.tracks:
        # update tracks
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        class_name= track.get_class()
        #fillout msg
        msg.data = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), track.track_id, class_name]
        # draw bbox on screen
        color = colors[int(track.track_id) % len(colors)] #for each bbox using different color
        color = [i * 255 for i in color] #from 0-1 to normal rgb value 0-255
        #bbox for object
        cv2.rectangle(cv_rgb, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2) 
        #box for showing 'class_name + track_id'
        cv2.rectangle(cv_rgb, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
        #putText in middle of above box by using bbox[1]-10 rather than bbox[1]-30
        cv2.putText(cv_rgb, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
        #cv2.rectangle(cv_rgb, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 1)
        #cv2.putText(cv_rgb, str(track.track_id),(int(bbox[2]), int(bbox[1])),0, 5e-3 * 200, (255,255,255),1)
        ##for showing trajectory
        center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
        pts[track.track_id].append(center)
        for j in range(1, len(pts[track.track_id])):
            if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
                continue
            thickness = int(np.sqrt(64/float(j+1))*2)
            cv2.line(cv_rgb, (pts[track.track_id][j-1]), (pts[track.track_id][j]), color, thickness)
    cv2.imshow("YOLOV4+SORT", cv_rgb)
    cv2.waitKey(3)
        

def main():
    global tracker  #remember global
    global encoder
    global msg
    global allowed_classes
    global pts
    # custom allowed classes (uncomment line below to customize tracker for only people)
    allowed_classes = ['person']
    ##for showing trajectory
    pts = [deque(maxlen=30) for _ in range(1000)]
    
    msg = IntList()
    # Definition of the parameters
    max_cosine_distance = 0.4  #max_cosine_distance = 0.2
    nn_budget = None   #nn_budget = 100
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)
    ## initialize deep sort
    # mars-small128.pb is a model just for people detection
    #model_filename = "/model_data/mars-small128.pb" #Change it to your directory
    model_filename = "/home/kevalen/yolov4_ws_deepsort/src/sort_track/src/deep_sort/model_data/mars-small128.pb" #Change it to your directory
    encoder = gdet.create_box_encoder(model_filename) #use to form appearence feature
    #Initialize ROS node
    rospy.init_node('sort_tracker', anonymous=True)
    rate = rospy.Rate(10)
    # Get the parameters
    (camera_topic, detection_topic, tracker_topic) = get_parameters()
    #Subscribe to darknet_ros to get BoundingBoxes from YOLO
    #sub_detection = rospy.Subscriber(detection_topic, BoundingBoxes , callback_det)
    #Subscribe to image topic
    image_sub = rospy.Subscriber(camera_topic,Image,callback_image)
    ##Subscribe to darknet_ros to get BoundingBoxes from YOLO
    sub_detection = rospy.Subscriber(detection_topic, BoundingBoxes , callback_det)
    while not rospy.is_shutdown():
        #Publish results of object tracking
        pub_trackers = rospy.Publisher(tracker_topic, IntList, queue_size=10)
        print(msg)
        pub_trackers.publish(msg)
        rate.sleep()


if __name__ == '__main__':
    try :
        main()
    except rospy.ROSInterruptException:
        pass

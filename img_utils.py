import datetime
import cv2
import numpy as np
from centroidtracker import CentroidTracker

def read_classes(file_Path):
    with open(file_Path, "r") as f:
        classes = [cname.strip() for cname in f.readlines()]
    return classes

def get_x2y2(bbox):
    x, y, w, h = bbox
    x2 = x + w
    y2  =y + h
    return x, y, x2, y2

def interaction_time(interaction_frmaes, fps):
    time_in_sec = interaction_frmaes / fps #"{:.2f}".format
    inter_time = datetime.timedelta(seconds=time_in_sec)
    return inter_time

def bb_intersection_over_union(boxAx, boxBx):
	# determine the (x, y)-coordinates of the intersection rectangle
    boxA = get_x2y2(boxAx)
    boxB = get_x2y2(boxBx)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
    return iou

def calculate_time(total_frames, fps):
    time_of_video = round(total_frames / fps)
    video_time  = datetime.timedelta(seconds=time_of_video)
    return str(video_time) #"{:.2f}".format(time_of_video)

def resize_one(img, size):
    out = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    return out

def create_area(frame):
    food_beat_pt_xy = (370, 0)
    food_beat_pt_wh = (100, 50)
    cv2.rectangle(frame, food_beat_pt_xy, (food_beat_pt_xy[0]+100, food_beat_pt_xy[1]+50), (0, 0, 0), 1)
    cv2.putText(frame, 'Fodder Beet', (food_beat_pt_xy[0]-130, food_beat_pt_xy[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)
    return frame, (food_beat_pt_xy[0], food_beat_pt_xy[1], food_beat_pt_wh[0], food_beat_pt_wh[1])


def process_yolo_frame(frame, model, COLORS, classes_name):
    pig_head_rects = []
    detected_pigs_feeding = 0
    detected_pigs_sniffing_jutebag = 0
    snout_boxes = []
    woodenbeem_boxes = []
    jutebag_boxes = []
    # perform object detection
    classes, scores, boxes = model.detect(frame, confThreshold=0.6, nmsThreshold=0.4)
    for (class_id, score, box) in zip(classes, scores, boxes): # box [x, y, width, height]
        color = [int(c) for c in COLORS[class_id]]
        label = "%s: %.2f" % (classes_name[class_id], score)
        cv2.rectangle(frame, box, color, 1)
        cv2.putText(frame, label, (box[0]+60, box[1]+60), cv2.FONT_HERSHEY_TRIPLEX, 1, color, 1)
        if classes_name[class_id] == "pig head":
            bx2 = int(box[0] + box[2])
            by2 = int(box[1] + box[3])
            cx, cy = int((box[0] + bx2) //2), int((box[1] + by2) // 2) 
            cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
            pig_head_rects.append(box)
        if classes_name[class_id] == "snout":
            snout_boxes.append(box)
        if classes_name[class_id] == "wooden beem":
            woodenbeem_boxes.append(box)
        if classes_name[class_id] == "jute bag":
            jutebag_boxes.append(box)
    pig_head_rects = np.array(pig_head_rects)
    # Initialize Tracker
    tracker = CentroidTracker(maxDisappeared=300, maxDistance=90)
    tracked_objects = tracker.update(pig_head_rects)
    for id, obj in tracked_objects.items():
        tracking_id = "Id_{}".format(id)
        cv2.putText(frame, tracking_id, (obj[0], obj[1]), cv2.FONT_HERSHEY_TRIPLEX, 0.7, [20, 255, 255], 1)

    frame, fodder_beet_area = create_area(frame=frame)
    # Pigs' feeding 
    for ibbox in pig_head_rects:
        iou_pigs_head = bb_intersection_over_union(fodder_beet_area , ibbox)
        if iou_pigs_head >= 0.2 or classes_name[class_id] == "pigs feeding":
            detected_pigs_feeding = 1
            # detected_pigs_feeding.append(frame)
            cv2.putText(frame, 'Pigs feeding', (fodder_beet_area[0]+100, fodder_beet_area[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    
    # Pigs' sniffing
    for jutbagxboxes in jutebag_boxes:
        for snoutxboxe in snout_boxes:
            iou_pigs_snount_jutbag = bb_intersection_over_union(jutbagxboxes, snoutxboxe)
            if iou_pigs_snount_jutbag >= 0.02:
                # detected_pigs_sniffing_jutebag.append(frame)
                detected_pigs_sniffing_jutebag = 1
                cv2.putText(frame, 'Pigs sniffing jute bag', (snoutxboxe[0]+10, snoutxboxe[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    return frame, detected_pigs_feeding, detected_pigs_sniffing_jutebag
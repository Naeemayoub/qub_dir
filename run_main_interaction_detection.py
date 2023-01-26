import cv2
import numpy as np
import pathlib
from utils.img_utils import resize_one, read_classes, interaction_time, \
                            process_yolo_frame, calculate_time
# Video Read and Writer
root_path = pathlib.Path.home()
save_path = str(root_path) + '/' + 'meta_databases/processed_data/results/'
file_path = "/Users/naeem/meta_databases/raw_data/qub_dir/Dataset/enrichment_videos/batch_1/all_time/311/20210506100000.mov" # 20210506100000.mov , test_video_1.mov, test_video_2, D13_20210506101139.mp4
vidfile_name = file_path.split('/')[-1].split('.')[0]
fourcc = cv2.VideoWriter_fourcc(*'avc1')
writer = cv2.VideoWriter(save_path + vidfile_name + "_output.mp4", fourcc, 15.0, (1280,  720))

cap = cv2.VideoCapture(file_path)
# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video file")

# video length
length_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
recod_vid_fps = cap.get(cv2.CAP_PROP_FPS)
total_video_time = calculate_time(length_frames , recod_vid_fps)


# Initialize Model
trained_wg = "trained_weights/yolov4-tiny-10000custom_best.weights"
cfg_path = "trained_weights/yolov4-tiny-6custom.cfg"
net = cv2.dnn.readNet(trained_wg, cfg_path)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(608, 608), scale=1/255)

# Initial classes and corresponding colors
classes_path = "trained_weights/classes6.txt"
classes_name = read_classes(classes_path)
# Initialize color
COLORS = np.random.randint(100, 255, size=(len(classes_name), 3), dtype='uint8')

count = 0
total_frames = []
detected_pigs_feed1 = []
detected_pigs_sniffing1 = []
while cap.isOpened():
    ret, frame = cap.read()
    count = count + 1
    frame = resize_one(frame, (1280, 720))
    frame, detected_pigs_feed, detected_pigs_sniffing = process_yolo_frame(frame, model=model, COLORS=COLORS, classes_name=classes_name)
    if detected_pigs_feed == 1:
        detected_pigs_feed1.append(detected_pigs_feed)
    if detected_pigs_sniffing == 1:
        detected_pigs_sniffing1.append(detected_pigs_sniffing)
    
    total_count_pigs_interaction_feeeding = len(detected_pigs_feed1)
    total_count_pigs_interaction_sniffing = len(detected_pigs_sniffing1)
    total_count_pigs_interaction = total_count_pigs_interaction_feeeding + total_count_pigs_interaction_sniffing
    cv2.putText(frame, "video length: " + total_video_time, (600, 550), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "Processing frame {0}/{1}".format(count, str(length_frames)), (600, 580), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, "Interaction Counts: " + str(total_count_pigs_interaction), (600, 610), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255), 1)
    fodderbeet_interaction_time = interaction_time(total_count_pigs_interaction_feeeding , recod_vid_fps)
    cv2.putText(frame, "interaction time with fodder beet " + str(fodderbeet_interaction_time) + " sec", (600, 640), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255), 1)
    sniffing_interaction_time = interaction_time(total_count_pigs_interaction_sniffing , recod_vid_fps)
    cv2.putText(frame, "interaction time with jute bag/sniffing " + str(sniffing_interaction_time) + " sec", (600, 670), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255), 1)

    # writer.write(frame)
    cv2.imshow('Frame', frame)
    print('processsing frame: {0}'.format(count))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
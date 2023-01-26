# This python program file contain interaction_detection with following main functions.

## run_main_interaction_detection.py: 
The application will start detection and displaying the interaction on the image frame in realtime.
The main function will import follwoing  function from img_utils.py
    1. resize_one: Resizes the image frame
    read_classes: Reads the classes from text file and create list of classes.
    calculate_time: Calculates the total time of video 
    interaction_time: Calculates the interaction time
    process_yolo_frame: Processes the frame in yolo model and produce the detection results. 
The addtional input files is 'img_utils.py' which contains all other function mentioned above.

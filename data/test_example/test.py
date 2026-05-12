import cv2, os, argparse, random
import numpy as np
import csv

def init_parameter():   
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--videos", type=str, default='foo_videos/', help="Dataset folder")
    parser.add_argument("--results", type=str, default='foo_results/', help="Results folder")
    args = parser.parse_args()
    return args

args = init_parameter()

# path CSV file of results
csv_path = os.path.join(args.results, "results.csv")

# Here you should initialize your method

################################################

with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Id Video", "Start(Seconds)"])
    # For all the test videos
    for video in os.listdir(args.videos):
        # Process the video
        ret = True
        cap = cv2.VideoCapture(video)
        while ret:
            ret, img = cap.read()
            # Here you should add your code for applying your method

            ########################################################
        cap.release()
        # Here you should add your code for writing the results
        pos_neg = random.randint(0, 1)
        if pos_neg == 0:
            start_instant_sec = ""
        else: 
            start_instant_sec = random.randint(0, 300) # the incident start time in seconds
        writer.writerow([video, start_instant_sec])
        ########################################################
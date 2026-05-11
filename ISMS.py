import cv2
import numpy as np
from ultralytics import YOLO
import os
from tkinter import Tk, filedialog
import csv
from datetime import datetime

# ---------------- SELECT VIDEO ----------------
Tk().withdraw()
video_path = filedialog.askopenfilename(
    title="Select Video File",
    filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
)

if not video_path:
    print("No file selected. Exiting.")
    exit()

print("Selected video:", video_path)

# ---------------- LOAD MODEL ----------------
model = YOLO("C:/Users/saiva/Downloads/Industrial Safety Monitoring/models/yolov8n.pt")

# ---------------- CREATE OUTPUT FOLDERS ----------------
os.makedirs("output", exist_ok=True)
os.makedirs("C:/Users/saiva/Downloads/Industrial Safety Monitoring/output/violations", exist_ok=True)

# ---------------- CREATE CSV LOG ----------------
log_file = open("C:/Users/saiva/Downloads/Industrial Safety Monitoring/output/logs/violation_log.csv", "w", newline='')
writer = csv.writer(log_file)
writer.writerow(["Date", "Time", "Violation Type", "Frame"])

# ---------------- VIDEO CAPTURE ----------------
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0

# prevent repeated capture
last_logged_frame = -100

# ---------------- OVERLAP FUNCTION ----------------
def box_overlap(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    return interArea > 0

# ---------------- HELPER FUNCTION ----------------
def log_violation(violation_type, frame):
    global last_logged_frame

    # save only once every 60 frames (~2 seconds)
    if frame_count - last_logged_frame < 60:
        return

    last_logged_frame = frame_count

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H-%M-%S")

    filename = f"C:/Users/saiva/Downloads/Industrial Safety Monitoring/output/violations/{violation_type}_{frame_count}.jpg"
    cv2.imwrite(filename, frame)

    writer.writerow([date, time, violation_type, frame_count])
    print(f"Violation Detected: {violation_type}")

# ---------------- MAIN LOOP ----------------
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    results = model(frame)[0]

    persons = []
    helmets = []
    vests = []

    # ----------- OBJECT DETECTION -----------
    for box in results.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        if conf < 0.45:
            continue

        # PERSON
        if label == "person":
            persons.append((x1, y1, x2, y2))
            color = (0,255,0)

        # HELMET
        elif label == "helmet":
            helmets.append((x1, y1, x2, y2))
            color = (255,0,0)

        # VEST
        elif label in ["vest", "safety_vest", "safety-vest", "jacket", "reflective_vest"]:
            vests.append((x1, y1, x2, y2))
            color = (0,165,255)

        else:
            color = (255,255,255)

        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,label,(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    # ----------- VIOLATION CHECKING -----------
    for px1, py1, px2, py2 in persons:

        helmet_found = False
        vest_found = False

        person_box = (px1, py1, px2, py2)

        # ---- HEAD REGION (TOP 35%) ----
        head_y2 = py1 + int((py2 - py1) * 0.35)
        head_box = (px1, py1, px2, head_y2)

        # ---- HELMET CHECK (HEAD ONLY) ----
        for h in helmets:
            if box_overlap(head_box, h):
                helmet_found = True
                break

        # ---- VEST CHECK (FULL BODY) ----
        for v in vests:
            if box_overlap(person_box, v):
                vest_found = True
                break

        # HELMET VIOLATION
        if not helmet_found:
            cv2.putText(frame,"NO HELMET!",
                        (px1, py1-40),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            log_violation("No_Helmet", frame)

        # VEST VIOLATION
        if not vest_found:
            cv2.putText(frame,"NO SAFETY VEST!",
                        (px1, py1-70),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            log_violation("No_Vest", frame)

    # ----------- DISPLAY -----------
    cv2.imshow("Industrial Safety Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
cap.release()
log_file.close()
cv2.destroyAllWindows()

print("Monitoring Finished.")
print("Check 'output/violations' folder for images.")
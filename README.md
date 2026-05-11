# Industrial Safety Monitoring System

This project is an AI-powered Industrial Safety Monitoring System developed using Python, YOLOv8, OpenCV, and Computer Vision techniques. The main objective of the system is to improve workplace safety by automatically detecting whether workers are wearing essential Personal Protective Equipment (PPE) such as helmets and safety vests.

The system processes video footage in real time and uses the YOLOv8 object detection model to identify persons, helmets, and safety vests in each frame. After detecting workers, the program analyzes specific regions of the body to verify safety compliance. If a worker is found without a helmet or safety vest, the system immediately flags it as a violation.

Whenever a violation is detected, the application displays warning messages on the video frame, captures a screenshot of the violation, and stores the details in a CSV log file containing the date, time, frame number, and violation type. This helps maintain safety records and improves monitoring efficiency in industrial and construction environments.

The project also includes a simple GUI-based video selection system using Tkinter and live monitoring visualization using OpenCV. The generated outputs can be used for workplace safety analysis, surveillance, and automated compliance checking.

## Features

* Real-time PPE detection using YOLOv8
* Person, helmet, and safety vest detection
* Automatic safety violation identification
* Violation screenshot capture
* CSV-based logging system
* Live video monitoring
* GUI-based video file selection

## Technologies Used

* Python
* OpenCV
* YOLOv8 (Ultralytics)
* NumPy
* Tkinter
* Computer Vision
* Deep Learning

## Applications

* Industrial safety monitoring
* Construction site surveillance
* Factory worker compliance checking
* Smart AI surveillance systems
* Automated workplace safety analysis

This project demonstrates the practical implementation of Artificial Intelligence and Deep Learning for real-world industrial safety and surveillance applications.

import cv2
from ultralytics import YOLO

# Open video
cap = cv2.VideoCapture("/home/youngcto13/aisandbox/myproj/video2.mp4")
assert cap.isOpened(), "Error reading video file"

# Get video properties
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video writer
video_writer = cv2.VideoWriter(
    "object_detection_output.avi",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h),
)

# Load YOLO model
model = YOLO("yolo26n.pt")

# Process video
while cap.isOpened():

    success, frame = cap.read()
    if not success:
        print("Video finished.")
        break

    # Run object detection
    results = model(frame)

    # Draw detections
    annotated_frame = results[0].plot()

    # Show frame
    cv2.imshow("Object Detection", annotated_frame)

    # Save frame
    video_writer.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
video_writer.release()
cv2.destroyAllWindows()

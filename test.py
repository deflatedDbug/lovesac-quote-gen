from ultralytics import YOLO
import cv2

model_weights = "runs/detect/train/weights/best.pt"
model = YOLO(model_weights)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera. Exiting.")
    exit()

print("Press 'space' to capture the image, 'ESC' to exit...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv2.imshow("Camera Feed - Press Space to Capture", frame)

    key = cv2.waitKey(1)
    if key % 256 == 32:
        # Save captured image to a file
        image_path = "captured_image.jpg"
        cv2.imwrite(image_path, frame)
        print("Image captured.")
        
        result = model(image_path, save=True)
        print("Processing result:", result)

        break
    elif key % 256 == 27:
        print("Exiting.")
        break

cap.release()
cv2.destroyAllWindows()  # Ensure this is called correctly to close windows

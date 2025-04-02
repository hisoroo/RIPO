import cv2
import torch
from PIL import Image

from core.detector import Detector
from core.embedder import Embedder
from db.database import Database


def register_user_from_camera(
    user_id: str, device="cuda" if torch.cuda.is_available() else "cpu"
):
    detector = Detector(device=device)
    embedder = Embedder(device=device)
    database = Database()

    cap = cv2.VideoCapture(0)
    print("📷 Starting camera... Look directly into the camera.")
    registered = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detection = detector.detect(frame)
        if detection:
            box = detection["box"]
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "Face detected - press 'c' to capture",
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Register User", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c") and detection:
            print("📸 Capturing image and creating embedding...")
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            embedding = embedder.get_embedding(image)

            if embedding is not None:
                database.save_embedding(user_id, embedding)
                print(f"✅ Registered new user: {user_id}")
                registered = True
            else:
                print("❌ Failed to generate embedding.")
            break
        elif key == ord("q"):
            print("❌ Registration canceled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return registered


def main():
    print("👤 User Registration")
    user_id = input("Enter user ID to register: ").strip()
    if user_id:
        success = register_user_from_camera(user_id)
        if success:
            print("✅ Registration complete.")
        else:
            print("❌ Registration failed.")
    else:
        print("❌ Invalid user ID. Exiting.")


if __name__ == "__main__":
    main()

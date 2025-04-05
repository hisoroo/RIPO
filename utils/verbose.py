import cv2


def print_logs(user_id, distance, detection):
    print(f"Odległość od wzorca: {distance:.4f}")
    print(f"Koordynaty obwiedni twarzy: {detection['box']}")
    print(f"Pewność detekcji twarzy: {detection['confidence']:.2f}")
    print()


def draw_overlay(frame, detection, distance=None):
    box = detection["box"]
    x1, y1, x2, y2 = box

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

    cv2.putText(
        frame,
        f"fdc: {detection['confidence']:.2f}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        1,
    )

    if distance is not None:
        cv2.putText(
            frame,
            f"dist: {distance:.4f}",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            1,
        )

import cv2


def setup_capture(mode, video_path):
    if mode == "file":
        print(f"Otwieranie pliku wideo: {video_path}")
        cap = cv2.VideoCapture(video_path)
    else:
        print("Uruchamianie kamery na żywo")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Nie udało się otworzyć źródła wideo!")

    return cap


def setup_recorder(cap, output_path):
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

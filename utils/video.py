import cv2


def setup_capture(mode, file_path):
    if mode == "video":
        print(f"Otwieranie pliku wideo: {file_path}")
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise RuntimeError("Nie udało się otworzyć pliku wideo!")
        return cap
    elif mode == "image":
        print(f"Otwieranie zdjęcia: {file_path}")
        frame = cv2.imread(file_path)
        if frame is None:
            raise RuntimeError("Nie udało się wczytać zdjęcia!")
        return frame
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

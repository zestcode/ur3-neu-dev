import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

CAMERA_INDICES = [1, 3]
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480

caps = []
for idx in CAMERA_INDICES:
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {idx}")
    caps.append(cap)
    print(f"Camera {idx} opened")

root = tk.Tk()
root.title("Dual Webcam Live Stream")
root.protocol("WM_DELETE_WINDOW", lambda: (
    [cap.release() for cap in caps], root.destroy()
))

label = tk.Label(root)
label.pack()

def update_frame():
    frames = []
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.putText(frame, f"cam{i} (idx {CAMERA_INDICES[i]})", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            frames.append(frame)

    if frames:
        combined = np.hstack(frames) if len(frames) > 1 else frames[0]
        img = ImageTk.PhotoImage(Image.fromarray(combined))
        label.imgtk = img
        label.configure(image=img)

    root.after(33, update_frame)  # ~30 fps

update_frame()
root.mainloop()

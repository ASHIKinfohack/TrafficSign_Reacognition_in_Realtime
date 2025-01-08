import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import threading
import sys
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageTk, ImageColor


after_id = None


model = tf.keras.models.load_model('model.h5')


cap = cv2.VideoCapture(0)


engine = pyttsx3.init()




female_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
engine.setProperty('voice', female_voice_id)

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (32, 32))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    return img



sign_names = {
    0: 'Speed limit (20 kilometer per hour)',
    1: 'Speed limit (30 kilometer per hour)',
    2: 'Speed limit (50 kilometer per hour)',
    3: 'Speed limit (60 kilometer per hour)',
    4: 'Speed limit (70 kilometer per hour)',
    5: 'Speed limit (100 kilometer per hour)',
    6: 'End of speed limit (80 kilometer per hour)',
    7: 'Speed limit (80 kilometer per hour)',
    8: 'Speed limit (120 kilometer per hour)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'vehicles over 3.5 metric tons',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

threshold = 0.68

def speak(sign_name):
    try:
        engine.startLoop(False)
        if sign_name != 'No Sign Detected':
            engine.say('Predicted traffic sign is ' + sign_name)
            print('Predicted Traffic Sign is:', sign_name)
        engine.iterate()
        engine.endLoop()
    except:
        sys.exc_info()

previous_sign = None

def update_ui(sign_name):
    global previous_sign
    if sign_name != 'No Sign Detected' and sign_name != previous_sign:
        detected_sign_name_label.config(text="Detected Sign Name: " + sign_name)
        previous_sign = sign_name
        root.update_idletasks()  # Update the UI to reflect changes immediately


def capture_frames():
    global after_id  # Declare after_id as global
    ret, frame = cap.read()
    img = preprocess(frame)
    pred = model.predict(np.array([img]), verbose=0)
    label = np.argmax(pred)
    probability = pred[0][label]

    if probability >= threshold:
        sign_name = sign_names.get(label, "Unknown Sign")
        update_ui(sign_name)
        speak_thread = threading.Thread(target=speak, args=(sign_name,))
        speak_thread.start()

    else:
        update_ui('No Sign Detected')
        speak_thread = threading.Thread(target=speak, args=('No Sign Detected',))
        speak_thread.start()

    cv2.imshow('Traffic Sign Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        root.after_cancel(after_id)  # Cancel the animation after closing the window
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)
    else:
        after_id = root.after(10, capture_frames)  # Recursive call for animation


def quit():
    root.after_cancel(after_id)  # Cancel the animation
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)


def generate_gradient(size=(800, 600), start_color="#FF6347", end_color="#1E90FF"):
    img = Image.new("RGB", size)
    draw = ImageDraw.Draw(img)
    width, height = size
    (r1, g1, b1) = ImageColor.getrgb(start_color)
    (r2, g2, b2) = ImageColor.getrgb(end_color)

    for y in range(height):
        r = int(r1 + (y / height) * (r2 - r1))
        g = int(g1 + (y / height) * (g2 - g1))
        b = int(b1 + (y / height) * (b2 - b1))
        draw.line((0, y, width, y), fill=(r, g, b))

    return ImageTk.PhotoImage(img)

root = tk.Tk()
root.title("Realtime Traffic Sign Detector")
gradient_bg = generate_gradient(size=(800, 600))
canvas = tk.Canvas(root, width=800, height=600)
canvas.create_image(0, 0, anchor=tk.NW, image=gradient_bg)
canvas.pack()
style = ttk.Style()
style.configure('TLabel', font=('Helvetica', 16), padding=10, foreground='blue')
style.configure('TButton', font=('Helvetica', 16), padding=10, background='green', foreground='white')
style.map('TButton', background=[('active', 'red')])
detected_sign_name_label = ttk.Label(root, text="Detected Sign Name: ", style='TLabel')
detected_sign_name_label.pack()



class VideoFrame(ttk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.video_label = tk.Label(self, bg='black')
        self.video_label.pack()


video_frame = VideoFrame(root, style='TVideoFrame.TFrame')
video_frame.pack()


def on_enter(event):
    curved_button1.config(bg="#45a049")


def on_leave(event):
    curved_button1.config(bg="#4CAF50")


curved_button1 = tk.Button(root, text="Start Detection", command=capture_frames, font=('Helvetica', 16, 'bold'),
                           bg='#4CAF50', fg='white', bd=0, highlightthickness=0,
                           activebackground='#45a049', activeforeground='white', relief=tk.FLAT, cursor="hand2")
curved_button1.bind("<Enter>", on_enter)
curved_button1.bind("<Leave>", on_leave)
curved_button1.pack(side="left")


quit_button = tk.Button(root, text="QUIT", command=quit, font=('Helvetica', 16, 'bold'),
                        bg='#4CAF50', fg='white', bd=0, highlightthickness=0,
                        activebackground='#45a049', activeforeground='white', relief=tk.FLAT, cursor="hand2")
quit_button.bind("<Enter>", on_enter)
quit_button.bind("<Leave>", on_leave)
quit_button.pack(side="right")


root.mainloop()

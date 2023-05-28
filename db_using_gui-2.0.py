import tkinter
import tkinter.messagebox
import customtkinter as ctk
import os
from tkinter import messagebox
import webbrowser
import json
import msal
import requests
from datetime import datetime, timedelta
import pyperclip
from PIL import Image, ImageTk
import sys
import time
import cv2 as cv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
import threading
import shutil
import fetch_meeting_func as fmf
import greeting_and_draw as g_and_d
import Apollo_on_the_couch as couch
from sklearn.model_selection import GridSearchCV
from playsound import playsound


ctktheme= "blue"
if ctktheme == "green":
    button_standard_color = ("#2CC985", "#2FA572")
elif ctktheme == "blue":
    button_standard_color = ("#3B8ED0", "#1F6AA5")
else:
    button_standard_color = ("#3a7ebf", "#1f538d")

ctk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme(ctktheme)  # Themes: "blue" (standard), "green", "dark-blue"
user_path = 'dataset/'

class FACELOADING:
    def __init__(self, directory):
        self.directory = directory
        self.target_size =(160,160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()
    
    def extract_face(self, filename):
        try:
            _, ext = os.path.splitext(filename)
            if ext.lower() not in ('.png', '.jpg', '.jpeg'):
                return None
            img = cv.imread(filename)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            face = img
            #x,y,w,h = self.detector.detect_faces(img)[0]['box']
            #x,y = abs(x), abs(y)
            #face = img[y:y+h,x:x+w]
            face_arr = cv.resize(face, self.target_size)
            return face_arr
        except:
            return None

    def load_faces(self, dir):
        FACES = []
        app.train_progressbar.button_1.configure(state='disabled',fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"))
        for im_name in os.listdir(dir):
            try:
                path = os.path.join(dir, im_name)
                _, ext = os.path.splitext(path)
                if ext.lower() not in ('.png', '.jpg', '.jpeg'):
                    continue
                single_face = self.extract_face(path)
                FACES.append(single_face)
                app.progress_during_training = app.progress_during_training + app.progress_after_1_image
                app.train_progressbar.progressbar.set(app.progress_during_training)
            except:
                pass
        return FACES
    
    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = os.path.join(self.directory, sub_dir)
            if not os.path.isdir(path):
                continue
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            print(f"Loaded successfully: {len(labels)}")
            self.X.extend(FACES)
            self.Y.extend(labels)
        return np.asarray(self.X), np.asarray(self.Y)
class FaceDetection(ctk.CTkToplevel):

    def __init__(self,
                 master: any = None,
                 directory: str = None):
        super().__init__()

        self.cap = cv.VideoCapture(0)
        self.padding = 0
        self.image_width = 1280
        self.image_height = 720
        self.mtcnn = MTCNN()
        self.caffe = cv.dnn.readNetFromCaffe('Data/deploy.prototxt.txt', 'Data/caffe.caffemodel')

        self.geometry(f"{1170}x{600}")
        #self.resizable(width=False, height=False)
        self.title("F.A.C.E")
        self.attributes("-topmost", True)
        self.rowconfigure(0, weight=1)
        self.selected_user = directory
        self.lift()

        self.icon = ctk.CTkImage(Image.open('Data/camera.png'), size=(30,30))
        self.button_capture_photo = ctk.CTkButton(self, width=50, height=50, text="", image=self.icon, corner_radius=40, command=self.capture_button_capture)
        self.button_capture_photo.grid(row=1, column=0,padx=(0,0), pady=(10,10))

        self.quit_button = ctk.CTkButton(self, width = 80, height = 40,text = "Quit" , command=self.quit_button)
        self.quit_button.grid(row = 1, column =1, padx=(0,0), pady=(0,10))
        self.quit_when_true = False

        
        self.scrollable_frame_users = []

        self.update_image_frame()

        self.recognized_users = []
        self.recognized_placed_users = []
        thread = threading.Thread(target=self.detect_face)
        thread.start()

    def capture_button_capture(self):
        thread = threading.Thread(target=self.capture_photo_button)
        thread.start()
        
    def button_pressed(self, image_path):
        # Open the image and resize it
        image = Image.open(image_path)
        image = image.resize((160, 160))
        
        # Create a PhotoImage from the resized image
        photo = ImageTk.PhotoImage(image)

        
        # Create a Label to display the image and add a title
        label = ctk.CTkLabel(master=self, image=photo, text='')
        label.image = photo # keep a reference to the image to prevent garbage collection
        label.grid(row=0, column=1, padx=20, pady=30, sticky='sw')
        
        title = ctk.CTkLabel(master=self, text="", font=("Arial", 0))
        title.grid(row=0, column=1, sticky='s')

    def capture_photo_button(self):
        print("Capture")
        self.play_sound("Data/camera-sound.wav")
        _, frame = self.cap.read()
        #frame = cv.flip(frame,1)

        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb_image = cv.cvtColor(self.cap.read()[1],cv.COLOR_BGR2RGB)

        # Create a blob from the input image
        blob = cv.dnn.blobFromImage(rgb_img, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Set input to the model
        self.caffe.setInput(blob)

        # Run inference and get the output
        detections = self.caffe.forward()

        save_path = 'dataset/'
        total_count_real = 0
        for file in os.listdir(f"{save_path}{self.selected_user}"):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                total_count_real += 1

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                # Get the coordinates of the bounding box for the face
                x = int(detections[0, 0, i, 3] * rgb_img.shape[1])
                y = int(detections[0, 0, i, 4] * rgb_img.shape[0])
                w = int(detections[0, 0, i, 5] * rgb_img.shape[1]) - x
                h = int(detections[0, 0, i, 6] * rgb_img.shape[0]) - y

                padding_x = 0
                padding_y = 0
                x1, y1 = max(0, x-padding_x), max(0, y-padding_y)
                face_image = Image.fromarray(cv.cvtColor(rgb_image[y1:y1+h+(2*padding_y), x1:x1+w+(2*padding_x)], cv.COLOR_RGBA2RGB))

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                filename = f"{save_path}{self.selected_user}/{total_count_real}.jpg"
                total_count_real += 1
                self.scrollable_frame.configure(label_text=f"{total_count_real} Images are available.")
                print("saved " + filename)
                face_image.save(filename)
                self.content = app.get_all_contents(app.users)
                app.create_user_elements(app.users, self.content)
                self.button_pressed(filename)
                self.update_image_frame()
                break

    def play_sound(self, voice_path):
        try:
            playsound(voice_path, block=False)
            return True
        except Exception as e:
            print(f"Error while playing sound {e}")
            return False
    
    def quit_button(self):
        print("quiting...")
        self.quit_when_true = True

    def update_image_frame(self):
        total_count = 0
        for file in os.listdir(f"{user_path}{self.selected_user}"):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                total_count += 1
        
        self.scrollable_frame = ctk.CTkScrollableFrame(self,width = 190, height = 300, label_text=f"{total_count} Images is available.", corner_radius=0)
        self.scrollable_frame.grid(row = 0, column = 1, sticky = "ne", padx=(0,0), pady=(0,0))
        
        for i in range(total_count):
            image_path = f"{user_path}{self.selected_user}/{i}.jpg"
            button = ctk.CTkButton(master=self.scrollable_frame, text="Image #" + str(i), command=lambda image_path=image_path: self.button_pressed(image_path))
            button.grid(row=i, column=0, padx=(0, 0), pady=(2, 0), sticky="w")
            self.scrollable_frame_users.append(button)

    def detect_face(self):
        while self.cap.isOpened():
            _, frame = self.cap.read()
            rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Define the input blob for the Caffe model
            blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

            # Pass the input blob to the Caffe model and get the output
            self.caffe.setInput(blob)
            detections = self.caffe.forward()

            # Iterate over the detections and draw bounding boxes around the faces
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    x1 = int(detections[0, 0, i, 3] * frame.shape[1])
                    y1 = int(detections[0, 0, i, 4] * frame.shape[0])
                    x2 = int(detections[0, 0, i, 5] * frame.shape[1])
                    y2 = int(detections[0, 0, i, 6] * frame.shape[0])
                    color = (114, 162, 47)
                    rgb_img = g_and_d.fancyDraw(rgb_img, x=x1, y=y1, x1=x2, y1=y2, color=color)

            frame_display = ctk.CTkImage(Image.fromarray(rgb_img), size=(960, 540))
            frame_display_label = ctk.CTkLabel(self, image=frame_display, text=None)
            frame_display_label.grid(row=0, column=0, sticky="nw", padx=(0,5), pady=(0,0))

            if self.quit_when_true:
                self.destroy()
                break

        self.cap.release()
        cv.destroyAllWindows()

    def update_scrollableframe(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        for user in self.recognized_placed_users:
            name = ctk.CTkButton(self.scrollable_frame, state = "disabled", text = user[0][2:-2] + " " + str(int(user[1]*100)), hover = False,fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"))
            name.pack(side="top",padx=5, pady=10)
class CTkMessagebox(ctk.CTkToplevel):
    
    def __init__(self,
                 master: any = None,
                 width: int = 400,
                 height: int = 200,
                 title: str = "CTkMessagebox",
                 message: str = "This is a CTkMessagebox!",
                 option_1: str = "OK",
                 option_2: str = None,
                 option_3: str = None,
                 border_width: int = 1,
                 border_color: str = "default",
                 button_color: str = "default",
                 bg_color: str = "default",
                 fg_color: str = "default",
                 text_color: str = "default",
                 title_color: str = "default",
                 button_text_color: str = "default",
                 button_width: int = None,
                 button_height: int = None,
                 cancel_button_color: str = "#c42b1c",
                 button_hover_color: str = "default",
                 icon: str = "info",
                 icon_size: tuple = None,
                 corner_radius: int = 15,
                 font: tuple = None,
                 header: bool = False,
                 topmost: bool = True,
                 fade_in_duration: int = 0):
        
        super().__init__()

        self.master_window = master
        self.width = 250 if width<250 else width
        self.height = 150 if height<150 else  height
            
        if self.master_window is None:
            self.spawn_x = int((self.winfo_screenwidth()-self.width)/2)
            self.spawn_y = int((self.winfo_screenheight()-self.height)/2)
        else:
            self.spawn_x = int(self.master_window.winfo_width() * .5 + self.master_window.winfo_x() - .5 * self.width + 7)
            self.spawn_y = int(self.master_window.winfo_height() * .5 + self.master_window.winfo_y() - .5 * self.height + 20)
            
        self.after(10)
        self.geometry(f"{self.width}x{self.height}+{self.spawn_x}+{self.spawn_y}")
        self.title(title)
        self.resizable(width=False, height=False)
        self.fade = fade_in_duration
        
        if self.fade:
            self.fade = 20 if self.fade<20 else self.fade
            self.attributes("-alpha", 0)
            
        if not header:
            self.overrideredirect(1)
    
    
        if topmost:
            self.attributes("-topmost", True)
        #else:
        #    self.transient(self.master_window)
    
        if sys.platform.startswith("win"):
            self.transparent_color = self._apply_appearance_mode(self._fg_color)
            self.attributes("-transparentcolor", self.transparent_color)
        elif sys.platform.startswith("darwin"):
            self.transparent_color = 'systemTransparent'
            self.attributes("-transparent", True)
        else:
            self.transparent_color = '#000001'
            corner_radius = 0

        self.lift()
        self.config(background=self.transparent_color)
        self.protocol("WM_DELETE_WINDOW", self.button_event)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)    
        self.x = self.winfo_x()
        self.y = self.winfo_y()
        self._title = title
        self.message = message
        self.font = font
        self.round_corners = corner_radius if corner_radius<=30 else 30
        self.button_width = button_width if button_width else self.width/4
        self.button_height = button_height if button_height else 28
        if self.fade: self.attributes("-alpha", 0)
        
        if self.button_height>self.height/4: self.button_height = self.height/4 -20
        self.dot_color = cancel_button_color
        self.border_width = border_width if border_width<6 else 5
    
        if bg_color=="default":
            self.bg_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["fg_color"])
        else:
            self.bg_color = bg_color

        if fg_color=="default":
            self.fg_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["top_fg_color"])
        else:
            self.fg_color = fg_color

        default_button_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkButton"]["fg_color"])
        
        if button_color=="default":
            self.button_color = (default_button_color, default_button_color, default_button_color)
        else:
            if type(button_color) is tuple:
                if len(button_color)==2:                
                    self.button_color = (button_color[0], button_color[1], default_button_color)
                elif len(button_color)==1:
                    self.button_color = (button_color[0], default_button_color, default_button_color)
                else:
                    self.button_color = button_color
            else:
                self.button_color = (button_color, button_color, button_color)

        if text_color=="default":
            self.text_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkLabel"]["text_color"])
        else:
            self.text_color = text_color

        if title_color=="default":
            self.title_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkLabel"]["text_color"])
        else:
            self.title_color = title_color
            
        if button_text_color=="default":
            self.bt_text_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkButton"]["text_color"])
        else:
            self.bt_text_color = button_text_color

        if button_hover_color=="default":
            self.bt_hv_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkButton"]["hover_color"])
        else:
            self.bt_hv_color = button_hover_color
            
        if border_color=="default":
            self.border_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["border_color"])
        else:
            self.border_color = border_color
            
        if icon_size:
            self.size_height = icon_size[1] if icon_size[1]<=self.height-100 else self.height-100
            self.size = (icon_size[0], self.size_height)
        else:
            self.size = (self.height/4, self.height/4)
        
        if icon in ["check", "cancel", "info", "question", "warning"]:
            self.icon = ctk.CTkImage(Image.open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Data', icon+'.png')), size=self.size)
        else:
            self.icon = ctk.CTkImage(Image.open(icon), size=self.size) if icon else None

        self.frame_top = ctk.CTkFrame(self, corner_radius=self.round_corners, width=self.width, border_width=self.border_width,
                                                bg_color=self.transparent_color, fg_color=self.bg_color, border_color=self.border_color)
        self.frame_top.grid(sticky="nswe")

        if button_width:
            self.frame_top.grid_columnconfigure(0, weight=1)
        else:
            self.frame_top.grid_columnconfigure((1,2,3), weight=1)

        if button_height:
            self.frame_top.grid_rowconfigure((0,1,3), weight=1)
        else:
            self.frame_top.grid_rowconfigure((0,1,2), weight=1)
            
        self.frame_top.bind("<B1-Motion>", self.move_window)
        self.frame_top.bind("<ButtonPress-1>", self.oldxyset)
        
        self.button_close = ctk.CTkButton(self.frame_top, corner_radius=10, width=10, height=10, hover=False,
                                           text="", fg_color=self.dot_color, command=self.button_event)
        self.button_close.configure(cursor="arrow")        
        self.button_close.grid(row=0, column=3, sticky="ne", padx=10, pady=10)

        self.title_label = ctk.CTkLabel(self.frame_top, width=1, text=self._title, text_color=self.title_color, font=self.font)
        self.title_label.grid(row=0, column=0, columnspan=4, sticky="nw", padx=(15,30), pady=5)
        self.title_label.bind("<B1-Motion>", self.move_window)
        self.title_label.bind("<ButtonPress-1>", self.oldxyset)
        
        self.info = ctk.CTkButton(self.frame_top,  width=1, height=self.height/2, corner_radius=0, text=self.message, font=self.font,
                                            fg_color=self.fg_color, hover=False, text_color=self.text_color, image=self.icon)
        self.info._text_label.configure(wraplength=self.width/2, justify="left")
        self.info.grid(row=1, column=0, columnspan=4, sticky="nwes", padx=self.border_width)
        
        if self.info._text_label.winfo_reqheight()>self.height/2:
            height_offset = int((self.info._text_label.winfo_reqheight())-(self.height/2) + self.height)
            self.geometry(f"{self.width}x{height_offset}")
            
        self.option_text_1 = option_1
        self.button_1 = ctk.CTkButton(self.frame_top, text=self.option_text_1, fg_color=self.button_color[0],
                                                width=self.button_width, font=self.font, text_color=self.bt_text_color,
                                                hover_color=self.bt_hv_color, height=self.button_height,
                                                command=lambda: self.button_event(self.option_text_1))
        
        self.button_1.grid(row=2, column=3, sticky="news", padx=(0,10), pady=10)

        if option_2:
            self.option_text_2 = option_2      
            self.button_2 = ctk.CTkButton(self.frame_top, text=self.option_text_2, fg_color=self.button_color[1],
                                                    width=self.button_width, font=self.font, text_color=self.bt_text_color,
                                                    hover_color=self.bt_hv_color, height=self.button_height,
                                                    command=lambda: self.button_event(self.option_text_2))
            self.button_2.grid(row=2, column=2, sticky="news", padx=10, pady=10)
            
        if option_3:
            self.option_text_3 = option_3
            self.button_3 = ctk.CTkButton(self.frame_top, text=self.option_text_3, fg_color=self.button_color[2],
                                                    width=self.button_width, font=self.font, text_color=self.bt_text_color,
                                                    hover_color=self.bt_hv_color, height=self.button_height,
                                                    command=lambda: self.button_event(self.option_text_3))
            self.button_3.grid(row=2, column=1, sticky="news", padx=(10,0), pady=10)

        if header:
            self.title_label.grid_forget()
            self.button_close.grid_forget()
            self.frame_top.configure(corner_radius=0)

        if self.winfo_exists():
            self.grab_set()
            
        if self.fade:
            self.fade_in()
        
    def fade_in(self):
        for i in range(0,110,10):
            if not self.winfo_exists():
                break
            self.attributes("-alpha", i/100)
            self.update()
            time.sleep(1/self.fade)
            
    def fade_out(self):
        for i in range(100,0,-10):
            if not self.winfo_exists():
                break
            self.attributes("-alpha", i/100)
            self.update()
            time.sleep(1/self.fade)

    def get(self):
        if self.winfo_exists():
            self.master.wait_window(self)
        return self.event
        
    def oldxyset(self, event):
        self.oldx = event.x
        self.oldy = event.y
    
    def move_window(self, event):
        self.y = event.y_root - self.oldy
        self.x = event.x_root - self.oldx
        self.geometry(f'+{self.x}+{self.y}')
        
    def button_event(self, event=None):
        try:
            self.button_1.configure(state="disabled")
            self.button_2.configure(state="disabled")
            self.button_3.configure(state="disabled")
        except AttributeError:
            pass

        if self.fade:
            self.fade_out()
        self.grab_release()
        self.destroy()
        self.event = event
class CTkFaceRecognizer(ctk.CTkToplevel):

    def __init__(self,
                 master: any = None):
        super().__init__()
        self.master_window = master
        self.facenet = FaceNet()
        self.faces_embeddings = np.load("Data/face_embeddings_done_4classes.npz")
        self.Y = self.faces_embeddings['Y']
        self.encoder = LabelEncoder()
        self.encoder.fit(self.Y)

        self.caffe = cv.dnn.readNetFromCaffe('Data/deploy.prototxt.txt', 'Data/caffe.caffemodel')
        self.model = pickle.load(open("Data/svm_model_160x160.pkl", 'rb'))

        self.cap = cv.VideoCapture(0)
        self.padding = 10

        self.mtcnn = MTCNN()

        self.image_width = 1280
        self.image_height = 720

        self.button_frame_height = 60

        self.user_found_width = 200

        self.width = self.image_width + self.user_found_width
        self.height = self.image_height
        
        self.spawn_x = int(self.master_window.winfo_width() * .5 + self.master_window.winfo_x() - .5 * self.width + 7)
        self.spawn_y = int(self.master_window.winfo_height() * .5 + self.master_window.winfo_y() - .5 * self.height + 20)

        self.after(10)
        self.geometry(f"{self.width}x{self.height}+{self.spawn_x}+{self.spawn_y}")

        self.resizable(width=False, height=False)
        self.title("F_A_C_E Recognition")

        self.attributes("-topmost", True)

        self.scrollable_frame = ctk.CTkScrollableFrame(self, width=self.user_found_width, height = self.image_height-self.button_frame_height)#
        self.scrollable_frame.grid( row= 0, column=1, sticky = "news", padx=(5,0))

        self.recognized_users = []
        self.recognized_placed_users = []

        self.button_quit_camera = ctk.CTkButton(self, width=140, height=40, command=self.quit_button, text="Quit")
        self.button_quit_camera.grid( row = 1, column = 1)

        self.quit_when_true = False

        thread = threading.Thread(target=self.recognize)
        thread.start()

    def recognize(self):
        cooldowns = {"default": 0}
        timeout = 5  # Timeout in seconds
        last_frame_time = time.time()

        while self.cap.isOpened():
            start_time = time.time()
            _, frame = self.cap.read()

            if frame is None:
                # No frame received, check for timeout
                current_time = time.time()
                if current_time - last_frame_time > timeout:
                    print("Timeout: No frames received")
                    break
                else:
                    continue

            last_frame_time = time.time()

            rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            h, w = frame.shape[:2]
            blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

            self.caffe.setInput(blob)
            detections = self.caffe.forward()

            if detections.shape[2] > 0:
                try:
                    face_detected = False  # Initialize the face_detected flag

                    for i in range(0, detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence < 0.5:
                            continue

                        face_detected = True  # Set the face_detected flag to True

                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # Check if the face region is within the bounds of the frame
                        if startX < 0:
                            startX = 0
                        if startY < 0:
                            startY = 0
                        if endX > w:
                            endX = w
                        if endY > h:
                            endY = h

                        img = rgb_img[startY:endY, startX:endX]
                        img = cv.resize(img, (160, 160))
                        img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
                        img = np.expand_dims(img_gray, axis=0)
                        img = np.expand_dims(img, axis=-1)
                        img = np.repeat(img, 3, axis=-1)
                        ypred = self.facenet.embeddings(img)
                        face_name = self.model.predict(ypred)
                        proba = self.model.predict_proba(ypred).max()

                        if proba > 0.75:
                            color = (114, 162, 47)  # Blue
                            text_color = (114, 162, 47)  # Blue 
                            final_name = face_name[0]
                            name_parts = str(face_name).split("_")
                            face_name = " ".join(name_parts)
                            self.update_recognized_users(face_name, proba)
                        else:
                            color = (98, 47, 165)  # Orange
                            text_color = (98, 47, 165)  # Orange
                            final_name = 'Unknown'
                            proba = 1 - proba

                        cv.putText(frame, '{} ({:.2%})'.format(final_name, proba), (startX, startY-20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, text_color, 2, cv.LINE_AA)
                        rgb_image = cv.cvtColor(g_and_d.fancyDraw(frame, x=startX, y=startY, x1=endX, y1=endY, color=color), cv.COLOR_BGR2RGB)
                    
                    if not face_detected:
                        # No faces detected
                        final_name = 'Empty'
                        rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                except Exception as e:
                    print(f"Error reading video frame: {e}")
                    continue
            else:
                # No faces detected
                final_name = 'Empty'
                rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            frame_display = ctk.CTkImage(Image.fromarray(rgb_image), size=(self.image_width, self.image_height))
            frame_display_label = ctk.CTkLabel(self, image=frame_display, text=None)
            frame_display_label.grid(row=0, column=0, sticky="news", rowspan=2, padx=(0, 0), pady=(0, 0))

            if self.quit_when_true:
                self.destroy()
                break
            print(final_name)
            if final_name in cooldowns and cooldowns[final_name] > time.time():
                cooldown_left = round(cooldowns[final_name] - time.time(), 2)
                print(f"{final_name} is still on cooldown for {cooldown_left} seconds!")
            elif final_name not in ['Unknown', 'Empty']:
                cooldowns[final_name] = time.time() + 60
                nextMeeting = fmf.fetch_next_meeting_time(os.path.join('dataset', final_name))
                g_and_d.greeting_and_reminder(final_name, meetingTime=nextMeeting)

            end_time = time.time()
            elapsed = end_time - start_time
            print(f"One loop takes: {elapsed:.3f} seconds!")

        self.cap.release()
        cv.destroyAllWindows()

    def quit_button(self):
        self.quit_when_true = True

    def update_recognized_users(self, face_name, proba):
        # Check if the face_name is already in the recognized_placed_users list
        for i, user in enumerate(self.recognized_placed_users):
            if user[0] == face_name:
                # If the new proba is higher than the previous one, update the proba value
                if proba > user[1]:
                    self.recognized_placed_users[i][1] = proba
                    self.update_scrollableframe()
                return

        # If the face_name is not already in the recognized_placed_users list, add it with its proba value
        self.recognized_placed_users.append([face_name, proba])
        self.update_scrollableframe()

    def update_scrollableframe(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        for user in self.recognized_placed_users:
            name = ctk.CTkButton(self.scrollable_frame, state = "disabled", text = user[0][2:-2] + " " + str(int(user[1]*100)), hover = False,fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"))
            name.pack(side="top",padx=5, pady=10)
class CTkProgressBox(ctk.CTkToplevel):
    
    def __init__(self,
                 master: any = None,
                 width: int = 450,
                 height: int = 250,
                 title: str = "CTkProgressBox",
                 message: str = "This is a CTkProgressBox!",
                 option_1: str = "OK",
                 border_width: int = 1,
                 border_color: str = "default",
                 button_color: str = "default",
                 bg_color: str = "default",
                 fg_color: str = "default",
                 text_color: str = "default",
                 title_color: str = "default",
                 button_text_color: str = "default",
                 button_width: int = None,
                 button_height: int = None,
                 cancel_button_color: str = "#c42b1c",
                 button_hover_color: str = "default",
                 icon: str = "info",
                 icon_size: tuple = None,
                 corner_radius: int = 15,
                 font: tuple = None,
                 header: bool = False,
                 topmost: bool = True,
                 start_recognition: bool = False,
                 close_on_finish: bool = False,
                 fade_in_duration: int = 0):
        
        super().__init__()

        self.master_window = master
        self.width = 250 if width<250 else width
        self.height = 188 if height<150 else  height
        self.start_recognition = start_recognition
        if self.master_window is None:
            self.spawn_x = int((self.winfo_screenwidth()-self.width)/2)
            self.spawn_y = int((self.winfo_screenheight()-self.height)/2)
        else:
            self.spawn_x = int(self.master_window.winfo_width() * .5 + self.master_window.winfo_x() - .5 * self.width + 7)
            self.spawn_y = int(self.master_window.winfo_height() * .5 + self.master_window.winfo_y() - .5 * self.height + 20)
            
        self.after(10)
        self.geometry(f"{self.width}x{self.height}+{self.spawn_x}+{self.spawn_y}")
        self.title(title)
        self.resizable(width=False, height=False)
        self.fade = fade_in_duration
        
        if self.fade:
            self.fade = 20 if self.fade<20 else self.fade
            self.attributes("-alpha", 0)
            
        if not header:
            self.overrideredirect(1)
    
    
        if topmost:
            self.attributes("-topmost", True)
    
        if sys.platform.startswith("win"):
            self.transparent_color = self._apply_appearance_mode(self._fg_color)
            self.attributes("-transparentcolor", self.transparent_color)
        elif sys.platform.startswith("darwin"):
            self.transparent_color = 'systemTransparent'
            self.attributes("-transparent", True)
        else:
            self.transparent_color = '#000001'
            corner_radius = 0

        self.lift()
        self.config(background=self.transparent_color)
        self.protocol("WM_DELETE_WINDOW", self.button_event)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)    
        self.x = self.winfo_x()
        self.y = self.winfo_y()
        self._title = title
        self.message = message
        self.font = font
        self.round_corners = corner_radius if corner_radius<=30 else 30
        self.button_width = button_width if button_width else self.width/4
        self.button_height = button_height if button_height else 28
        if self.fade: self.attributes("-alpha", 0)
        
        #if self.button_height>self.height/4: self.button_height = self.height/4 -20
        #self.dot_color = cancel_button_color
        self.border_width = border_width if border_width<6 else 5
    
        if bg_color=="default":
            self.bg_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["fg_color"])
        else:
            self.bg_color = bg_color

        if fg_color=="default":
            self.fg_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["top_fg_color"])
        else:
            self.fg_color = fg_color

        default_button_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkButton"]["fg_color"])
        
        if button_color=="default":
            self.button_color = (default_button_color, default_button_color, default_button_color)
        else:
            if type(button_color) is tuple:
                if len(button_color)==2:                
                    self.button_color = (button_color[0], button_color[1], default_button_color)
                elif len(button_color)==1:
                    self.button_color = (button_color[0], default_button_color, default_button_color)
                else:
                    self.button_color = button_color
            else:
                self.button_color = (button_color, button_color, button_color)

        if text_color=="default":
            self.text_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkLabel"]["text_color"])
        else:
            self.text_color = text_color

        if title_color=="default":
            self.title_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkLabel"]["text_color"])
        else:
            self.title_color = title_color
            
        if button_text_color=="default":
            self.bt_text_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkButton"]["text_color"])
        else:
            self.bt_text_color = button_text_color

        if button_hover_color=="default":
            self.bt_hv_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkButton"]["hover_color"])
        else:
            self.bt_hv_color = button_hover_color
            
        if border_color=="default":
            self.border_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["border_color"])
        else:
            self.border_color = border_color
            
        if icon_size:
            self.size_height = icon_size[1] if icon_size[1]<=self.height-100 else self.height-100
            self.size = (icon_size[0], self.size_height)
        else:
            self.size = (self.height/4, self.height/4)
        
        if icon in ["check", "cancel", "info", "question", "warning"]:
            self.icon = ctk.CTkImage(Image.open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Data', icon+'.png')),size=self.size)
        else:
            self.icon = ctk.CTkImage(Image.open(icon), size=self.size) if icon else None

        self.frame_top = ctk.CTkFrame(self, corner_radius=self.round_corners, width=self.width, border_width=self.border_width, bg_color=self.transparent_color, fg_color=self.bg_color, border_color=self.border_color)
        self.frame_top.grid(sticky="nswe")


        self.frame_top.grid_columnconfigure((0,1,2,3), weight=1)
        self.frame_top.grid_rowconfigure((0,1,2,3), weight=1)
            
        self.frame_top.bind("<B1-Motion>", self.move_window)
        self.frame_top.bind("<ButtonPress-1>", self.oldxyset)
        

        self.title_label = ctk.CTkLabel(self.frame_top, width=1, text=self._title, text_color=self.title_color, font=self.font)
        self.title_label.grid(row=0, column=0, columnspan=4, sticky="nw", padx=(15,30), pady=5)
        self.title_label.bind("<B1-Motion>", self.move_window)
        self.title_label.bind("<ButtonPress-1>", self.oldxyset)
        
        self.info = ctk.CTkButton(self.frame_top,  width=0, height=(self.height/2)-20, corner_radius=0, text=self.message, font=self.font,
                                            fg_color=self.fg_color, hover=False, text_color=self.text_color, image=self.icon) #, text_color=self.text_color
        self.info._text_label.configure(wraplength=self.width/2, justify="left")
        self.info.grid(row=1, column=0, columnspan=4, sticky="nwes", padx=self.border_width, pady=(0,0))
        
        if self.info._text_label.winfo_reqheight()>self.height/2:
            height_offset = int((self.info._text_label.winfo_reqheight())-(self.height/2) + self.height)
            self.geometry(f"{self.width}x{height_offset}")


        self.progress_label = ctk.CTkButton(master = self.frame_top, text = "training..", corner_radius=0, font=self.font,fg_color=self.fg_color, hover = False, text_color=self.text_color)
        self.progress_label.grid(row=2, column=0, columnspan=4, sticky="nwes", padx=self.border_width, pady=(0,0))


        self.progressbar = ctk.CTkProgressBar(master = self.frame_top,mode='determinate', height = 10)
        self.progressbar.grid(row=3,column = 0, columnspan = 4, sticky = "news", padx=(4,4))
        self.progressbar.set(0)
        
        


        self.option_text_1 = option_1
        self.button_1 = ctk.CTkButton(self.frame_top, text=self.option_text_1, fg_color=self.button_color[0],width=self.button_width, font=self.font, text_color=self.bt_text_color,hover_color=self.bt_hv_color, height=self.button_height,command=lambda: self.button_event(self.option_text_1))
        
        self.button_1.grid(row=4, column=3, sticky="news", padx=(0,10), pady=10)

        if header:
            self.title_label.grid_forget()
            self.button_close.grid_forget()
            self.frame_top.configure(corner_radius=0)

        if self.winfo_exists():
            self.grab_set()
            
        if self.fade:
            self.fade_in()
        
    def fade_in(self):
        for i in range(0,110,10):
            if not self.winfo_exists():
                break
            self.attributes("-alpha", i/100)
            self.update()
            time.sleep(1/self.fade)
            
    def fade_out(self):
        for i in range(100,0,-10):
            if not self.winfo_exists():
                break
            self.attributes("-alpha", i/100)
            self.update()
            time.sleep(1/self.fade)

    def get(self):
        if self.winfo_exists():
            self.master.wait_window(self)
        return self.event
        
    def oldxyset(self, event):
        self.oldx = event.x
        self.oldy = event.y
    
    def move_window(self, event):
        self.y = event.y_root - self.oldy
        self.x = event.x_root - self.oldx
        self.geometry(f'+{self.x}+{self.y}')
        
    def button_event(self, event=None):
        if self.start_recognition:
            self.recognition_F_A_C_E = CTkFaceRecognizer(master=self)
            self.recognition_F_A_C_E.grab_set()
        try:
            self.button_1.configure(state="disabled")
        except AttributeError:
            pass

        if self.fade:
            self.fade_out()
        self.grab_release()
        self.destroy()
        self.event = event
class NewUserPopup(ctk.CTkToplevel):

    def __init__(self,
                 master: any = None):
        super().__init__()
        ############################################ Code for center of window############################################

        self.width = 250
        self.height = 350
        self.master_window = master
        if self.master_window is None:
            self.spawn_x = int((self.winfo_screenwidth()-self.width)/2)
            self.spawn_y = int((self.winfo_screenheight()-self.height)/2)
        else:
            self.spawn_x = int(self.master_window.winfo_width() * .5 + self.master_window.winfo_x() - .5 * self.width + 7)
            self.spawn_y = int(self.master_window.winfo_height() * .5 + self.master_window.winfo_y() - .5 * self.height + 20)
        self.geometry(f"{self.width}x{self.height}+{self.spawn_x}+{self.spawn_y}")

        ############################################ Code for center of window############################################

        # set window title
        self.title("New User")

        # create label
        label = ctk.CTkLabel(self, text="Fill in your user details below")
        label.pack(side="top", pady=10)

        # create firstname input field
        firstname_label = ctk.CTkLabel(self, text="Firstname:")
        firstname_label.pack(side="top", padx=10, pady=5)
        firstname_entry = ctk.CTkEntry(self)
        firstname_entry.pack(side="top", padx=10, pady=5)

        # create lastname input field
        lastname_label = ctk.CTkLabel(self, text="Lastname:")
        lastname_label.pack(side="top", padx=10, pady=5)
        lastname_entry = ctk.CTkEntry(self)
        lastname_entry.pack(side="top", padx=10, pady=5)

        # create role selection dropdown
        role_label = ctk.CTkLabel(self, text="Select a role:")
        role_label.pack(side="top", padx=10, pady=5)
        role_var = ctk.StringVar()
        role_options = ["Standard", "Auto-Open", "Janitor"]
        role_menu = ctk.CTkOptionMenu(
            self, variable=role_var, values=role_options)
        role_menu.pack(side="top", padx=10, pady=5)

        # create save button
        save_button = ctk.CTkButton(self, text="Save", command=self.save_user)
        save_button.pack(side="bottom", padx=10, pady=10)

    def save_user(self):
        # get user input
        # assume the first child widget is the firstname entry
        firstname_entry = self.nametowidget(self.winfo_children()[2])
        firstname = firstname_entry.get().strip().capitalize()
        # assume the fifth child widget is the lastname entry
        lastname_entry = self.nametowidget(self.winfo_children()[4])
        lastname = lastname_entry.get().strip().capitalize()

        # check if user already exists
        user_folder_name = firstname + "_" + lastname
        user_folder_path = os.path.join(user_path, user_folder_name)
        if os.path.exists(user_folder_path):
            messagebox.showerror("Error", "This user already exists")
            return

        # create user folder
        os.makedirs(user_folder_path)
        app.users = app.get_subdirs( user_path)
        app.content = app.get_all_contents( app.users)
        app.create_user_elements( app.users, app.content)
        # close popup
        self.destroy()
        pass
class CTkClickedUser(ctk.CTkToplevel):
    
    def __init__(self,
                 master: any = None,
                 width: int = 400,
                 height: int = 200,
                 title: str = "Edit User",
                 selected_user = "Firstname Lastname",
                 border_width: int = 1,
                 border_color: str = "default",
                 button_color: str = "default",
                 bg_color: str = "default",
                 fg_color: str = "default",
                 text_color: str = "default",
                 title_color: str = "default",
                 button_text_color: str = "default",
                 button_width: int = None,
                 button_height: int = None,
                 cancel_button_color: str = "#c42b1c",
                 button_hover_color: str = "default",
                 icon: str = "info",
                 icon_size: tuple = None,
                 corner_radius: int = 15,
                 font: tuple = None,
                 header: bool = False,
                 topmost: bool = True):
        
        super().__init__()

        name = selected_user.split(" ")
        self.subdir = '_'.join(name)
        self.master_window = master
        self.width = 550
        self.height = 250
        
        
        if self.master_window is None:
            self.spawn_x = int((self.winfo_screenwidth()-self.width)/2)
            self.spawn_y = int((self.winfo_screenheight()-self.height)/2)
        else:
            self.spawn_x = int(self.master_window.winfo_width() * .5 + self.master_window.winfo_x() - .5 * self.width + 7)
            self.spawn_y = int(self.master_window.winfo_height() * .5 + self.master_window.winfo_y() - .5 * self.height + 20)
            
        self.after(10)
        self.geometry(f"{self.width}x{self.height}+{self.spawn_x}+{self.spawn_y}")
        self.title(title)
        self.resizable(width=False, height=False)
        
            
        if not header:
            self.overrideredirect(1)
    
    
        if topmost:
            self.attributes("-topmost", True)

        if sys.platform.startswith("win"):
            self.transparent_color = self._apply_appearance_mode(self._fg_color)
            self.attributes("-transparentcolor", self.transparent_color)
        elif sys.platform.startswith("darwin"):
            self.transparent_color = 'systemTransparent'
            self.attributes("-transparent", True)
        else:
            self.transparent_color = '#000001'
            corner_radius = 0

        self.lift()
        self.config(background=self.transparent_color)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)    
        self.x = self.winfo_x()
        self.y = self.winfo_y()
        self._title = title
        self.font = font
        self.round_corners = corner_radius if corner_radius<=30 else 30
        self.button_width = button_width if button_width else self.width/4
        self.button_height = button_height if button_height else 28

        
        if self.button_height>self.height/4: self.button_height = self.height/4 -20
        self.dot_color = cancel_button_color
        self.border_width = border_width if border_width<6 else 5
    
        if bg_color=="default":
            self.bg_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["fg_color"])
        else:
            self.bg_color = bg_color

        if fg_color=="default":
            self.fg_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["top_fg_color"])
        else:
            self.fg_color = fg_color

        default_button_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkButton"]["fg_color"])
        
        if button_color=="default":
            self.button_color = (default_button_color, default_button_color, default_button_color)
        else:
            if type(button_color) is tuple:
                if len(button_color)==2:                
                    self.button_color = (button_color[0], button_color[1], default_button_color)
                elif len(button_color)==1:
                    self.button_color = (button_color[0], default_button_color, default_button_color)
                else:
                    self.button_color = button_color
            else:
                self.button_color = (button_color, button_color, button_color)

        if text_color=="default":
            self.text_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkLabel"]["text_color"])
        else:
            self.text_color = text_color

        if title_color=="default":
            self.title_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkLabel"]["text_color"])
        else:
            self.title_color = title_color
            
        if button_text_color=="default":
            self.bt_text_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkButton"]["text_color"])
        else:
            self.bt_text_color = button_text_color

        if button_hover_color=="default":
            self.bt_hv_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkButton"]["hover_color"])
        else:
            self.bt_hv_color = button_hover_color
            
        if border_color=="default":
            self.border_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["border_color"])
        else:
            self.border_color = border_color
            
        if icon_size:
            self.size_height = icon_size[1] if icon_size[1]<=self.height-100 else self.height-100
            self.size = (icon_size[0], self.size_height)
        else:
            self.size = (self.height/4, self.height/4)
        
        if icon in ["check", "cancel", "info", "question", "warning"]:
            self.icon = ctk.CTkImage(Image.open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Data', icon+'.png')), size=self.size)
        else:
            self.icon = ctk.CTkImage(Image.open(icon), size=self.size) if icon else None

        self.frame_top = ctk.CTkFrame(self, corner_radius=self.round_corners, width=self.width, border_width=self.border_width, bg_color=self.transparent_color, fg_color=self.bg_color, border_color=self.border_color)
        self.frame_top.grid(sticky="nswe")

        if button_width:
            self.frame_top.grid_columnconfigure(0, weight=1)
        else:
            self.frame_top.grid_columnconfigure((1,2,3), weight=1)

        if button_height:
            self.frame_top.grid_rowconfigure((0,1,3), weight=1)
        else:
            self.frame_top.grid_rowconfigure((0,1,2), weight=1)

        temp_path  = os.path.join(user_path, self.subdir)
        temp_path = os.path.join(temp_path, "ms_graph_api_token.json")

        self.frame_top.bind("<B1-Motion>", self.move_window)
        self.frame_top.bind("<ButtonPress-1>", self.oldxyset)
        self.button_close = ctk.CTkButton(self.frame_top, corner_radius=10, width=40, height=20,  text="Close", fg_color=self.dot_color, command=self.button_event)
        self.button_close.configure(cursor="arrow")        
        self.button_close.grid(row=0, column=3, sticky="ne", padx=10, pady=10)


        self.title_label = ctk.CTkLabel(self.frame_top, width=1, text=self._title, text_color=self.title_color, font=self.font)
        self.title_label.grid(row=0, column=0, columnspan=4, sticky="nw", padx=(15,30), pady=5)
        self.title_label.bind("<B1-Motion>", self.move_window)
        self.title_label.bind("<ButtonPress-1>", self.oldxyset)
        
        self.info = ctk.CTkButton(self.frame_top,  width=1, height=self.height/2, corner_radius=0, text=selected_user, font=self.font, fg_color=self.fg_color, hover=False, text_color=self.text_color, image=self.icon)
        self.info._text_label.configure(wraplength=self.width/2, justify="left")
        self.info.grid(row=1, column=0, columnspan=4, sticky="nwes", padx=self.border_width)
        
        if self.info._text_label.winfo_reqheight()>self.height/2:
            height_offset = int((self.info._text_label.winfo_reqheight())-(self.height/2) + self.height)
            self.geometry(f"{self.width}x{height_offset}")
            
        self.option_text_1 = "Add Picture"
        self.button_1 = ctk.CTkButton(self.frame_top, text=self.option_text_1, fg_color=self.button_color[0], width=self.button_width, font=self.font, text_color=self.bt_text_color, height=self.button_height, command=self.add_pictures)
        self.button_1.grid(row=2, column=0, sticky="news", padx=(0,10), pady=10)

        if os.path.exists(temp_path):
            self.option_text_2 = "Replace Calendar"
        else:
            self.option_text_2 = "Add Calendar"    
         
        self.button_2 = ctk.CTkButton(self.frame_top, text=self.option_text_2, fg_color=self.button_color[1], width=self.button_width, font=self.font, text_color=self.bt_text_color, height=self.button_height, command=lambda: self.add_calendar(self.subdir))
        self.button_2.grid(row=2, column=1, sticky="news", padx=10, pady=10)


        
        self.option_text_3 = "Delete Calendar"
        self.button_3 = ctk.CTkButton(self.frame_top, text=self.option_text_3, width=self.button_width, font=self.font, text_color=self.bt_text_color, height=self.button_height, command=lambda: self.delete_calendar(self.subdir))
        self.button_3.grid(row=2, column=2, sticky="news", padx=(10,0), pady=10)



        if not os.path.exists(temp_path):
            self.button_3.configure(state="disabled", fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"))

        self.option_text_4 = "Delete " + self.subdir.split("_")[0]
        self.button_4 = ctk.CTkButton(self.frame_top, text=self.option_text_4, fg_color=self.button_color[2], width=self.button_width, font=self.font, text_color=self.bt_text_color, height=self.button_height, command=lambda: self.delete_user(self.subdir))
        self.button_4.grid(row=2, column=3, sticky="news", padx=(10,0), pady=10)
            

    def get(self):
        if self.winfo_exists():
            self.master.wait_window(self)
        return self.event
        
    def oldxyset(self, event):
        self.oldx = event.x
        self.oldy = event.y
    
    def move_window(self, event):
        self.y = event.y_root - self.oldy
        self.x = event.x_root - self.oldx
        self.geometry(f'+{self.x}+{self.y}')

    def add_pictures(self):
        #self.selected_user = self.user_var.get()
        detection = FaceDetection(self, directory=self.subdir)
        detection.grab_set()
        detection.focus_set()

    def delete_user(self,selected_subdir):
        
        folder_path = os.path.join(user_path,selected_subdir)

        split_name = selected_subdir.split("_")
        converted_name = ' '.join(split_name)
        msg = CTkMessagebox(master=self, title="Delete?", message=("Do you want to delete " + converted_name), icon="question", option_1="No", option_2="Yes")

        if msg.get() == "Yes":
            shutil.rmtree(folder_path)
            couch.delete_user(selected_subdir)
        
        app.users = app.get_subdirs(user_path)
        app.content = app.get_all_contents(app.users)
        app.create_user_elements(app.users, app.content)
        app.radio_button_event()

    def delete_calendar(self, path):
        folder_path = os.path.join(user_path,path)
        split_name = path.split("_")
        converted_name = ' '.join(split_name)
        msg = CTkMessagebox(master=self, title="Delete?", message=("Do you want to delete " + converted_name + "'s calendar"), icon="question", option_1="No", option_2="Yes")

        if msg.get() == "Yes":
            calendar_file = os.path.join(folder_path, 'ms_graph_api_token.json')
            print(calendar_file)
            os.remove(calendar_file)
            couch.delete_document(path)
        
        app.users = app.get_subdirs(user_path)
        app.content = app.get_all_contents(app.users)
        app.create_user_elements(app.users, app.content)
        app.radio_button_event()

    def add_calendar(self, path):
        #app_id = '3cff8cea-21c6-4547-8620-7a809a0615f1'    # Max old
        app_id = 'c4614fe4-b441-48bc-af4e-211f00c1b27e'     #Jesper new
        scopes = ['Calendars.Read']

        user = path
        filename = 'ms_graph_api_token.json'
        filepath = os.path.join(user, filename)
        filepath = os.path.join(user_path, filepath)

        access_token_cache = msal.SerializableTokenCache()

        if os.path.exists(filepath):
            access_token_cache.deserialize(open(filepath, "r").read())
            token_detail = json.load(open(filepath,))
            token_detail_key = list(token_detail['AccessToken'].keys())[0]
            token_expiration = datetime.fromtimestamp(
                int(token_detail['AccessToken'][token_detail_key]['expires_on']))
            if datetime.now() > token_expiration:
                os.remove(filepath)
                access_token_cache = msal.SerializableTokenCache()

        # assign a SerializableTokenCache object to the client instance
        client = msal.PublicClientApplication(client_id=app_id, token_cache=access_token_cache)

        accounts = client.get_accounts()
        if accounts:
            # load the session
            token_response = client.acquire_token_silent(scopes, accounts[0])
        else:
            # authenticate your account as usual
            flow = client.initiate_device_flow(scopes=scopes)
            if 'user_code' in flow:
                self.code = flow['user_code']
                print('user_code: ' + flow['user_code'])
                msg = CTkMessagebox(master=self, title="User token", message="User token = " + flow['user_code'] + " do you want to copy?", icon="question", option_1="No", option_2="Yes", topmost=False)
                if msg.get() == "No":
                    msg.destroy()
                else:
                    msg.destroy()
                    pyperclip.copy(flow['user_code'])
                webbrowser.open('https://microsoft.com/devicelogin')
                token_response = client.acquire_token_by_device_flow(flow)
            else:
                print(
                    f"Device flow authentication failed: {flow.get('error')}")
                token_response = None

        if token_response:
            with open(filepath, 'w') as _f:
                _f.write(access_token_cache.serialize())
        app.content = app.get_all_contents(app.users)
        app.create_user_elements(app.users, app.content)
        app.radio_button_event()

    def button_event(self, event=None):
        self.grab_release()
        self.destroy()
        self.event = event
class UserButton(ctk.CTkButton):
    def __init__(self, master=None, subdir=None, mainwindow = None, **kwargs):
        super().__init__(master, **kwargs)
        self.subdir = subdir
        self.mainwindow = mainwindow
        self.bind('<Button-1>', self.button_clicked)
    
    def button_clicked(self, event):
        configure_user = CTkClickedUser(selected_user = self.subdir, master=self.mainwindow) #master = self.mainwindow,
        configure_user.grab_set()
        configure_user.focus_set()        

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        couch.download_everything()

        self.users = []
        self.users = self.get_subdirs(user_path)
        self.content = []
        self.content = self.get_all_contents(self.users)

        # configure window
        self.title("Face recognition")
        self.geometry(f"{700}x{480}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=5, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Face recognition", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))


        self.sidebar_button_new_user = ctk.CTkButton(self.sidebar_frame, text="New User", command=self.sidebar_button_new_user)
        self.sidebar_button_new_user.grid(row=3, column=0, padx=20, pady=10)

        self.sidebar_button_train_recognition = ctk.CTkButton(self.sidebar_frame, text="Upload", command=self.sidebar_button_train_recognition)
        self.sidebar_button_train_recognition.grid(row=1, column=0, padx=20, pady=10)

        self.sidebar_button_face_recognition = ctk.CTkButton(self.sidebar_frame, text="Test Recognition", command=self.sidebar_button_face_recognition)
        self.sidebar_button_face_recognition.grid(row=2, column=0, padx=20, pady=10)

        self.sidebar_button_quit = ctk.CTkButton(self.sidebar_frame, text="Quit", command=self.sidebar_button_quit_confirmation)
        self.sidebar_button_quit.grid(row=10, column=0, padx=20, pady=10)

        # create userframe
        users = self.get_subdirs(user_path)
        self.user_frame = ctk.CTkScrollableFrame(self, label_text="User", height=500, corner_radius=0)
        self.user_frame.grid(row=0, column=1, rowspan=2, padx=(10, 0), pady=(10, 0), sticky="nsew",)

        self.user_frame.grid_rowconfigure((0, 5, 2), weight=1)
        self.user_frame.grid_columnconfigure((2, 3), weight=0)

        self.user_frame_interaction = ctk.CTkFrame(self, height=50, corner_radius=00)
        self.user_frame_interaction.grid(row=4, column=1, padx=(10, 0), pady=(0, 0), sticky="nsew")

        self.user_var = tkinter.IntVar(value=0)
        self.scrollable_frame_users = []

        self.content = self.get_all_contents(users)
        self.create_user_elements(users, self.content)


    def create_user_elements(self, users, content):
        for widget in self.user_frame.winfo_children():
            widget.destroy()

        for i in range(len(users)):
            name = users[i].split("_")
            converted_name = ' '.join(name)
            custom_button = UserButton(master=self.user_frame, text=converted_name, subdir=converted_name, mainwindow=self)
            custom_button.grid(row=i, column=0, padx=(10, 0), pady=(10, 0), sticky="w")

            image_count = str(sum([1 for file in content[i] if file.lower().endswith((".jpg", ".png", ".jpeg"))])) + " images"
            ctkButton = ctk.CTkLabel(self.user_frame, text=image_count)
            ctkButton.grid(row=i, column=1, padx=(20, 20), pady=(10, 0), sticky="e")

            checkbox = ctk.CTkCheckBox(master=self.user_frame, state="disabled", text="Calendar")
            checkbox.grid(row=i, column=2, padx=(0, 10), pady=(10, 0), sticky="e")
            if "ms_graph_api_token.json" in content[i]:
                checkbox.select()

    def radio_button_event(self):
        next_user = self.user_var.get()

        # clear previous checkboxes
        for checkbox in self.scrollable_frame_switches:
            checkbox.destroy()
        self.scrollable_frame_switches = []

        # create new checkboxes
        for i in range(len(self.content[next_user])):
            switch = ctk.CTkCheckBox(master=self.scrollable_frame, text=str(self.content[next_user][i]))
            switch.grid(row=i, column=0, padx=10, pady=(10, 10), sticky="w")
            self.scrollable_frame_switches.append(switch)
        self.scrollable_frame.update()

    def get_subdirs(self, path):
        subdirs = []
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                subdirs.append(item)
        return subdirs

    def get_contents(self, path):
        contents = os.listdir(path)
        return [item for item in contents if item != '.DS_Store']

    def get_all_contents(self, users):
        list = []
        for i in range(len(users)):
            list.append(self.get_contents(os.path.join(user_path, users[i])))
        return list

    def sidebar_button_quit_confirmation(self):
        msg = CTkMessagebox(master=self, title="Exit?", message="Do you want to close the program?", icon="question", option_1="No", option_2="Yes")
        if msg.get() == "Yes":
            self.destroy()

    def sidebar_button_new_user(self):
        popup = NewUserPopup(master=self)
        popup.grab_set()

    def sidebar_button_train_recognition(self):
        self.number_of_images_progressbar = self.count_images()
        self.progress_after_1_image = 0.90/self.number_of_images_progressbar
        self.progress_during_training = 0

        self.train_progressbar = CTkProgressBox(master = self, option_1 = "Exit", title = "Training face recognition", message = "Training neural network..")
        # Start the training function in a separate thread
        training_thread = threading.Thread(target=self.train_F_A_C_E_Model)
        training_thread.start()

        # Continue running the GUI loop
        #self.master.mainloop()

    def sidebar_button_face_recognition(self):
        msg = CTkMessagebox(master=self, title="Train?", message="Do you want to train neural network before testing?", option_1="No", option_2="Yes", option_3="Exit")
        answer = msg.get()
        if answer == "Yes":
            self.number_of_images_progressbar = self.count_images()
            self.progress_after_1_image = 0.90/self.number_of_images_progressbar
            self.progress_during_training = 0

            self.train_progressbar = CTkProgressBox(master = self, option_1 = "Exit", title = "Training face recognition", message = "Training neural network..", start_recognition=True)
            # Start the training function in a separate thread
            training_thread = threading.Thread(target=self.train_F_A_C_E_Model)
            training_thread.start()

        elif answer == "No":
            self.recognition_F_A_C_E = CTkFaceRecognizer(master=self)
            self.recognition_F_A_C_E.grab_set()
        else:
            pass

    def get_embedding(self, face_img, embedder):
        face_img = face_img.astype('float32')
        face_img = np.expand_dims(face_img, axis=0)
        yhat = embedder.embeddings(face_img)
        return yhat[0]
    
    def count_images(self):
        total_count = 0
        for root, dirs, files in os.walk(user_path):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    total_count += 1
        return total_count
    
    def train_F_A_C_E_Model(self):
        start_time = time.time()
        embedder = FaceNet()
        faceloading = FACELOADING('dataset')
        
        X, Y = faceloading.load_classes()
        #print(f"loading: {(time.time()-start_time):.2f} seconds")
        encoder = LabelEncoder()
        encoder.fit(Y)
        #print(f"fit 1: {(time.time()-start_time):.2f} seconds")
        print(encoder.classes_)
        Y_encoded = encoder.transform(Y)

        EMBEDDED_X = []
        for img in X:
            EMBEDDED_X.append(self.get_embedding(img,embedder))

        EMBEDDED_X = np.asarray(EMBEDDED_X)

        # save the embedded faces and encoded labels to a compressed numpy file
        np.savez_compressed('Data/face_embeddings_done_4classes.npz', EMBEDDED_X=EMBEDDED_X, Y=Y_encoded)

        X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y, shuffle=True, random_state=17)

        param_grid = {
            'kernel': ['linear', 'rbf'],
            'C': [0.1, 1, 10, 100],
            'gamma': [0.1, 1, 10, 100]
        }

        # instantiate the SVM model
        model = SVC(probability=True)

        # create a GridSearchCV object with the model and parameter grid
        grid_search = GridSearchCV(model, param_grid, cv=5)

        # fit the grid search object on the training data
        grid_search.fit(X_train, Y_train)

        # print the best parameters found
        print(grid_search.best_params_)

        # use the best hyperparameters to train the model on the full training set
        model = SVC(kernel=grid_search.best_params_['kernel'], C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'], probability=True)
        model.fit(X_train, Y_train)
        self.train_progressbar.progressbar.set(1)
        #print(f"fit 2: {(time.time()-start_time):.2f} seconds")
        ypread_train = model.predict(X_train)
        ypread_test = model.predict(X_test)

        with open('Data/svm_model_160x160.pkl','wb' ) as f:
            pickle.dump(model,f)

        end_time = time.time()
        self.train_progressbar.progressbar.set(0)
        self.train_progressbar.progress_label.configure(text = "Uploading data..")
        ################################################################################################
        #                   Här kan du ladda upp data på databasen

        couch.upload_everything()

        ################################################################################################
        self.train_progressbar.progressbar.set(1)
        self.train_progressbar.progress_label.configure(text = "Done!")
        self.train_progressbar.button_1.configure(state = "normal", fg_color = button_standard_color, border_width = 0)
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        
if __name__ == "__main__":
    app = App()
    app.mainloop()

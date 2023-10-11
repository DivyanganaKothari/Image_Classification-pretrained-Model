import tkinter as tk
from tkinter import simpledialog,messagebox
import cv2 as cv
import os
import model
import camera
import PIL

from PIL import Image,ImageTk

LOGO_PATH= "logo.jpg"


class App:

    def __init__(self, window=tk.Tk(), window_title="Camera Classifier"):

        self.window = window
        self.window_title = window_title
        self.window.title("Fraunhofer")

        self.window.configure(bg="light gray")  # Change the background color
        self.num_classes = None
        self.is_model_trained = False
        self.counters = []
        self.class_name_labels = []

        # Create a frame for the logo label
        logo_frame = tk.Frame(self.window)
        logo_frame.pack(side=tk.LEFT, padx=10, pady=10, anchor=tk.NW)
        # Load and display the logo image
        logo_image = Image.open(LOGO_PATH)
        logo_image = ImageTk.PhotoImage(logo_image)
        self.logo_label = tk.Label(logo_frame, image=logo_image)
        self.logo_label.image = logo_image  # Keep a reference to prevent image from being garbage collected
        self.logo_label.pack()
        #self.model_instance = model.Model(num_classes=self.num_classes)
        self.auto_predict = False

        self.camera = camera.Camera()

        self.delay = 15
        self.init_gui()
        self.window.attributes("-topmost", True)
        self.update()
        self.current_row = 1

        self.window.mainloop()

    def class_name_frame(self):
        # Create a frame to hold the class names if it doesn't already exist
        if not hasattr(self, 'class_names_frame'):
            self.class_names_frame = tk.LabelFrame(self.window, text="Class Names", font=("Arial", 12))
            self.class_names_frame.pack(side=tk.RIGHT, padx=10, pady=10, anchor=tk.NE, fill=tk.Y)

        self.class_names = []
        self.class_labels = []
        self.class_buttons = []



    def init_gui(self):

        # a frame for the camera feed
        camera_frame = tk.Frame(self.window)
        camera_frame.pack(side=tk.LEFT, padx=10, pady=10)


        # a label for the camera feed section
        camera_label = tk.Label(camera_frame, text="Camera Feed", font=("Arial", 14))
        camera_label.pack()

        self.camera = camera.Camera()

        self.canvas = tk.Canvas(camera_frame, width=self.camera.width, height=self.camera.height)
        self.canvas.pack()

        self.class_label = tk.Label(camera_frame, text="CLASS",bg="#D0D0E9")
        self.class_label.config(font=("Arial", 25))
        self.class_label.pack(anchor=tk.CENTER, expand=True)

        self.class_name_frame()

        # Create a button to add classes dynamically
        self.add_class_button = tk.Button(self.window, text="Add Object", bg="#D0D0E9", width=50,
                                          command=self.add_object)
        self.add_class_button.pack(anchor=tk.CENTER, expand=True)

        self.btn_train = tk.Button(self.window, text="Train Model", bg="#D0D0E9", width=50, command=self.train_model)
        self.btn_train.pack(anchor=tk.CENTER, expand=True)

        self.btn_predict = tk.Button(self.window, text="Predict", bg="#D0D0E9", width=50, command=self.predict)
        self.btn_predict.pack(anchor=tk.CENTER, expand=True)

        self.btn_auto_predict = tk.Button(self.window, text="Auto Predict", bg="#D0D0E9", width=50,
                                          command=self.auto_predict_toggle)
        self.btn_auto_predict.pack(anchor=tk.CENTER, expand=True)

        self.btn_reset = tk.Button(self.window, text="Reset",bg="#D0D0E9", width=50, command=self.reset)
        self.btn_reset.pack(anchor=tk.CENTER, expand=True)


    def add_object(self):
            class_name = simpledialog.askstring("Add Object", "Enter the name of the object:", parent=self.window)

            if class_name:
                self.class_names.append(class_name)
                # Create a label for the new class
                class_label = tk.Label(self.class_names_frame, text=class_name)
                class_label.config(font=("Arial", 12))
                class_label.pack(anchor=tk.NW, padx=10, pady=5)
                self.class_labels.append(class_label)

                # Create a button for the new class
                class_folder = len(self.class_names) - 1  # Assign each object to an integer folder

                class_button = tk.Button(self.window, text=f"Add images for {class_name}", bg="#87CEEB", width=50,
                                         command=lambda folder=class_folder: self.save_for_class(folder))

                class_button.pack(anchor=tk.CENTER, expand=True)

                self.class_buttons.append(class_button)

                # Initialize counters for the new class
                self.counters.append(1)

                self.current_row += 1

                # Update num_classes to reflect the current number of classes
                self.num_classes = len(self.class_names)

                # Create a new Model instance with the updated num_classes
                self.model = model.Model(num_classes=self.num_classes)



    def auto_predict_toggle(self):
        if not self.is_model_trained:
            if not self.btn_auto_predict:
                return  # Don't show the error message if the predict button is disabled
            messagebox.showerror("Error", "Model is not trained yet.")
            return
        else:
            self.auto_predict = not self.auto_predict

        if self.auto_predict:
            self.btn_auto_predict.config(bg="green")
        else:
            self.btn_auto_predict.config(bg="#D0D0E9")

    def update(self):
       if self.auto_predict:
            print(self.predict())

       ret, frame = self.camera.get_frame()

       if ret:
           self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
           self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

       self.window.after(self.delay, self.update)

    def train_model(self):

        if not self.class_names:
            messagebox.showerror("Error", "No objects have been added yet.")
            return

            # Assuming you have counters for each class
        class_counters = [1] * len(self.class_names)

        self.model.train(class_counters, num_epochs=10)
        self.is_model_trained = True

    def predict(self):
        if not self.is_model_trained:
            if not self.btn_predict:
                return  # Don't show the error message if the predict button is disabled
            messagebox.showerror("Error", "Model is not trained yet.")
            return

        ret, frame = self.camera.get_frame()

        if not ret:
            print("Failed to capture frame.")
            return

        prediction = self.model.predict(frame)

        if prediction >= 0 and prediction <= self.num_classes - 1:
            predicted_class = self.class_names[prediction]
            self.class_label.config(text=predicted_class)
            return predicted_class
        else:
            # Clear the class label if no valid prediction is made
            self.class_label.config(text="CLASS")

    def reset(self):
        if not self.is_model_trained:
            if not self.btn_reset:
                return
            messagebox.showerror("Error", "Model is not trained yet.")
            return
        result = messagebox.askyesno("Confirmation", "Reset will delete all saved images. Continue?")
        if result:
            for folder in os.listdir():
                if folder.isdigit():  # Check if the folder name is an integer
                    folder_path = os.path.join('', folder)
                    for file in os.listdir(folder_path):
                        file_path = os.path.join(folder_path, file)
                        if os.path.isfile(file_path):
                            with PIL.Image.open(file_path) as img:
                                img.close()
                            os.unlink(file_path)


            # Clear the model's knowledge of objects
            self.model.reset()

            self.counters = [1] * self.num_classes
            self.model = model.Model(num_classes=self.num_classes)

            self.is_model_trained = False
            self.class_label.config(text="CLASS")

            # Clear the class names as well
            self.class_names = []

            # Clear the existing class labels and buttons
            for class_label in self.class_labels:
                class_label.destroy()
            for class_button in self.class_buttons:
                class_button.destroy()

            self.class_name_frame()
            self.btn_train.config(state=tk.NORMAL)
            self.btn_predict.config(state=tk.NORMAL)

    def save_for_class(self, class_num):
        ret, frame = self.camera.get_frame()
        if not os.path.exists(str(class_num)):
            os.mkdir(str(class_num))

        file_path = f'{class_num}/frame{self.counters[class_num - 1]}.jpg'
        print("Saving image to:", file_path)

        cv.imwrite(file_path, frame)
        img = Image.open(file_path)
        img.thumbnail((150, 150), Image.LANCZOS)
        img.save(file_path)

        self.counters[class_num - 1] += 1

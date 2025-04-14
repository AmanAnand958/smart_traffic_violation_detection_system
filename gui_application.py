import customtkinter as ctk
from tkinter import messagebox
from PIL import Image, ImageTk
from AppKit import NSOpenPanel, NSOKButton
from tkinterdnd2 import TkinterDnD, DND_FILES

# --- Custom Detection Imports (placeholders for your actual logic) ---
from helmet_violation import detect_helmet_violation
from signal_violation import detect_signal_violation
from overspeeding import detect_speed_violation
from illegal_parking import detect_illegal_parking
from wrong_side_violation import wrong_side_driving
from overloading import detect_overloading
from phone_violation import detect_mobile_phone_usage

# Set appearance mode and theme
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

# ------------------ FUNCTIONS ------------------

def select_video():
    panel = NSOpenPanel.openPanel()
    panel.setAllowedFileTypes_(["mp4", "avi", "mov"])
    if panel.runModal() == NSOKButton:
        file_path = panel.URL().path()
        if file_path:
            video_path.set(file_path)

def on_drop(event):
    file_path = event.data.strip('{}')
    if file_path.endswith(('.mp4', '.avi', '.mov')):
        video_path.set(file_path)
    else:
        messagebox.showwarning("Invalid File", "Please drop a video file (mp4, avi, mov).")

def run_detection(detection_type):
    file = video_path.get()
    if not file or file == "Drag and Drop Video File Here":
        messagebox.showerror("Error", "Please select or drag-and-drop a video file.")
        return

    detection_functions = {
        "Signal Breach": detect_signal_violation,
        "Helmet Check": detect_helmet_violation,
        "Speed Limit": detect_speed_violation,
        "Illegal Parking": detect_illegal_parking,
        "Wrong Way Driving": wrong_side_driving,
        "Over Loading": detect_overloading,
        "Phone Use": detect_mobile_phone_usage
    }

    detection_functions[detection_type](file)
    messagebox.showinfo("Completed", f"{detection_type} detection completed!")

# ------------------ GUI SETUP ------------------

root = TkinterDnD.Tk()
root.title("Traffic Violation Detection")
root.geometry("1000x650")
root.config(bg="#FFFFFF")

# Background Image (optional)
try:
    bg_image = Image.open("img/bk.jpg").resize((1000, 650), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = ctk.CTkLabel(root, image=bg_photo, text="")
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
except FileNotFoundError:
    print("Background image not found. Skipping...")

# Title
title_label = ctk.CTkLabel(root, text="TRAFFIC VIOLATION DETECTION SYSTEM",
                           font=("Arial", 28, "bold"), fg_color="#0077B6", text_color="white")
title_label.pack(pady=0, fill="x")

video_path = ctk.StringVar(value="Path to video file")

# --------------- Drag and Drop Box ---------------
video_frame = ctk.CTkFrame(root, fg_color="white", border_width=2, corner_radius=10)
video_frame.pack(pady=20, padx=20, fill="x")
video_frame.columnconfigure(2, weight=1)

video_frame.drop_target_register(DND_FILES)
video_frame.dnd_bind('<<Drop>>', on_drop)

try:
    footage_img = Image.open("img/footage.png").resize((50, 50), Image.LANCZOS)
    footage_icon = ImageTk.PhotoImage(footage_img)
    icon_label = ctk.CTkLabel(video_frame, image=footage_icon, text="", fg_color="white")
except FileNotFoundError:
    icon_label = ctk.CTkLabel(video_frame, text="📹", font=("Arial", 30), fg_color="white")

icon_label.grid(row=0, column=0, padx=(15, 5), pady=15, sticky="e")
drag_label = ctk.CTkLabel(video_frame, text="Drag and Drop", font=("Helvetica", 16, "bold"),
                          text_color="#0077B6", fg_color="white")
drag_label.grid(row=0, column=1, sticky="w", pady=15)
entry = ctk.CTkEntry(video_frame, textvariable=video_path, font=("Helvetica", 14),
                     fg_color="white", text_color="grey", border_width=0, corner_radius=0)
entry.grid(row=0, column=2, sticky="ew", padx=(15, 5), pady=15)

select_button = ctk.CTkButton(video_frame, text="BROWSE", command=select_video,
                              font=("Helvetica", 12, "bold"), fg_color="#FF007F",
                              hover_color="#FF3399", text_color="white",
                              width=100, height=40, corner_radius=10)
select_button.grid(row=0, column=3, padx=(5, 15), pady=15)

# --------------- Detection Buttons ---------------
button_frame = ctk.CTkFrame(root, fg_color="white")
button_frame.pack(pady=7)

buttons = [
    ("Signal Breach", 0, 0),
    ("Helmet Check", 0, 1),
    ("Speed Limit", 0, 2),
    ("Illegal Parking", 1, 0),
    ("Wrong Way Driving", 1, 1),
    ("Over Loading", 1, 2),
    ("Phone Use", 2, 1),
]

for text, row, col in buttons:
    btn = ctk.CTkButton(button_frame, text=text, command=lambda d=text: run_detection(d),
                        font=("Helvetica", 16, "bold"), fg_color="#0077B6",
                        hover_color="#0099E6", text_color="white",
                        width=200, height=50, corner_radius=10)
    btn.grid(row=row, column=col, padx=25, pady=15)

# --------------- Larger Exit Button ---------------
exit_button = ctk.CTkButton(root, text="EXIT", command=root.quit,
                            font=("Helvetica", 14, "bold"), fg_color="#FF007F",
                            hover_color="#FF3399", text_color="white",
                            width=150, height=50, corner_radius=12)
exit_button.pack(pady=10)

root.mainloop()
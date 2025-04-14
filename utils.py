import requests
import xml.etree.ElementTree as ET
import customtkinter as ctk
from tkinter import messagebox
import re
import json
from fine_calculator import FineCalculator
from message_alerts import MessageAlerts

API_URL = "http://www.regcheck.org.uk/api/reg.asmx/CheckIndia"
USERNAME = "antoshenkoa"
DEFAULT_IMAGE_URL = "https://imgd.aeplcdn.com/664x374/n/bw/models/colors/hero-select-model-black-grey-stripe-1706531424390.png?q=80"

MOCK_VEHICLE_DATA = {
    "DL5SCY0736": {
        "owner_name": "Pratish Bansal",
        "model": "Tvs Splendor",
        "image_url": DEFAULT_IMAGE_URL
    },
    "DEFAULT": {
        "owner_name": "Mock Owner",
        "model": "Mock Model",
        "image_url": DEFAULT_IMAGE_URL
    }
}


def fetch_vehicle_details(plate):
    payload = {"RegistrationNumber": plate, "username": USERNAME}
    try:
        response = requests.post(API_URL, data=payload, headers={"Content-Type": "application/x-www-form-urlencoded"})
        if response.status_code == 200:
            root = ET.fromstring(response.text)
            vehicle_json_str = root.find(".//{http://regcheck.org.uk}vehicleJson")
            if vehicle_json_str is not None and vehicle_json_str.text:
                try:
                    vehicle_data = json.loads(vehicle_json_str.text)
                    owner = vehicle_data.get("Owner", "Unknown Owner")
                    model = vehicle_data.get("CarModel", {}).get("CurrentTextValue", "Unknown Model")
                    if owner == "":
                        return MOCK_VEHICLE_DATA.get(plate, MOCK_VEHICLE_DATA["DEFAULT"])
                    return {
                        "owner_name": owner,
                        "model": model,
                        "image_url": vehicle_data.get("ImageUrl", DEFAULT_IMAGE_URL)
                    }
                except json.JSONDecodeError:
                    return MOCK_VEHICLE_DATA.get(plate, MOCK_VEHICLE_DATA["DEFAULT"])
        return MOCK_VEHICLE_DATA.get(plate, MOCK_VEHICLE_DATA["DEFAULT"])
    except requests.RequestException:
        return MOCK_VEHICLE_DATA.get(plate, MOCK_VEHICLE_DATA["DEFAULT"])


def validate_phone(phone):
    return re.match(r"^\+\d{10,15}$", phone) is not None


def show_phone_input_gui(detected_plates, violation_data, violation_type="Overloading"):
    if not detected_plates:
        root = ctk.CTk()
        root.withdraw()
        messagebox.showinfo("Detected Number Plates", "No plates detected")
        root.quit()
        root.destroy()
        return

    # Set appearance and theme
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.geometry("900x700")

    root.resizable(True, True)

    phone_entries = {}
    fine_calculator = FineCalculator()
    message_alerts = MessageAlerts()
    status_var = ctk.StringVar(value="Ready")

    def send_all_challans():
        status_var.set("Processing...")
        results = []
        for plate, entry in phone_entries.items():
            phone = entry.get()
            if not validate_phone(phone):
                results.append(f"Invalid phone number for {plate}: {phone}")
                continue
            fine = fine_calculator.calculate_fine(plate, violation_data.get(plate, None), violation_type)
            success, msg = message_alerts.send_challan(plate, phone, violation_data.get(plate, None), fine,
                                                       violation_type)
            results.append(msg)
        messagebox.showinfo("Challan Results", "\n".join(results))
        status_var.set("Completed" if all("Failed" not in r for r in results) else "Completed with errors")
        if all("Failed" not in r for r in results):
            root.after(1000, root.destroy)

    def validate_single_plate(plate, entry):
        phone = entry.get()
        if validate_phone(phone):
            entry.configure(border_color="#22c55e")  # Green for valid
        else:
            entry.configure(border_color="#ef4444")  # Red for invalid

    def on_closing():
        if messagebox.askokcancel("Quit", "Are you sure you want to exit?"):
            root.quit()
            root.destroy()

    # Header
    header_frame = ctk.CTkFrame(root, fg_color="#f8fafc", height=80)
    header_frame.pack(fill="x", pady=0)
    header_frame.pack_propagate(False)

    ctk.CTkLabel(header_frame,
                 text=f"{violation_type}  Management",
                 font=("Segoe UI", 24, "bold"),
                 text_color="#1e3a8a").pack(pady=20)

    # Main content
    main_frame = ctk.CTkFrame(root, fg_color="transparent")
    main_frame.pack(fill="both", expand=True, padx=20, pady=10)

    scrollable_frame = ctk.CTkScrollableFrame(main_frame, fg_color="transparent")
    scrollable_frame.pack(fill="both", expand=True)

    for plate in detected_plates:
        plate_frame = ctk.CTkFrame(scrollable_frame,
                                   fg_color="#ffffff",
                                   corner_radius=15,
                                   border_width=2,
                                   border_color="#e2e8f0")
        plate_frame.pack(fill="x", pady=15, padx=10)

        vehicle_details = fetch_vehicle_details(plate)

        # Left info section
        info_frame = ctk.CTkFrame(plate_frame, fg_color="transparent")
        info_frame.pack(side="left", fill="x", expand=True, padx=15, pady=10)

        plate_label = ctk.CTkLabel(info_frame,
                                   text=f"Plate: {plate}",
                                   font=("Segoe UI", 14, "bold"),
                                   text_color="#1e3a8a",
                                   anchor="w")
        plate_label.pack(fill="x", pady=(0, 5))

        ctk.CTkLabel(info_frame,
                     text=f"Owner: {vehicle_details['owner_name']}",
                     font=("Segoe UI", 12),
                     text_color="#475569",
                     anchor="w").pack(fill="x", pady=2)

        ctk.CTkLabel(info_frame,
                     text=f"Model: {vehicle_details['model']}",
                     font=("Segoe UI", 12),
                     text_color="#475569",
                     anchor="w").pack(fill="x", pady=2)

        # Right input section
        entry_frame = ctk.CTkFrame(plate_frame, fg_color="transparent")
        entry_frame.pack(side="right", padx=15, pady=10)

        ctk.CTkLabel(entry_frame,
                     text="Phone Number:",
                     font=("Segoe UI", 12, "bold"),
                     text_color="#1e3a8a").pack(pady=(0, 5))

        entry = ctk.CTkEntry(entry_frame,
                             width=250,
                             font=("Segoe UI", 12),
                             placeholder_text="+91XXXXXXXXXX",
                             text_color="#1e3a8a",
                             fg_color="#f8fafc",
                             border_color="#94a3b8")
        entry.pack()
        entry.insert(0, "+918287933022")
        entry.bind("<KeyRelease>", lambda e, p=plate, en=entry: validate_single_plate(p, en))
        phone_entries[plate] = entry

    # Footer
    footer_frame = ctk.CTkFrame(root, fg_color="#f1f5f9", height=60)
    footer_frame.pack(fill="x", side="bottom")
    footer_frame.pack_propagate(False)

    status_label = ctk.CTkLabel(footer_frame,
                                textvariable=status_var,
                                font=("Segoe UI", 12),
                                text_color="#64748b")
    status_label.pack(side="left", padx=10)

    send_button = ctk.CTkButton(footer_frame,
                                text="Send All Challans",
                                command=send_all_challans,
                                font=("Segoe UI", 14, "bold"),
                                fg_color="#3b82f6",
                                hover_color="#2563eb",
                                width=200)
    send_button.pack(side="right", padx=10, pady=10)

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    detected_plates = {"DL5SCY0736"}
    violation_data = {"DL5SCY0736": 3}
    show_phone_input_gui(detected_plates, violation_data, "Phone Violation")
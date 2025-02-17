import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import subprocess
import os
from datetime import datetime

# Load base configuration from JSON file
BASE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "base_config.json")
if os.path.exists(BASE_CONFIG_PATH):
    with open(BASE_CONFIG_PATH, "r") as file:
        config = json.load(file)
        # If the config working directory is not found, set it to the current directory
        if not os.path.exists(config["General"].get("working_directory", "")):
            config["General"]["working_directory"] = os.path.dirname(os.path.dirname(__file__))
else:
    messagebox.showerror("Error", "Base configuration file not found!")
    config = {}

class InfraGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Infrastructure Evaluation GUI")
        self.size = (600, 600)
        self.root.geometry(f"{self.size[0]}x{self.size[1]}+{int((root.winfo_screenwidth() - self.size[0]) / 2)}+{int((root.winfo_screenheight() - self.size[1]) / 2)}")

        self.data = config.copy()  # Copy the base configuration to self.data
        self.entries = {}  # Dictionary to hold entry widgets

        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill="both")

        # Create tabs
        self.general_tab = ttk.Frame(self.notebook)
        self.road_tab = ttk.Frame(self.notebook)
        self.rail_tab = ttk.Frame(self.notebook)

        # Add tabs to the notebook
        self.notebook.add(self.general_tab, text="General")
        self.notebook.add(self.road_tab, text="Road")
        self.notebook.add(self.rail_tab, text="Rail")

        # Initialize the UI components for each tab
        self.create_general_tab()
        self.create_settings_tab(self.road_tab, "Road")
        self.create_settings_tab(self.rail_tab, "Rail")

    def create_general_tab(self):
        # General tab UI components
        # Working directory label
        directory_frame = ttk.LabelFrame(self.general_tab, text="Directories")
        directory_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(directory_frame, text="Working Directory:").pack(pady=5)
        # Entry to display the working directory and fill with the stated working directory
        self.working_dir_entry = ttk.Entry(directory_frame, width=75)
        self.working_dir_entry.insert(0, self.data["General"].get("working_directory", ""))
        self.working_dir_entry.pack(pady=5)
        ttk.Button(directory_frame, text="Browse", command=self.set_working_directory).pack(pady=5)

        # Display the resulting data and output directories
        ttk.Label(directory_frame, text="Resulting Data Directory:").pack(pady=5)
        data_dir_entry = ttk.Entry(directory_frame, width=75)
        data_dir = os.path.join(self.data["General"].get("working_directory", ""), "data")
        data_dir_entry.insert(0, data_dir)
        data_dir_entry.pack(pady=5)
        data_dir_entry.configure(state="readonly")

        # Control buttons
        # Save and load labelframe
        save_load_frame = ttk.LabelFrame(self.general_tab, text="Load/Safe Configuration")
        ttk.Button(save_load_frame, text="Import JSON", command=self.import_json).pack(pady=5)
        ttk.Button(save_load_frame, text="Export JSON", command=self.export_json).pack(pady=5)
        save_load_frame.pack(fill="x", padx=10, pady=5)
        # Run labelframe
        run_frame = ttk.LabelFrame(self.general_tab, text="Run")
        ttk.Button(run_frame, text="Run Road", command=lambda: self.run_script("road.py")).pack(pady=5)
        ttk.Button(run_frame, text="Run Rail", command=lambda: self.run_script("rail.py")).pack(pady=5)
        ttk.Button(run_frame, text="Run Both", command=self.run_both).pack(pady=5)
        run_frame.pack(fill="x", padx=10, pady=5)


    def set_working_directory(self):
        # Open a directory dialog and set the working directory
        directory = filedialog.askdirectory()
        if directory:
            self.working_dir_entry.delete(0, tk.END)
            self.working_dir_entry.insert(0, directory)
            self.data["General"]["working_directory"] = directory

    def create_settings_tab(self, tab, category):
        # Create a scrollable canvas for settings
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        # Configure the scrollable frame
        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.entries[category] = {}
        
        # Populate the settings tab with entries
        for subcategory, values in self.data[category].items():
            frame = ttk.LabelFrame(scroll_frame, text=subcategory, padding=5)
            frame.pack(fill="x", padx=10, pady=5)
            self.entries[category][subcategory] = {}
            for key, value in values.items():
                row = ttk.Frame(frame)
                row.pack(fill="x", padx=5, pady=2)
                ttk.Label(row, text=key, width=25).pack(side="left")
                entry = ttk.Entry(row, width=50)
                entry.insert(0, str(value))
                entry.pack(side="right", expand=True)
                entry.bind("<FocusOut>", lambda e, k=key, s=subcategory, c=category, en=entry: self.update_data(k, s, c, en))
                self.entries[category][subcategory][key] = entry

    def update_data(self, key, subcategory, category, entry):
        # Update the data dictionary when an entry loses focus
        try:
            value = json.loads(entry.get())
            self.data[category][subcategory][key] = value
        except json.JSONDecodeError:
            messagebox.showerror("Input Error", f"Invalid input for {key}. Please enter a valid number.")

    def run_script(self, script):
        # Run a script in the working directory
        self.safe_executed_config()
        working_dir = self.data["General"].get("working_directory", "")
        # Add the directory of the script with infraScan{capitalized(script)} to the working directory and without .py use os functions
        working_dir = os.path.join(working_dir, f"infraScan{os.path.splitext(script)[0].capitalize()}")
        print(working_dir)
        if not os.path.exists(os.path.join(working_dir, script)):
            messagebox.showerror("Error", f"{script} not found in working directory.")
            return
        subprocess.run(["python", os.path.join(working_dir, script)], creationflags=subprocess.CREATE_NEW_CONSOLE)
        # Message for the user that the program is running
        messagebox.showinfo("Running", f"Running {os.path.join(working_dir, script)} script. Please wait for the process to start.")

    def run_both(self):
        # Run both road.py and rail.py scripts
        self.run_script("road.py")
        self.run_script("rail.py")

    def safe_executed_config(self):
        # Save the current configuration to a JSON file in a folder past_runs before running a script with time stamp
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        file_path = os.path.join(self.data["General"].get("working_directory", ""), "GUI", "past_runs", f"config_{current_time}.json")
        # Create the past_runs directory if it does not exist
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        with open(file_path, "w") as file:
            json.dump(self.data, file, indent=4)

    def import_json(self):
        # Import configuration from a JSON file
        file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if file_path:
            with open(file_path, "r") as file:
                self.data = json.load(file)
            self.refresh_ui()
            messagebox.showinfo("Success", "Configuration loaded successfully.")

    def export_json(self):
        # Export the current configuration to a JSON file
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
        if file_path:
            with open(file_path, "w") as file:
                json.dump(self.data, file, indent=4)
            messagebox.showinfo("Success", "Configuration saved successfully.")

    def refresh_ui(self):
        # Refresh the UI with the current data
        for category, subcategories in self.entries.items():
            for subcategory, fields in subcategories.items():
                for key, entry in fields.items():
                    entry.delete(0, tk.END)
                    entry.insert(0, str(self.data[category][subcategory].get(key, "")))

if __name__ == "__main__":
    root = tk.Tk()
    app = InfraGUI(root)
    root.mainloop()

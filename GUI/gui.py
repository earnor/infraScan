# Import Package
import tkinter as tk # For GUI
from tkinter import ttk, filedialog, messagebox # For GUI
import json # For data operation
import subprocess # For running sub moduls (e.g. road.py, rail.py)
import os
os.environ['USE_PYGEOS'] = '0'  # Force GeoPandas to use Shapely
import sys
import multiprocessing # For running sub moduls (e.g. road, rail)
from icecream import ic # For debugging
import logging

from datetime import datetime # For time operation
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches
import contextily as ctx

# Get the parent directory of GUI (i.e., InfraScan)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)  # Add InfraScan to Python's module search path
# Import the Rail and Road classes (they are outside the GUI directory)
# For that reason, we need to add the parent directory to the Python path
# And go back in the directory structure to import the classes
from logging_config import logger  # Import central logger
try:
    from infraScanRail.rail import Rail
    from infraScanRoad.road import Road
except ImportError as e:
    logging.error(f"Error importing scripts: {e}")
    messagebox.showerror("Import Error", f"Failed to import Rail or Road script.\n{e}")

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


def start_process(process_class, config):
    """Runs the given process class in a separate process."""
    try:
        process = process_class(config)  # Create an instance with config data
        process.run()  # Start execution
        logger.notice(f"{process_class.__name__} process completed.")
    except Exception as e:
        logger.error(f"Error running {process_class.__name__}: {e}")

class InfraGUI:
    def __init__(self, root):
        """
        Initialize the InfraGUI class.

        Parameters:
        root (tk.Tk): The root window of the Tkinter application.
        """
        self.root = root
        self.root.title("Infrastructure Evaluation GUI")
        self.size = (800, 800)
        self.root.geometry(f"{self.size[0]}x{self.size[1]}+{int((root.winfo_screenwidth() - self.size[0]) / 2)}+{int((root.winfo_screenheight() - self.size[1]) / 2)}")

        self.data = config.copy()  # Copy the base configuration to self.data
        self.original_data = json.loads(json.dumps(config))  # Deep copy to store original values
        self.entries = {}  # Dictionary to hold entry widgets

        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill="both")

        # Create tabs
        self.general_tab = ttk.Frame(self.notebook)
        self.data_tab = ttk.Frame(self.notebook)
        self.road_tab = ttk.Frame(self.notebook)
        self.rail_tab = ttk.Frame(self.notebook)
        self.spatial_limits_tab = ttk.Frame(self.notebook)

        # Add tabs to the notebook
        self.notebook.add(self.general_tab, text="General")
        self.notebook.add(self.data_tab, text="Data")
        self.notebook.add(self.road_tab, text="Road")
        self.notebook.add(self.rail_tab, text="Rail")
        self.notebook.add(self.spatial_limits_tab, text="Spatial Limits")

        # Initialize the UI components for each tab
        self.create_general_tab()
        self.create_data_tab()
        self.create_settings_tab(self.road_tab, "Road")
        self.create_settings_tab(self.rail_tab, "Rail")
        self.create_spatial_limit_tab()

    def create_general_tab(self):
        """
        Creates the UI components for the General tab for
        configuration control buttons, and run script buttons.

        Parameters:
        self (object): The instance of the class containing this method.
        """
        # General tab UI components
        
        # Control buttons
        control_frame = ttk.LabelFrame(self.general_tab, text="Configuration")
        # Label that shows the current name of the configuration file
        ttk.Label(control_frame, text="Config File Name:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.config_file_var = tk.StringVar()
        self.config_file_var.set(self.data["General"].get("config_file_name", ""))
        ttk.Label(control_frame, textvariable=self.config_file_var).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(control_frame, text="Import JSON", command=self.import_json).grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        ttk.Button(control_frame, text="Export JSON", command=self.export_json).grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        ttk.Button(control_frame, text="Reset to Default", command=self.reset_to_default).grid(row=1, column=2, columnspan=2, padx=5, pady=5, sticky="ew")
        control_frame.pack(fill="x", padx=10, pady=5)

        # Process Selection (Radio Buttons)
        process_frame = ttk.LabelFrame(self.general_tab, text="Process Selection")
        process_frame.pack(fill="x", padx=10, pady=5)

        self.process_vars = {}
        row_frame = ttk.Frame(process_frame)
        row_frame.pack(fill="x", padx=5, pady=5)

        for idx, (process, value) in enumerate(self.data["Process"].items()):
            var = tk.IntVar(value=value)
            self.process_vars[process] = var

            col_frame = ttk.Frame(row_frame)
            col_frame.grid(row=0, column=idx, padx=20, pady=5)

            ttk.Label(col_frame, text=process).pack(anchor="center")
            ttk.Radiobutton(col_frame, text="True", variable=var, value=1, command=lambda p=process: self.update_process(p, 1)).pack(side="left", anchor="center")
            ttk.Radiobutton(col_frame, text="False", variable=var, value=0, command=lambda p=process: self.update_process(p, 0)).pack(side="right", anchor="center")

        # Infrastructure Selection (Radio Buttons)
        infra_frame = ttk.LabelFrame(self.general_tab, text="Infrastructure Selection")
        infra_frame.pack(fill="x", padx=10, pady=5)

        self.infrastructure_vars = {}
        for infra_category, items in self.data["Infrastructure"].items():
            category_frame = ttk.LabelFrame(infra_frame, text=infra_category)
            category_frame.pack(fill="x", padx=5, pady=5)
            row_frame = ttk.Frame(category_frame)
            row_frame.pack(fill="x", padx=5, pady=5)
            for idx, (infra, value) in enumerate(items.items()):
                var = tk.IntVar(value=value)
                self.infrastructure_vars[infra] = var

                col_frame = ttk.Frame(row_frame)
                col_frame.grid(row=0, column=idx, padx=20, pady=5)

                ttk.Label(col_frame, text=infra).pack(anchor="center")
                ttk.Radiobutton(col_frame, text="True", variable=var, value=1,
                        command=lambda i=infra_category, 
                        k=infra: self.update_infrastructure(i, k, 1)).pack(side="left", anchor="center")
                ttk.Radiobutton(col_frame, text="False", variable=var, value=0,
                        command=lambda i=infra_category,
                        k=infra: self.update_infrastructure(i, k, 0)).pack(side="right", anchor="center")
            ttk.Button(category_frame, text=f"Run {infra_category}", command=lambda: self.run_script(f"{infra_category.lower()}")).pack(side="right", padx=5, pady=5)
        
        # Run Both Button
        run_frame = ttk.Frame(self.general_tab)
        run_frame.pack(fill="x", padx=10, pady=5)
        ttk.Button(run_frame, text="Run Both", command=self.run_both).pack(pady=5)


    def update_process(self, key, value):
        """Updates the Process dictionary when a radio button is selected."""
        self.data["Process"][key] = value

    def update_infrastructure(self, category, key, value):
        """Updates the Infrastructure dictionary when a radio button is selected."""
        self.data["Infrastructure"][category][key] = value

    def create_data_tab(self):
        """
        Creates the UI components for the Data tab, allowing users to change the directory paths
        """
        # Working directory label
        directory_frame = ttk.LabelFrame(self.data_tab, text="Directories")
        directory_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(directory_frame, text="Working Directory:").pack(pady=5)
        
        # Entry to display the working directory and fill with the stated working directory
        self.working_dir_entry = ttk.Entry(directory_frame, width=75)
        self.working_dir_entry.insert(0, self.data["General"].get("working_directory", ""))
        self.working_dir_entry.pack(pady=5)

        # Bind events to detect manual input and trigger directory update
        self.working_dir_entry.bind("<FocusOut>", self.manual_directory_update)
        self.working_dir_entry.bind("<Return>", self.manual_directory_update)

        ttk.Button(directory_frame, text="Browse", command=self.set_working_directory).pack(pady=5)

        def create_directory_entry(parent, label_text: str, subdirs: list):
            """Creates a read-only directory entry with a label.
            
            Args:
            parent: Parent widget
            label_text (str): Text for the label
            subdirs (list): List of subdirectories to append to working directory
            """
            ttk.Label(parent, text=label_text).pack(pady=5)
            entry = ttk.Entry(parent, width=75)
            dir_path = os.path.join(self.data["General"].get("working_directory", ""), *subdirs)
            entry.insert(0, dir_path)
            entry.pack(pady=5)
            entry.configure(state="readonly")
            return entry

        # Create directory entries
        self.rail_data_entry = create_directory_entry(directory_frame, "Rail Data Directory:", ["InfraScanRail", "data"])
        self.road_data_entry = create_directory_entry(directory_frame, "Road Data Directory:", ["InfraScanRoad", "data"])
        

    def set_working_directory(self):
        """
        Opens a directory dialog to select a working directory and updates
        the working directory entry field and the data dictionary with the selected path.
        """
        directory = filedialog.askdirectory()
        if directory:
            # Update UI
            self.working_dir_entry.delete(0, tk.END)
            self.working_dir_entry.insert(0, directory)

            # Update data dictionary
            self.data["General"]["working_directory"] = directory

            # Update derived data directories
            self.refresh_data_paths()

    def refresh_data_paths(self):
        """ Refreshes the automatically generated data directories. """
        if not hasattr(self, "rail_data_entry") or not hasattr(self, "road_data_entry"):
            return  # Ensure entries exist before updating

        rail_path = os.path.join(self.data["General"].get("working_directory", ""), "InfraScanRail", "data")
        road_path = os.path.join(self.data["General"].get("working_directory", ""), "InfraScanRoad", "data")

        self.rail_data_entry.configure(state="normal")
        self.road_data_entry.configure(state="normal")

        self.rail_data_entry.delete(0, tk.END)
        self.road_data_entry.delete(0, tk.END)

        self.rail_data_entry.insert(0, rail_path)
        self.road_data_entry.insert(0, road_path)

        self.rail_data_entry.configure(state="readonly")
        self.road_data_entry.configure(state="readonly")

        # Update JSON data
        self.data["General"]["Rail Data Directory"] = rail_path
        self.data["General"]["Road Data Directory"] = road_path

    def manual_directory_update(self, event=None):
        """Updates the working directory when the user manually types it in and refreshes data paths."""
        new_directory = self.working_dir_entry.get().strip()  # Get user input

        if os.path.isdir(new_directory):  # Ensure it's a valid directory
            self.data["General"]["working_directory"] = new_directory
            self.refresh_data_paths()  # Update the data directory paths
        else:
            messagebox.showerror("Invalid Directory", "The directory you entered does not exist.")


    def create_settings_tab(self, tab: tk.Widget, category: str):
        """
        Creates a settings tab with scrollable content for a given category.
        This method generates a tab containing a scrollable canvas with labeled frames
        and entry widgets for each subcategory and key-value pair in the provided category data.
        Each entry widget is pre-filled with the corresponding value and updates the data
        on focus out.
        Args:
            tab (tk.Widget): The parent widget where the settings tab will be created.
            category (str): The category of settings to be displayed in the tab.
        Returns:
            None
        """
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.entries[category] = {}

        for subcategory, values in self.data[category].items():
            frame = ttk.LabelFrame(scroll_frame, text=subcategory, padding=5)
            frame.pack(fill="x", padx=10, pady=5)
            self.entries[category][subcategory] = {}

            for key, value in values.items():
                row = ttk.Frame(frame)
                row.pack(fill="x", padx=5, pady=2)

                display_value = value["value"]
                unit = value["unit"]

                label_text = f"{key} ({unit})" if unit else key
                ttk.Label(row, text=label_text, width=35).pack(side="left")

                entry = ttk.Entry(row, width=40)
                entry.insert(0, str(display_value))
                entry.pack(side="right", expand=True)
                entry.bind("<FocusOut>", lambda e, k=key, s=subcategory, c=category, en=entry: self.update_data(k, s, c, en))
                self.entries[category][subcategory][key] = entry

    def update_data(self, key: str, subcategory: str, category: str, entry: ttk.Entry):
        """
        Updates the data dictionary with the value from the entry widget.

        Parameters:
        key (str): The key in the data dictionary to update.
        subcategory (str): The subcategory in the data dictionary to update.
        category (str): The category in the data dictionary to update.
        entry (ttk.Entry): The entry widget containing the new value.
        """
        try:
            value = json.loads(entry.get())

            # Check if the original value is a dictionary with "value" and "unit"
            if isinstance(self.data[category][subcategory][key], dict) and "unit" in self.data[category][subcategory][key]:
                self.data[category][subcategory][key]["value"] = value  # Preserve the unit
            else:
                self.data[category][subcategory][key] = {"value": value, "unit": ""}  # Fallback

        except json.JSONDecodeError:
            messagebox.showerror("Input Error", f"Invalid input for {key}. Please enter a valid number.")

    def create_spatial_limit_tab(self):
        """
        Creates the UI components for the Spatial Limits tab, allowing users to change 
        the limits set in the JSON configuration and visualize the spatial extent with GeoPandas.
        """
        # Ensure "General" and "Spatial Limits" exist in self.entries
        if "General" not in self.entries:
            self.entries["General"] = {}
        if "Spatial Limits" not in self.entries["General"]:
            self.entries["General"]["Spatial Limits"] = {}

        spatial_limits = self.data["General"].get("Spatial Limits", {})

        # Frame for Inputs
        input_frame = ttk.Frame(self.spatial_limits_tab)
        input_frame.pack(fill="x", padx=10, pady=5, side="top")

        # Frame for the Plot
        plot_frame = ttk.Frame(self.spatial_limits_tab)
        plot_frame.pack(fill="both", expand=True, padx=10, pady=5, side="bottom")

        # Entry Widgets for Spatial Limits
        self.spatial_entries = {}

        for key, value in spatial_limits.items():
            row = ttk.Frame(input_frame)
            row.pack(fill="x", padx=10, pady=5)

            ttk.Label(row, text=key, width=20).pack(side="left")
            entry = ttk.Entry(row, width=40)
            entry.insert(0, str(value))
            entry.pack(side="right", expand=True)
            entry.bind("<FocusOut>", lambda e, k=key, en=entry: self.update_spatial_limit(k, en))

            self.spatial_entries[key] = entry

        # Create an initial plot with GeoPandas
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Initial Plot Update
        self.update_spatial_plot()

    def update_spatial_limit(self, key: str, entry: ttk.Entry):
        """
        Updates the spatial limits in the data dictionary with the value from the entry widget.
        Refreshes the spatial plot when values are changed.
        """
        try:
            value = float(entry.get())
            self.data["General"]["Spatial Limits"][key] = value

            # Update the spatial plot whenever limits change
            self.update_spatial_plot()

        except ValueError:
            messagebox.showerror("Input Error", f"Invalid input for {key}. Please enter a valid number.")



    def update_spatial_plot(self):
        """
        Updates the spatial extent plot based on the values in the Spatial Limits tab.
        Uses GeoPandas for improved visualization and context layers.
        """
        try:
            # Get spatial extent values from the entries
            e_min = float(self.spatial_entries["e_min"].get())
            e_max = float(self.spatial_entries["e_max"].get())
            n_min = float(self.spatial_entries["n_min"].get())
            n_max = float(self.spatial_entries["n_max"].get())

            # Create a bounding box as a GeoDataFrame
            bbox_geom = box(e_min, n_min, e_max, n_max)
            bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:2056")  # LV95 Projection

            # Reproject the bounding box to EPSG:3857 for Contextily
            bbox_gdf = bbox_gdf.to_crs(epsg=3857)

            # Clear the previous plot
            self.ax.clear()
            
            # Plot bounding box
            bbox_gdf.plot(ax=self.ax, color="none", edgecolor="blue", linewidth=2, label="Bounding Box")

            # Compute bounding box dimensions
            dx = bbox_gdf.total_bounds[2] - bbox_gdf.total_bounds[0]  # Width
            dy = bbox_gdf.total_bounds[3] - bbox_gdf.total_bounds[1]  # Height

            # Ensure the plot is square: Take the larger dimension
            max_dim = max(dx, dy)

            # Apply a minimum 25% buffer
            buffer_size = max_dim * 0.25
            e_center = (bbox_gdf.total_bounds[2] + bbox_gdf.total_bounds[0]) / 2
            n_center = (bbox_gdf.total_bounds[3] + bbox_gdf.total_bounds[1]) / 2

            # Compute new square extent
            e_min_plot = e_center - (max_dim / 2) - buffer_size
            e_max_plot = e_center + (max_dim / 2) + buffer_size
            n_min_plot = n_center - (max_dim / 2) - buffer_size
            n_max_plot = n_center + (max_dim / 2) + buffer_size
            
            # Add basemap from Contextily
            try:
                ctx.add_basemap(self.ax, source=ctx.providers.SwissFederalGeoportal.NationalMapColor)
            except:
                logger.notice("Basemap can not be downloaded")

            # Set limits to enforce square plot
            self.ax.set_xlim(e_min_plot, e_max_plot)
            self.ax.set_ylim(n_min_plot, n_max_plot)
            self.ax.set_aspect('equal', adjustable='box')  # Ensure square aspect ratio
            self.ax.grid(True, which='both', linestyle='--', linewidth=0.5) # Add grid
            # Turn of all edges
            self.ax.spines['top'].set_visible(False)
            self.ax.spines['right'].set_visible(False)
            self.ax.spines['bottom'].set_visible(False)
            self.ax.spines['left'].set_visible(False)

            # Set labels and title
            self.ax.set_xlabel("Easting (m)")
            self.ax.set_ylabel("Northing (m)")
            self.ax.set_title("Spatial Extent (WGS 84)")

            # Manually create legend handles
            legend_patches = [mpatches.Patch(edgecolor="blue", facecolor="none", label="Bounding Box")]

            # Manually set legend with patches
            self.ax.legend(handles=legend_patches, loc="upper right")

            # Redraw the canvas
            self.canvas.draw()

        except ValueError:
            messagebox.showerror("Input Error", "Invalid spatial extent values. Please enter valid numbers.")



    def reset_to_default(self):
        """
        Resets all values to their default (original) state.
        """
        confirm = messagebox.askyesno("Reset", "Are you sure you want to reset all values to their defaults?")
        if confirm:
            self.data = json.loads(json.dumps(self.original_data))  # Restore original data
            self.refresh_ui()
            messagebox.showinfo("Reset", "Configuration has been reset to default values.")

    def run_script(self, script: str):
        """
        Runs the Rail or Road class in a separate process.

        Parameters:
        script (str): The name of the script ("rail" or "road").
        """
        logger.notice(f"Starting {script} process...")

        # Save configuration before running
        working_dir = self.data["General"].get("working_directory", "")
        save_path = os.path.join(working_dir, "GUI", "past_runs", f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        # Set Config File Name in json itself (if name is base_config.json, it will be overwritten)
        if self.data["General"]["config_file_name"] == "base_config.json":
            self.data["General"]["config_file_name"] = os.path.basename(save_path)
        # Save the configuration to a JSON file
        self.save_config(save_path)

        # Select the appropriate class
        if script == "rail":
            process_class = Rail
        elif script == "road":
            process_class = Road
        else:
            messagebox.showerror("Error", f"Unknown script: {script}")
            return

        # Start the process using multiprocessing
        process = multiprocessing.Process(target=start_process, args=(process_class, self.data))
        process.start()

    def run_both(self):
        """
        Executes two scripts: road and rail.

        This method sequentially runs the scripts road.py and rail by calling the run_script method for each script.
        """
        # Run both road and rail scripts
        self.run_script("road")
        self.run_script("rail")

    def save_config(self, file_path: str):
        """
        Save the current configuration to a specified JSON file.

        This method updates the metadata with the current date and time,
        ensures the past_runs directory exists, and writes the configuration
        data to the specified file path.

        Parameters:
        file_path (str): The path where the configuration file will be saved.
        """
        # Update metadata before exporting
        self.data["Metadata"]["date_created"] = datetime.now().strftime("%Y-%m-%d")
        self.data["Metadata"]["time_created"] = datetime.now().strftime("%H:%M:%S")
        # Create the past_runs directory if it does not exist
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        with open(file_path, "w") as file:
            json.dump(self.data, file, indent=4)

    def import_json(self):
        """ Imports configuration data from a JSON file and updates the UI """
        file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if file_path:
            with open(file_path, "r") as file:
                self.data = json.load(file)
            self.refresh_ui()
            messagebox.showinfo("Success", "Configuration loaded successfully.")


    def export_json(self):
        """Export the current configuration to a JSON file, ensuring unit preservation."""
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])

        if file_path:
            # Update metadata before exporting
            self.data["Metadata"]["date_created"] = datetime.now().strftime("%Y-%m-%d")
            self.data["Metadata"]["time_created"] = datetime.now().strftime("%H:%M:%S")
            self.data["General"]["config_file_name"] = os.path.basename(file_path)

            # Ensure units are correctly stored before saving
            for category, subcategories in self.entries.items():
                for subcategory, fields in subcategories.items():
                    for key, entry in fields.items():
                        value = entry.get()

                        # Ensure the existing unit is not lost
                        if isinstance(self.data[category][subcategory][key], dict) and "unit" in self.data[category][subcategory][key]:
                            self.data[category][subcategory][key]["value"] = json.loads(value)

            # Save to file
            with open(file_path, "w") as file:
                json.dump(self.data, file, indent=4)
            
            messagebox.showinfo("Success", "Configuration saved successfully.")


    def refresh_ui(self):
        """ Refreshes the UI with the current data after an import or reset. """

        # Refresh config file name
        self.config_file_var.set(self.data["General"].get("config_file_name", ""))
        
        # Refresh process selection radio buttons
        for process, var in self.process_vars.items():
            var.set(self.data["Process"].get(process, 0))

        # Refresh infrastructure selection radio buttons
        for infra_category, items in self.data["Infrastructure"].items():
            for infra, var in self.infrastructure_vars.items():
                if infra in items:
                    var.set(items[infra])  # Update radio button value

        # Refresh working directory entry
        self.working_dir_entry.delete(0, tk.END)
        self.working_dir_entry.insert(0, self.data["General"].get("working_directory", ""))

        # Refresh automatically generated data directories
        self.refresh_data_paths()

        # Refresh all numerical entry fields
        for category, subcategories in self.entries.items():
            for subcategory, fields in subcategories.items():
                for key, entry in fields.items():
                    value = self.data[category][subcategory].get(key, "")

                    # Handle case where value is stored as {"value": x, "unit": y}
                    if isinstance(value, dict) and "value" in value:
                        display_value = value["value"]
                    else:
                        display_value = value  # Fallback for old format

                    entry.delete(0, tk.END)
                    entry.insert(0, str(display_value))

        # Refresh spatial limits tab
        self.refresh_spatial_tab()

    def refresh_spatial_tab(self):
        """ Refreshes the spatial limits tab and updates the plot. """
        if hasattr(self, "spatial_entries"):
            for key, entry in self.spatial_entries.items():
                value = self.data["General"]["Spatial Limits"].get(key, "")
                entry.delete(0, tk.END)
                entry.insert(0, str(value))

        # Redraw the spatial plot
        self.update_spatial_plot()






if __name__ == "__main__":
    root = tk.Tk()
    app = InfraGUI(root)
    root.mainloop()

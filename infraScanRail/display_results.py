import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import os


def create_scenario_analysis_viewer(csv_file):
    """
    Creates a GUI to analyze scenarios from a CSV file.
    
    Parameters:
    - csv_file (str): Path to the CSV file containing the data.

    The CSV file should have columns containing 'Net Benefit', 'Monetized Savings',
    'Construction Cost', and 'Maintenance Costs' data for different scenarios,
    along with other descriptive columns.
    """
    # Load the data file
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        messagebox.showerror("Error", f"The data file '{csv_file}' was not found.")
        return
    except pd.errors.EmptyDataError:
        messagebox.showerror("Error", f"The data file '{csv_file}' is empty or corrupted.")
        return

    # Function to display details for the selected scenario
    def display_scenario_details():
        selected_scenario = scenario_var.get()
        
        if selected_scenario == "Select a scenario" or not selected_scenario:
            messagebox.showerror("Error", "Please select a valid scenario.")
            return

        # Filter the columns for the selected scenario
        monetized_savings_column = f"Monetized Savings {selected_scenario} [in CHF]"
        net_benefit_column = f"Net Benefit {selected_scenario} [in Mio. CHF]"
        construction_cost_column = "Construction Cost [in Mio. CHF]"
        maintenance_cost_column = "Maintenance Costs [in Mio. CHF]"
        
        # Check if all required columns exist
        required_columns = [
            "development", "Sline", 
            construction_cost_column, maintenance_cost_column, 
            monetized_savings_column, net_benefit_column
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            messagebox.showerror(
                "Error", f"The following required columns are missing: {', '.join(missing_columns)}"
            )
            return

        # Filter and reorder columns
        filtered_df = df[required_columns]

        # Round the numeric values in relevant columns
        for col in [construction_cost_column, maintenance_cost_column, monetized_savings_column, net_benefit_column]:
            filtered_df[col] = filtered_df[col].round(0).astype(int)

        # Sort the data by the selected scenario's net benefit column (least negative = best)
        filtered_df = filtered_df.sort_values(by=net_benefit_column, ascending=False)

        # Clear the treeview
        for item in table.get_children():
            table.delete(item)

        # Insert the column headers
        table["columns"] = filtered_df.columns.tolist()
        table["show"] = "headings"

        for col in filtered_df.columns:
            table.heading(col, text=col)
            table.column(col, width=200, anchor=tk.CENTER)

        # Insert the rows into the table
        best_row_index = filtered_df.index[0]  # The first row has the best net benefit
        for index, row in filtered_df.iterrows():
            row_tag = "best" if index == best_row_index else "normal"
            table.insert("", tk.END, values=row.tolist(), tags=(row_tag,))

        # Apply tag styling
        table.tag_configure("best", background="lightgreen")
        table.tag_configure("normal", background="white")

    # Extract scenario names from the column names
    available_scenarios = list(set(
        col.replace("Net Benefit ", "").split(" [")[0] 
        for col in df.columns if "Net Benefit" in col
    ))

    # Create the GUI
    root = tk.Tk()
    root.title("Best Development Finder")
    root.geometry("1400x700")
    root.configure(padx=20, pady=20)

    # Dropdown for selecting a scenario
    scenario_var = tk.StringVar(value="Select a scenario")
    scenario_label = tk.Label(root, text="Select a scenario:", font=("Arial", 12))
    scenario_label.pack(pady=5)
    scenario_dropdown = tk.OptionMenu(root, scenario_var, *available_scenarios)
    scenario_dropdown.pack(pady=10)

    # Button to display details for the selected scenario
    show_button = tk.Button(root, text="Show Scenario Details", font=("Arial", 12), command=display_scenario_details)
    show_button.pack(pady=10)

    # Treeview widget for the table
    table_frame = tk.Frame(root)
    table_frame.pack(fill=tk.BOTH, expand=True, pady=10)

    table = ttk.Treeview(table_frame)
    table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Add a scrollbar for the table
    scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=table.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    table.configure(yscrollcommand=scrollbar.set)

    # Run the GUI
    root.mainloop()


# Main script usage
if __name__ == "__main__":
    # set the working directory o this file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # Print working directory
    print("Current working directory:", os.getcwd())
    
    # Specify the path to your CSV file
    csv_file_path = "data/costs/total_costs_with_geometry.csv"
    # Call the function to create and display the GUI
    create_scenario_analysis_viewer(csv_file_path)

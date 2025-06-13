import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
from matplotlib.ticker import EngFormatter


def plot_tt_development_over_time():
    tt_savings_path = r"C:\Users\Silvano Fuchs\OneDrive - ETH Zurich\MA\06-Developments\network_performance_Ak2035 plot\traveltime_savings.csv"
    # Load the data
    tt_savings_1 = pd.read_csv(tt_savings_path)
    tt_savings_2 = tt_savings_1.copy()
    # Apply the 0.9 multiplier to tt_savings_2
    tt_savings_2["status_quo_tt"] *= 0.9  # TODO
    # Group by year and compute mean and std
    grouped_1 = tt_savings_1.groupby("year")["status_quo_tt"].agg(["mean", "std"]).reset_index()
    grouped_2 = tt_savings_2.groupby("year")["status_quo_tt"].agg(["mean", "std"]).reset_index()
    # Plotting
    plt.figure(figsize=(10, 6))
    # Plot for tt_savings_1
    plt.plot(grouped_1["year"], grouped_1["mean"], label="Status Quo - No Railway Expansion", color="blue")
    plt.fill_between(grouped_1["year"],
                     grouped_1["mean"] - grouped_1["std"],
                     grouped_1["mean"] + grouped_1["std"],
                     color="blue", alpha=0.2)
    # Plot for tt_savings_2
    plt.plot(grouped_2["year"], grouped_2["mean"], label="Status Quo - With Railway Expansion (0.9x)", color="green")
    plt.fill_between(grouped_2["year"],
                     grouped_2["mean"] - grouped_2["std"],
                     grouped_2["mean"] + grouped_2["std"],
                     color="green", alpha=0.2)
    # Styling
    plt.title("Development of Total Travel Time Across Scenarios")
    plt.xlabel("Year")
    plt.ylabel("Total Travel Time (status_quo_tt)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_scenarios():
    directory = r"C:\Users\Silvano Fuchs\OneDrive - ETH Zurich\MA\06-Developments\szenario_plot"
    with open(
            r"C:\Users\Silvano Fuchs\OneDrive - ETH Zurich\MA\06-Developments\szenario_plot\scenario_data_for_plots.pkl",
            'rb') as f:
        components = pickle.load(f)

    # Extrahiere die einzelnen Komponenten
    population_scenarios = components['population_scenarios']
    modal_split_scenarios = components['modal_split_scenarios']
    distance_per_person_scenarios = components['distance_per_person_scenarios']

    def plot_scenarios_with_range(
            scenarios_df: pd.DataFrame,
            save_path,
            value_col: str = "population",
            bezirk: str = None
    ):
        """
        Plot the range of all scenarios for a given value column as a shaded area
        and a single example scenario, with customized axis formatting.

        Parameters:
        - scenarios_df: DataFrame with columns "scenario", "year", and the specified value column
        - save_path: path where the plot will be saved
        - value_col: name of the column in scenarios_df containing the values to plot
        - bezirk: name of the bezirk (if plotting population scenarios)
        """
        # compute per-year stats
        year_stats = (
            scenarios_df
            .groupby("year")[value_col]
            .agg(min="min", max="max", mean="mean", std="std")
            .reset_index()
        )

        # Berechne +/- 1.65 Standardabweichungen (90% Konfidenzintervall)
        year_stats["mean_plus_1_65std"] = year_stats["mean"] + 1.65 * year_stats["std"]
        year_stats["mean_minus_1_65std"] = year_stats["mean"] - 1.65 * year_stats["std"]

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        # shaded range
        ax.fill_between(
            year_stats["year"],
            year_stats["min"],
            year_stats["max"],
            color='grey', alpha=0.3,
            label="Gesamter Bereich"
        )

        # +/- 1.65 Std. Abw. Linien (90% Konfidenzintervall)
        ax.plot(
            year_stats["year"],
            year_stats["mean_plus_1_65std"],
            color='red', linestyle='-', alpha=0.7,
            label="+1,65σ (95%)"
        )

        ax.plot(
            year_stats["year"],
            year_stats["mean_minus_1_65std"],
            color='red', linestyle='-', alpha=0.7,
            label="-1,65σ (5%)"
        )

        # mean line
        ax.plot(
            year_stats["year"],
            year_stats["mean"],
            color='grey', linestyle='--', alpha=0.8,
            label="Mittelwert"
        )

        # pick a random scenario to highlight
        sample_id = scenarios_df["scenario"].drop_duplicates().sample(n=1).iloc[0]
        sample_df = scenarios_df[scenarios_df["scenario"] == sample_id]
        ax.plot(
            sample_df["year"],
            sample_df[value_col],
            color='blue', linewidth=2,
            label=f"Beispielszenario {sample_id}"
        )

        # Hinzufügen des Markers für das Bundeszenario im Jahr 2050 (+/- 5% um den Mittelwert)
        if 2050 in year_stats["year"].values:
            mean_2050 = year_stats.loc[year_stats["year"] == 2050, "mean"].values[0]
            lower_bound = mean_2050 * 0.95  # -5%
            upper_bound = mean_2050 * 1.05  # +5%

            # Füge vertikale Linie für 2050 hinzu
            ax.vlines(x=2050, ymin=lower_bound, ymax=upper_bound,
                      colors='green', linestyles='solid', linewidth=3,
                      label="Bundes-Szenario 2050 (±5%)")

            # Füge Marker an den Enden hinzu
            ax.plot([2050], [lower_bound], marker="_", markersize=10, color='green')
            ax.plot([2050], [upper_bound], marker="_", markersize=10, color='green')

        # labels & styling
        col_title = value_col.replace('_', ' ').title()
        ax.set_xlabel("Jahr")

        # Formatierung der Y-Achse je nach Werttyp
        if value_col == "population":
            # Bevölkerung in Tausend ohne Dezimalstellen
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x / 1000):,}k".replace(",", ".")))
            ax.set_ylabel("Bevölkerung (in Tausend)")
            title_prefix = "Bevölkerung"
            if bezirk:
                title_prefix += f" - Bezirk {bezirk}"
        elif value_col == "modal_split":
            # Modal Split in Prozent
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
            ax.set_ylabel("Modal-Split (%)")
            title_prefix = "Modal-Split"
        else:
            # Standardformatierung mit SI-Präfix
            ax.yaxis.set_major_formatter(EngFormatter(unit='', places=2))
            ax.set_ylabel(col_title)
            title_prefix = col_title

        ax.set_title(f"{title_prefix}-Szenarien: Bereich, Mittelwert und 90% Konfidenzintervall")
        ax.grid(True)

        # Legende in der linken oberen Ecke
        ax.legend(loc='upper left')

        fig.tight_layout()

        # Save the plot, creating a filename based on the value column
        filename = f"{value_col.lower().replace(' ', '_')}"
        if bezirk:
            filename += f"_{bezirk.lower().replace(' ', '_')}"
        filename += "_scenarios.png"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path)
        plt.show()
        plt.close(fig)

    # Plots für Bevölkerung nach Bezirk
    pop_scenarios = {bezirk: population_scenarios[bezirk] for bezirk in list(population_scenarios.keys())}
    for bezirk, pop_scenario in pop_scenarios.items():
        plot_scenarios_with_range(pop_scenario, directory, 'population', bezirk)

    # Plots für Modal Split und Distanz pro Person
    plot_scenarios_with_range(modal_split_scenarios, directory, 'modal_split')
    plot_scenarios_with_range(distance_per_person_scenarios, directory, 'distance_per_person')

#plot_tt_development_over_time()
plot_scenarios()
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
from matplotlib.ticker import EngFormatter


def plot_tt_development_over_time(save_fig=False, output_dir=None, show_std_dev=True):
    """
    Erzeugt einen Plot zur Entwicklung der Reisezeit über die Jahre und vergleicht zwei Szenarien:
    AK2035 (Ausbaukonzept 2035) und aktuelles Netz (Stand 2024).

    Parameters:
    - save_fig: Boolean, ob die Figur gespeichert werden soll
    - output_dir: Zielverzeichnis zum Speichern der Figur (falls save_fig=True)
    - show_std_dev: Boolean, ob die Standardabweichung als Fläche dargestellt werden soll
    """
    data_dir = r"C:\Users\Silvano Fuchs\OneDrive - ETH Zurich\MA\06-Developments\network_performance_Ak2035 plot"
    tt_savings2035_path = os.path.join(data_dir, "traveltime_savings_2035.csv")
    tt_savings2024_path = os.path.join(data_dir, "traveltime_savings_2024.csv")

    # Daten laden
    tt_savings_2035 = pd.read_csv(tt_savings2035_path)
    tt_savings_2024 = pd.read_csv(tt_savings2024_path)

    # Daten nach Jahr gruppieren und Mittelwert/Standardabweichung berechnen
    grouped_1 = tt_savings_2035.groupby("year")["status_quo_tt"].agg(["mean", "std"]).reset_index()
    grouped_2 = tt_savings_2024.groupby("year")["status_quo_tt"].agg(["mean", "std"]).reset_index()

    # Differenz der Reisezeiten berechnen
    merged_data = pd.merge(grouped_1, grouped_2, on="year", suffixes=('_2035', '_2024'))
    merged_data['difference'] = merged_data['mean_2024'] - merged_data['mean_2035']  # Differenz (2024 - 2035)

    # Figur mit zwei Y-Achsen erstellen
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    ax2 = ax1.twinx()  # Zweite Y-Achse erstellen

    # Dunklere Orange-Farbe für bessere Lesbarkeit
    darker_orange = '#D35400'  # Dunkleres Orange

    # Differenz als Balken auf der zweiten Y-Achse (zuerst zeichnen, damit sie im Hintergrund sind)
    bars = ax2.bar(merged_data["year"], merged_data["difference"],
                   alpha=0.6, color=darker_orange, width=0.5,
                   label="Reisezeit-Differenz (2024 - 2035)")

    # Ausbaukonzept 2035 auf der ersten Y-Achse
    if show_std_dev:
        ax1.fill_between(grouped_1["year"],
                         grouped_1["mean"] - grouped_1["std"],
                         grouped_1["mean"] + grouped_1["std"],
                         color="green", alpha=0.2)

    # Aktuelles Netz (Stand 2024) auf der ersten Y-Achse
    if show_std_dev:
        ax1.fill_between(grouped_2["year"],
                         grouped_2["mean"] - grouped_2["std"],
                         grouped_2["mean"] + grouped_2["std"],
                         color="blue", alpha=0.2)

    # Linien im Vordergrund zeichnen (nach den Balken)
    line1 = ax1.plot(grouped_1["year"], grouped_1["mean"],
                     label="Mit Ausbaukonzept 2035",
                     color="green", linewidth=2, zorder=10)

    line2 = ax1.plot(grouped_2["year"], grouped_2["mean"],
                     label="Netz Stand 2024",
                     color="blue", linewidth=2, zorder=10)

    # Y-Achsenformatierung für die erste Y-Achse
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x / 1000):,}k".replace(",", ".")))

    # Y-Achsenformatierung für die zweite Y-Achse - ohne Tausender-Formatierung
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}" if x.is_integer() else f"{x:.0f}"))

    # Setze die zweite Y-Achse auf den Maximalwert 1000
    ax2.set_ylim(0, 1000)

    # Stelle sicher, dass beide Achsen dasselbe Raster haben
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2.grid(False)  # Zweites Raster ausschalten, damit nur ein Raster sichtbar ist

    # Styling und Beschriftungen
    ax1.set_title("Entwicklung der Gesamt-Reisezeit bei verschiedenen Netzausbauständen")
    ax1.set_xlabel("Jahr")
    ax1.set_ylabel("Gesamt-Reisezeit (in Tausend Minuten)")
    ax2.set_ylabel("Reisezeit-Differenz (in Minuten)")

    # Farben der Y-Achsenbeschriftungen anpassen - dunkleres Orange für bessere Lesbarkeit
    ax1.yaxis.label.set_color('black')
    ax2.yaxis.label.set_color(darker_orange)

    # Farben der Y-Achsen-Ticks anpassen
    ax2.tick_params(axis='y', colors=darker_orange)

    # Kombinierte Legende für beide Y-Achsen
    lines = line1 + line2 + [bars]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best", framealpha=0.9)

    # Kompakte Darstellung
    fig.tight_layout()

    # Optional: Figur speichern
    if save_fig and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, "travel_time_development_with_diff.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Abbildung gespeichert unter: {fig_path}")

    plt.show()
    return fig, ax1, ax2


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
            bezirk: str = None,
            federal_2050_range: tuple = None,  # Kann absolute Werte oder Faktoren enthalten
            range_is_factor: bool = False  # Gibt an, ob federal_2050_range Faktoren enthält
    ):
        """
        Plot the range of all scenarios for a given value column as a shaded area
        and a single example scenario, with customized axis formatting.

        Parameters:
        - scenarios_df: DataFrame with columns "scenario", "year", and the specified value column
        - save_path: path where the plot will be saved
        - value_col: name of the column in scenarios_df containing the values to plot
        - bezirk: name of the bezirk (if plotting population scenarios)
        - federal_2050_range: optional tuple (min, max) für den Bundesmarker:
            - Für population: Faktor vom Mittelwert (z.B. 0.885, 1.115 für ±11.5%)
              wenn range_is_factor=True, sonst absolute Werte
            - Für modal_split und distance_per_person: immer absolute Werte (z.B. 18.7, 24.3)
        - range_is_factor: Wenn True, werden die Werte in federal_2050_range als Faktoren interpretiert
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

        # Hinzufügen des Markers für das Bundeszenario im Jahr 2050
        if 2050 in year_stats["year"].values:
            mean_2050 = year_stats.loc[year_stats["year"] == 2050, "mean"].values[0]

            # Verwende benutzerdefinierte Werte wenn angegeben, sonst standardmäßig
            if federal_2050_range:
                if value_col == "population" and range_is_factor:
                    # Für Bevölkerung: Faktoren vom Mittelwert
                    factor_lower, factor_upper = federal_2050_range
                    lower_bound = mean_2050 * factor_lower
                    upper_bound = mean_2050 * factor_upper
                    marker_description = f"Bundes-Szenario 2050 (±{((factor_upper - 1) * 100):.1f}%)"
                else:
                    # Für Modal Split und Distance: absolute Werte
                    lower_bound, upper_bound = federal_2050_range
                    if value_col == "modal_split":
                        marker_description = f"Bundes-Szenario 2050 ({lower_bound*100:.1f}-{upper_bound*100:.1f}%)"
                    else:
                        marker_description = f"Bundes-Szenario 2050 ({lower_bound:.2f}-{upper_bound:.2f} km)"
            else:
                # Standardmäßig: ±5% vom Mittelwert
                lower_bound = mean_2050 * 0.95
                upper_bound = mean_2050 * 1.05
                marker_description = "Bundes-Szenario 2050 (±5%)"

            # Metallenes Orange für den Marker
            marker_color = '#E08D3C'  # Metallene Orange-Farbe

            # Füge vertikale Linie für 2050 hinzu mit reduzierter Strichdicke
            ax.vlines(x=2050, ymin=lower_bound, ymax=upper_bound,
                      colors=marker_color, linestyles='solid', linewidth=2,
                      label=marker_description)

            # Füge Marker an den Enden hinzu
            ax.plot([2050], [lower_bound], marker="_", markersize=10, color=marker_color)
            ax.plot([2050], [upper_bound], marker="_", markersize=10, color=marker_color)

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
            # Modal Split in Prozent (Werte kommen als Dezimalzahlen und müssen für die Anzeige mit 100 multipliziert werden)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))
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

    # Plots für Bevölkerung nach Bezirk mit ±11.5% vom Mittelwert
    pop_scenarios = {bezirk: population_scenarios[bezirk] for bezirk in list(population_scenarios.keys())}
    for bezirk, pop_scenario in pop_scenarios.items():
        plot_scenarios_with_range(pop_scenario, directory, 'population', bezirk,
                                      federal_2050_range=(0.885, 1.115), range_is_factor=True)

    # Plot für Modal Split mit absoluten Werten (18.7-24.3%)
    plot_scenarios_with_range(modal_split_scenarios, directory, 'modal_split',
                                  federal_2050_range=(0.187, 0.243))

    # Plot für Distance per Person mit absoluten Werten (34.77-39.47 km)
    plot_scenarios_with_range(distance_per_person_scenarios, directory, 'distance_per_person',
                                  federal_2050_range=(34.77, 39.47))
fig_dir = r"C:\Users\Silvano Fuchs\OneDrive - ETH Zurich\MA\06-Developments\network_performance_Ak2035 plot"
#plot_tt_development_over_time(save_fig=True, output_dir=fig_dir, show_std_dev=False)
plot_scenarios()
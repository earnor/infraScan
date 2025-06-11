import pandas as pd
import paths
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from tqdm import tqdm
from scipy.stats import norm, qmc
import numpy as np
from typing import Dict, List
from joblib import Parallel, delayed
import pickle
import os

def get_bezirk_population_scenarios():
    # Read the Swiss population scenario CSV with "," separator
    df_ch = pd.read_csv(paths.POPULATION_SCENARIO_CH_BFS_2055, sep=",")
    # Extract the relevant values for 2018 and 2050
    pop_2018 = df_ch.loc[df_ch['Jahr'] == 2018, 'Beobachtungen'].values
    pop_2050 = df_ch.loc[df_ch['Jahr'] == 2050, 'Referenzszenario A-00-2025'].values
    # Compute the growth factor: population_2050 / population_2018
    swiss_growth_factor_18_50 = pop_2050[0] / pop_2018[0]
    # Read the CSV file with ";" as separator
    df = pd.read_csv(paths.POPULATION_SCENARIO_CANTON_ZH_2050, sep=';')
    # Step 1: Aggregate total population per district and year
    population_summary = (
        df.groupby(['bezirk', 'jahr'])['anzahl']
        .sum()
        .reset_index()
        .rename(columns={'anzahl': 'total_population'})
    )
    # Step 2: Create full grid for all districts and all years from 2011 to 2050
    all_years = pd.Series(range(2011, 2051), name='jahr')
    all_districts = population_summary['bezirk'].unique()
    full_index = pd.MultiIndex.from_product([all_districts, all_years], names=['bezirk', 'jahr'])
    # Reindex to ensure each district has all years, fill missing population with 0
    population_complete = (
        population_summary.set_index(['bezirk', 'jahr'])
        .reindex(full_index)
        .fillna({'total_population': 0})
        .reset_index()
    )
    # Step 3: Calculate year-over-year growth rate per district
    population_complete['growth_rate'] = (
        population_complete
        .sort_values(['bezirk', 'jahr'])
        .groupby('bezirk')['total_population']
        .pct_change()
    )
    # Step 4: Split the complete dataset into a dictionary by district
    district_tables = {}
    for district, group in population_complete.groupby('bezirk'):
        group = group.reset_index(drop=True)

        # Extract population for 2018 and 2050
        pop_2018 = group.loc[group['jahr'] == 2018, 'total_population'].values
        pop_2050 = group.loc[group['jahr'] == 2050, 'total_population'].values

        growth_factor_18_50 = pop_2050[0] / pop_2018[0]

        # Compute yearly relative growth factor vs. CH
        relative_growth = (growth_factor_18_50 - 1) / (swiss_growth_factor_18_50 - 1)
        yearly_growth_factor = relative_growth ** (
                    1 / 32)  # this factor is only applicable to yearly growth RATES in the form of 0.015 for example, not FACTORS 1.015!!!

        # Store in DataFrame attributes
        group.attrs['growth_factor_18_50'] = growth_factor_18_50
        group.attrs['yearly_growth_factor_district_to_CH'] = yearly_growth_factor

        district_tables[district] = group
    # Step 5: Read Swiss growth rates from Eurostat Excel file
    eurostat_df = pd.read_excel(paths.POPULATION_SCENARIO_CH_EUROSTAT_2100)
    # Convert all column names to strings FIRST (important!)
    eurostat_df.columns = eurostat_df.columns.map(str)
    # Filter for the row where unit == 'GROWTH_RATE'
    growth_rate_row = eurostat_df[eurostat_df['unit'] == 'GROWTH_RATE']
    # Define year columns as strings
    year_columns = [str(year) for year in range(2051, 2101)]
    # Extract growth rates from that row
    ch_growth_rates = growth_rate_row[year_columns].iloc[0].astype(float)
    # Step 6: Extend each district with projected growth rates and populations
    for district, df_district in district_tables.items():
        # Get the last known population (for 2050)
        last_population = df_district.loc[df_district['jahr'] == 2050, 'total_population'].values[0]

        # Get the district-specific yearly growth factor to scale national growth rates
        scaling_factor = df_district.attrs['yearly_growth_factor_district_to_CH']

        # Prepare data for years 2051–2100
        new_rows = []
        current_population = last_population

        for year in range(2051, 2101):
            base_growth_rate = ch_growth_rates[str(year)]  # national growth rate (e.g., 0.012)
            adjusted_growth_rate = base_growth_rate * scaling_factor

            current_population *= (1 + adjusted_growth_rate)

            new_rows.append({
                'bezirk': district,
                'jahr': year,
                'total_population': current_population,
                'growth_rate': adjusted_growth_rate
            })

        # Convert new rows to DataFrame and append
        extension_df = pd.DataFrame(new_rows)
        df_extended = pd.concat([df_district, extension_df], ignore_index=True)
        district_tables[district] = df_extended.reset_index(drop=True)
    return district_tables



def generate_population_scenarios(ref_df: pd.DataFrame,
                                  start_year: int,
                                  end_year: int,
                                  n_scenarios: int = 1000,
                                  start_std_dev: float = 0.01,
                                  end_std_dev: float = 0.03,
                                  std_dev_shocks: float = 0.02) -> pd.DataFrame:
    """
    Generate stochastic population scenarios using Latin Hypercube Sampling and a random walk process.
    The main growth rates are perturbed using LHS and a time-varying std dev. Random shocks are added separately.

    Parameters:
    - ref_df: DataFrame with columns "jahr", "total_population", "growth_rate"
              - Only the "total_population" value at start_year is used as the initial population.
              - "growth_rate" is used as the base deterministic growth.
    - start_year: year to begin scenario generation
    - end_year: year to end scenario generation
    - n_scenarios: number of scenarios to generate
    - start_std_dev: starting std deviation applied to growth rate perturbation
    - end_std_dev: ending std deviation applied to growth rate perturbation
    - std_dev_shocks: std deviation of yearly additive shocks

    Returns:
    - DataFrame with columns: "scenario", "year", "population", "growth_rate"
    """
    # Filter and sort reference data
    ref_df = ref_df.sort_values("jahr")
    ref_df = ref_df[(ref_df["jahr"] >= start_year) & (ref_df["jahr"] <= end_year)].reset_index(drop=True)

    years = ref_df["jahr"].values
    ref_growth = ref_df["growth_rate"].values  # deterministic base growth per year
    n_years = len(years)
    initial_population = ref_df[ref_df["jahr"] == start_year]["total_population"].values[0]

    # Linearly interpolate std devs across years for growth rate variation
    growth_std_devs = np.linspace(start_std_dev, end_std_dev, n_years)

    # Latin Hypercube Sampling: growth rate perturbations
    sampler = qmc.LatinHypercube(d=n_years)
    lhs_samples = sampler.random(n=n_scenarios)  # shape: (n_scenarios, n_years)
    growth_perturbations = norm.ppf(lhs_samples) * growth_std_devs  # shape: (n_scenarios, n_years)

    # Perturbed growth rate: base + scenario-specific offset
    scenario_growth = ref_growth + growth_perturbations  # shape: (n_scenarios, n_years)
    # Setze Wachstumsrate für das erste Jahr auf 0 (kein Wachstum im ersten Jahr)
    scenario_growth[:, 0] = 0

    # Random shocks: et ~ N(0, std_dev_shocks)
    shock_sampler = qmc.LatinHypercube(d=n_years)
    lhs_shocks = shock_sampler.random(n=n_scenarios)
    et = norm.ppf(lhs_shocks) * std_dev_shocks
    # Setze Schocks für das erste Jahr auf 0
    et[:, 0] = 0

    # Cumulative shocks per scenario
    cumulative_shocks = np.cumsum(et, axis=1)  # shape: (n_scenarios, n_years)

    # Deterministic growth: cumulative product of (1 + growth_rate)
    deterministic_growth = np.cumprod(1 + scenario_growth, axis=1)  # shape: (n_scenarios, n_years)

    # Population index = deterministic path × stochastic shocks
    population_index = deterministic_growth + cumulative_shocks

    # Scale by initial population
    pop_scenarios = initial_population * population_index

    # Assemble output DataFrame
    scenario_data = []
    for i in range(n_scenarios):
        for t in range(n_years):
            # Berechne growth_index: 100 am Anfang und dann entsprechend der relativen Bevölkerungsentwicklung
            growth_index = 100 * (pop_scenarios[i, t] / initial_population)

            # Berechne die effektive Wachstumsrate inklusive Schocks
            if t == 0:
                # Für das erste Jahr ist die Wachstumsrate definitionsgemäß 0
                effective_growth_rate = 0.0
            else:
                # Berechne die prozentuale Änderung zur Bevölkerung des Vorjahres
                effective_growth_rate = (pop_scenarios[i, t] / pop_scenarios[i, t - 1]) - 1

            scenario_data.append({
                "scenario": i,
                "year": years[t],
                "population": pop_scenarios[i, t],
                "growth_rate": effective_growth_rate,
                "growth_index": growth_index
            })

    return pd.DataFrame(scenario_data)


def generate_modal_split_scenarios(avg_growth_rate: float,
                                   start_value: float,
                                   start_year: int,
                                   end_year: int,
                                   n_scenarios: int = 1000,
                                   start_std_dev: float = 0.01,
                                   end_std_dev: float = 0.03,
                                   std_dev_shocks: float = 0.02) -> pd.DataFrame:
    """
    Generate stochastic modal split scenarios using Latin Hypercube Sampling and a random walk process.

    Parameters:
    - avg_growth_rate: average annual growth rate to apply (can be positive or negative)
    - start_value: initial modal split value at start_year
    - start_year: year to begin scenario generation
    - end_year: year to end scenario generation
    - n_scenarios: number of scenarios to generate
    - start_std_dev: starting std deviation applied to growth rate perturbation
    - end_std_dev: ending std deviation applied to growth rate perturbation
    - std_dev_shocks: std deviation of yearly additive shocks

    Returns:
    - DataFrame with columns: "scenario", "year", "modal_split", "growth_rate", "growth_index"
    """
    # Erstelle temporären Referenzdatensatz mit konstanter Wachstumsrate
    years = np.arange(start_year, end_year + 1)
    n_years = len(years)

    # Berechne die Werte mit konstanter Wachstumsrate
    growth_factors = np.ones(n_years) * (1 + avg_growth_rate)
    growth_factors[0] = 1  # Erster Faktor ist 1, da es der Startwert ist

    # Kumulatives Wachstum berechnen
    cumulative_growth = np.cumprod(growth_factors)
    modal_split_values = start_value * cumulative_growth

    # Erstelle Array von Wachstumsraten (erster Wert ist 0, danach konstant)
    growth_rates = np.zeros(n_years)
    growth_rates[1:] = avg_growth_rate  # Konstante Wachstumsrate für alle Jahre außer dem ersten

    # Erstelle temporären DataFrame
    ref_df = pd.DataFrame({
        "jahr": years,
        "total_population": modal_split_values,
        "growth_rate": growth_rates
    })

    # Verwende die bestehende Funktion
    modal_split_scenarios_df = generate_population_scenarios(
        ref_df=ref_df,
        start_year=start_year,
        end_year=end_year,
        n_scenarios=n_scenarios,
        start_std_dev=start_std_dev,
        end_std_dev=end_std_dev,
        std_dev_shocks=std_dev_shocks
    )

    # Umbenennen der Spalte "population" in "modal_split"
    modal_split_scenarios_df = modal_split_scenarios_df.rename(columns={"population": "modal_split"})

    return modal_split_scenarios_df


def generate_distance_per_person_scenarios(avg_growth_rate: float,
                                           start_value: float,
                                           start_year: int,
                                           end_year: int,
                                           n_scenarios: int = 1000,
                                           start_std_dev: float = 0.01,
                                           end_std_dev: float = 0.03,
                                           std_dev_shocks: float = 0.02) -> pd.DataFrame:
    """
    Generate stochastic trips per person scenarios using Latin Hypercube Sampling and a random walk process.

    Parameters:
    - avg_growth_rate: average annual growth rate to apply (can be positive or negative)
    - start_value: initial trips per person value at start_year
    - start_year: year to begin scenario generation
    - end_year: year to end scenario generation
    - n_scenarios: number of scenarios to generate
    - start_std_dev: starting std deviation applied to growth rate perturbation
    - end_std_dev: ending std deviation applied to growth rate perturbation
    - std_dev_shocks: std deviation of yearly additive shocks

    Returns:
    - DataFrame with columns: "scenario", "year", "trips_per_person", "growth_rate", "growth_index"
    """
    # Nutze die bestehende Funktion für Modal-Split-Szenarien
    scenarios_df = generate_modal_split_scenarios(
        avg_growth_rate=avg_growth_rate,
        start_value=start_value,
        start_year=start_year,
        end_year=end_year,
        n_scenarios=n_scenarios,
        start_std_dev=start_std_dev,
        end_std_dev=end_std_dev,
        std_dev_shocks=std_dev_shocks
    )

    # Benenne die Spalte "modal_split" in "trips_per_person" um
    scenarios_df = scenarios_df.rename(columns={"modal_split": "distance_per_person"})

    return scenarios_df


def plot_population_scenarios(scenarios_df: pd.DataFrame, n_to_plot: int = 10):
    """
    Plot a sample of population scenarios.

    Parameters:
    - scenarios_df: DataFrame with columns "scenario", "year", "population"
    - n_to_plot: number of scenarios to randomly plot
    """
    plt.figure(figsize=(10, 6),dpi=300)

    sample_ids = scenarios_df["scenario"].drop_duplicates().sample(n=min(n_to_plot, scenarios_df["scenario"].nunique()))
    sample_df = scenarios_df[scenarios_df["scenario"].isin(sample_ids)]

    for scenario_id in sample_df["scenario"].unique():
        data = sample_df[sample_df["scenario"] == scenario_id]
        plt.plot(data["year"], data["population"] / 1e3, label=f"Scenario {scenario_id}")

    plt.xlabel("Year")
    plt.ylabel("Population (thousands)")
    plt.title(f"Sample of {n_to_plot} Population Scenarios")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_scenarios_with_range(
    scenarios_df: pd.DataFrame,
    save_path,
    value_col: str = "population"
    
):
    """
    Plot the range of all scenarios for a given value column as a shaded area
    and a single example scenario, with automatically scaled SI-prefix axis.

    Parameters:
    - scenarios_df: DataFrame with columns "scenario", "year", and the specified value column
    - value_col: name of the column in scenarios_df containing the values to plot
    """
    # compute per-year stats
    year_stats = (
        scenarios_df
        .groupby("year")[value_col]
        .agg(min="min", max="max", mean="mean")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # shaded range
    ax.fill_between(
        year_stats["year"],
        year_stats["min"],
        year_stats["max"],
        color='grey', alpha=0.3,
        label="Possible range"
    )

    # mean line
    ax.plot(
        year_stats["year"],
        year_stats["mean"],
        color='grey', linestyle='--', alpha=0.8,
        label="Mean"
    )

    # pick a random scenario to highlight
    sample_id = scenarios_df["scenario"].drop_duplicates().sample(n=1).iloc[0]
    sample_df = scenarios_df[scenarios_df["scenario"] == sample_id]
    ax.plot(
        sample_df["year"],
        sample_df[value_col],
        color='blue', linewidth=2,
        label=f"Example scenario {sample_id}"
    )

    # apply automatic SI‐prefix scaling on the Y axis
    ax.yaxis.set_major_formatter(EngFormatter(unit='', places=2))

    # labels & styling
    col_title = value_col.replace('_', ' ').title()
    ax.set_xlabel("Year")
    ax.set_ylabel(col_title)
    ax.set_title(f"{col_title} scenarios: range and example")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    # Save the plot, overwriting any existing file
    plt.savefig(save_path)
    plt.close(fig)


def build_station_to_communes_mapping(
        communes_to_stations: pd.DataFrame
) -> Dict[str, List[int]]:
    """
    Baut ein Mapping: station_id -> List[Commune_BFS_code].
    """
    return communes_to_stations.groupby('ID_point')['Commune_BFS_code'] \
        .apply(list) \
        .to_dict()


def compute_growth_od_matrix_optimized(
        initial_od: pd.DataFrame,
        station_communes: Dict[str, List[int]],
        communes_population: pd.DataFrame,
        population_scenarios: Dict[str, pd.DataFrame],
        scenario: int,
        year: int,
        start_year: int,
        station_commune_lookup: Dict[str, pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Optimierte Version von compute_growth_od_matrix
    """
    # --- 1) Struktur aufsetzen
    stations = initial_od.columns.tolist()
    from_stations = initial_od.index.tolist()

    # --- 2) Wachstumsindex für alle Stationen berechnen
    station_growth = {}

    # Wenn lookup nicht existiert, erstellen wir einen
    if station_commune_lookup is None:
        station_commune_lookup = {}

    for station in stations:
        # Lookup für diese Station verwenden wenn vorhanden
        if station not in station_commune_lookup:
            communes = station_communes.get(int(station), [])
            if not communes:
                station_growth[station] = 1.0
                continue

            # Vorfiltern der relevanten Gemeinden für diese Station
            station_data = []
            for commune in communes:
                row = communes_population[communes_population['gemeinde_bfs_nr'] == commune]
                if not row.empty:
                    district = row['bezirk'].iat[0]
                    pop_start_commune = row['anzahl'].iat[0]
                    station_data.append((commune, district, pop_start_commune))

            station_commune_lookup[station] = station_data

        sum_start = 0.0
        sum_curr = 0.0

        for commune, district, pop_start_commune in station_commune_lookup[station]:
            # Population im Szenario für Start- und Ziel-Jahr
            scen = population_scenarios[district]

            # Effizienterer Zugriff mit vorgefilterten Daten
            scenario_data = scen[(scen['scenario'] == scenario)]
            pop_d_start = scenario_data[scenario_data['year'] == start_year]['population'].iloc[0]
            pop_d_curr = scenario_data[scenario_data['year'] == year]['population'].iloc[0]

            # Bezirksfaktor
            factor_d = (pop_d_curr / pop_d_start) if pop_d_start > 0 else 1.0

            sum_start += pop_start_commune
            sum_curr += pop_start_commune * factor_d

        station_growth[station] = (sum_curr / sum_start) if sum_start > 0 else 1.0

    # --- 3) OD-Matrix mit Wachstumsfaktoren erzeugen
    growth_od = pd.DataFrame(1.0, index=from_stations, columns=stations)

    # Vektorisierte Anwendung der Faktoren
    sqrt_factors = {station: np.sqrt(factor) for station, factor in station_growth.items()}

    # Zeilenweise Multiplikation
    for station in set(list(map(str, from_stations))).intersection(sqrt_factors.keys()):
        growth_od.loc[int(station), :] *= sqrt_factors[station]

    # Spaltenweise Multiplikation
    for station in set(stations).intersection(sqrt_factors.keys()):
        growth_od.loc[:, station] *= sqrt_factors[station]

    return growth_od


def apply_modal_trips_optimized(
        initial_od: pd.DataFrame,
        growth_od: pd.DataFrame,
        modal_factors: Dict[tuple, float],
        distance_factors: Dict[tuple, float],
        scenario: int,
        start_year: int,
        year: int
) -> pd.DataFrame:
    """
    Optimierte Version von apply_modal_trips mit Lookup-Table
    """
    # Faktoren aus Lookup-Tabelle holen
    scenario_year_key = (scenario, year)
    m_factor = modal_factors.get(scenario_year_key, 1.0)
    d_factor = distance_factors.get(scenario_year_key, 1.0)

    # Einen Schritt berechnen
    return (initial_od * growth_od * m_factor * d_factor).astype('float32')


def precompute_modal_distance_factors(
        modal_df: pd.DataFrame,
        distance_df: pd.DataFrame,
        start_year: int
) -> tuple:
    """
    Vorausberechnung der Modal- und Distance-Faktoren
    """
    modal_factors = {}
    distance_factors = {}

    scenarios = modal_df['scenario'].unique()
    years = modal_df['year'].unique()

    # Modal split factors
    for s in scenarios:
        m_start = modal_df.loc[(modal_df['scenario'] == s) &
                               (modal_df['year'] == start_year), 'modal_split'].iat[0]

        for y in years:
            if y == start_year:
                modal_factors[(s, y)] = 1.0
                continue

            m_curr = modal_df.loc[(modal_df['scenario'] == s) &
                                  (modal_df['year'] == y), 'modal_split'].iat[0]
            m_factor = (m_curr / m_start) if m_start > 0 else 1.0
            modal_factors[(s, y)] = m_factor

    # Distance per person factors
    for s in scenarios:
        d_start = distance_df.loc[(distance_df['scenario'] == s) &
                                  (distance_df['year'] == start_year), 'distance_per_person'].iat[0]

        for y in years:
            if y == start_year:
                distance_factors[(s, y)] = 1.0
                continue

            d_curr = distance_df.loc[(distance_df['scenario'] == s) &
                                     (distance_df['year'] == y), 'distance_per_person'].iat[0]
            d_factor = (d_curr / d_start) if d_start > 0 else 1.0
            distance_factors[(s, y)] = d_factor

    return modal_factors, distance_factors


def generate_od_growth_scenarios(
        initial_od_matrix: pd.DataFrame,
        communes_to_stations: pd.DataFrame,
        communes_population: pd.DataFrame,
        start_year: int,
        end_year: int,
        num_of_scenarios: int,
        do_plot: bool = False,
        n_jobs: int = -1
) -> Dict[str, Dict[int, pd.DataFrame]]:
    """
    Optimierte Version von generate_od_growth_scenarios mit Multiprocessing
    """

    initial_od_matrix = initial_od_matrix.set_index('from_station')
    # 1) Bezirkspopulationsszenarien
    bezirk_pop_scenarios = get_bezirk_population_scenarios()
    population_scenarios = {
        bezirk: generate_population_scenarios(df, start_year, end_year, num_of_scenarios)
        for bezirk, df in bezirk_pop_scenarios.items()
    }

    # 2) Modal-Split- & Distance-per-Person-Szenarien
    modal_split_scenarios = generate_modal_split_scenarios(
        avg_growth_rate=0.0045,
        start_value=0.209,
        start_year=start_year,
        end_year=end_year,
        n_scenarios=num_of_scenarios,
        start_std_dev=0.002,
        end_std_dev=0.01,
        std_dev_shocks=0.01
    )
    distance_per_person_scenarios = generate_distance_per_person_scenarios(
        avg_growth_rate=-0.0027,
        start_value=39.79,
        start_year=start_year,
        end_year=end_year,
        n_scenarios=num_of_scenarios,
        start_std_dev=0.002,
        end_std_dev=0.005,
        std_dev_shocks=0.01
    )
    first_three_bezirk = list(population_scenarios.keys())[:3]
    first_three_scenarios = {bezirk: population_scenarios[bezirk] for bezirk in first_three_bezirk}
    for pop_scenario in first_three_scenarios.values():
        plot_scenarios_with_range(pop_scenario, paths.PLOT_SCENARIOS, 'population')
    plot_scenarios_with_range(modal_split_scenarios, paths.PLOT_SCENARIOS,'modal_split')
    plot_scenarios_with_range(distance_per_person_scenarios, paths.PLOT_SCENARIOS,'distance_per_person')


    # Vorauswertung aller Modal/Distance Faktoren
    print("Berechne Modal und Distance Faktoren...")
    modal_factors, distance_factors = precompute_modal_distance_factors(
        modal_split_scenarios, distance_per_person_scenarios, start_year
    )

    # 3) Station→Commune-Mapping (einmalig)
    print("Erstelle Station-Commune Mapping...")
    station_communes = build_station_to_communes_mapping(communes_to_stations)
    station_commune_lookup = {}  # Cache für station-commune Beziehungen

    # 4) Funktion für parallele Verarbeitung
    def process_scenario(s):
        key = f"scenario_{s + 1}"
        results_s = {}

        for y in range(start_year, end_year + 1):
            pop_growth_od = compute_growth_od_matrix_optimized(
                initial_od_matrix,
                station_communes,
                communes_population,
                population_scenarios,
                s, y, start_year,
                station_commune_lookup
            )

            final_od = apply_modal_trips_optimized(
                initial_od_matrix,
                pop_growth_od,
                modal_factors,
                distance_factors,
                s, start_year, y
            )
            results_s[y] = final_od

        return key, results_s

    # 5) Parallele Verarbeitung mit Fortschrittsbalken
    print(f"Berechne {num_of_scenarios} Szenarien mit {n_jobs} Prozessen...")
    scenario_results = Parallel(n_jobs=n_jobs, verbose = 100 )( #backend="loky", max_nbytes=None
        delayed(process_scenario)(s) for s in range(num_of_scenarios)
    )
    #scenario_results = [process_scenario(s) for s in range(num_of_scenarios)]
    print("fertig")
    # 6) Ergebnisse zusammenführen
    results = dict(scenario_results)
    return results


# bezirk_pop_scen = get_bezirk_population_scenarios()
# affoltern_df = bezirk_pop_scen['Affoltern']
# pop_scenarios_df = generate_population_scenarios(affoltern_df, 2022, 2100,n_scenarios=100, start_std_dev=0.005, end_std_dev=0.01, std_dev_shocks=0.02)
#
# plot_population_scenarios(pop_scenarios_df, n_to_plot=100)
# ms_scenario_df = generate_modal_split_scenarios(0.0045, 0.209, 2022, 2100, n_scenarios=100, start_std_dev=0.002, end_std_dev=0.005, std_dev_shocks=0.01)#growth rate assumption from verkehrsperspektiven 2017 til 2060
# trips_per_person_scenario_df = generate_distance_per_person_scenarios(-0.0027, 39.79, 2022, 2100, n_scenarios=100, start_std_dev=0.002, end_std_dev=0.005, std_dev_shocks=0.01)#growth rate assumption from verkehrsperspektiven 2017 til 2050 via gesamte verkehrsleistung, computed with chatgpt
#
# plot_scenarios_with_range(pop_scenarios_df,'population')
# plot_scenarios_with_range(ms_scenario_df, 'modal_split')
# plot_scenarios_with_range(trips_per_person_scenario_df, 'distance_per_person')



def load_scenarios_from_cache(cache_dir):
    """
    Load scenarios from individual .pkl files in the cache directory.
    
    Parameters:
    - cache_dir: Directory containing scenario .pkl files
    
    Returns:
    - Dictionary of loaded scenarios
    """
    scenarios = {}
    if os.path.exists(cache_dir):
        for file in os.listdir(cache_dir):
            if file.endswith('.pkl'):
                scenario_name = file.replace('.pkl', '')
                with open(os.path.join(cache_dir, file), 'rb') as f:
                    scenarios[scenario_name] = pickle.load(f)
        print(f"Loaded {len(scenarios)} scenarios from {cache_dir}")
    return scenarios

def get_random_scenarios(start_year=2018, end_year=2100, num_of_scenarios=100, use_cache=False, do_plot=False):
    """
    Retrieve or generate random OD growth scenarios.

    Parameters:
    - start_year: The starting year for the scenarios.
    - end_year: The ending year for the scenarios.
    - num_of_scenarios: The number of scenarios to generate.
    - use_cache: If True, load scenarios from cache instead of regenerating.
    - do_plot: If True, plot the scenarios after generation.

    Returns:
    - scenarios: The scenario DataFrame.
    """
    cache_dir = paths.RANDOM_SCENARIO_CACHE_PATH

    if use_cache:
        # Load cached scenarios from individual .pkl files
        scenarios = load_scenarios_from_cache(cache_dir)
        if scenarios:
            return scenarios
        else:
            print("No cached scenarios found. Generating new scenarios.")

    # Generate new scenarios
    scenarios = generate_od_growth_scenarios(
        pd.read_csv(paths.OD_STATIONS_KT_ZH_PATH),
        pd.read_excel(paths.COMMUNE_TO_STATION_PATH),
        pd.read_csv(paths.POPULATION_PER_COMMUNE_ZH_2018),
        start_year=start_year,
        end_year=end_year,
        num_of_scenarios=num_of_scenarios,
        do_plot=do_plot
    )

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)  # Ensure directory exists
    for scenario_name, scenario_data in scenarios.items():
        scen_path = os.path.join(cache_dir, f"{scenario_name}.pkl")
        with open(scen_path, 'wb') as f:
            pickle.dump(scenario_data, f)
    print(f"Saved {len(scenarios)} scenarios to {cache_dir}")
    
    return scenarios


if __name__ == '__main__':
    get_random_scenarios(start_year=2018, end_year=2100, num_of_scenarios=100, use_cache=False, do_plot=True)
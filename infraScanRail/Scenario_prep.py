import pandas as pd
import paths
import matplotlib.pyplot as plt
from scipy.stats import norm, qmc
import numpy as np

def get_bezirk_population_scenarios():
    global eurostat_df
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


# def generate_population_scenarios(ref_df: pd.DataFrame,
#                                   start_year: int,
#                                   end_year: int,
#                                   n_scenarios: int = 1000,
#                                   start_std_dev: float = 0.01,
#                                   end_std_dev: float = 0.03) -> pd.DataFrame:
#     """
#     Generate stochastic population scenarios using Latin Hypercube Sampling and a random walk process,
#     with standard deviation increasing linearly from start_std_dev to end_std_dev.
#
#     Parameters:
#     - ref_df: DataFrame with columns "jahr", "total_population", "growth_rate"
#               - Only the "total_population" value at start_year is used as the initial population.
#               - "growth_rate" should be in decimal (e.g., 0.01 for 1%)
#     - start_year: year to begin scenario generation
#     - end_year: year to end scenario generation
#     - n_scenarios: number of scenarios to generate
#     - start_std_dev: starting standard deviation for growth rate perturbation
#     - end_std_dev: ending standard deviation for growth rate perturbation
#
#     Returns:
#     - DataFrame with columns: "scenario", "year", "population", "growth_rate"
#     """
#     # Filter and sort reference data
#     ref_df = ref_df.sort_values("jahr")
#     ref_df = ref_df[(ref_df["jahr"] >= start_year) & (ref_df["jahr"] <= end_year)].reset_index(drop=True)
#
#     years = ref_df["jahr"].values
#     ref_growth = ref_df["growth_rate"].values
#     initial_population = ref_df[ref_df["jahr"] == start_year]["total_population"].values[0]
#     n_years = len(years)
#
#     # Linearly interpolate std devs over years
#     std_devs = np.linspace(start_std_dev, end_std_dev, n_years)
#
#     # Generate Latin Hypercube Samples in [0, 1]
#     sampler = qmc.LatinHypercube(d=n_years)
#     lhs_samples = sampler.random(n=n_scenarios)  # shape: (n_scenarios, n_years)
#
#     # Convert LHS samples into standard normal values
#     normal_samples = stats.norm.ppf(lhs_samples)  # shape: (n_scenarios, n_years)
#
#     # Scale each year's sample by its std deviation
#     growth_devs = normal_samples * std_devs  # shape: (n_scenarios, n_years)
#
#     # Apply additive noise
#     scenarios_growth = ref_growth + growth_devs  # assuming ref_growth is shape (n_years,)
#
#     # Initialize population trajectories
#     pop_scenarios = np.zeros((n_scenarios, n_years))
#     pop_scenarios[:, 0] = initial_population
#
#     for i in range(n_scenarios):
#         for t in range(1, n_years):
#             pop_scenarios[i, t] = pop_scenarios[i, t - 1] * (1 + scenarios_growth[i, t - 1])
#
#     # Compile results into DataFrame
#     scenario_data = []
#     for i in range(n_scenarios):
#         for t in range(n_years):
#             scenario_data.append({
#                 "scenario": i,
#                 "year": years[t],
#                 "population": pop_scenarios[i, t],
#                 "growth_rate": scenarios_growth[i, t]
#             })
#
#     return pd.DataFrame(scenario_data)



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

    # Random shocks: et ~ N(0, std_dev_shocks)
    shock_sampler = qmc.LatinHypercube(d=n_years)
    lhs_shocks = shock_sampler.random(n=n_scenarios)
    et = norm.ppf(lhs_shocks) * std_dev_shocks

    # Cumulative shocks per scenario
    cumulative_shocks = np.cumsum(et, axis=1)  # shape: (n_scenarios, n_years)

    # Deterministic growth: cumulative product of (1 + growth_rate)
    deterministic_growth = np.cumprod(1 + scenario_growth, axis=1)  # shape: (n_scenarios, n_years)

    # Population index = deterministic path × stochastic shocks
    population_index = deterministic_growth * np.exp(cumulative_shocks)

    # Scale by initial population
    pop_scenarios = initial_population * population_index

    # Assemble output DataFrame
    scenario_data = []
    for i in range(n_scenarios):
        for t in range(n_years):
            scenario_data.append({
                "scenario": i,
                "year": years[t],
                "population": pop_scenarios[i, t],
                "growth_rate": scenario_growth[i, t]
            })

    return pd.DataFrame(scenario_data)


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

bezirk_pop_scen = get_bezirk_population_scenarios()
affoltern_df = bezirk_pop_scen['Affoltern']
scenarios_df = generate_population_scenarios(affoltern_df, 2022, 2100,n_scenarios=100, start_std_dev=0.001, end_std_dev=0.003, std_dev_shocks=0.002)
plot_population_scenarios(scenarios_df, n_to_plot=100)
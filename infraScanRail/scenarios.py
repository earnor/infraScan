import pandas as pd
import random

import settings
from data_import import *
from rasterio.features import geometry_mask
from rasterstats import zonal_stats

import boto3 
import rasterio as rio
from rasterio.session import AWSSession
from shapely.geometry import Polygon
import os
import rasterio


import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon


################################################################################################################################################
#########################################################################################################################################################################
#new methode for scenarios 
#for population
 
def future_scenario_pop(n):
    # Load boundaries of municipalities (GEMEINDEN)
    #n=3
    boundaries = gpd.read_file(r"data\Scenario\Boundaries\Gemeindegrenzen\UP_GEMEINDEN_F.shp")

    # Define the inner boundary polygon based on the corridor limits
    e_min, e_max = 2687000, 2708000
    n_min, n_max = 1237000, 1254000

    def polygon_from_points(e_min, e_max, n_min, n_max, margin=0):
        return Polygon([
            (e_min - margin, n_min - margin),
            (e_min - margin, n_max + margin),
            (e_max + margin, n_max + margin),
            (e_max + margin, n_min - margin)
        ])

    # Create the inner boundary polygon
    innerboundary = polygon_from_points(e_min, e_max, n_min, n_max)

    # Filter for municipalities that intersect with the inner boundary
    boundaries_in_corridor = boundaries[boundaries.intersects(innerboundary)]

    scenario_zh = pd.read_csv(r"data/Scenario/KTZH_00000705_00001741.csv", sep=";")
    total_growth = scenario_zh[scenario_zh['jahr'] == 2050]['anzahl'].sum() - scenario_zh[scenario_zh['jahr'] == 2020]['anzahl'].sum()

    # Read the CSV file and select relevant columns
    pop_gemeinden = pd.read_excel(r"data/_basic_data/KTZH_00000127_00001245.xlsx", 
                                  sheet_name="Gemeinden", header=5).dropna()[['BFS-NR', 'GEMEINDE', 'TOTAL_2021']]

    def calculate_growth(pop_gemeinden, total_growth, k_urban, k_equal, k_rural):
        # Urban growth
        pop_gemeinden['weighted_population_urban'] = pop_gemeinden['TOTAL_2021'] ** k_urban
        total_weighted_urban = pop_gemeinden['weighted_population_urban'].sum()
        pop_gemeinden['growth_allocation_urban'] = (pop_gemeinden['weighted_population_urban'] / total_weighted_urban) * total_growth
        pop_gemeinden['relative_growth_allocation_urban'] = (pop_gemeinden['growth_allocation_urban'] / pop_gemeinden['TOTAL_2021']) + 1  # Adding +1
        
        # Equal growth
        pop_gemeinden['weighted_population_equal'] = pop_gemeinden['TOTAL_2021'] ** k_equal
        total_weighted_equal = pop_gemeinden['weighted_population_equal'].sum()
        pop_gemeinden['growth_allocation_equal'] = (pop_gemeinden['weighted_population_equal'] / total_weighted_equal) * total_growth
        pop_gemeinden['relative_growth_allocation_equal'] = (pop_gemeinden['growth_allocation_equal'] / pop_gemeinden['TOTAL_2021']) + 1  # Adding +1
        
        # Rural growth
        pop_gemeinden['weighted_population_rural'] = pop_gemeinden['TOTAL_2021'] ** k_rural
        total_weighted_rural = pop_gemeinden['weighted_population_rural'].sum()
        pop_gemeinden['growth_allocation_rural'] = (pop_gemeinden['weighted_population_rural'] / total_weighted_rural) * total_growth
        pop_gemeinden['relative_growth_allocation_rural'] = (pop_gemeinden['growth_allocation_rural'] / pop_gemeinden['TOTAL_2021']) + 1  # Adding +1
        
        # Select columns for output
        return pop_gemeinden[[ 'GEMEINDE',
            'relative_growth_allocation_urban',
            'relative_growth_allocation_equal',
            'relative_growth_allocation_rural']]

    # Calculate scenarios
    scenarios = {}
    for i in range(1, n + 1):
        growth_factor = 1 + ((i - (n // 2 + 1)) / n)
        adjusted_growth = total_growth * growth_factor
        
        # Calculate and get results for each scenario
        k_urban, k_equal, k_rural = 1.06, 1.0, 0.95  # exponents for urban, equal, rural
        growth_results = calculate_growth(pop_gemeinden, adjusted_growth, k_urban, k_equal, k_rural)

        # Add +1 to the relative growth values (for each growth allocation)
        growth_results['relative_growth_allocation_urban'] 
        growth_results['relative_growth_allocation_equal'] 
        growth_results['relative_growth_allocation_rural'] 

        # Rename the columns to include scenario numbering
        growth_results = growth_results.rename(columns={
            'relative_growth_allocation_urban': f'pop_urban_{i}',
            'relative_growth_allocation_equal': f'pop_equal_{i}',
            'relative_growth_allocation_rural': f'pop_rural_{i}'
        })

        # Store the scenario results
        scenarios[f'scenario_{i}'] = growth_results
    
    # Merge all scenarios into a single DataFrame
    merged_scenarios = pop_gemeinden[['GEMEINDE']]
    for scenario_data in scenarios.values():  # Iterate over the DataFrame values, not the keys
        merged_scenarios = merged_scenarios.merge(scenario_data, on='GEMEINDE', how='left')


    # Filter merged_scenarios to only include municipalities within boundaries_in_corridor
    merged_scenarios = merged_scenarios[merged_scenarios['GEMEINDE'].isin(boundaries_in_corridor['GEMEINDENA'])]

    # Perform a merge to add BFS and geometry columns based on matching municipalities
    merged_scenarios = merged_scenarios.merge(
        boundaries_in_corridor[['GEMEINDENA', 'BFS', 'geometry']],
        left_on='GEMEINDE', 
        right_on='GEMEINDENA',
        how='inner')

    # Drop the extra column used for merging if not needed
    merged_scenarios = merged_scenarios.drop(columns=['GEMEINDENA'])

    # Convert merged_scenarios to a GeoDataFrame
    merged_scenarios = gpd.GeoDataFrame(
        merged_scenarios, geometry=merged_scenarios['geometry'], 
        crs="EPSG:2056")  # Replace with the Swiss coordinates system

    # Save the GeoDataFrame to a shapefile
    merged_scenarios.to_file(r"data\temp\data_scenario_pop.shp")
    return

def scenario_to_raster_pop(frame=False):
    # Load the shapefile
    scenario_polygon = gpd.read_file(r"data\temp\data_scenario_pop.shp")
    #frame = [2680600, 1227700, 2724300, 1265600]

    if frame != False:
        # Create a bounding box polygon
        bounding_poly = box(frame[0], frame[1], frame[2], frame[3])
        len = (frame[2]-frame[0])/100
        width = (frame[3]-frame[1])/100
        print(f"frame: {len, width} it should be 377, 437")

        # Calculate the difference polygon
        # This will be the area in the bounding box not covered by existing polygons
        difference_poly = bounding_poly
        for geom in scenario_polygon['geometry']:
            difference_poly = difference_poly.difference(geom)

        new_row = {'geometry': difference_poly}
        for pop_scen in settings.pop_scenarios:
            new_row[pop_scen] = scenario_polygon[pop_scen].mean()

        print("New row added")
        scenario_polygon = gpd.GeoDataFrame(pd.concat([pd.DataFrame(scenario_polygon), pd.DataFrame(pd.Series(new_row)).T], ignore_index=True))

    
    #growth_rate_columns_pop = ["s1_pop", "s2_pop", "s3_pop"]
    path_pop = r"data\independent_variable\processed\raw\pop20.tif"

    #growth_rate_columns_empl = ["s1_empl", "s2_empl", "s3_empl"]
    #path_empl = r"data\independent_variable\processed\raw\empl20.tif"

    growth_to_tif(scenario_polygon, path=path_pop, columns=settings.pop_scenarios)
    #growth_to_tif(scenario_polygon, path=path_empl, columns=growth_rate_columns_empl)
    print('Scenario_To_Raster complete')


    base_path = r"data/independent_variable/processed/scenario"
    output_file = r"data/independent_variable/processed/scenario/pop_combined.tif"

    create_single_tif_with_bands(base_path, settings.pop_scenarios, output_file)
    return

#########################################################################################################################################################################
#########################################################################################################################################################################
#new methode for scenarios 
#for employment

def future_scenario_empl(n):
    # Load boundaries of municipalities (GEMEINDEN)
    boundaries = gpd.read_file(r"data\Scenario\Boundaries\Gemeindegrenzen\UP_GEMEINDEN_F.shp")
    # Aggregate geometries by municipality name to handle exclaves
    boundaries = boundaries.dissolve(by='GEMEINDENA').reset_index()

    # Define the inner boundary polygon based on the corridor limits
    e_min, e_max = 2687000, 2708000
    n_min, n_max = 1237000, 1254000

    def polygon_from_points(e_min, e_max, n_min, n_max, margin=0):
        return Polygon([
            (e_min - margin, n_min - margin),
            (e_min - margin, n_max + margin),
            (e_max + margin, n_max + margin),
            (e_max + margin, n_min - margin)
        ])

    # Create the inner boundary polygon
    innerboundary = polygon_from_points(e_min, e_max, n_min, n_max)

    # Filter for municipalities that intersect with the inner boundary
    boundaries_in_corridor = boundaries[boundaries.intersects(innerboundary)]

    # Load employment data and calculate growth
    empl_dev = pd.read_csv(r"data\Scenario\KANTON_ZUERICH_596.csv", sep=";", encoding="unicode_escape")
    empl_dev = empl_dev[["BFS_NR", "INDIKATOR_JAHR", "INDIKATOR_VALUE"]]
    empl_dev = empl_dev.rename(columns={"BFS_NR": "BFS", "INDIKATOR_JAHR": "jahr", "INDIKATOR_VALUE": "anzahl"})
    empl_dev = empl_dev[empl_dev["BFS"] != 0].reset_index(drop=True)
    empl_dev = empl_dev[(empl_dev["jahr"] == 2011) | (empl_dev["jahr"] == 2021)]
    empl_dev = empl_dev.pivot(index="BFS", columns="jahr", values="anzahl").reset_index()

    # Import BFS mapping
    bfs_nr = gpd.read_file(r"data\Scenario\Boundaries\Gemeindegrenzen\UP_GEMEINDEN_F.shp")
    bfs_nr = bfs_nr[["BFS", "GEMEINDENA"]]
    empl_dev = empl_dev.merge(bfs_nr, on="BFS", how="left")
    empl_dev.columns.name = None
    empl_dev.columns = ["BFS", "empl_2011", "empl_2021", "Gemeindename"]
    empl_dev["rel_10y"] = empl_dev["empl_2021"] / empl_dev["empl_2011"] - 1
    empl_dev = empl_dev.dropna()

    # Project canton-wide employment growth for 2050
    canton_empl_2021 = empl_dev["empl_2021"].sum()
    canton_rel_10y = canton_empl_2021 / empl_dev["empl_2011"].sum() - 1
    canton_total_growth_2050 = canton_empl_2021 * (1 + canton_rel_10y * 2.9)
    total_employment_growth = canton_total_growth_2050 - canton_empl_2021

    # Prepare employment data for municipalities
    emp_gemeinden = empl_dev.rename(columns={"BFS": "BFS-NR", "Gemeindename": "GEMEINDE"})[["BFS-NR", "GEMEINDE", "empl_2021"]]
    emp_gemeinden = emp_gemeinden.rename(columns={"empl_2021": "TOTAL_2021"})

    # Growth allocation calculation
    def calculate_growth(emp_gemeinden, total_growth, k_urban, k_equal, k_rural):
        emp_gemeinden['weighted_employment_urban'] = emp_gemeinden['TOTAL_2021'] ** k_urban
        total_weighted_urban = emp_gemeinden['weighted_employment_urban'].sum()
        emp_gemeinden['growth_allocation_urban'] = (emp_gemeinden['weighted_employment_urban'] / total_weighted_urban) * total_growth
        emp_gemeinden['relative_growth_allocation_urban'] = (emp_gemeinden['growth_allocation_urban'] / emp_gemeinden['TOTAL_2021']) + 1

        emp_gemeinden['weighted_employment_equal'] = emp_gemeinden['TOTAL_2021'] ** k_equal
        total_weighted_equal = emp_gemeinden['weighted_employment_equal'].sum()
        emp_gemeinden['growth_allocation_equal'] = (emp_gemeinden['weighted_employment_equal'] / total_weighted_equal) * total_growth
        emp_gemeinden['relative_growth_allocation_equal'] = (emp_gemeinden['growth_allocation_equal'] / emp_gemeinden['TOTAL_2021']) + 1

        emp_gemeinden['weighted_employment_rural'] = emp_gemeinden['TOTAL_2021'] ** k_rural
        total_weighted_rural = emp_gemeinden['weighted_employment_rural'].sum()
        emp_gemeinden['growth_allocation_rural'] = (emp_gemeinden['weighted_employment_rural'] / total_weighted_rural) * total_growth
        emp_gemeinden['relative_growth_allocation_rural'] = (emp_gemeinden['growth_allocation_rural'] / emp_gemeinden['TOTAL_2021']) + 1

        return emp_gemeinden[['GEMEINDE', 
                              'relative_growth_allocation_urban', 
                              'relative_growth_allocation_equal', 
                              'relative_growth_allocation_rural']]

    # Scenario calculations
    scenarios = {}
    for i in range(1, n + 1):
        growth_factor = 1 + ((i - (n // 2 + 1)) / n)
        adjusted_growth = total_employment_growth * growth_factor
        k_urban, k_equal, k_rural = 1.06, 1.0, 0.95
        growth_results = calculate_growth(emp_gemeinden, adjusted_growth, k_urban, k_equal, k_rural)
        growth_results = growth_results.rename(columns={
            'relative_growth_allocation_urban': f'empl_urban_{i}',
            'relative_growth_allocation_equal': f'empl_equal_{i}',
            'relative_growth_allocation_rural': f'empl_rural_{i}'
        })
        scenarios[f'scenario_{i}'] = growth_results

    # Merge all scenarios
    merged_scenarios = emp_gemeinden[['GEMEINDE']]
    for scenario_data in scenarios.values():
        merged_scenarios = merged_scenarios.merge(scenario_data, on='GEMEINDE', how='left')

    # Filter for municipalities in the corridor
    merged_scenarios = merged_scenarios[merged_scenarios['GEMEINDE'].isin(boundaries_in_corridor['GEMEINDENA'])]

    # Add geometry information
    merged_scenarios = merged_scenarios.merge(
        boundaries_in_corridor[['GEMEINDENA', 'BFS', 'geometry']],
        left_on='GEMEINDE', 
        right_on='GEMEINDENA',
        how='inner'
    ).drop(columns=['GEMEINDENA'])

    # Convert to GeoDataFrame
    merged_scenarios = gpd.GeoDataFrame(
        merged_scenarios, geometry=merged_scenarios['geometry'], 
        crs="EPSG:2056"
    )

    # Save to shapefile
    merged_scenarios = merged_scenarios.drop_duplicates(subset=['GEMEINDE'])
    merged_scenarios.to_file(r"data\temp\data_scenario_empl.shp")
    return

####################################################################################################################################################

def scenario_to_raster_emp(frame=False):
    # Load the shapefile
    scenario_polygon = gpd.read_file(r"data\temp\data_scenario_empl.shp")
    #frame = [2680600, 1227700, 2724300, 1265600]

    if frame != False:
        # Create a bounding box polygon
        bounding_poly = box(frame[0], frame[1], frame[2], frame[3])
        len = (frame[2]-frame[0])/100
        width = (frame[3]-frame[1])/100
        print(f"frame: {len, width} it should be 377, 437")

        # Calculate the difference polygon
        # This will be the area in the bounding box not covered by existing polygons
        difference_poly = bounding_poly
        for geom in scenario_polygon['geometry']:
            difference_poly = difference_poly.difference(geom)

        # Calculate the mean values for the three columns
        #mean_values = scenario_polygon.mean()

        # Create a new row for the difference polygon
        #new_row = {'geometry': difference_poly, 's1_pop': mean_values['s1_pop'], 's2_pop': mean_values['s2_pop'],
        #           's3_pop': mean_values['s3_pop'], 's1_empl': mean_values['s1_empl'], 's2_empl': mean_values['s2_empl'], 's3_empl': mean_values['s3_empl']}
        #new_row = {'geometry': difference_poly, 's1_pop': scenario_polygon['s1_pop'].mean(),
        #          's2_pop': scenario_polygon['s2_pop'].mean(), 's3_pop': scenario_polygon['s3_pop'].mean(),
        #          's1_empl': scenario_polygon['s1_empl'].mean(), 's2_empl': scenario_polygon['s2_empl'].mean(),
        #          's3_empl': scenario_polygon['s3_empl'].mean()}

        new_row = {'geometry': difference_poly}
        for empl_scen in settings.empl_scenarios:
            new_row[empl_scen] = scenario_polygon[empl_scen].mean()

        print("New row added")
        scenario_polygon = gpd.GeoDataFrame(pd.concat([pd.DataFrame(scenario_polygon), pd.DataFrame(pd.Series(new_row)).T], ignore_index=True))
    
    #growth_rate_columns_empl = ["s1_empl", "s2_empl", "s3_empl"]
    path_empl = r"data\independent_variable\processed\raw\empl20.tif"

    growth_to_tif(scenario_polygon, path=path_empl, columns=settings.empl_scenarios)
    #growth_to_tif(scenario_polygon, path=path_empl, columns=growth_rate_columns_empl)
    print('Scenario_To_Raster complete')


    base_path = r"data/independent_variable/processed/scenario"
    output_file = r"data/independent_variable/processed/scenario/empl_combined.tif"

    create_single_tif_with_bands(base_path, settings.empl_scenarios, output_file)
    return


###########################################################################################################################
###########################################################################################################################
# for both empl and pop to convert to tif and to stack the tif files with bands in to a single tif for each file

def growth_to_tif(polygons, path, columns):
    # Load the raster data
    aws_session = AWSSession(requester_pays=True)
    with rio.Env(aws_session):
        with rasterio.open(path) as src:
            raster = src.read(1)  # Assuming a single band raster

            # Iterate over each growth rate column
            for col in columns:
                # Create a new copy of the original raster to apply changes for each column
                modified_raster = raster.copy()

                for index, row in polygons.iterrows():
                    polygon = row['geometry']
                    change_rate = row[col]  # Use the current growth rate column

                    # Create a mask to identify raster cells within the polygon
                    mask = geometry_mask([polygon], out_shape=modified_raster.shape, transform=src.transform, invert=True)

                    # Apply the change rate to the cells within the polygon
                    modified_raster[mask] *= (change_rate)  # You may need to adjust this based on how your change rates are stored

                # Save the modified raster data to a new TIFF file
                output_tiff = f'data\independent_variable\processed\scenario\{col}.tif'
                with rasterio.open(output_tiff, 'w', **src.profile) as dst:
                    dst.write(modified_raster, 1)
    return

def create_single_tif_with_bands(base_path, file_names, output_file):
    """
    Reads .tif files, combines their bands into a single multi-band .tif file.

    Parameters:
        base_path (str): The base directory where the .tif files are located.
        file_names (list): List of file names (without extensions) to process.
        output_file (str): The path to save the combined .tif file.

    Returns:
        None
    """
    bands_data = []  # List to store bands
    meta = None  # To store metadata for the new file

    for file_name in file_names:
        input_file = os.path.join(base_path, f"{file_name}.tif")
        try:
            with rasterio.open(input_file) as src:
                # Read the first band
                band = src.read(1)
                bands_data.append(band)

                # Save the metadata from the first file (assuming consistent metadata)
                if meta is None:
                    meta = src.meta

                print(f"Band from {file_name} added.")
        except FileNotFoundError:
            print(f"File not found: {input_file}")
        except Exception as e:
            print(f"Error processing {input_file}: {e}")

    if meta is not None and bands_data:
        # Update metadata for multi-band file
        meta.update(count=len(bands_data))

        # Write all bands to the output file
        with rasterio.open(output_file, 'w', **meta) as dst:
            for i, band in enumerate(bands_data, start=1):
                dst.write(band, i)  # Write each band to the respective index

        print(f"All bands combined and saved as: {output_file}")
    else:
        print("No bands were processed. Check input files.")


def dummy_generate_scenarios(n_scenarios: int, seed: int = None):
    if seed is not None:
        random.seed(seed)

    scenarios = []
    for _ in range(n_scenarios):
        pop_growth = random.uniform(0.10, 0.35)
        empl_growth = random.uniform(0.15, 0.50)
        oev_modal_split_growth = random.uniform(0.05, 0.50)
        mobility_growth_rate = random.uniform(-0.15, 0.40)

        # Convert to growth factors (1 + rate)
        pop_factor = 1 + pop_growth
        empl_factor = 1 + empl_growth
        oev_factor = 1 + oev_modal_split_growth
        mobility_factor = 1 + mobility_growth_rate

        general_factor = pop_factor * empl_factor * oev_factor * mobility_factor

        scenarios.append({
            'pop_growth': pop_growth,
            'empl_growth': empl_growth,
            'oev_modal_split_growth': oev_modal_split_growth,
            'mobility_growth_rate': mobility_growth_rate,
            'general_factor': general_factor
        })

    return scenarios
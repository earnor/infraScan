import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
import os
import re
from rasterstats.io import bounds_window
from rasterio.windows import bounds

from rasterio.merge import merge
from tqdm import tqdm

def stack_tif_files(var):

    # List of your TIFF file paths
    tiff_files = [f"\s1_{var}.tif", f"\s2_{var}.tif", f"\s3_{var}.tif"]

    # Open the first file to retrieve the metadata
    with rasterio.open(r"data\independent_variable\processed\scenario"+tiff_files[0]) as src0:
        meta = src0.meta

    # Update metadata to reflect the number of layers
    meta.update(count=len(tiff_files))

    out_fp = fr"data\independent_variable\processed\scenario\scen_{var}.tif"
    # Read each layer and write it to stack
    with rasterio.open(out_fp, 'w', **meta) as dst:
        for id, layer in enumerate(tiff_files, start=1):
            with rasterio.open(r"data\independent_variable\processed\scenario"+layer) as src1:
                dst.write_band(id, src1.read(1))


# # 0 Who will drive by car
# We assume peak hour demand is generated by population residence at origin and employment opportunites at destination.
def GetCommunePopulation(y0):    # We find population of each commune.
    rawpop = pd.read_excel('data\_basic_data\KTZH_00000127_00001245.xlsx',sheet_name='Gemeinden',header=None)
    rawpop.columns = rawpop.iloc[5]
    rawpop = rawpop.drop([0,1,2,3,4,5,6])
    pop = pd.DataFrame(data=rawpop,columns=['BFS-NR  ','TOTAL_'+str(y0)+'  ']).sort_values(by='BFS-NR  ')
    popvec = np.array(pop['TOTAL_'+str(y0)+'  '])
    return popvec


def GetCommuneEmployment(y0): # we find employment in each commune.
    rawjob = pd.read_excel('data\_basic_data\KANTON_ZUERICH_596.xlsx')
    rawjob = rawjob.loc[(rawjob['INDIKATOR_JAHR'] == y0) & (rawjob['BFS_NR'] > 0) & (rawjob['BFS_NR'] != 291)]

    #rawjob=rawjob.loc[(rawjob['INDIKATOR_JAHR']==y0)&(rawjob['BFS_NR']>0)&(rawjob['BFS_NR']!=291)]
    job=pd.DataFrame(data=rawjob,columns=['BFS_NR','INDIKATOR_VALUE']).sort_values(by='BFS_NR')
    jobvec = np.array(job['INDIKATOR_VALUE'])
    return jobvec


def GetHighwayPHDemandPerCommune():
    # now we extract an od matrix for pt from year 2019
    # we then modify the OD matrix to fit our needs of expressing peak hour pt travel demand
    rawod = pd.read_excel('data\_basic_data\KTZH_00001982_00003903.xlsx')
    communalOD = rawod.loc[(rawod['jahr']==2018) & (rawod['kategorie']=='Verkehrsaufkommen') & (rawod['verkehrsmittel']=='oev')]
    #communalOD = data.drop(['jahr','quelle_name','quelle_gebietart','ziel_name','ziel_gebietart',"kategorie","verkehrsmittel","einheit","gebietsstand_jahr","zeit_dimension"],axis=1)
    #sum(communalOD['wert'])
    # 1 Who will go on highway?
    # # # Not binnenverkehr ... removes about 50% of trips
    communalOD['wert'].loc[(communalOD['quelle_code']==communalOD['ziel_code'])]=0
    #sum(communalOD['wert'])
    # # Take share of OD
    tau = 0.1  #Data is in trips per OD combination per day. Now we assume the number of trips gone in peak hour
    # This ratio explains the interzonal trips made in peak hour as a ratio of total interzonal trips made per day.
    communalOD['wert'] = (communalOD['wert']*tau)
    # # # Not those who travel < 15 min ?  Not yet implemented.
    return communalOD


def GetODMatrix(od):
    od_int = od.loc[(od['quelle_code']<9999) & (od['ziel_code']<9999)]
    odmat = od_int.pivot(index='quelle_code',columns='ziel_code',values='wert')
    return odmat


def GetCommuneShapes(raster_path): #todo this might be unnecessary if you already have these shapes.
    communalraw = gpd.read_file(r"data\_basic_data\Gemeindegrenzen\UP_GEMEINDEN_F.shp")
    communalraw = communalraw.loc[(communalraw['ART_TEXT']=='Gemeinde')]
    communedf = gpd.GeoDataFrame(data=communalraw,geometry=communalraw['geometry'],columns=['BFS','GEMEINDENA'],crs="epsg:2056").sort_values(by='BFS')

    # Read the reference TIFF file
    with rasterio.open(raster_path) as src:
        profile = src.profile
        profile.update(count=1)

    # Rasterize
    with rasterio.open('data\_basic_data\Gemeindegrenzen\gemeinde_zh.tif', 'w', **profile) as dst:
        rasterized_image = rasterize(
            [(shape, value) for shape, value in zip(communedf.geometry, communedf['BFS'])],
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            all_touched=False,
            dtype=rasterio.int32
        )
        dst.write(rasterized_image, 1)

    # Convert the rasterized image to a numpy array
    commune_raster = np.array(rasterized_image)

    return commune_raster, communedf

def  GetCatchmentold():

    return

def compute_TT():

    # Change working directory
    os.chdir(r"C:\Users\phili\polybox\ETH_RE&IS\Master Thesis\06-Developments\01-Code\infraScanRail")

    # Directories containing CSV files (relative paths since the working directory is now set)
    directories = [
        r"data\traffic_flow\od\rail",
        r"data\Network\travel_time\developments"
    ]

    # Loop through each directory
    for directory in directories:
        if os.path.exists(directory):  # Check if directory exists
            # List all CSV files in the directory
            csv_files = [file for file in os.listdir(directory) if file.endswith(".csv")]
            
            # Read each CSV file and assign as a DataFrame variable
            for csv_file in csv_files:
                file_path = os.path.join(directory, csv_file)
                try:
                    # Create a variable name from the CSV file name without the extension
                    var_name = csv_file.replace(".csv", "")
                    
                    # Load the CSV as a DataFrame
                    df = pd.read_csv(file_path)
                    
                    # Filter and rename columns
                    df_filtered = df[['OriginStation', 'Destination', 'TotalTravelTime']].rename(
                        columns={'OriginStation': 'from_id', 'Destination': 'to_id', 'TotalTravelTime': 'time'}
                    )
                    
                    # Assign the filtered DataFrame to the dynamically created variable
                    globals()[var_name] = df_filtered
                    
                    print(f"Loaded and processed {csv_file} as {var_name}")
                except Exception as e:
                    print(f"Error reading or processing {csv_file}: {e}")
        else:
            print(f"Directory not found: {directory}")

    # Example: access the processed DataFrame by its variable name (if applicable)
    # For example, if there was a file '100000.csv', it will be accessible as `100000`




def GetVoronoiOD_old(voronoidf, scen_empl_path, scen_pop_path, voronoi_tif_path):

    # define dev (=ID of the polygons of a development)

    #todo When we iterate over devs and scens, maybe we can check if the VoronoiDF already has the communal data and then skip the following five lines
    popvec = GetCommunePopulation(y0="2021")
    jobvec = GetCommuneEmployment(y0=2021)
    od = GetHighwayPHDemandPerCommune()
    odmat = GetODMatrix(od)

    # This function returns a np array of raster data storing the bfs number of the commune in each cell
    commune_raster, commune_df = GetCommuneShapes(raster_path=voronoi_tif_path)

    if jobvec.shape[0] != odmat.shape[0]:
        print("Error: The number of communes in the OD matrix and the number of communes in the employment data do not match.")
    #com_idx = np.unique(od['quelle_code']) # previously od_mat

    #todo the following line setting up rasterdf. We could probably pick a better GDF to start with. Most important is to have the raster, location of raster cells
    """
    rasterdf = pop_input_20.drop(['VZA20_0-50','VZA20_500-','VZA20_1000','VZA20_2000','VZA20_3500','VZA20_5000'],axis=1) #todo
    rasterdf.geometry = rasterdf.centroid
    rasterdf = rasterdf.sjoin(communeShapes,how="inner") #here we want to know which Commune's BFS ID corresponds to which raster cell
    rasterdf = rasterdf.drop(['index_right'],axis=1)
    rasterdf = rasterdf.sjoin(voronoidf,how="inner") #here we want to know which Voronoi's ID corresponds to which raster cell
    rasterdf = rasterdf.drop(['index_right'],axis=1)
    """
    # Until now scenarios were given as gpd squares and not as tif raster file

    # 1. Define a new raster file that stores the Commune's BFS ID as cell value
    # Think if new band or new tif makes more sense
    # using communeShapes



    # I guess here iterate over all developments
    #voronoidf = voronoidf.loc[(voronoidf['ID_develop'] == dev)] # Work with temp gdf of voronoi
    # If possible simplify all the amount of developments

    # Open scenario (medium) raster data    (low = band 2, high = band 3)
    with rasterio.open(scen_pop_path) as src:
        # Read the raster into a NumPy array (assuming you want the first band)
        scen_pop_medium_tif = src.read(1)

    with rasterio.open(scen_empl_path) as src:
        # Read the raster into a NumPy array (assuming you want the first band)
        scen_empl_medium_tif = src.read(1)

    # Open voronoi raster data
    with rasterio.open(voronoi_tif_path) as src:
        # Read the raster into a NumPy array (assuming you want the first band)
        voronoi_tif = src.read(1)
    unique_voronoi_id = np.sort(np.unique(voronoi_tif))
    #vor_idx = unique_voronoi_id.tolist()
    vor_idx = unique_voronoi_id.size
    #vor_idx = voronoidf['ID_point'].sort_by('ID_point')

    # Get voronoi tif boundaries and filter the commune_df that lay in it or touch it
    # Get the bounds of the voronoi tif
    bounds = src.bounds
    # Get the commune_df that are within the bounds
    commune_df_filtered = commune_df.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]
    # Get "BFS" value of the commune_df_filtered that are within the bounds
    commune_df_filtered = commune_df_filtered["BFS"].to_numpy()

    # Do a copy of odmat and filter the rows and columns that are not in commune_df_filtered
    odmat_frame = odmat.loc[commune_df_filtered, commune_df_filtered]


    #todo By addressing this monster of five for loops, we can win a lot of computational performance.

    #od_mn = np.zeros([len(vor_idx),len(vor_idx)])

    # Assume vectorized functions are defined for the below operations

    # Step 1: generate unit_flow matrix from each commune to each other commune
    cout_r = odmat / np.outer(popvec, jobvec)

    # Step 2: Get all pairs of combinations from communes to polygons
    unique_commune_id = np.sort(np.unique(commune_raster))
    pairs = pd.DataFrame(columns=['commune_id', 'voronoi_id'])
    pop_empl = pd.DataFrame(columns=['commune_id', 'voronoi_id', "empl", "pop"])

    for i in tqdm(unique_voronoi_id, desc='Processing Voronoi IDs'):
        # Get the voronoi raster
        mask_voronoi = voronoi_tif == i
        for j in unique_commune_id:
            if j > 0:
                # Get the commune raster
                mask_commune = commune_raster == j
                # Combined mask
                mask = mask_commune & mask_voronoi
                # Check if there are overlaying values
                if np.nansum(mask) > 0:
                    pairs = pairs.append({'commune_id': j, 'voronoi_id': i}, ignore_index=True)

                    # Get the population and employment values
                    pop = scen_pop_medium_tif[mask]
                    empl = scen_empl_medium_tif[mask]
                    pop_empl = pop_empl.append({'commune_id': j, 'voronoi_id': i, 'empl': np.nansum(empl), 'pop': np.nansum(pop)}, ignore_index=True)
            else:
                continue

    # Print array shapes to compare
    print(f"cout_r: {cout_r.shape}")
    print(f"pairs: {pairs.shape}")
    print(f"pop_empl: {pop_empl.shape}")

    # Step 3 complete exploded matrix
    # Initialize the OD matrix DataFrame with zeros or NaNs
    tuples = list(zip(pairs['voronoi_id'], pairs['commune_id']))
    multi_index = pd.MultiIndex.from_tuples(tuples, names=['voronoi_id', 'commune_id'])
    od_matrix = pd.DataFrame(index=multi_index, columns=multi_index).fillna(0)

    # Handle raster without values
    # Drop pairs with 0 pop or empl

    set_id_destination = [col[1] for col in od_matrix.columns]

    # Get unique values from the second level of the index
    unique_values_second_index = od_matrix.index.get_level_values(1).unique()

    # Iterate over each cell in the od_matrix to fill it with corresponding values from other_matrix
    for commune_id_origin in unique_values_second_index:
    #for (polygon_id_o, commune_id_o), _ in tqdm(od_matrix.index.to_series().iteritems(), desc='Allocating unit_values to OD matrix'):

        # Extract the row for commune_id_o
        row_values = cout_r.loc[commune_id_origin]

        # Use the valid columns to extract values
        extracted_values = row_values[set_id_destination].to_numpy()

        # Create a boolean mask for rows where the second element of the index matches commune_id_o
        mask = od_matrix.index.get_level_values(1) == commune_id_origin

        # Update the rows in od_matrix where the mask is True
        od_matrix.loc[mask] = extracted_values


    """
    # Iterate over each cell in the od_matrix to fill it with corresponding values from other_matrix
    for (polygon_id_o, commune_id_o), _ in tqdm(od_matrix.index.to_series().iteritems(), desc='Allocating unit_values to OD matrix'):
        for (polygon_id_d, commune_id_d), _ in od_matrix.columns.to_series().iteritems():
            # Find the value in other_matrix for the commune_id pair
            value_to_add = cout_r.loc[commune_id_o, commune_id_d]
            # Add this value to the od_matrix cell
            od_matrix.loc[(polygon_id_o, commune_id_o), (polygon_id_d, commune_id_d)] += value_to_add
    """

    """
    od_matrix_np = od_matrix.to_numpy()
    cout_r_np = cout_r.to_numpy()

    # Assuming the order of rows and columns in od_matrix aligns with the indices in cout_r
    for i, (polygon_id_o, commune_id_o) in tqdm(enumerate(od_matrix.index), desc='Allocating unit_values to OD matrix'):
        # Index of commune_id_o in cout_r
        idx_o = cout_r.index.get_loc(commune_id_o)

        for j, (polygon_id_d, commune_id_d) in enumerate(od_matrix.columns):
            # Index of commune_id_d in cout_r
            idx_d = cout_r.columns.get_loc(commune_id_d)

            # Add the value from cout_r to od_matrix
            od_matrix_np[i, j] += cout_r_np[idx_o, idx_d]

    # Convert the updated NumPy array back to a DataFrame, if needed
    od_matrix = pd.DataFrame(od_matrix_np, index=od_matrix.index, columns=od_matrix.columns)
    """

    """
    # Allocate values to the OD matrix
    for idx, value in tqdm(cout_r.iterrows(), desc='Allocating unit_values to OD matrix'):
        print(value, idx)
        commune_id = idx
        # Update the OD matrix using the commune_id
        od_matrix.loc[pd.IndexSlice[:, commune_id], pd.IndexSlice[:, commune_id]] += value
    """

    pop_empl = pop_empl.set_index(['voronoi_id', 'commune_id'])
    for polygon_id, row in tqdm(pop_empl.iterrows(), desc='Allocating pop and empl to OD matrix'):
        # Multiply all values in the row/column
        od_matrix.loc[polygon_id] *= row['pop']
        od_matrix.loc[:, polygon_id] *= row['empl']

    # Step 4: Group the OD matrix by polygon_id
    # Reset the index to turn the MultiIndex into columns
    od_matrix_reset = od_matrix.reset_index()

    # Sum the values by 'polygon_id' for both the rows and columns
    od_grouped = od_matrix_reset.groupby('voronoi_id').sum()

    # Now od_grouped has 'polygon_id' as the index, but we still need to group the columns
    # First, transpose the DataFrame to apply the same operation on the columns
    od_grouped = od_grouped.T

    # Again group by 'polygon_id' and sum, then transpose back
    od_grouped = od_grouped.groupby('voronoi_id').sum().T

    # Drop column commune_id
    od_grouped = od_grouped.drop(columns='commune_id')

    # Set diagonal values to 0
    temp_sum = od_grouped.sum().sum()
    np.fill_diagonal(od_grouped.values, 0)
    # Compute the sum after changing the diagonal
    temp_sum2 = od_grouped.sum().sum()
    # Print difference
    print(f"Sum of OD matrix before {temp_sum} and after {temp_sum2}")

    # Save pd df to csv
    od_grouped.to_csv(r"data\traffic_flow\od\od_matrix_2020.csv")
    #odmat.to_csv(r"data\traffic_flow\od\od_matrix_raw.csv")

    # Print sum of all values in od df
    # Sum over all values in pd df
    sum_com = odmat.sum().sum()
    sum_poly = od_grouped.sum().sum()
    sum_com_frame = odmat_frame.sum().sum()
    print(f"Total trips before {sum_com_frame} ({odmat_frame.shape} communes) and after {sum_poly} ({od_grouped.shape} polygons)")
    print(f"Total trips before {sum_com} ({odmat.shape} communes) and after {sum_poly} ({od_grouped.shape} polygons)")

    # Sum all columns of od_grouped
    origin = od_grouped.sum(axis=1).reset_index()
    origin.colum = ["voronoi_id", "origin"]
    # Sum all rows of od_grouped
    destination = od_grouped.sum(axis=0)
    destination = destination.reset_index()

    # merge origin and destination to voronoidf based on voronoi_id
    voronoidf = voronoidf.merge(origin, how='left', left_on='ID_point', right_on='voronoi_id')
    voronoidf = voronoidf.merge(destination, how='left', left_on='ID_point', right_on='voronoi_id')
    voronoidf = voronoidf.rename(columns={'0_x': 'origin', '0_y': 'destination'})
    voronoidf.to_file(r"data\traffic_flow\od\OD_voronoidf.gpkg", driver="GPKG")


    # Same for odmat and commune_df
    origin_commune = odmat_frame.sum(axis=1).reset_index()
    origin_commune.colum = ["commune_id", "origin"]
    destination_commune = odmat_frame.sum(axis=0).reset_index()
    destination_commune.colum = ["commune_id", "destination"]
    commune_df = commune_df.merge(origin_commune, how='left', left_on='BFS', right_on='quelle_code')
    commune_df = commune_df.merge(destination_commune, how='left', left_on='BFS', right_on='ziel_code')
    commune_df = commune_df.rename(columns={'0_x': 'origin', '0_y': 'destination'})
    commune_df.to_file(r"data\traffic_flow\od\OD_commune_filtered.gpkg", driver="GPKG")


    #potential = np.outer(scen_pop_medium_tif.flatten().reshape(1, -1), scen_empl_medium_tif.flatten().reshape(1, -1))

    #commune_raster = commune_raster.flatten().reshape(1, -1)
    #unit_flow = pd.DataFrame(index=commune_raster, columns=commune_raster)
    # fill this using the values from cout_r
    # mutliply by potential = flow from each cell to each other cell

    # map id of each cell (municipality) to the id of the voronoi polygon it is in and group over this new index

    # know how much cell of which municipality each voronoi polygon contains
    # values in municipality row multiplied by amount of
    #od_mn = cont_v * pop_m * job_n
    """
    # Main loop - consider parallelizing this loop
    for m in range(vor_idx):
        mask_origin = voronoi_tif == m
        r_subset = commune_raster.astype(float)
        r_subset[~mask_origin]= np.nan
        com_idx = np.unique(r_subset)

        pop_m = scen_pop_medium_tif.astype(float)
        pop_m[~mask_origin] = np.nan

        for n in range(vor_idx):
            mask_destination = voronoi_tif == n
            job_n = scen_empl_medium_tif.astype(float)
            job_n[~mask_destination] = np.nan

            if r_subset.size == 0:
                continue

            for i in com_idx:
                mask_commune = r_subset == i

            # Vectorized computations replacing nested loops
            cont_r = compute_cont_r(odmat, popvec, jobvec)
            cont_v = compute_cont_v(cont_r, pop_m, job_n)

            od_mn[m, n] = cont_v * np.nansum(pop_m * job_n)
        #od_mn[m, :] = cont_v * pop_m * job_n
    """


    """

        if r_subset.size == 0:
            continue

        # Overlay scen_tif and municipality_tif
        pop_m = scen_pop_medium_tif[mask]
        #pop_m = voronoidf.loc['ID_point'==m]['s'+scen+'_pop']
        for n in vor_idx:
            # Overlay scen_tif and municipality_tif
            job_n = scen_empl_medium_tif[mask]
            #job_n = voronoidf.loc['ID_point'==n]['s'+scen+'_empl']
            Cont_v=0
            for r in r_subset:
                ii=0
                Cont_r =0
                #for i in range(odmat.shape[0]):
                for i in odmat.index:
                    pop_i = popvec[ii]
                    ii=ii+1
                    jj=0
                    #for j in range(odmat.shape[0]):
                    for j in odmat.index:
                        job_j = jobvec[jj]
                        jj=jj+1
                        Cont_r = odmat.loc[i, j] / (pop_i * job_j)
                Cont_v = Cont_v + Cont_r
            od_mn[m, n] = Cont_v*pop_m*job_n
    """
    return od_grouped


def GetVoronoiOD_multi_old(scen_empl_path, scen_pop_path, voronoi_tif_path):

    popvec = GetCommunePopulation(y0="2021")
    jobvec = GetCommuneEmployment(y0=2021)
    od = GetHighwayPHDemandPerCommune()
    odmat = GetODMatrix(od)

    # This function returns a np array of raster data storing the bfs number of the commune in each cell
    commune_raster, commune_df = GetCommuneShapes(raster_path=voronoi_tif_path)

    if jobvec.shape[0] != odmat.shape[0]:
        print("Error: The number of communes in the OD matrix and the number of communes in the employment data do not match.")

    # Open scenario (medium) raster data    (low = band 2, high = band 3)
    with rasterio.open(scen_pop_path) as src:
        # Read the raster into a NumPy array (assuming you want the first band)
        scen_pop_medium_tif = src.read(1)

    with rasterio.open(scen_empl_path) as src:
        # Read the raster into a NumPy array (assuming you want the first band)
        scen_empl_medium_tif = src.read(1)

    # Step 1: generate unit_flow matrix from each commune to each other commune
    cout_r = odmat / np.outer(popvec, jobvec)

    # Directory path to developments
    directory_path = "data/Network/travel_time/developments/"

    # List to hold extracted values
    xx_values = []

    # Iterate through files in the directory
    for filename in os.listdir(directory_path):
        # Check if the filename matches the pattern 'devXX_source_id_raster.tif'
        match = re.match(r'dev(\d+)_source_id_raster\.tif', filename)
        if match:
            # Extract XX value and add to the list
            xx = match.group(1)
            xx_values.append(xx)

    # Convert values to integers if needed
    xx_values = [int(xx) for xx in xx_values]
    # print(xx_values)

    for xx in tqdm(xx_values, desc='Processing Voronoi IDs'):
        # Construct the file path
        file_path = f"{directory_path}dev{xx}_source_id_raster.tif"

        # Open the file with rasterio
        with rasterio.open(file_path) as src:
            # Read the raster data
            voronoi_tif = src.read(1)

        unique_voronoi_id = np.sort(np.unique(voronoi_tif))

        # Step 2: Get all pairs of combinations from communes to polygons
        unique_commune_id = np.sort(np.unique(commune_raster))
        pairs = pd.DataFrame(columns=['commune_id', 'voronoi_id'])
        pop_empl = pd.DataFrame(columns=['commune_id', 'voronoi_id', "empl", "pop"])

        for i in unique_voronoi_id:
            # Get the voronoi raster
            mask_voronoi = voronoi_tif == i
            for j in unique_commune_id:
                if j > 0:
                    # Get the commune raster
                    mask_commune = commune_raster == j
                    # Combined mask
                    mask = mask_commune & mask_voronoi
                    # Check if there are overlaying values
                    if np.nansum(mask) > 0:
                        pairs = pairs.append({'commune_id': j, 'voronoi_id': i}, ignore_index=True)

                        # Get the population and employment values
                        pop = scen_pop_medium_tif[mask]
                        empl = scen_empl_medium_tif[mask]
                        pop_empl = pop_empl.append({'commune_id': j, 'voronoi_id': i, 'empl': np.nansum(empl), 'pop': np.nansum(pop)}, ignore_index=True)
                else:
                    continue

        # Step 3 complete exploded matrix
        # Initialize the OD matrix DataFrame with zeros or NaNs
        tuples = list(zip(pairs['voronoi_id'], pairs['commune_id']))
        multi_index = pd.MultiIndex.from_tuples(tuples, names=['voronoi_id', 'commune_id'])
        od_matrix = pd.DataFrame(index=multi_index, columns=multi_index).fillna(0)

        # Handle raster without values
        # Drop pairs with 0 pop or empl

        # Get the set of destination commune id
        set_id_destination = [col[1] for col in od_matrix.columns]

        # Get unique values from the second level of the index
        unique_values_second_index = od_matrix.index.get_level_values(1).unique()

        # Iterate over each cell in the od_matrix to fill it with corresponding values from other_matrix
        for commune_id_origin in unique_values_second_index:
            # for (polygon_id_o, commune_id_o), _ in tqdm(od_matrix.index.to_series().iteritems(), desc='Allocating unit_values to OD matrix'):

            # Extract the row for commune_id_o
            row_values = cout_r.loc[commune_id_origin]

            # Use the valid columns to extract values
            extracted_values = row_values[set_id_destination].to_numpy()

            # Create a boolean mask for rows where the second element of the index matches commune_id_o
            mask = od_matrix.index.get_level_values(1) == commune_id_origin

            # Update the rows in od_matrix where the mask is True
            od_matrix.loc[mask] = extracted_values

        pop_empl = pop_empl.set_index(['voronoi_id', 'commune_id'])
        for polygon_id, row in pop_empl.iterrows():
            # Multiply all values in the row/column
            od_matrix.loc[polygon_id] *= row['pop']
            od_matrix.loc[:, polygon_id] *= row['empl']

        # Step 4: Group the OD matrix by polygon_id
        # Reset the index to turn the MultiIndex into columns
        od_matrix_reset = od_matrix.reset_index()

        # Sum the values by 'polygon_id' for both the rows and columns
        od_grouped = od_matrix_reset.groupby('voronoi_id').sum()

        # Now od_grouped has 'polygon_id' as the index, but we still need to group the columns
        # First, transpose the DataFrame to apply the same operation on the columns
        od_grouped = od_grouped.T

        # Again group by 'polygon_id' and sum, then transpose back
        od_grouped = od_grouped.groupby('voronoi_id').sum().T

        # Drop column commune_id
        od_grouped = od_grouped.drop(columns='commune_id')

        # Set diagonal values to 0
        np.fill_diagonal(od_grouped.values, 0)

        # Save pd df to csv
        od_grouped.to_csv(fr"data\traffic_flow\od\developments\od_matrix_dev{xx}.csv")
        #odmat.to_csv(r"data\traffic_flow\od\od_matrix_raw.csv")

    return

def GetCatchmentOD_old():
    # Other parts of the function remain unchanged...

    # for each of these scenarios make an own copy of od_matrix named od_matrix+scen
    for scen in pop_empl_scenarios:
        print(f"Processing scenario {scen}")
        od_matrix_temp = od_matrix.copy()

        for polygon_id, row in tqdm(pop_empl.iterrows(), desc='Allocating pop and empl to OD matrix'):
            # Multiply all values in the row/column
            od_matrix_temp.loc[polygon_id] *= row[f'{scen}']
            od_matrix_temp.loc[:, polygon_id] *= row[f'{scen}']

        # Save the ungrouped OD matrix as a CSV
        ungrouped_od_matrix_path = fr"data/traffic_flow/od/rail/od_matrix_temp_{scen}.csv"
        od_matrix_temp.to_csv(ungrouped_od_matrix_path)
        print(f"Saved ungrouped OD matrix for scenario {scen} at {ungrouped_od_matrix_path}")

        ###############################################################################################################################
        # Step 4: Group the OD matrix by polygon_id
        # Reset the index to turn the MultiIndex into columns
        od_matrix_reset = od_matrix_temp.reset_index()

        # Sum the values by 'polygon_id' for both the rows and columns
        od_grouped = od_matrix_reset.groupby('catchment_id').sum()

        # Now od_grouped has 'polygon_id' as the index, but we still need to group the columns
        # First, transpose the DataFrame to apply the same operation on the columns
        od_grouped = od_grouped.T

        # Again group by 'polygon_id' and sum, then transpose back
        od_grouped = od_grouped.groupby('catchment_id').sum().T

        # Drop column commune_id
        od_grouped = od_grouped.drop(columns='commune_id')

        # Set diagonal values to 0
        temp_sum = od_grouped.sum().sum()
        np.fill_diagonal(od_grouped.values, 0)
        # Compute the sum after changing the diagonal
        temp_sum2 = od_grouped.sum().sum()
        # Print difference
        print(f"Sum of OD matrix before {temp_sum} and after {temp_sum2} removing diagonal values")

        # Save the grouped OD matrix
        grouped_od_matrix_path = fr"data/traffic_flow/od/rail/od_matrix_{scen}.csv"
        od_grouped.to_csv(grouped_od_matrix_path)
        print(f"Saved grouped OD matrix for scenario {scen} at {grouped_od_matrix_path}")

        # Print sum of all values in od df
        # Sum over all values in pd df
        sum_com = odmat.sum().sum()
        sum_poly = od_grouped.sum().sum()
        sum_com_frame = odmat_frame.sum().sum()
        print(
            f"Total trips before {sum_com_frame} ({odmat_frame.shape} communes) and after {sum_poly} ({od_grouped.shape} polygons)")
        print(
            f"Total trips before {sum_com} ({odmat.shape} communes) and after {sum_poly} ({od_grouped.shape} polygons)")

        # Sum all columns of od_grouped
        origin = od_grouped.sum(axis=1).reset_index()
        origin.colum = ["catchment_id", "origin"]
        # Sum all rows of od_grouped
        destination = od_grouped.sum(axis=0)
        destination = destination.reset_index()

        # merge origin and destination to catchmentdf based on catchment_id
        # Make a copy of catchmentdf
        catchmentdf_temp = catchmentdf.copy()
        catchmentdf_temp.rename(columns={'id': 'ID_point'}, inplace=True)
        catchmentdf_temp = catchmentdf_temp.merge(origin, how='left', left_on='ID_point', right_on='catchment_id')
        catchmentdf_temp = catchmentdf_temp.merge(destination, how='left', left_on='ID_point', right_on='catchment_id')
        catchmentdf_temp = catchmentdf_temp.rename(columns={'0_x': 'origin', '0_y': 'destination'})
        catchmentdf_temp.to_file(fr"data/traffic_flow/od/catchment_id_{scen}.gpkg", driver="GPKG")

        # Same for odmat and commune_df
        if scen == "20":
            origin_commune = odmat_frame.sum(axis=1).reset_index()
            origin_commune.colum = ["commune_id", "origin"]
            destination_commune = odmat_frame.sum(axis=0).reset_index()
            destination_commune.colum = ["commune_id", "destination"]
            commune_df = commune_df.merge(origin_commune, how='left', left_on='BFS', right_on='quelle_code')
            commune_df = commune_df.merge(destination_commune, how='left', left_on='BFS', right_on='ziel_code')
            commune_df = commune_df.rename(columns={'0_x': 'origin', '0_y': 'destination'})
            commune_df.to_file(r"data/traffic_flow/od/OD_commune_filtered.gpkg", driver="GPKG")

    return

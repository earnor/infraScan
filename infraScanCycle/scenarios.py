import pandas as pd
from data_import import *
from rasterio.features import geometry_mask
from rasterstats import zonal_stats

import boto3
import rasterio as rio
from rasterio.session import AWSSession
from shapely.validation import make_valid

def future_scenario_zuerich_2022(df_input):
    """
    This function represents the changes in population and employment based on data from the canton of Zürich.
    :param df: DataFrame defining the growth by region
    :param lim: List of coordinates defining the perimeter investigated in the analysis
    :return:
    """

    # import boundaries of each region
    boundaries = gpd.read_file(r"data/Scenario/Boundaries/Gemeindegrenzen/UP_BEZIRKE_F.shp")

    # group per location and time
    df = df_input.copy()
    df = df.groupby(["bezirk", "jahr"])[["anzahl"]].sum()
    df = df.reset_index()

    # filter regions of interest
    df = df[(df["jahr"] >= 2020)] # & (df["bezirk"].isin(['Bülach', 'Hinwil', 'Meilen', 'Pfäffikon', 'Uster', 'Zürich', 'Winterthur']))]
    df = df.pivot(index='bezirk', columns='jahr', values='anzahl')


    def relative_number(x):
        return x / df.iloc[:, 0]
    # Compute the relative growth in each district and for each year from 2020 to 2050
    df_rel = df.apply(relative_number,args=(), axis=0)

    # plot development per region in one
    boundaries = boundaries.merge(df_rel, left_on="BEZIRK", right_on="bezirk", how="right")

    df_scenario = boundaries[["BEZIRK", "geometry", 2050]]
    df_scenario.rename(columns={2050: 's1_pop'}, inplace=True)

    # import boundaries of each region
    empl_dev = pd.read_csv(r"data/Scenario/KANTON_ZUERICH_596.csv", sep=";", encoding='unicode_escape')
    empl_dev = empl_dev[["BFS_NR", "GEBIET_NAME", "INDIKATOR_JAHR", "INDIKATOR_VALUE"]]

    bfs_nr = gpd.read_file(r"data/Scenario/Boundaries/Gemeindegrenzen/UP_GEMEINDEN_F.shp")
    bfs_nr = bfs_nr[["BFS", "BEZIRKSNAM"]]

    empl_dev = empl_dev.merge(right=bfs_nr, left_on="BFS_NR", right_on="BFS", how="left")
    empl_dev = empl_dev.drop_duplicates(subset=['BFS_NR', 'GEBIET_NAME', 'INDIKATOR_JAHR'], keep='first')
    empl_dev = empl_dev.rename(columns={"BEZIRKSNAM":"bezirk", "INDIKATOR_JAHR":"jahr", "INDIKATOR_VALUE":"anzahl"})

    # group per location and time
    empl_dev = empl_dev.groupby(["bezirk", "jahr"])[["anzahl"]].sum()
    empl_dev = empl_dev.reset_index()
    #print(empl_dev.head(10).to_string())

    empl_dev = empl_dev[(empl_dev["jahr"] == 2011) | (empl_dev["jahr"] == 2021)]
    empl_dev = empl_dev.pivot(index='bezirk', columns='jahr', values='anzahl').reset_index()

    # Rename the columns
    empl_dev.columns.name = None
    empl_dev.columns = ['bezirk', '2011', '2021']
    empl_dev["rel_10y"] = empl_dev["2021"] / empl_dev["2011"] - 1
    #empl_dev["empl50"] = (empl_dev["2021"] * (1 + empl_dev["rel_10y"] * 2.9)).astype(int)
    empl_dev["s1_empl"] = (1 + empl_dev["rel_10y"] * 2.9)
    empl_dev = empl_dev[["bezirk", "s1_empl"]]

    print(empl_dev.head(10).to_string())

    # plot development per region in one
    df_scenario = df_scenario.merge(empl_dev, left_on="BEZIRK", right_on="bezirk", how="right")
    #print(df_scenraio.head(10).to_string())

    # df_scenraio = boundaries[["BEZIRK", "geometry", 2050]]

    df_scenario["s2_pop"] = df_scenario["s1_pop"] - (df_scenario["s1_pop"] -1) / 3
    df_scenario["s3_pop"] = df_scenario["s1_pop"] + (df_scenario["s1_pop"] - 1) / 3

    df_scenario["s2_empl"] = df_scenario["s1_empl"] - (df_scenario["s1_empl"] -1) / 3
    df_scenario["s3_empl"] = df_scenario["s1_empl"] + (df_scenario["s1_empl"] - 1) / 3

    print(df_scenario.columns)

    df_scenario.to_file(r"data/temp/data_scenario_n.shp")
    return


def scenario_to_raster(frame=False):
    # Load the shapefile
    scenario_polygon = gpd.read_file(r"data/temp/data_scenario_n.shp")

    if frame != False:
        # Create a bounding box polygon
        bounding_poly = box(frame[0], frame[1], frame[2], frame[3])
        n_cols = (frame[2] - frame[0]) / 100
        n_rows = (frame[3] - frame[1]) / 100

        # Calculate the difference polygon
        # This will be the area in the bounding box not covered by existing polygons

        difference_poly = bounding_poly
        for geom in scenario_polygon['geometry']:
            if geom is not None and not geom.is_empty:
                geom = make_valid(geom)
                difference_poly = difference_poly.difference(geom)


        # Calculate the mean values for the three columns
        #mean_values = scenario_polygon.mean()

        # Create a new row for the difference polygon
        #new_row = {'geometry': difference_poly, 's1_pop': mean_values['s1_pop'], 's2_pop': mean_values['s2_pop'],
        #           's3_pop': mean_values['s3_pop'], 's1_empl': mean_values['s1_empl'], 's2_empl': mean_values['s2_empl'], 's3_empl': mean_values['s3_empl']}
        new_row = {'geometry': difference_poly, 's1_pop': scenario_polygon['s1_pop'].mean(),
                   's2_pop': scenario_polygon['s2_pop'].mean(), 's3_pop': scenario_polygon['s3_pop'].mean(),
                   's1_empl': scenario_polygon['s1_empl'].mean(), 's2_empl': scenario_polygon['s2_empl'].mean(),
                   's3_empl': scenario_polygon['s3_empl'].mean()}
        print("New row added")
        #scenario_polygon = scenario_polygon.append(new_row, ignore_index=True)
        scenario_polygon = gpd.GeoDataFrame(pd.concat([pd.DataFrame(scenario_polygon), pd.DataFrame(pd.Series(new_row)).T], ignore_index=True))

    growth_rate_columns_pop = ["s1_pop", "s2_pop", "s3_pop"]
    path_pop = r"data/independent_variable/processed/raw/pop20.tif"

    growth_rate_columns_empl = ["s1_empl", "s2_empl", "s3_empl"]
    path_empl = r"data/independent_variable/processed/raw/empl20.tif"

    growth_to_tif(scenario_polygon, path=path_pop, columns=growth_rate_columns_pop)
    growth_to_tif(scenario_polygon, path=path_empl, columns=growth_rate_columns_empl)

    return


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
                output_tiff = f'data/independent_variable/processed/scenario/{col}.tif'
                with rasterio.open(output_tiff, 'w', **src.profile) as dst:
                    dst.write(modified_raster, 1)
    return


def scenario_to_voronoi(polygons_gdf, euclidean=False):

    # List of your raster files
    raster_path = r"data/independent_variable/processed/scenario"
    raster_files = ['s1_empl.tif', 's2_empl.tif', 's3_empl.tif', 's1_pop.tif', 's2_pop.tif', 's3_pop.tif']

    # Loop over the raster files and calculate zonal stats for each
    for i, raster_file in enumerate(raster_files, start=1):
        with rasterio.open(raster_path+'//' + raster_file) as src:
            affine = src.transform
            array = src.read(1)
            nodata = src.nodata
            nodata = -999
            #polygons_gdf[(raster_file).removesuffix('.tif')] = pd.DataFrame(zonal_stats(vectors=polygons_gdf['geometry'], raster=src, stats='mean'))['mean']
            # Calculate zonal statistics
            stats = zonal_stats(polygons_gdf, array, affine=affine, stats=['sum'], nodata=nodata, all_touched=True)


            # Extract the 'sum' values and assign them to a new column in the geodataframe
            polygons_gdf[(raster_file).removesuffix('.tif')] = [stat['sum'] for stat in stats]

    # Now polygons_gdf has new columns with the sum of raster values for each polygon
    # You can save this geodataframe to a new file if desired
    #print(polygons_gdf.head(10).to_string())
    if euclidean:
        polygons_gdf.to_file(r"data/Voronoi/voronoi_developments_euclidian_values.shp")
    else:
        polygons_gdf.to_file(r"data/Voronoi/voronoi_developments_tt_values.shp")

    return


def plot_scenarios(corridor_polygon=None):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    from matplotlib.patches import Patch
    import numpy as np

    scenario_gdf = gpd.read_file(r"data/temp/data_scenario_n.shp")
    voronoi_gdf  = gpd.read_file(r"data/Voronoi/voronoi_developments_euclidian_values.shp")

    if scenario_gdf.crs is None: scenario_gdf = scenario_gdf.set_crs("EPSG:2056")
    if voronoi_gdf.crs  is None: voronoi_gdf  = voronoi_gdf.set_crs("EPSG:2056")

    scenarios   = ['s1', 's2', 's3']
    scen_labels = ['S1 — High growth', 'S2 — Medium growth', 'S3 — Low growth']
    variables   = ['pop', 'empl']
    var_labels  = ['Population (relative to 2020)', 'Employment (relative to 2011)']

    # ---------------------------------------------------------------
    # FIGURE 1: District-level growth rates  (2 rows × 3 cols)
    # ---------------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 3, figsize=(20, 12))

    for row, (var, var_label) in enumerate(zip(variables, var_labels)):
        cols = [f'{s}_{var}' for s in scenarios]
        vmin = scenario_gdf[cols].min().min()
        vmax = scenario_gdf[cols].max().max()
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.RdYlGn

        for col_idx, (scen, scen_label, col) in enumerate(zip(scenarios, scen_labels, cols)):
            ax = axes1[row, col_idx]
            scenario_gdf.plot(
                column=col, ax=ax, cmap=cmap, norm=norm,
                edgecolor='white', linewidth=0.6, alpha=0.85
            )

            # District labels
            for _, row_data in scenario_gdf.iterrows():
                if row_data.geometry is not None and not row_data.geometry.is_empty:
                    cx, cy = row_data.geometry.centroid.x, row_data.geometry.centroid.y
                    val = row_data[col]
                    ax.annotate(f"{row_data['BEZIRK']}\n×{val:.2f}",
                                xy=(cx, cy), ha='center', va='center',
                                fontsize=6, color='black')

            if corridor_polygon is not None:
                gpd.GeoDataFrame({'geometry': [corridor_polygon]}, crs="EPSG:2056").boundary.plot(
                    ax=ax, color='black', linewidth=1.5, linestyle='--', zorder=5)

            ax.set_title(f'{scen_label}\n{var_label}', fontsize=10, fontweight='bold')
            ax.set_aspect('equal')
            ax.axis('off')

        # Shared colorbar per row
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig1.colorbar(sm, ax=axes1[row, :], shrink=0.6, pad=0.02,
                             label=f'Growth multiplier — {var_label}')

    fig1.suptitle('Cantonal Growth Scenarios 2020→2050\nby District (Canton of Zürich)',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('data/Voronoi/plot_scenarios_districts.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved → data/Voronoi/plot_scenarios_districts.png")

    # ---------------------------------------------------------------
    # FIGURE 2: Voronoi — aggregated demand per catchment (2 rows × 3 cols)
    # ---------------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 3, figsize=(20, 12))

    for row, (var, var_label) in enumerate(zip(variables, var_labels)):
        cols = [f'{s}_{var}' for s in scenarios]

        # Drop NaN for colorbar range
        valid = voronoi_gdf[cols].replace(0, np.nan).dropna()
        vmin  = valid.min().min()
        vmax  = valid.max().max()
        norm  = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap  = cm.YlOrRd

        for col_idx, (scen, scen_label, col) in enumerate(zip(scenarios, scen_labels, cols)):
            ax = axes2[row, col_idx]
            voronoi_gdf.plot(
                column=col, ax=ax, cmap=cmap, norm=norm,
                edgecolor='gray', linewidth=0.4, alpha=0.85,
                missing_kwds={'color': 'lightgray', 'label': 'No data'}
            )

            if corridor_polygon is not None:
                gpd.GeoDataFrame({'geometry': [corridor_polygon]}, crs="EPSG:2056").boundary.plot(
                    ax=ax, color='black', linewidth=1.5, linestyle='--', zorder=5)

            ax.set_title(f'{scen_label}\n{var_label}', fontsize=10, fontweight='bold')
            ax.set_aspect('equal')
            ax.axis('off')

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig2.colorbar(sm, ax=axes2[row, :], shrink=0.6, pad=0.02,
                      label=f'Total {var} per Voronoi catchment')

    fig2.suptitle('Scenario Demand aggregated to Voronoi Catchments\n(Euclidean access areas)',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('data/Voronoi/plot_scenarios_voronoi.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved → data/Voronoi/plot_scenarios_voronoi.png")

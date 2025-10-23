infraScanRail — Overview and Run Guide

- Purpose: Evaluate potential rail infrastructure developments by generating candidate links, computing travel-time impacts across scenarios, monetizing benefits, estimating costs, and visualizing results.
- Entry point: call `infrascanrail()` from `infraScanRail/main.py`.

Quick Start

- Prepare data: Ensure the `data` directory contains the referenced network, OD, and shape files (see Paths and Inputs). Most paths are configured in `infraScanRail/paths.py`.
- Configure settings: Adjust `infraScanRail/settings.py` for network selection, OD logic, scenario settings, and caching.
- Create env: Use `infraScanRail/environment.yml` or `infraScanRail/requirements.txt` to install dependencies.
- Run: `python -c "from infraScanRail.main import infrascanrail; infrascanrail()"`

Execution Flow (what runs, order, data in/out)

1) Initialize focus area
- Code: `create_focus_area()` in `main.py`.
- Input: hard-coded LV95 bounds in `main.py`.
- Output: inner/outer corridor polygons stored via helper `save_focus_area_shapefile` (in codebase) for later clipping/plots.

2) Import basic context data
- Code: `get_lake_data()`, `import_cities()` in `data_import.py`.
- Input: shapefiles and CSVs under `data/landuse_landcover`, `data/manually_gathered_data`.
- Output: small helper layers written to `data/landuse_landcover/processed` and `data/manually_gathered_data`.

3) Import and preprocess rail network
- Code: `import_process_network(use_cache)` in `main.py` → uses `data_import.py` and helpers.
- Inputs (from `paths.py`):
  - Rail services GeoPackages: one of
    - `paths.RAIL_SERVICES_AK2035_PATH`
    - `paths.RAIL_SERVICES_AK2035_EXTENDED_PATH`
    - `paths.RAIL_SERVICES_2024_PATH`
    - `paths.RAIL_SERVICES_AK2024_EXTENDED_PATH`
  - Station nodes: `paths.RAIL_NODES_PATH` (CSV)
- Key steps (outputs under `data/Network/processed`):
  - Reformat nodes/edges, compute endpoints, split/clean geometry: `reformat_rail_nodes()`, `reformat_rail_edges()`
  - Create service variants: `create_railway_services_AK2035[_extended]()` / `create_railway_services_2024_extended()`
  - Attach construction attributes from `data/Network/Rail-Service_Link_construction_cost.csv`: `add_construction_info_to_network()`
  - Persist processed points as `points.gpkg`, edges as `edges.gpkg` and derived files used downstream.

4) Generate developments (candidate new links)
- Code: `generate_infra_development(use_cache, mod_type)` in `main.py` (functions implemented in `generate_infrastructure.py` and `scoring.py`).
- Two mechanisms (controlled by `settings.infra_generation_modification_type`):
  - Extend lines near current endpoints: `generate_rail_edges(n, radius)` → writes `new_links.gpkg`, refines to `filtered_new_links.gpkg`, in-corridor subset, then `calculate_new_service_time()` annotates lengths and times; optionally augments with new S-Bahn lines derived from bus/S-Bahn geometries.
  - Add direct connections for missing links: builds a graph from the current network (`prepare_Graph`) and computes plausible new lines (`get_missing_connections`, `generate_new_railway_lines`, `export_new_railway_lines`). These are added into the candidate link set (`add_railway_lines_to_new_links`).
- Finally, the network is updated with new links and one network per development is written to `paths.DEVELOPMENT_DIRECTORY` (GeoPackages). A combined network is saved to `paths.NETWORK_WITH_ALL_MODIFICATIONS`.

5) Build OD demand
- Selection controlled by `settings.OD_type`:
  - `'canton_ZH'`: aggregate communal public-transport OD to station-to-station OD.
    - Code: `getStationOD(use_cache, perimeter_stations, only_demand_from_to_perimeter)` in `main.py`.
    - Inputs: communal OD from `data/_basic_data/KTZH_00001982_00003903.xlsx` and commune→station mapping `paths.COMMUNE_TO_STATION_PATH`.
    - Output: station OD CSV at `paths.OD_STATIONS_KT_ZH_PATH`.
  - `'pt_catchment_perimeter'`: build station catchments from bus/rail stop buffers and derive OD from catchment assignment.
    - Code: `get_catchment()` and `GetCatchmentOD()` in `catchment_pt.py` and `traveltime_delay.py`.
    - Inputs: ZVV stops/lines, rail stations, corridor polygons.
    - Outputs: catchment polygons and an OD table under `data/traffic_flow/od/rail`.

6) Create travel-time graphs and compute OD travel times
- Code: `create_travel_time_graphs(network_selection, use_cache, dev_id_lookup_table)` in `main.py` backed by `TT_Delay.py`.
- Inputs:
  - Status-quo network: path resolved via `paths.get_rail_services_path(settings.rail_network)`.
  - Development networks: every `.gpkg` in `paths.DEVELOPMENT_DIRECTORY`.
- Process:
  - `create_graphs_from_directories` loads each network, builds a directed graph with “entry/exit/sub” nodes per station and edges weighted by minutes (includes comfort-weighted change time from `cost_parameters.py`).
  - `calculate_od_pairs_with_times_by_graph` runs all-pairs Dijkstra between stations for each graph; returns one OD-time DataFrame per graph (status quo and per development).
- Output cache: `data/Network/travel_time/cache/od_times.pkl` (when `settings.use_cache_traveltime_graph=True`).

7) Generate scenarios (population/modal split)
- Code: `get_random_scenarios(...)` in `random_scenarios.py` (used when `OD_type='canton_ZH'`).
- Inputs: population references (`paths.POPULATION_SCENARIO_*`) and Eurostat growth; produces stochastic paths using LHS.
- Output cache: `paths.RANDOM_SCENARIO_CACHE_PATH`.

8) Total travel times and monetization
- Code (in `main.py` using helpers from `TT_Delay.py`/`scoring.py`):
  - `calculate_total_travel_times` aggregates scenario-weighted OD times to Total Travel Time (TTT) for status quo and for each development.
  - `calculate_monetized_tt_savings` compares TTTs, multiplies by `VTTS` from `cost_parameters.py`, and writes `data/costs/traveltime_savings.csv` along with in-memory lists: `dev_list`, `monetized_tt`, `scenario_list`.
- Output cache: `paths.TTS_CACHE` when enabled.

9) Costs, discounting, aggregation
- Code: in `scoring.py`:
  - `construction_costs(...)` reads `data/Network/Rail-Service_Link_construction_cost.csv`, merges new link info, and computes construction and maintenance costs using parameters from `cost_parameters.py`.
  - `create_cost_and_benefit_df(...)` composes a per-development cost/benefit timeline; `discounting(...)` applies the discount rate; `rearange_costs(...)` aggregates and reshapes.
- Key outputs (see `paths.py`):
  - `paths.TOTAL_COST_RAW`, `paths.COST_AND_BENEFITS_DISCOUNTED`, and `paths.TOTAL_COST_WITH_GEOMETRY` (CSV with geometry joins used for plotting/GUI).

10) Visualization and (optional) GUI
- Code: `visualize_results()` in `scoring.py` uses `plots.py` to export development-specific layers and tables. Static plots go to `plots/`.
- Optional GUI: `display_results.py` → `create_scenario_analysis_viewer(paths.TOTAL_COST_WITH_GEOMETRY)` to browse scenarios in a table.

Run Order (default pipeline)

- Set `settings.py` as needed (notably `rail_network`, caches, `infra_generation_modification_type`, `OD_type`, scenario years, and flags).
- Run `infrascanrail()` which performs, in order:
  1. Focus area setup
  2. Context data import (lakes, city labels)
  3. Network import and preprocessing
  4. Development generation (extend lines, new direct connections) and per-development network export
  5. OD build (by stations or by catchment, depending on `OD_type`)
  6. Travel-time graph creation and OD time computation (status quo and each development)
  7. Scenario generation/load and TTT aggregation
  8. Monetized travel-time savings
  9. Cost estimation, discounting, aggregation
  10. Visualization and outputs

Key Modules (what they do and main inputs)

- `main.py`
  - Orchestrates the full pipeline; provides `infrascanrail()`.
  - Calls helpers to create networks, OD, scenarios, travel times, costs, and plots.

- `settings.py`
  - Central switches: network selection (`rail_network`), cache flags, `infra_generation_modification_type` (`EXTEND_LINES`, `NEW_DIRECT_CONNECTIONS`, `ALL`), `OD_type` (`canton_ZH` or `pt_catchment_perimeter`), scenario span and counts, plotting toggles.
  - Spatial parameters: corridor polygon and raster size.

- `paths.py`
  - Absolute `MAIN` folder and data file constants (all inputs/outputs). Function `get_rail_services_path(rail_network_settings)` resolves the active service gpkg.

- `data_import.py`
  - Import helpers: lakes, city labels, corridor polygon creation.
  - Network preparation: split/normalize edges/nodes, create/add lines (`add_new_line`), create “extended” service sets for 2024/2035; write `edges.gpkg`/`points.gpkg`.

- `generate_infrastructure.py`
  - Generate candidate links near endpoints (`generate_rail_edges`), filter redundancy (`filter_unnecessary_links`), compute realistic travel time per link (`calculate_new_service_time`).
  - Identify missing direct connections and generate new line geometries (`prepare_Graph`, `get_missing_connections`, `generate_new_railway_lines`, `export_new_railway_lines`, `add_railway_lines_to_new_links`).
  - Update the combined network with new links and export one network per development (GeoPackages in `paths.DEVELOPMENT_DIRECTORY`).

- `TT_Delay.py`
  - Build directed graphs with boarding/alighting/change penalties (`create_directed_graph`).
  - Create graphs from network files (`create_graphs_from_directories`), compute all-pairs OD times per graph (`calculate_od_pairs_with_times_by_graph`).
  - Scenario processing utilities: OD preprocessing and Numba-accelerated weighted time aggregation.

- `scoring.py`
  - Edge splitting/merging utilities; compute costs (`construction_costs`), build cost/benefit timelines (`create_cost_and_benefit_df`, `discounting`), aggregate/reshape for plotting.
  - Misc. helpers used across network prep and plotting.

- `scenarios.py` and `random_scenarios.py`
  - Deterministic and stochastic population/employment scenario generation; cache under `paths.RANDOM_SCENARIO_CACHE_PATH`. `result_plots.py` provides analysis plots.

- `catchment_pt.py` and `traveltime_delay.py`
  - Build station catchments from S-Bahn/bus stops (buffering, overlap resolution), clip to inner boundary, and derive OD when `OD_type='pt_catchment_perimeter'`.

- `plots.py`, `plot_parameter.py`, `display_results.py`
  - Export scenario-specific cost/benefit tables with geometry, maps of developments, cumulative plots; optional Tkinter GUI to browse scenarios.

- `cost_parameters.py`
  - Economic parameters: VTTS, construction/maintenance cost rates, discount rate, valuation periods, comfort-weighted change time.

Paths and Inputs (selected)

- Network and nodes
  - `paths.RAIL_SERVICES_*` → status-quo rail service GeoPackages
  - `paths.RAIL_NODES_PATH` → Rail_Node.csv (station/node attributes)
  - Processed outputs under `data/Network/processed/` (points.gpkg, edges.gpkg, new_links.gpkg, developments/*.gpkg)

- OD and scenarios
  - Communal OD: `data/_basic_data/KTZH_00001982_00003903.xlsx`
  - Station OD out: `paths.OD_STATIONS_KT_ZH_PATH`
  - Scenario sources: `paths.POPULATION_SCENARIO_*`, `paths.POPULATION_PER_COMMUNE_ZH_2018`
  - Scenario cache dir: `paths.RANDOM_SCENARIO_CACHE_PATH`

- Costs
  - Construction input: `data/Network/Rail-Service_Link_construction_cost.csv`
  - Aggregated outputs: `paths.TOTAL_COST_*`, `paths.COST_AND_BENEFITS_DISCOUNTED`

Outputs (where to find results)

- Development networks: `paths.DEVELOPMENT_DIRECTORY` (one .gpkg per development)
- Travel time caches: `data/Network/travel_time/cache/*.pkl`
- TT savings: `data/costs/traveltime_savings.csv`
- Aggregated costs with geometry: `paths.TOTAL_COST_WITH_GEOMETRY`
- Plots and scenario layers: `plots/` and scenario-specific `.gpkg` exports created by `plots.py`

Notes and Tips

- Caching: Toggle `settings.use_cache_*` flags to skip recomputation of network prep, catchments, OD, scenarios, travel times, and TTT aggregation.
- Network selection: `settings.rail_network` controls which status-quo service file is used via `paths.get_rail_services_path`.
- Absolute path: `paths.MAIN` is set to an absolute Windows path; if you cloned the repo elsewhere, update it (or `cd` into `infraScanRail` before running) to ensure relative reads/writes work.
- OD choice: For most Zurich-focused workflows set `OD_type='canton_ZH'`. Use `'pt_catchment_perimeter'` only if the catchment-based OD derivation has been prepared.

Detailed Module Reference (file by file)

- main.py
  - Role: Orchestrator. Defines `infrascanrail()` which runs the entire pipeline end-to-end.
  - Key steps: focus area creation; data import; `import_process_network`; `generate_infra_development`; OD build (`getStationOD` or `GetCatchmentOD`); travel-time graphs (`create_travel_time_graphs`); scenario generation (`get_random_scenarios`); TTT and monetization (`compute_tts`); costs (`construction_costs`, `create_cost_and_benefit_df`, `discounting`, `rearange_costs`); visualization.
  - Inputs: Controlled via `settings.py` and `paths.py`.
  - Outputs: Runtimes log, caches in `data/Network/.../cache/`, costs in `data/costs`, plots under `plots/`.

- settings.py
  - Role: Central configuration flags and parameters: network selection (`rail_network`), cache toggles, infrastructure generation mode (`infra_generation_modification_type`), OD mode (`OD_type`), scenario settings (years, counts), plotting flags, corridor polygons.

- paths.py
  - Role: File path constants for all inputs/outputs and `get_rail_services_path(rail_network_settings)` resolver for the active rail services GeoPackage.
  - Note: `MAIN` is an absolute path; ensure it matches your local clone or `cd` into `infraScanRail` before running.

- cost_parameters.py
  - Role: Economic parameters used across scoring: VTTS, construction and maintenance unit costs, discount rate, valuation period, and comfort-weighted change time (boarding/alighting/change penalties in minutes).

- data_import.py
  - Role: Utility imports and network preparation helpers.
  - Functions:
    - `get_lake_data()`, `import_cities()`: import and persist auxiliary layers for plotting.
    - `polygon_from_points(...)`: build rectangular polygons from LV95 bounds.
    - `reformat_rail_edges(rail_network)`: reads selected rail services gpkg, computes endpoint points, attaches origin/destination coordinates, writes `data/Network/processed/edges.gpkg`.
    - `reformat_rail_nodes()`: simplifies the node set and writes processed station points to `data/Network/processed/points.gpkg` (intersection vs access points).
    - `add_new_line(...)`: programmatically appends a new service between given station names (creates both directions A/B, handles `Via`, sets travel times, frequencies, geometry).
    - `create_railway_services_2024_extended()`: example of extending the 2024 network with additional services using `add_new_line` and existing station points.

- generate_infrastructure.py
  - Role: Generate candidate links and networks for developments.
  - Extend lines around endpoints:
    - `generate_rail_edges(n, radius)`: identify endpoints in the current network and connect nearby stations (within `radius` km, up to `n` nearest) to create candidate links; writes intermediate `generated_nodeset.gpkg`, `endnodes.gpkg`, and lines.
    - `assign_services_to_generated_points(...)`: derives terminating services for endpoints.
    - `filter_unnecessary_links(rail_network)`: drops candidate links that are already part of existing S-lines; writes `filtered_new_links.gpkg`.
    - `calculate_new_service_time()`: splits S-Bahn lines at stops, builds a graph of segments, computes shortest path times per candidate link, and writes `paths.NEW_LINKS_UPDATED_PATH` with `shortest_path_length` and `time` columns.
  - New direct connections:
    - `prepare_Graph(df_network, df_points)`: builds an undirected MultiGraph of the current network, splits edges with `Via`, removes shortcut edges, tags nodes with station names, computes positions, and pickles graph+pos (`paths.GRAPH_POS_PATH`).
    - `get_missing_connections(...)`, `generate_new_railway_lines(...)`: detect gaps between centers and generate viable multi-station paths as potential new lines.
    - `export_new_railway_lines(new_lines, pos, file_path)`: export generated lines to a gpkg with attributes (start/end stations, node ids, path string, geometry).
    - `add_railway_lines_to_new_links(...)`: converts multi-station lines into per-segment candidate links, adds travel time (reuse if present, else derive from length), appends to `paths.NEW_LINKS_UPDATED_PATH`.
  - Network update and per-dev export:
    - `update_network_with_new_links(...)`, `update_stations(...)`, `create_network_foreach_dev()`: merge candidate links into the base network, update station set, and write one `.gpkg` per development to `paths.DEVELOPMENT_DIRECTORY`.

- TT_Delay.py
  - Role: Build directed graphs of the rail network (status quo and each development) and compute OD travel times.
  - Graph model:
    - `create_directed_graph(df, change_time)`: entry/exit nodes per station, sub-nodes per service-direction; edges represent in-vehicle travel, boarding/alighting (3 min), and intra-station transfers (comfort-weighted change time from `cost_parameters.py`).
  - Batch processing:
    - `create_graphs_from_directories(directories, n_jobs)`: read each `.gpkg` or `.csv`, convert to graphs, return list of `nx.DiGraph`.
    - `calculate_od_pairs_with_times_by_graph(graphs)`: for each graph, run all-pairs Dijkstra between station entry/exit nodes; returns a DataFrame of OD times with station names and a `graph_id`.
  - Scenario aggregation:
    - `preprocess_OD_matrix(...)`: aligns an OD table to station names and returns a matrix + index map.
    - `process_scenario_year_numba(...)` and `compute_weighted_times(...)`: fast, Numba-accelerated accumulation of scenario-weighted total times.

- scoring.py
  - Role: Data transformations for network edges, cost computation, and cost/benefit aggregation.
  - Utilities: `split_via_nodes(df)` expands edges with `Via` intermediate nodes into atomic segments; `merge_lines(df)` merges parallel segments ignoring direction and sums frequency/time.
  - Development ingestion: `read_development_files()` loads all `.gpkg` from `paths.DEVELOPMENT_DIRECTORY`, filters `new_dev == 'Yes'`; `process_via_column(df)` normalizes various `Via` encodings.
  - Costs: `construction_costs(...)` merges network cost attributes, calculates construction and maintenance cost per development using rates in `cost_parameters.py` and duration.
  - Aggregation: `create_cost_and_benefit_df(...)`, `discounting(...)`, `aggregate_costs(...)`, `transform_and_reshape_cost_df()`, and `rearange_costs(...)` build discounted cost/benefit time series and reshaped outputs. `plot_costs_benefits_example(...)` provides an example figure.

- scenarios.py
  - Role: Deterministic, corridor-focused scenario generation for population/employment and export to rasters.
  - `future_scenario_pop(n)`, `future_scenario_empl(n)`: compute relative growth allocations (urban/equal/rural biases) per municipality intersecting the corridor, based on reference totals.
  - `Scenario_To_Raster`: write scenario columns to TIFFs and create combined multi-band rasters; utilities for stacking and merging.

- random_scenarios.py
  - Role: Stochastic scenario generation over years using Latin Hypercube Sampling and random-walk-like shocks.
  - `get_bezirk_population_scenarios()`: builds district-level growth baselines and extends to 2100 using Eurostat growth rates.
  - `generate_population_scenarios(...)`, `generate_modal_split_scenarios(...)`: generate scenario ensembles with per-year growth perturbations and shocks; returns long-format DataFrames.
  - Used by `main.py` via `get_random_scenarios(...)` to populate the scenario cache consumed during TTT aggregation/monetization.

- catchment_pt.py
  - Role: Build and reconcile catchment polygons for public transport access when `OD_type='pt_catchment_perimeter'`.
  - `create_train_buffers(...)` (1000 m buffers for S-Bahn stations), `create_bus_buffers(...)` (650 m merged buffers grouped by nearest train station), `resolve_overlaps(...)` (subtract train buffers from bus buffers, merge by station), `clip_and_fill_polygons(...)` (clip to inner boundary and fill uncovered with -1), `add_diva_nr_to_points_with_buffer(...)` (enrich station points with stop identifiers).

- traveltime_delay.py
  - Role: Raster utilities and legacy catchment-based OD support.
  - `stack_tif_files(var)`: stack scenario rasters into multi-band TIFFs; `GetCommuneShapes(...)` rasterizes communes; `GetVoronoiOD_old(...)` demonstrates an older approach to map commune-based OD and scenario rasters to Voronoi polygons.

- plots.py
  - Role: Post-processing exports and visualizations for developments and scenarios.
  - `plotting(input_file, output_file, node_file)`: joins development cost/benefit outputs with node mappings to export per-scenario GeoPackages (adds source/target station IDs and names, normalizes cost fields).
  - `plot_developments_and_table_for_scenarios(input_dir, output_dir)`: draws all developments on OSM background and creates accompanying tables.

- result_plots.py
  - Role: Comparative plots across networks/scenarios.
  - `plot_tt_development_over_time(...)`: compares status-quo travel time evolution for different network assumptions (e.g., 2024 vs AK2035) with mean/std bands and differences.

- display_results.py
  - Role: Optional Tkinter GUI to browse the aggregated per-development results table.
  - `create_scenario_analysis_viewer(csv_file)`: loads `paths.TOTAL_COST_WITH_GEOMETRY`, offers scenario selection, and displays per-development Construction/Maintenance, Monetized Savings, and Net Benefit (best highlighted).

- ODPrep_rail.py
  - Role: Prototype for building a multi-scale OD preparation based on communes and tessellations; includes TAZ categorization, access-point assignment, and communal OD utilities. Not part of the default `main.py` pipeline.

- traveltime_comp.py
  - Role: Experimental/legacy code for network conversion and demand mapping with additional modeling utilities; not called by `main.py`.

- requirements.txt / requirements-fm.txt / environment.yml
  - Role: Environment definitions and dependencies. Prefer `environment.yml` (conda) or `requirements.txt` (pip) for reproducible setups.

- unused_*.py
  - Role: Legacy or exploratory modules retained for reference. Not used by the main pipeline.

Data Inputs (schemas and contents)

- Rail services (status quo and variants)
  - `data/temp/railway_services_ak2035.gpkg`, `data/temp/railway_services_ak2035_extended.gpkg`, `data/temp/network_railway-services.gpkg`, `data/temp/network2024_railway_services_extended.gpkg`
  - Type: GeoPackage (LineString rows; both directions present via `Direction` A/B)
  - Key columns: `FromNode` (int), `ToNode` (int), `FromStation` (str), `ToStation` (str), `Service` (str, e.g., S‑line id), `Frequency` (int per hour), `Direction` ('A'/'B'), `Via` (str encodings of intermediate node ids, e.g., "[2526]" or custom), `TravelTime` (min), `InVehWait` (min), optional `FromEnd`/`ToEnd` (endpoint flags), plus cost enrichments if merged: `NumOfTracks`, `Bridges m`, `Tunnel m`, `TunnelTrack`, `tot length m`, `length of 1`, `length of 2 `, `length of 3 and more`.
  - CRS: EPSG:2056 (LV95) for geometry.
  - Used by: `data_import.reformat_rail_edges`, `generate_infrastructure.filter_unnecessary_links`, `generate_infrastructure.prepare_Graph`, `main.create_travel_time_graphs` (via `TT_Delay.create_graphs_from_directories`), `main.add_construction_info_to_network`.

  Schema (example)

  | Column        | Type   | Example                |
  |---------------|--------|------------------------|
  | FromNode      | int    | 1234                   |
  | ToNode        | int    | 5678                   |
  | FromStation   | str    | Uster                  |
  | ToStation     | str    | Zürich Stadelhofen     |
  | Service       | str    | S5                     |
  | Frequency     | int    | 4                      |
  | Direction     | str    | A                      |
  | Via           | str    | [2526]                 |
  | TravelTime    | float  | 7                      |
  | InVehWait     | float  | 0                      |
  | FromEnd/ToEnd | 0/1    | 0 / 0                  |
  | geometry      | Line   | LineString(...)        |

- Rail nodes (raw)
  - `data/Network/Rail_Node.csv`
  - Type: CSV (Swiss-German encoding in sources)
  - Key columns: `NR` (node id), `NAME` (station name), `XKOORD`, `YKOORD` (LV95 meters), possibly station attributes (`HST`, etc.). Used to build `points.gpkg` and name mapping.
  - Used by: `data_import.reformat_rail_nodes`, `plots.plotting` (node name joins).

  Schema (example)

  | Column | Type | Example        |
  |--------|------|----------------|
  | NR     | int  | 1234           |
  | NAME   | str  | Uster          |
  | XKOORD | int  | 2689500        |
  | YKOORD | int  | 1243500        |
  | HST    | str  | 123-45 (opt.)  |

- Processed station points
  - `data/Network/processed/points.gpkg`
  - Type: GeoPackage (Point)
  - Key columns: `ID_point` (int), `NAME` (station), `XKOORD`, `YKOORD`, `HST`/`index` where present; spatial flags like `within_corridor`, `on_corridor_border` if created; may have `DIVA_NR` after enrichment.
  - CRS: EPSG:2056.
  - Used by: `generate_infrastructure.generate_rail_edges`, `generate_infrastructure.prepare_Graph` (positions), `catchment_pt.add_diva_nr_to_points_with_buffer`, plotting utilities.

  Schema (example)

  | Column            | Type   | Example     |
  |-------------------|--------|-------------|
  | ID_point          | int    | 1234        |
  | NAME              | str    | Uster       |
  | XKOORD / YKOORD   | int    | 2689500/... |
  | within_corridor   | bool   | True        |
  | on_corridor_border| bool   | False       |
  | DIVA_NR           | int    | 8503000     |
  | geometry          | Point  | POINT(...)  |

- Processed edges
  - `data/Network/processed/edges.gpkg` (and `edges_with_attribute.gpkg` in some utilities)
  - Type: GeoPackage (LineString)
  - Key columns mirror rail services, after reformatting/splitting (`FromNode`, `ToNode`, stations, `Service`, `Frequency`, `TravelTime`, `InVehWait`, `Via`, and geometry).
  - CRS: EPSG:2056.
  - Used by: network diagnostics and experiments (e.g., `traveltime_comp.py`), plotting.

  Schema (example)

  | Column      | Type  | Example       |
  |-------------|-------|---------------|
  | FromNode    | int   | 1234          |
  | ToNode      | int   | 5678          |
  | Service     | str   | S5            |
  | TravelTime  | float | 7             |
  | InVehWait   | float | 0             |
  | geometry    | Line  | LineString... |

- Candidate new links / developments
  - `data/Network/processed/new_links.gpkg` → initial generated links
  - `data/Network/processed/filtered_new_links.gpkg` → redundancy removed
  - `data/Network/processed/filtered_new_links_in_corridor.gpkg` → clipped to corridor
  - `data/Network/processed/updated_new_links.gpkg` → annotated lengths/times and dev ids
  - Type: GeoPackage (LineString)
  - Key columns: `from_ID_new` (int), `to_ID` (int), `Sline`/`name` (str id), `dev_id` (int; see `settings.py` ranges), `shortest_path_length` (m), `time` (min), `geometry`.
  - CRS: EPSG:2056.
  - Per-development networks: `data/Network/processed/developments/*.gpkg` (full network per dev; used to compute OD times per graph).
  - Used by: `generate_infrastructure.filter_unnecessary_links`, `generate_infrastructure.calculate_new_service_time`, `generate_infrastructure.add_railway_lines_to_new_links`, `generate_infrastructure.update_network_with_new_links`, `main.create_travel_time_graphs` (reads per-dev networks).

  Schema (example)

  | Column               | Type   | Example |
  |----------------------|--------|---------|
  | from_ID_new          | int    | 1234    |
  | to_ID                | int    | 5678    |
  | Sline / name         | str    | X7      |
  | dev_id               | int    | 101032  |
  | shortest_path_length | float  | 8423.5  |
  | time                 | float  | 9.8     |
  | geometry             | Line   | ...     |

- Combined network with modifications
  - `data/Network/processed/combined_network_with_all_modifications.gpkg`
  - Type: GeoPackage (LineString)
  - Contents: base services + all appended candidate links.
  - Used by: GIS inspection; potential downstream analyses.

- Graph cache (positions)
  - `data/Network/processed/graph_data.pkl`
  - Type: pickle with dict `{ 'G': nx.Graph, 'pos': {node_id: (x,y)} }` from `prepare_Graph`.
  - Used by: debugging/plotting positions; accelerates re-use of `G` + `pos`.

- Communal OD (Kanton ZH)
  - `data/_basic_data/KTZH_00001982_00003903.xlsx`
  - Type: Excel
  - Used filters: `jahr` (e.g., 2018/2019), `kategorie == 'Verkehrsaufkommen'`, `verkehrsmittel in {'oev','miv'}`
  - Key columns: `quelle_code` (int BFS/zone), `ziel_code` (int), `wert` (trips per day), plus descriptors (`quelle_name`, `ziel_name`, etc.).
  - Used by: `scoring.GetOevDemandPerCommune` (called in `main.getStationOD`) and OD preparation modules; also referenced in `traveltime_delay.py` and `ODPrep_rail.py`.

  Schema (example)

  | Column       | Type  | Example     |
  |--------------|-------|-------------|
  | jahr         | int   | 2018        |
  | kategorie    | str   | Verkehrsaufkommen |
  | verkehrsmittel | str | oev         |
  | quelle_code  | int   | 191         |
  | ziel_code    | int   | 261         |
  | wert         | float | 1234.0      |

- Station OD (derived)
  - `data/traffic_flow/od/rail/ktzh/od_matrix_stations_ktzh_20.csv`
  - Type: CSV matrix (index and columns are station ids/names; values are peak-hour OD trips after aggregation).
  - Used by: `main.plot_passenger_flows_on_network` (reads `paths.OD_STATIONS_KT_ZH_PATH`); can be used to weight OD time savings.

  Shape (example)

  |        | Uster | Zürich HB | ... |
  |--------|------:|----------:|-----|
  | Uster  |   0   |     350   | ... |
  | Pfäffikon ZH | 120 |      90 | ... |

- Commune→station mapping
  - `data/Network/processed/Communes_to_railway_stations_ZH.xlsx`
  - Type: Excel
  - Columns: commune identifiers (BFS, names) and station id/name mappings for aggregation.
  - Used by: `main.getStationOD` via `aggregate_commune_od_to_station_od` (in codebase utilities).

  Schema (example)

  | Column     | Type | Example      |
  |------------|------|--------------|
  | BFS        | int  | 191          |
  | GEMEINDE   | str  | Uster        |
  | STATION_ID | int  | 1234         |
  | STATION    | str  | Uster        |

- PT stop and line data (ZVV)
  - Lines: `data/Network/Buslines/Linien_des_offentlichen_Verkehrs_-OGD.gpkg` (layer `ZVV_S_BAHN_Linien_L`)
    - LineString segments; used for splitting and service-time inference.
  - Stops: `data/Network/Buslines/Haltestellen_des_offentlichen_Verkehrs_-OGD.gpkg`
    - Points; key columns: `DIVA_NR` (stop id), `VTYP` (e.g., 'S-Bahn'), `geometry`.
  - CRS: typically EPSG:2056.
  - Used by: `generate_infrastructure.calculate_new_service_time` (splitting and path times), `catchment_pt` (buffers, overlaps), enrichment of station points.

  Stops schema (example)

  | Column  | Type | Example   |
  |---------|------|-----------|
  | DIVA_NR | int  | 8503000   |
  | VTYP    | str  | S-Bahn    |
  | geometry| Point| POINT(...)|

  Lines schema (example)

  | Column    | Type  | Example      |
  |-----------|-------|--------------|
  | LINIEN_ID | str   | S5           |
  | geometry  | Line  | LineString.. |
  | layer     | str   | ZVV_S_BAHN_Linien_L |

- Boundaries and polygons
  - Districts: `data/_basic_data/Gemeindegrenzen/UP_BEZIRKE_F.shp`
  - Communes: `data/_basic_data/Gemeindegrenzen/UP_GEMEINDEN_F.shp`
    - Columns: `BFS` (int), `GEMEINDENA` (name), `ART_TEXT` ('Gemeinde'), `geometry` (Polygon/MultiPolygon).
  - Focus area outputs: inner/outer corridor shapefiles saved by `create_focus_area()`.
  - Lakes: `data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp` filtered by `GEWAESSERN` in {'Zürichsee','Greifensee','Pfäffikersee'}; exported to `processed/lake_data_zh.gpkg`.
  - Used by: `scenarios.future_scenario_pop/empl`, `catchment_pt.clip_and_fill_polygons`, `ODPrep_rail`, plotting contexts.

- Scenarios (population/employment)
  - Canton population: `data/Scenario/KTZH_00000705_00001741.csv`
    - Columns: `bezirk` (district), `jahr`, `anzahl` (population).
  - Switzerland population (BFS): `data/Scenario/pop_scenario_switzerland_2055.csv`
    - Columns: `Jahr`, `Beobachtungen`, `Referenzszenario A-00-2025` (used to compute national growth factor).
  - Eurostat projections: `data/Scenario/Eurostat_population_CH_2100.xlsx`
    - Key row where `unit == 'GROWTH_RATE'`; columns `2051`…`2100` (floats) drive extension beyond 2050.
  - Commune population base: `data/_basic_data/KTZH_00000127_00001245.xlsx` (sheet 'Gemeinden', header row at index 5)
    - Columns include `BFS-NR`, `GEMEINDE`, `TOTAL_YYYY` (e.g., `TOTAL_2021`).
  - Employment base: `data/Scenario/KANTON_ZUERICH_596.csv` (or `.xlsx`)
    - Columns normalized to: `BFS`, `jahr`, `anzahl` (employment); used to compute `rel_10y` and 2050 projection per commune.
  - Scenario rasters: outputs in `data/independent_variable/processed/scenario/*.tif` (single- or multi-band per variable, e.g., `pop_combined.tif`).
  - Used by: `random_scenarios.get_bezirk_population_scenarios` and generators; `scenarios.py` for corridor-focused variants; `traveltime_delay.stack_tif_files`.

  Example (KTZH_00000705_00001741.csv)

  | bezirk | jahr | anzahl |
  |--------|-----:|------:|
  | Uster  | 2018 |  82000 |
  | Uster  | 2050 | 100000 |

- Costs
  - Base construction attributes: `data/Network/Rail-Service_Link_construction_cost.csv` (sep=';', decimal=',')
    - Columns include `FromNode`, `ToNode` and engineering measures: `NumOfTracks`, `Bridges m`, `Tunnel m`, `TunnelTrack`, `tot length m`, `length of 1`, `length of 2 `, `length of 3 and more`.
  - Aggregated outputs: `data/costs/total_costs_raw.csv`, `data/costs/costs_and_benefits_dev_discounted.csv`, `data/costs/total_costs_with_geometry.csv` (joined with geometries for plotting/GUI).
  - Connection curves: `data/costs/costs_connection_curves.xlsx` (aux helper).
  - Used by: `scoring.construction_costs`, `main.add_construction_info_to_network`; outputs consumed by `plots.py` and `display_results.py`.

  Schema (construction_cost.csv example)

  | Column           | Type  | Example |
  |------------------|-------|---------|
  | FromNode         | int   | 1234    |
  | ToNode           | int   | 5678    |
  | NumOfTracks      | int   | 2       |
  | Bridges m        | float | 250.0   |
  | Tunnel m         | float | 1200.0  |
  | TunnelTrack      | int   | 2       |
  | tot length m     | float | 3800.0  |
  | length of 1      | float | 1000.0  |
  | length of 2      | float | 2000.0  |
  | length of 3 and more | float | 800.0 |

  Outputs (total_costs_with_geometry.csv example)

  | Column                                  | Example                 |
  |-----------------------------------------|-------------------------|
  | development                              | Development_101032      |
  | Construction Cost [in Mio. CHF]          | 350.2                   |
  | Maintenance Costs [in Mio. CHF]          | 95.7                    |
  | monetized_savings_total_od_matrix_...    | 412.5                   |
  | Net Benefit Urban Medium [in Mio. CHF]   | 62.3                    |
  | geometry (WKT or separate gpkg outputs)  | LINESTRING(...)         |

- OD/time caches and results
  - Travel-time graphs/OD times: `data/Network/travel_time/cache/od_times.pkl` → dict with `od_times_dev`, `od_times_status_quo`, `G_status_quo`, `G_developments`.
  - TTT cache: `data/Network/travel_time/cache/compute_tts_cache.pkl` when enabled.
  - Monetized savings: `data/costs/traveltime_savings.csv` (per scenario and development).
  - Used by: `main.create_travel_time_graphs` (reads/writes OD times cache), `main.compute_tts` (writes TTT and monetized savings cache), plotting/GUI.

- Misc. plotting/background
  - OSM extract: `data/_basic_data/planet_8.4,47.099_9.376,47.492.osm.pbf` used for map backgrounds in `plots.py`.
  - City labels: `data/manually_gathered_data/City_map.csv` with columns `x`, `y` (LV95 meters) used to write `cities.shp`.

Conventions

- CRS: All GeoPackages and shapefiles are expected in EPSG:2056 unless noted. Raster indices and corridor bounds also align to LV95 meters.
- Decimals: Many Swiss CSV/Excel sources use `;` separator and `,` decimal; loaders in code set `sep=';'`, `decimal=','` where needed.
- Via encoding: Legacy networks may store `Via` as strings like "[2526]" or custom concatenations (e.g., "1,8,8,5,,,3..."); `scoring.process_via_column` normalizes these to lists of ints or `-99`.

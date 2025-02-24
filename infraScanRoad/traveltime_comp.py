import math

import pandas as pd
import scipy.io
import numpy as np
from scipy.optimize import minimize, Bounds, least_squares
import timeit
import os
import sys
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
from shapely.geometry import Point
import networkx as nx
from itertools import islice
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components
import data_import

# Get the parent directory of GUI (i.e., InfraScan)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)  # Add InfraScan to Python's module search path
from logging_config import logger  # Import central logger

#os.chdir(r"C:\Users\Fabrice\Desktop\HS23\Thesis\Code")
os.chdir(r"C:\Users\spadmin\PycharmProjects\infraScan")
## INPUT TRAFFIC NETWORK
#--- Options
output_lab='Portugal'
network='Port271221'
figureoption=1 #0: no figures, otherwise, 1


def convert_data_to_input():
    # Import gpkg file with the network points
    points = gpd.read_file(r"data\Network\processed\points_with_attribute.gpkg")
    #point_in_corridor = gpd.read_file(r"data\Network\processed\points_corridor.gpkg")
    #points_on_border = gpd.read_file(r"data\Network\processed\points_corridor_border.gpkg")
    # Import gpkg file with the network edges
    edges = gpd.read_file(r"data\Network\processed\edges_with_attribute.gpkg")

    # Set crs for points and edges to epsg:2056
    points = points.set_crs("epsg:2056", allow_override=True)
    edges = edges.set_crs("epsg:2056", allow_override=True)
    # Print crs of points and edges
    print(f"Points crs: {points.crs}")
    print(f"Edges crs: {edges.crs}")

    # Change "corridor_border" to False if "within_corridor" is True
    points.loc[points["within_corridor"] == True, "on_corridor_border"] = False

    # Define new column to state if node generates traffic in model
    # When point is in corridor or on border, it generates traffic
    points["generate_traffic"] = points["within_corridor"] | points["on_corridor_border"]


    #######################################################################################################################
    # Store values as needed for the model

    # Assert nodes and edges are sorted by ID
    points = points.sort_values(by=["ID_point"])
    edges = edges.sort_values(by=["ID_edge"])

    # Nodes: store dict of coordinates of all nodes nodes = [[x1, y1], [x2, y2], ...] <class 'numpy.ndarray'>
    nodes_lv95 = points[["geometry"]].to_numpy()
    # Same with coordinates converted to wgs84
    nodes_wgs84 = points[["geometry"]].to_crs("epsg:4326").to_numpy()

    # Edges: store dict of coordinates of all edges links = [[id_start_node_1, id_end_node_1], [id_start_node_2, id_end_node_2], ...] <class 'numpy.ndarray'>
    links = edges[["start", "end"]].to_numpy()

    # Length of edges stored a array link_length_i = [[length_1], [length_2], ...] <class 'numpy.ndarray'>
    link_length_i = edges["geometry"].length.to_numpy()

    # Number of edges
    nlinks = len(edges)

    # Travel time on each link assuming free flow speed
    # Get edge length
    edges["length"] = edges["geometry"].length / 1000  # in kilometers
    # Calculate free flow travel time on all edges
    edges["fftt_i"] = edges["length"] / edges["ffs"]  # in hours
    # Store these values in a dict as fftt_i = [[fftt_1], [fftt_2], ...] <class 'numpy.ndarray'>
    fftt_i = edges[["fftt_i"]].to_numpy()

    # Capacity on each link
    # Store these values in a dict as capacity_i = [[capacity_1], [capacity_2], ...] <class 'numpy.ndarray'>
    Xmax_i = edges[["capacity"]].to_numpy()

    # Store same alpha and gamma for all links as alpha_i = [[alpha_1], [alpha_2], ...] <class 'numpy.ndarray'>
    alpha = 0.25
    gamma = 2.4
    alpha_i = np.tile(alpha, Xmax_i.shape)
    gamma_i = np.tile(gamma, Xmax_i.shape)

    par = {"fftt_i": fftt_i, "Xmax_i": Xmax_i, "alpha_i": alpha_i, "gamma_i": gamma_i}



    return nodes_lv95, nodes_wgs84, links, link_length_i, nlinks, par


def get_nw_data():
    # Adapt OD matrix
    # nodes within perimeter and on border
    ####################################################
    ### Define zones
    ### Check for 2050

    # Import OD matrix and flatten it (D_od)
    OD_matrix = pd.read_csv(r"data\traffic_flow\od\od_matrix_2020.csv", sep=",", index_col=0)
    print(OD_matrix.head(5).to_string())
    # Import polygons of the corridor
    #voronoi_OD = gpd.read_file(r"data\traffic_flow\od\OD_voronoidf.gpkg")
    voronoi_OD = gpd.read_file(r"data\Network\travel_time\Voronoi_statusquo.gpkg")
    print(voronoi_OD.head(5).to_string())

    # Import gpkg file with the network points
    points = gpd.read_file(r"data\Network\processed\points_with_attribute.gpkg")
    points = points.sort_values(by=["ID_point"])

    # Filter points to only keep those within the corridor or on border
    points_in = points[(points["within_corridor"] == True) | (points["on_corridor_border"] == True)]
    print(f"Points in corridor or border: {points_in.shape}")

    # Get common ID_point and voronoi_ID as list
    common_ID = list(set(points_in["ID_point"]).intersection(voronoi_OD["ID_point"]))
    print(f"Point in polygon and with voronoi: {len(common_ID)}")

    # new column "generate_demand" where ID_point is in common_ID
    points["generate_demand"] = points["ID_point"].isin(common_ID)

    # Filter OD matrix to only keep common ID in rows and columns
    # Convert ID elements to the appropriate type if necessary
    common_ID = [int(id) for id in common_ID]
    OD_matrix.index = OD_matrix.index.map(lambda x: int(float(x)))
    OD_matrix.columns = OD_matrix.columns.map(lambda x: int(float(x)))

    OD_matrix = OD_matrix.loc[common_ID, common_ID]
    print(f"Shape OD matrix: {OD_matrix.shape}")

    # flatten OD matrix to 1D array as D_od
    D_od = OD_matrix.to_numpy().flatten()


    # Get the amount of values in the OD matrix as nOD
    nOD = len(D_od)
    print(nOD)

    # Map the single zones of the OD to actual nodes in the network
    # nodes within perimeter and on border
    # Get nodes in zones -> then build network with these nodes
    # Import edges
    edges = gpd.read_file(r"data\Network\processed\edges_with_attribute.gpkg")
    edges = edges.sort_values(by=["ID_edge"])

    print(edges.head(5).to_string())
    # Build network using networkx
    # Map the coordinates to the edges DataFrame
    # Set the index of the nodes DataFrame to be the 'ID_point' column
    points.set_index('ID_point', inplace=True)

    # Create an empty NetworkX graph
    G = nx.MultiGraph()
    print(points.head(5).to_string())
    # Add nodes with IDs to the graph
    for node_id, row in points.iterrows():
        G.add_node(node_id, pos=(row['geometry'].x, row['geometry'].y), demand=row['generate_demand'])

    # Add edges to the graph
    # Make sure 'start' and 'end' in edges_gdf refer to the node IDs, add attribute
    for idx, row in edges.iterrows():
        G.add_edge(row['start'], row['end'], key=row['ID_edge'], fftt=row['ffs'] / row.geometry.length)
    """
    # Plot graph small points with coordinates as position and color based on demand attribute
    nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), node_size=3, node_color=list(nx.get_node_attributes(G, 'demand').values()), edge_color='black', width=0.5)
    #nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), node_size=1, node_color='black', edge_color='black', width=0.5)
    plt.show()
    """


    # Compute the route for each OD pair (maybe limit to max 5)
    # Get routes for all points with demand = True in the network G
    # Get all nodes with demand = True
    #nx.all_simple_edge_paths() # (u,v,k) -> u,v are nodes, k is the key of the edge

    # Step 1: Identify nodes with demand
    demand_nodes = [n for n, attr in G.nodes(data=True) if attr.get('demand') == True]
    print(f"Number of points considered: {len(demand_nodes)} ({len(demand_nodes)*len(demand_nodes)})")
    """
    # Find connected components
    connected_components = list(nx.connected_components(G))
    # Filter components to include only those with at least one node having 'demand' == True
    #connected_components_with_demand = [comp for comp in connected_components if any(G.nodes[node]['demand'] for node in comp)]

    print(connected_components)

    # Convert the generator to a list to get its length
    connected_components_list = list(connected_components)

    # To plot each connected component in a different color, we can use a color map
    colors = plt.cm.rainbow(np.linspace(0, 1, len(connected_components_list)))

    # Create a plot
    plt.figure(figsize=(8, 6))

    # For each connected component, using a different color for each
    for comp, color in zip(connected_components_list, colors):
        # Separate nodes with demand and without demand
        nodes_with_demand = [node for node in comp if G.nodes[node]['demand']]
        nodes_without_demand = [node for node in comp if not G.nodes[node]['demand']]

        # Get positions for all nodes in the component
        pos = nx.get_node_attributes(G, 'pos')  # Use existing positions if available

        # Draw nodes with demand
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_with_demand, node_color=[color], node_size=20)
        # Draw nodes without demand
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_without_demand, node_color=[color], node_size=3)
        # Draw edges
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v in G.edges() if u in comp and v in comp])

        # Draw node labels in small size
        nx.draw_networkx_labels(G, pos, font_size=2)

    # Show the plot
    plt.title("Connected Components with Individual Colors")
    plt.axis('off')  # Turn off axis
    # safe plot
    plt.savefig(fr"plot\results\connected_components.png", dpi=500)
    plt.show()
    """

    index_routes = 0
    index_OD_pair = 0
    delta_odr = np.zeros((nOD, 1000000))
    routelinks_list = []
    #delta_ir = np.zero((nlinks, 10000)
    """
    for i in range(len(demand_nodes)):
        #print(f"New origin    {i}")
        for j in range(len(demand_nodes)):
            #print(f"    {j}")
            source = demand_nodes[i]
            target = demand_nodes[j]
            paths = nx.all_simple_edge_paths(G, source, target)
            #print(f"Number of paths: {len(paths)}")


            # iterate over all paths
            for path in paths:
                # Get cell in delta_odr based on index_OD_pair and index_routes
                delta_odr[index_OD_pair][index_routes] = 1
                #print("Yes")
                # get the key of the edges of each path (u,v,k)
                routelinks_list.append([k for _, _, k in path])
                # Add array of edge keys of path to routelinks as dict with nbr_routes (as

                index_routes = index_routes + 1
            index_OD_pair = index_OD_pair + 1
    """

    # Convert MultiDiGraph or MultiGraph to DiGraph or Graph
    G_simple = nx.Graph(G)

    def k_shortest_paths_edge_ids(G, source, target, k, weight=None):
        return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


    for i in range(len(demand_nodes)):
        for j in range(len(demand_nodes)):
            source = demand_nodes[i]
            target = demand_nodes[j]
            unique_paths_ij = []

            paths = k_shortest_paths_edge_ids(G_simple, source, target, 2, weight='fftt')

            for path in paths:
                edge_ids = [list(G[u][v])[0] for u, v in zip(path[:-1], path[1:])]  # Assuming G is a DiGraph or Graph

                # Check if this path is already in unique_paths
                if edge_ids not in unique_paths_ij:
                    unique_paths_ij.append(edge_ids)
            #print(unique_paths_ij)
            for edge_ids in unique_paths_ij:
                routelinks_list.append(edge_ids)
                delta_odr[index_OD_pair][index_routes] = 1
                index_routes += 1


            """
            for path in paths:
                edge_ids = []
                for u, v in zip(path[:-1], path[1:]):
                    # Assuming G is a DiGraph or Graph, this gets the key for the edge (u, v)
                    edge_key = list(G[u][v])[0]  # Get the first (and only) key
                    edge_ids.append(edge_key)

                routelinks_list.append(edge_ids)
                delta_odr[index_OD_pair][index_routes] = 1
                index_routes += 1
            """
            index_OD_pair += 1
            # print(f"Number of routes: {index_routes} and OD pairs: {index_OD_pair}")


            """
            try:
                # Find the shortest path based on 'fftt' weight
                shortest_path = nx.shortest_path(G, source, target, weight='fftt')
                # Convert path to edge keys
                edge_keys = [min(G[u][v], key=lambda key: G[u][v][key]['fftt']) for u, v in
                             nx.utils.pairwise(shortest_path)]
                # Add edge keys of the shortest path to routelinks_list
                routelinks_list.append(edge_keys)

                # Update delta_odr and other indices as needed
                delta_odr[index_OD_pair][index_routes] = 1
                index_routes += 1

            except nx.NetworkXNoPath:
                # Handle the case where no path exists between source and target
                print(f"No path between {source} and {target}")
        index_OD_pair += 1
            """



    print(f"Number of routes: {index_routes}")
    print(f"Number of OD pairs: {index_OD_pair}")

    max_length = max(len(arr) for arr in routelinks_list)
    # Replace all 0s with -1
    routelinks_list = [[-1 if element == 0 else element for element in sublist] for sublist in routelinks_list]

    # Create a 2D array, padding shorter arrays with zeros
    routelinks_same_len = [arr + [0] * (max_length - len(arr)) for arr in routelinks_list]
    routelinks = np.array(routelinks_same_len)

    # Get the number of rows in array1
    n_routes = routelinks.shape[0]

    ## Assert the algorithm works well
    column_sums = np.sum(delta_odr, axis=0)
    print(f"The number of routes {np.sum(column_sums >= 1)} and {n_routes}")
    if np.sum(column_sums >= 1) == n_routes:
        delta_odr = delta_odr[:, :n_routes]
    else:
        print("not same amount of routes in computation - check code for errors")

    # Example edge keys and 2D numpy array
    edge_keys = [k for u, v, k in G.edges(keys=True)]  # Replace with your actual edge keys
    # Replace all 0s with -1
    edge_keys = [-1 if element == 0 else element for element in edge_keys]

    # Initialize DataFrame with edge IDs as columns and zeros
    delta_ir_df = pd.DataFrame(0, index=np.arange(len(routelinks)), columns=edge_keys)

    # Fill the DataFrame
    ####################################################################################################################
    ######## This is not filling correctly

    # From a matrix with all edge IDs for each route (routelinks) route ID (x) and edge ID (y) just in a list
    # to a matrix with all routes (x) and all edges (y) as delta_ir (binary if edge in route)
    for row_idx, path in enumerate(routelinks):
        #print(f"row_idx: {row_idx} and path: {path}")
        for edge in path:
            if edge != 0:
                delta_ir_df.at[row_idx, edge] = 1
        # print entire row
        #print(delta_ir_df.iloc[row_idx])

    # Sort in ascending edge_id and store it as array
    print(delta_ir_df.sort_index(axis=1).transpose().head(10).to_string())
    delta_ir = delta_ir_df.sort_index(axis=1).transpose().to_numpy()

    print(f"Amount of links used: {np.sum(delta_ir > 0)} (delta_ir) and {np.sum(routelinks > 0)} (routelinks)")

    print(f"Shape delta_odr {delta_odr.shape} (OD pairs x #routes)")
    print(f"Shape delta_ir {delta_ir.shape} (#links (edge ID sorted ascending) x #routes)")
    print(f"Shape routelinks {routelinks.shape} (#routes x amount of links of longest route)")
    print(f"Shape D_od {D_od.shape[0]} ({math.sqrt(D_od.shape[0])} OD zones)")
    print(f"nOD number of OD pairs {nOD}")
    print(f"number of routes (n_routes) {n_routes}")
            #paths = list(k_shortest_paths(G, source, target, 3))


    # Store a matrix with edge_ID for each route (routelinks) route ID (x) and edge ID (y)

    # Store a matrix with all OD pairs (x) and all route (y) as delta_odr (binary if route serves OD pair)

    # Store a matrix with all links (x) and all edges (y) as delta_ir (binary if edge in route)
    return delta_ir, delta_odr, routelinks, D_od, nOD, n_routes


def Topology(network):
    filename = 'data/traffic_flow/mat/'+network+'.mat'
    matfile = scipy.io.loadmat(filename)
    #Network topography
    #load([network,'.mat'],'nodes','Infolinks') #Infolinks:[Node i Node j Length(km) fftt_i(hr) Capacity(pcu/h/lane) typeroad]
    #
    nodes = matfile['nodes']
    links = matfile['Infolinks'][:,[0,1]]
    linklength_i = matfile['Infolinks'][:,[2]]
    nlinks = len(links)
    #links=Infolinks(:,[1 2]);
    #linklength_i=Infolinks(:,3);
    #nlinks=length(links);
    return [nodes,links,nlinks,linklength_i]


def CostFunctionParm(network):
    filename = 'data/traffic_flow/mat/'+network+'.mat'
    matfile = scipy.io.loadmat(filename)
    #Definition of cost function and free-flow travel times
    #load([network,'.mat'],'Infolinks','alpha','gamma') #Infolinks:[Node i Node j Length(km) fftt_i(hr) Capacity(pcu/h/lane) typeroad]
    #
    typeroad_i=matfile['Infolinks'][:,[5]]
    #typeroad_i=Infolinks(:,6);
    #
    par = {}
    par['fftt_i']=matfile['Infolinks'][:,[3]]
    par['Xmax_i']=matfile['Infolinks'][:,[4]]
    par['alpha_i']=np.tile(matfile['alpha'][0,0], typeroad_i.shape)
    par['alpha_i'][typeroad_i==2] = matfile['alpha'][0,1]
    par['gamma_i']=np.tile(matfile['gamma'][0,0], typeroad_i.shape)
    par['gamma_i'][typeroad_i==2] = matfile['gamma'][0,1]
    return [par,typeroad_i]


def ODs_Definition():
    #--- Define the OD pairs
    extremenodes=[     1,    10,    12,    22,    23,    35,    36,    42,    43,    44,    45]
    n=len(extremenodes)
    nOD=n*(n-1) #initial memory allocation
    ODs=np.zeros((nOD,2))
    #ODs=zeros(nOD,2)
    pos=0
    for i in range(0,n-1):
        for j in range(i+1,n):
            ODs[pos,0]=extremenodes[i]
            ODs[pos,1]=extremenodes[j]
            pos=pos+1


    quitar=list([999,998])#find(ODs(:,1)==36 & ODs(:,2)==43);
    ind=list(filter(lambda a: a not in quitar, range(0,pos)))
    #ind=setdiff(0:pos,quitar)
    inv = list(zip(ODs[ind,1],ODs[ind,0]))
    newODs1 = ODs[ind,:]
    newODs2 = np.array(inv)
    ODs=np.row_stack((newODs1,newODs2)) #The other direction is added
    nOD=len(ODs)
    return [ODs,nOD]


def CostFun(Xi,par):
    #Computes the cost function for the flow Xi, with the parameters 'par', Xi has the adequate size

    #BPR
    s1 = np.power(np.divide((Xi,par['Xmax_i'])),par['gamma_i'])
    s2 = np.multiply(par['alpha_i'],s1)
    Ci = np.multiply(par['fftt_i'],np.add(1,s2))
    #Ci=par.fftt_i.*(1+par.alpha_i.*(Xi./par.Xmax_i).^par.gamma_i);
    return Ci


def IntCostFun(Xi,par):
    #Computes the integral of the cost function for the flow Xi, with the
    #parameters 'par'. Xi has the adequate size
    Xi=2
    #BPR
    # s1 = (Xi./par.Xmax_i).^par.gamma_i            (Flow/MaxCapacity)^gamma
    # s2 = Xi*par.alpha_i.*par.fftt_i.*s1           Flow*alpha*fftt*s1
    # Ci = Xi.*par.fftt_i + s2./(par.gamma_i + 1)   Flow*fftt + s2/(gamma+1)

    s1 = np.power(np.divide(Xi,par['Xmax_i']),par['gamma_i']) #(Xi./par.Xmax_i).^par.gamma_i
    s2 = np.multiply(Xi,np.multiply(par['alpha_i'],np.multiply(par['fftt_i'],s1))) # (Xi.*par.alpha_i.*par.fftt_i.*s1)
    Ci = np.multiply(Xi,par['fftt_i']) + np.divide(s2,(np.add(par['gamma_i'],1))) #Xi.*par.fftt_i + s2./(par.gamma_i + 1);
    #Ci=Xi.*par.fftt_i + (Xi.*par.alpha_i.*par.fftt_i.*(Xi./par.Xmax_i).^par.gamma_i)./(par.gamma_i + 1);

    # returns travel cost for each link
    return Ci


def SUE_C_Logit(nroutes,D_od,par,delta_ir,delta_odr,cf_r,theta):
    #De acuerdo con C logit SUE_Zhou (2010)

    #--- Optimizacion NO lineal
    #objfun = @(D_r)( sum(IntLinksTimes(D_r)) + 1/theta*sum(D_r.*log(D_r)) + sum(D_r.*cf_r) );
    #A=[];b=[];  # A x <= b
    #Aeq=delta_odr;beq=D_od; # Aeq x = beq
    #lb=zeros(nroutes,1);ub=max(D_od)*ones(nroutes,1);
    #D_r0=delta_odr.H*(D_od./sum(delta_odr,2)); #estimacion: reparto equitativamente #zeros(nroutes,1);#
    ##options=optimset('Algorithm','interior-point','MaxFunEvals',1e5);#,'Display','off'); #matlab viejo
    #options=optimoptions('fmincon','Algorithm','sqp','MaxFunEvals',1e5,'Display','off');
    #[D_r,fval,exitflag,output]=fmincon(objfun,D_r0,A,b,Aeq,beq,lb,ub,[],options);
    #x_i=delta_ir*D_r

    #--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    #--- Non-linear optimisation using Sequential least squares programming

    def IntLinksTimes(D_r):
        # Demand on each link from demand on routes
        x_i=np.matmul(delta_ir,D_r)
        intTrec_i=IntCostFun(x_i,par) #integral de la funcion de coste
        return intTrec_i

    #def bpr(x):


    def fun(x):
        ############3 Thy not theta?
        thetavec=np.ones_like(x)
        #eqval = np.sum(IntLinksTimes(x),axis=0) + 1/theta*np.sum(np.multiply(x,np.log(x)),axis=1) + np.sum((np.multiply(x,cf_r)),axis=1)

        # X contains 2150 zeros

        #print(np.multiply(x,np.log(x)))
        #print(np.sum(np.divide(np.multiply(x,np.log(x)),thetavec)))

        # IntLinksTimes()           Travel time on each link

        # np.multiply(x,cf_r)       Commonality factor on each route with optimize flow
        ############################################################################################################
        ### Wha to divide by 1?
        #print(np.sum(IntLinksTimes(x)) + np.sum(np.divide(np.multiply(x, np.log(x)), thetavec)) + np.sum((np.multiply(x, cf_r))))
        #print(np.multiply(x, np.log(x)))
        # Count the amount of nan in x
        #print(np.sum(np.isnan(np.multiply(x, np.log(x)))))  # This creates nan (like 30)
        #print(np.sum(np.isnan(np.multiply(x,cf_r))))
        #print(cf_r)
        #print(np.sum(np.isinf(cf_r)))
        #print(np.multiply(x,cf_r))

        # Check values in np.multiply(x,cf_r), number of nan and number of inf
        #print(f"Number of nan in np.multiply(x,cf_r): {np.sum(np.isnan(np.multiply(x,cf_r)))} and number of inf: {np.sum(np.isinf(np.multiply(x,cf_r)))}")
        # Same for np.multiply(x,np.log(x))
        #print(f"Number of nan in np.multiply(x,np.log(x)): {np.sum(np.isnan(np.multiply(x,np.log(x))))} and number of inf: {np.sum(np.isinf(np.multiply(x,np.log(x))))}")
        # Same for x
        #print(f"Number of nan in x: {np.sum(np.isnan(x))}, number of inf: {np.sum(np.isinf(x))}, number of zeros: {np.sum(x==0)}")
        # And for log(x)
        #print(f"Number of nan in log(x): {np.sum(np.isnan(np.log(x)))}, number of inf: {np.sum(np.isinf(np.log(x)))}, number of zeros: {np.sum(np.log(x)==0)}")


        """
        #Check if 0 in x if so print the optimization iteration
        if 0 in x:
            #print("0 in x")
            print(f"There are 0 in x   {np.sum(x==0)}")
        
        # Check for NaN or Inf in x
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            print("NaN or Inf found in x:", x)
        """
        # If x is smaller than 0 replace with 0.001
        x[x<0] = 0.001
        temp_log = np.log(x)

        #replace inf with 0
        temp_log[np.isinf(temp_log)] = 0.1
        temp_log[np.isnan(temp_log)] = 0.1
        """
        # Check if inf or nan in np.log(x)
        if np.sum(np.isinf(np.log(x))) > 0:
            print(f"inf in log(x)  {np.sum(np.isinf(np.log(x)))}")
        if np.sum(np.isnan(np.log(x))) > 0:
            print(f"nan in log(x)  {np.sum(np.isnan(np.log(x)))}")
        """


        # Replace all nan values in x with 0
        #x[np.isnan(x)] = 0

        result =  np.sum(IntLinksTimes(x)) + np.sum(np.divide(np.multiply(x,temp_log),thetavec)) + np.sum((np.multiply(x,cf_r)))
        if np.sum(IntLinksTimes(x)<0) > 0:
            print(f"Neagtive numbers in IntLinksTimes(x): {np.sum(IntLinksTimes(x)>0)}")
        if np.isnan(np.sum(IntLinksTimes(x))):
            print("NaN in IntLinksTimes:", result)
        if np.isnan(np.sum(np.divide(np.multiply(x,temp_log),thetavec))):
            print("NaN in np.divide(np.multiply(x,np.log(x)),thetavec):", np.sum(np.isnan(np.divide(np.multiply(x,temp_log),thetavec))))
            print("NaN in np.multiply(x,np.log(x))):", np.sum(np.isnan(np.multiply(x,temp_log))))
            print("NaN in x:", np.sum(np.isnan(x)))
            # Print amount of negative values in x
            print(f"Negative values in x: {np.sum(x<0)}")
        if np.isnan(np.sum((np.multiply(x,cf_r)))):
            print("NaN in np.multiply(x,cf_r):", result)
        # same for inf
        if np.isinf(np.sum(IntLinksTimes(x))):
            print("Inf in IntLinksTimes:", result)
        if np.isinf(np.sum(np.divide(np.multiply(x,temp_log),thetavec))):
            print("Inf in np.divide(np.multiply(x,np.log(x)),thetavec):", result)
        if np.isinf(np.sum((np.multiply(x,cf_r)))):
            print("Inf in np.multiply(x,cf_r):", result)
        return result


        #s=np.squeeze(eqval)
        #val=float(s)
        #val=float(eqval)
        #return eqval
    # def fun_der(x):
    #     der = np.zeros_like(x)
    #     s0 = np.matmul(delta_ir,x)
    #     s1 = np.power(np.divide(s0,par['Xmax_i']),par['gamma_i'])
    #     s2 = np.multiply(par['alpha_i'],s1)
    #     s3 = np.multiply(par['fftt_i'],s2)
    #     s4 = np.matmul(delta_ir.transpose(),s3)
    #     a1 = np.matmul(delta_ir.transpose(),par['fftt_i'])
    #     der = np.divide((np.log(x)+1),theta)+cf_r+a1+s4
    #     return der.flatten()

    def callback_function(*args):
        global iteration_count
        iteration_count += 1
        xk = args[0]  # The first argument is the current solution vector
        objective_value = fun(xk)
        print(f"Iteration {iteration_count}, Objective Function Value: {objective_value}")

    ineq_cons = {'type': 'ineq',
             'fun' : lambda x: x}#,
             #'jac' : lambda x: np.array([])}
    eq_cons = {'type': 'eq',
            'fun' : lambda x: (np.matmul(delta_odr,x)-(D_od)).flatten()}#,
            #'jac' : lambda x: -delta_odr.flatten()}

    ###################################################################################################################
    # Check if there are nan values in the matrix
    # replace nan values with 0
    tt = (np.divide(D_od.transpose(), np.sum(delta_odr, axis=1))).transpose()
    tt[np.isinf(tt)] = tt.max()*10

    D_r0 = np.matmul(delta_odr.transpose(),tt)


    #D_r0=np.matmul(delta_odr.transpose(),(np.divide(D_od.transpose(),np.sum(delta_odr,axis=1))).transpose())
    lb = np.zeros((np.shape(D_r0))).flatten()
    # Add a very small value to lb to avoid 0 in x
    lb = lb + 0.01
    ub = (max(D_od)*np.ones(np.shape(D_r0))).flatten()
    ub = ub*5
    bounds = Bounds(lb, ub)
    print(f"lb: {lb}")
    print(f"ub: {ub}")
    #res=least_squares(fun, D_r0.flatten(),jac=fun_der,bounds=bounds)

    # D_r to be optimized -> demand on each route
    res = minimize(fun, D_r0.flatten(),
                    method='trust-constr',
                    #method='SLSQP',
                    #jac=fun_der,
                    constraints=[eq_cons, ineq_cons],
                    options={#'ftol': 1e5,
                             'maxiter':3,
                             'verbose': 0,
                             'disp': True},
                    bounds=bounds,
                    callback=callback_function
                    )
                    #)
    # Describe variables
    # D_r is the demand on each route
    D_r = res.x
    # check if values in D_r are negative if so print the amount
    if np.sum(D_r<0) > 0:
        print(f"Negative values in D_r: {np.sum(D_r<0)}")
    #Same for 0
    if np.sum(D_r==0) > 0:
        print(f"0 in D_r: {np.sum(D_r==0)}")

    D_r[D_r <= 0] = 0.001

    # x_i is the demand on each link
    x_i=delta_ir*D_r
    # Get travel time on each route
    intTrec_i=IntCostFun(x_i,par)


    thetavec=np.ones_like(D_r)

    # fval is the objective function value
    fval = np.sum(IntLinksTimes(D_r)) + np.sum(np.divide(np.multiply(D_r,np.log(D_r)),thetavec)) + np.sum((np.multiply(D_r,cf_r)))
    #fval = sum(IntLinksTimes(D_r)) + 1/theta*sum(np.multiply(D_r,np.log(D_r))) + sum(np.multiply(D_r,cf_r))
    #--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    return [x_i,D_r,intTrec_i,fval]


def Commonality(betaCom,delta_ir,delta_odr,fftt_i,fftt_r):
    [nOD,nroutes]=delta_odr.shape
    print(f"nOD {nOD} and nroutes {nroutes}")

    # Initialize commonality factor (per route)
    cf_r=np.zeros((nroutes,1))

    # Iterate over all pairs of OD
    for od in range(0,nOD):
        # Find routes for OD pair
        routes = np.argwhere(np.ravel(delta_odr[od,:])==1)
        #print(routes)

        # Check if there is more than  route for the OD pair
        if len(routes) <= 1:
            # In this case there is no commonality factor
            continue
        else:
            #routes=find(delta_odr[od,:]==1)
            for r1 in routes:
                # Freeflow travel time for route
                t0_1=fftt_r[r1]
                for r2 in routes:
                    # Freeflow travel time for route
                    t0_2=fftt_r[r2]
                    aa = np.argwhere(np.ravel(delta_ir[:,r1])==1)
                    bb = np.argwhere(np.ravel(delta_ir[:,r2])==1)
                    # Get common links among routes routes compared
                    common = np.array(list(set(aa.flatten()).intersection(bb.flatten())))
                    #common=intersect(find(delta_ir(:,r1)==1),find(delta_ir(:,r2)==1));

                    # Check if common is empty
                    if len(common) == 0:
                        t0_1_2=0
                    
                    else:
                        # Sum of freeflow travel time for common links
                        t0_1_2=sum(fftt_i[common])

                    # Update commonality factor
                    cf_r[r1] = cf_r[r1] + t0_1_2 / (t0_1**.5 * t0_2**.5)
                    """
                    if (t0_1**.5*t0_2**.5) == 0:
                        print(f"t0_1 or t0_2 is 0 for OD pair {od} and routes {r1} and {r2}")
                    """
    cf_r = betaCom * np.log(cf_r)
    cf_r[np.isnan(cf_r)] = 0 #for routes with just one link -->cf_r=0;
    cf_r[np.isinf(cf_r)] = 0 #for routes with just one link -->cf_r=0;
    return cf_r


## READING DATA AND FITTING THE MODEL
"""
#--- Topography of the network
[nodes,links,nlinks,linklength_i]=Topology(network)

#--- Cost function parameters
[par,typeroad_i]=CostFunctionParm(network)
#if figureoption, PlotNetwork(nodes,links,find(typeroad_i==2),[network,' Network'],1,1), end
"""
"""
# print for all variable ist daty structure and the data itselve
print("nodes: ", type(nodes), nodes)
print("links: ", type(links), links)
print("nlinks: ", type(nlinks), nlinks)
print("linklength_i: ", type(linklength_i), linklength_i)
print("par: ", type(par), par)
print("typeroad_i: ", type(typeroad_i), typeroad_i)
"""


# Same with own data
nodes_lv95, nodes_wgs84, links, link_length_i, nlinks, par = convert_data_to_input()
delta_ir, delta_odr, routelinks, D_od, nOD, nroutes = get_nw_data()

# Remove all trip from origin to same destination
# Delete all diagonals in OD matrix
# Get the amount of values in the OD matrix as nOD
OD_single = int(math.sqrt(nOD))
# Remove every first and then every (OD_single + 1)th element
idx = np.arange(0, nOD, OD_single+1)
# Delete every (OD_single + 1)th element
D_od = np.delete(D_od, idx)
# same for columns of delta_odr
delta_odr = np.delete(delta_odr, idx, axis=0)

print(f"Shape delta_odr {delta_odr.shape} (OD pairs x #routes)")
print(f" Shape D_od {D_od.shape} (OD pairs x 1)")


"""
# Convert the matrix to a connectivity matrix (1 for connected, 0 for not connected)
connectivity_matrix = np.where(delta_odr > 0, 1, 0)

# Find connected components
n_components, labels = connected_components(csgraph=connectivity_matrix, directed=False)

# Extract sets of connected nodes
connected_sets = [np.where(labels == i)[0] for i in range(n_components)]

print(connected_sets)
"""

"""
#--- Definition of ODs and Routes
filename = 'data/traffic_flow/mat/'+network+'_RoutesOD.mat'
matfile = scipy.io.loadmat(filename)
#load([network,'_RoutesOD.mat'],'D_od','nOD','routelinks','nroutes','delta_ir','delta_odr')

delta_ir = matfile['delta_ir']
delta_odr = matfile['delta_odr']
routelinks = matfile['routelinks']
D_od = matfile['D_od']
nOD = matfile['nOD']
nroutes = matfile['nroutes']
"""
#--- Definition of parameters for SUE

# Freeflow traval time per route
fftt_r=(np.matmul(par['fftt_i'].transpose(),delta_ir)).transpose()
#######################################################################################################################
# No travel time when OD same point (intra cellular move)
#fftt_r[fftt_r == 0] = 0.0001

betaCom=1
# Get the common links among routes
cf_r = Commonality(betaCom,delta_ir,delta_odr,par['fftt_i'],fftt_r)
# print amount of nan in cf_r
print(f"Amount of nan in cf_r: {np.sum(np.isnan(cf_r))}")
print(f"Amount of inf in cf_r: {np.sum(np.isinf(cf_r))}")
theta=1.2

iteration_count=0
## INPUT DATA ANALYSIS

var_labs=list(np.zeros((nlinks)))
for i in range(0,nlinks):
    var_labs[i] = print("L_{%3.0f}" % i)
    #var_labs{i}=sprintf('L_{#-.0f}',i);

done=0  #calculate iterations
factor=0.01 #fftt and capacity will be multiplied bu this factor

## CALCULATION OF INCREASE OF TOTAL TRAVEL COST
if not done:
    t = timeit.default_timer()
    #t = time.process_time()

    #--- Reference value: network with no damage

    [Xi,D_r1,intTrec_i,ref]=SUE_C_Logit(nroutes,D_od,par,delta_ir,delta_odr,cf_r,theta)
    Results = {'Xi':Xi, "D_r1":D_r1,'ref':ref}

    # Sum values of Xi for each row
    Xi_sum = np.sum(Xi, axis=1)

    # Store Xi_sum as csv but using number through pd df
    pd.DataFrame(Xi_sum).to_csv("data/traffic_flow/Xi_sum.csv", header=False, index=False)

    # Store D_r1 as csv but using number through pd df
    pd.DataFrame(D_r1).to_csv("data/traffic_flow/D_r1.csv", header=False, index=False)

    # Store intTrec_i as csv using pd df
    pd.DataFrame(intTrec_i).to_csv("data/traffic_flow/intTrec_i.csv", header=False, index=False)


    #--- Damaging link by link
    #Results=zeros(nlinks,1) #sol for each i
    #tic = time.time()
    #parfor i=1:nlinks
    #    #Reduction of capacity and ffftt of the affected link
    #    par2=par;
    #    par2.fftt_i(i)=par2.fftt_i(i)/factor;
    #    par2.Xmax_i(i)=par2.Xmax_i(i)*factor;
    #    # cost evaluation
    #    [~,~,fval]=SUE_C_Logit(nroutes,D_od,par2,delta_ir,delta_odr,cf_r,theta);
    #    fprintf('Evaluation i=#-3.0f  -> #10.2f (#10.2f ##)\n',i,fval, (fval-ref)/ref*100)
    #    Results(i)=fval;
    #end
    #toc = time.time()
    #print(toc-tic, ' sec elapsed')
    comptime=timeit.default_timer()-t;
    print(f'CPU time (seconds): {comptime}')
    #
    print(Results)
    scipy.io.savemat('data/traffic_flow/mat/Critical_v01.mat',Results)
else: #thus, it is done
    Results = scipy.io.loadmat('data/traffic_flow/mat/Critical_v01',Results)
 #done


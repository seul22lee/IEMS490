# Data processing for melt pool temperature and melt pool width

import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import Rbf, RBFInterpolator
import matplotlib.pyplot as plt
import os


def get_meltpool_temp_width_depth(GAMMA,solidus_temp, timestep, indexes_laser_on):
    # get node location
    nodes_df = pd.DataFrame(GAMMA.domain.nodes.get(),columns=['x','y','z'])
    nodes_df['node_number'] = np.arange(len(nodes_df))
    nodes_df["temperature"] = GAMMA.heat_solver.temperature.get()
    mask_remove_negative_z = nodes_df["z"] > -3.4
    nodes_df = nodes_df[mask_remove_negative_z]
    
    laser_location = GAMMA.heat_solver.laser_loc

    # select nodes that is birth
    birth_temperature = 400
    df_results, nodes = get_nodes_above_solidus_dataframe(GAMMA.heat_solver.temperature, 400 ,timestep)

    if np.sum(GAMMA.heat_solver.laser_direction.get()) != 0:
        tool_path_df = np.round(GAMMA.heat_solver.laser_direction.get().reshape(1,-1))
        tool_path_df = pd.DataFrame(data=tool_path_df,columns=['x','y','z'])
        tool_path_df['time'] = timestep
        tool_path_df.fillna(0,inplace=True)

        # find the current top layer and the nodes belongs to the top layer
        nodes_activated = nodes_df[nodes_df['temperature'] > birth_temperature]
        max_z = nodes_activated['z'].max()
        nodes_top_layer = nodes_activated[nodes_activated['z']==max_z]
        

        # select a number of nodes to fit an RBF
        square_region = 6 #mm
        in_square_cond = (nodes_top_layer['x']<=float(laser_location[0])+square_region/2) & \
                        (nodes_top_layer['x']>=float(laser_location[0])-square_region/2) & \
                            (nodes_top_layer['y']<=float(laser_location[1])+square_region/2) & \
                                (nodes_top_layer['y']>=float(laser_location[1])-square_region/2)
        nodes_top_layer_for_RBF = nodes_top_layer[in_square_cond]
        # fit RBF
        try:
            rbf_surf = Rbf(nodes_top_layer_for_RBF['x'], nodes_top_layer_for_RBF['y'], nodes_top_layer_for_RBF['temperature'], function='linear')
            x_range = np.linspace(np.max([laser_location[0].get()-square_region/2,-20]), np.min([laser_location[0].get()+square_region/2,20]), 30)  
            y_range = np.linspace(np.max([laser_location[1].get()-square_region/2,-20]), np.min([laser_location[1].get()+square_region/2,20]), 30)

            X, Y = np.meshgrid(x_range, y_range)
            Z = rbf_surf(X, Y) 

            # find which top layer nodes is within the scanning range of the hyper spectrum camera
            scan_radius = 0.45 # mm
            top_nodes_in_range = nodes_top_layer[(nodes_top_layer['x']- float(laser_location[0]))**2 + (nodes_top_layer['y'] - float(laser_location[1]))**2 <= scan_radius**2]

            Xr = X.ravel()
            Yr = Y.ravel()
            in_circle = np.sqrt((Xr-float(laser_location[0].get()))**2+(Yr-float(laser_location[1].get()))**2)
            mask = np.where(in_circle<=(scan_radius))
            Zr = Z.ravel()
            Z_in_circle = Zr[mask]
            melt_pool_temp = np.mean(Z_in_circle)
        
        
            # Melt pool width
            if tool_path_df["x"].item() == 1 or tool_path_df["x"].item() == -1 and tool_path_df["y"].item() == 0: # moving horizontally: look at the diff on y axis
                above_meltpool = Zr>=solidus_temp
                if sum(above_meltpool) >= 2:
                    melt_pool_width = np.max(Yr[above_meltpool]) - np.min(Yr[above_meltpool])
                else:
                        melt_pool_width = 0.0002
            elif tool_path_df["y"].item() == 1 or tool_path_df["y"].item() == -1 and tool_path_df["x"].item() == 0: # moving vertically: look at the diff on x axis
                above_meltpool = Zr>=solidus_temp
                if sum(above_meltpool) >= 2:
                    melt_pool_width = np.max(Xr[above_meltpool]) - np.min(Xr[above_meltpool])
                else:
                        melt_pool_width = 0.0002
            else:
                melt_pool_width = 0
        except:
            melt_pool_width = 0
            melt_pool_temp = 400
                
        # Melt pool depth
        if tool_path_df["x"].item() == 1 or tool_path_df["x"].item() == -1 :#and tool_path_df["y"].item() == 0: # look at x direction
            try: 
                # select the range of nodes based on laser_location
                x_range = 1.5   # 2 
                y_range = 1.5   # 1
                z_range = 4  # 2
                in_cube_cond =  (nodes_activated['x']<=float(laser_location[0])+x_range/2) & \
                                (nodes_activated['x']>=float(laser_location[0])-x_range/2) & \
                                (nodes_activated['y']<=float(laser_location[1])+y_range/2) & \
                                (nodes_activated['y']>=float(laser_location[1])-y_range/2) &\
                                (nodes_activated['z']>=float(laser_location[2])-z_range) 
                nodes_melt_pool_range = nodes_activated[in_cube_cond]
                # print("moving on x direction")
                            
                # interpolation
                #rbf_cube = Rbf(nodes_melt_pool_range['x'], nodes_melt_pool_range['y'], nodes_melt_pool_range['z'],nodes_melt_pool_range['temperature'])
                rbf_cube = RBFInterpolator(nodes_melt_pool_range[['x', 'y', 'z']].values, nodes_melt_pool_range['temperature'].values)
                grid_x, grid_y, grid_z = np.mgrid[float(laser_location[0])-x_range/2:float(laser_location[0])+x_range/2:5j,\
                                                float(laser_location[1])-y_range/2:float(laser_location[1])+y_range/2:5j,\
                                                max(float(laser_location[2])-z_range,min(nodes_melt_pool_range['z'])):float(laser_location[2]):50j]
                
                points_3D = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
                temp_interp = rbf_cube(points_3D)
                temp_interp = temp_interp.reshape(grid_x.shape)
                
                mask_above_solidus = temp_interp >= solidus_temp
                mp_z = grid_z[mask_above_solidus]

                melt_pool_depth = np.max(mp_z) - np.min(mp_z) - 0.75
            except:
                melt_pool_depth = 0        
            
        # Melt pool depth
        elif tool_path_df["y"].item() == 1 or tool_path_df["y"].item() == -1: #and tool_path_df["x"].item() == 0: # look at y direction
            try:
                # select the range of nodes based on laser_location
                y_range = 1.5
                x_range = 1.5
                z_range = 4
                in_cube_cond = (nodes_activated['x']<=float(laser_location[0])+x_range/2) & \
                                (nodes_activated['x']>=float(laser_location[0])-x_range/2) & \
                                (nodes_activated['y']<=float(laser_location[1])+y_range/2) & \
                                (nodes_activated['y']>=float(laser_location[1])-y_range/2) &\
                                (nodes_activated['z']>=float(laser_location[2])-z_range) 
                nodes_melt_pool_range = nodes_activated[in_cube_cond]
                
                #print(nodes_melt_pool_range)
                # interpolation
                rbf_cube = RBFInterpolator(nodes_melt_pool_range[['x', 'y', 'z']].values, nodes_melt_pool_range['temperature'].values)
                grid_x, grid_y, grid_z = np.mgrid[float(laser_location[0])-x_range/2:float(laser_location[0])+x_range/2:5j,\
                                                float(laser_location[1])-y_range/2:float(laser_location[1])+y_range/2:5j,\
                                                max(float(laser_location[2])-z_range,min(nodes_melt_pool_range['z'])):float(laser_location[2]):50j]
                
                points_3D = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
                temp_interp = rbf_cube(points_3D)
                temp_interp = temp_interp.reshape(grid_x.shape)
                mask_above_solidus = temp_interp >= solidus_temp
                mp_z = grid_z[mask_above_solidus]
                melt_pool_depth = np.max(mp_z) - np.min(mp_z) - 0.75
            except:
                melt_pool_depth = 0
                   
            
        else: 
            melt_pool_depth = 0


        return melt_pool_width, melt_pool_temp, melt_pool_depth
            
    
    else:
         return 0, 400, 0


def get_nodes_above_solidus_dataframe(temperature, solidus_temperature, timestep):
    """
    Collect node numbers that are above the solidus temperature over time and save in a DataFrame.
    
    :param file_path: Path to the Zarr array.
    :param solidus_temperature: The solidus temperature to compare against.
    :return: A pandas DataFrame with time steps and node numbers above solidus temperature.
    """
        
    # Convert the Zarr array to a NumPy array
    temp = np.array(temperature.get())
    
    # List to accumulate results
    results = []
    
    # Iterate over each time step (row in the numpy array)

        # Find indices (node numbers) where temperature is above the solidus temperature

    nodes = np.where(temp > solidus_temperature)[0]
    
    # Append a dictionary for each time step with nodes above the solidus temperature
    results.append({"Time Step": timestep, "Nodes Above Solidus": nodes})

    # Convert results to a DataFrame
    df_results = pd.DataFrame(results)

    return df_results, nodes


def fit_temperature_surface(nodes_top_layer):
    if len(nodes_top_layer) <2 or nodes_top_layer.empty:
        return None, None
    else:
        rbf = Rbf(nodes_top_layer['x'], nodes_top_layer['y'], nodes_top_layer['temperature'], function='linear')
        return rbf, nodes_top_layer
    
def calculate_line(unit_vector, max_temp_point, line_length):
        if (abs(unit_vector['x'].astype('int')) > abs(unit_vector['y'].astype('int'))).any():
            line_x = np.linspace(max_temp_point['x'] - line_length/2, max_temp_point['x'] + line_length/2, num=100)
            line_y = np.full(100, max_temp_point['y'])
        else:
            line_x = np.full(100, max_temp_point['x'])
            line_y = np.linspace(max_temp_point['y'] - line_length/2, max_temp_point['y'] + line_length/2, num=100)
        return line_x, line_y

def calculate_melt_pool_width(line_x, line_y, line_z, solidus_temp):
        intersections = np.where(np.diff(np.sign(line_z - solidus_temp)) != 0)[0]
        if len(intersections) >= 2:
            x_coords = line_x[intersections]
            y_coords = line_y[intersections]
            width = np.sqrt((x_coords[-1] - x_coords[0])**2 + (y_coords[-1] - y_coords[0])**2)
            return width
        return 0
import pandas as pd
import numpy as np
# from sklearn.preprocessing import StandardScaler

def engineer_features(df):
    """
    The function creates new continuous variables, interaction terms, and groups the Soil_Type
    binary variables into climatic and geological zones based on the USFS ELU codes.
    """
    
    # Euclidean distance to hydrology
    df["Distance_To_Hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2 +
        df["Vertical_Distance_To_Hydrology"] ** 2
    )
    
    # Average hillshade
    df["Average_Hillshade"] = df[["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]].mean(axis=1)
    
    # Combined distances
    df["combined_distances"] = df[[
        "Horizontal_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points"
    ]].mean(axis=1)
    
    # Transformations of Elevation
    df["Elevation_log"] = np.log1p(df["Elevation"])
    df["Elevation_square"] = df["Elevation"] ** 2
    
    # Climatic zones based on Soil_Type (first digit of the ELU)
    climatic_zones = {
        "climatic_zone_2": list(range(1, 7)),
        "climatic_zone_3": [7, 8],
        "climatic_zone_4": list(range(9, 14)),
        "climatic_zone_5": [14, 15],
        "climatic_zone_6": [16, 17, 18],
        "climatic_zone_7": list(range(19, 35)),
        "climatic_zone_8": list(range(35, 41))
    }
    
    for zone_name, soil_list in climatic_zones.items():
        soil_columns = [f"Soil_Type{i}" for i in soil_list]
        df[zone_name] = df[soil_columns].sum(axis=1).clip(upper=1)
    
    # Geological zones based on Soil_Type (second digit of the ELU)
    geological_zones = {
        "geological_zone_1": [14, 15, 16, 17, 19, 20, 21],
        "geological_zone_2": [9, 22, 23],
        "geological_zone_5": [7, 8],
        "geological_zone_7": [1,2,3,4,5,6,10,11,12,13,18,24,25,26,
                              27,28,29,30,31,32,33,34,35,36,37,38,39,40]
    }
    
    for zone_name, soil_list in geological_zones.items():
        soil_columns = [f"Soil_Type{i}" for i in soil_list]
        df[zone_name] = df[soil_columns].sum(axis=1).clip(upper=1)
    
    # Unique wilderness area index (assuming that each observation has only one active wilderness area)
    df["wilderness_area_index"] = (
        df["Wilderness_Area1"] * 1 +
        df["Wilderness_Area2"] * 2 +
        df["Wilderness_Area3"] * 3 +
        df["Wilderness_Area4"] * 4
    )
    df["wilderness_area_elevation_interaction"] = df["Elevation"] * df["wilderness_area_index"]
    
    #Normalization:
    # continuous_cols = ["Elevation", "Aspect", "Slope", "Distance_To_Hydrology",
    #                    "combined_distances", "Elevation_log", "Elevation_square",
    #                    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    # scaler = StandardScaler()
    # df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
    
    return df

import pandas as pd
import pickle
import numpy as np
import rasterio
import pystac_client
from datetime import datetime
from odc.stac import stac_load
from planetary_computer import sign


def load_data(filepath):
    raw_data = pd.read_csv(filepath)
    data = pd.DataFrame(raw_data)
    print("How many data:",len(data))
    return data


class SentinelDataProcessor:
    def __init__(self, stac_url="https://planetarycomputer.microsoft.com/api/stac/v1"):
        self.stac = pystac_client.Client.open(stac_url)
        
    def fetch_data(self, bounds, time_window, cloud_cover_threshold=30, scale=None):
        """
        Fetch Sentinel-2 data based on parameters
        """
        search = self.stac.search(
            bbox=bounds,
            datetime=time_window,
            collections=["sentinel-2-l2a"],
            query={"eo:cloud_cover": {"lt": cloud_cover_threshold}},
        )
        self.items = list(search.get_items())
        print(f'Number of scenes found: {len(self.items)}')
        
        # Load the data
        self.data = stac_load(
            self.items,
            bands=["B01", "B02", "B03", "B04", "B05", "B06", 
                  "B07", "B08", "B8A", "B11", "B12"],
            crs="EPSG:4326",
            resolution=scale,
            chunks={"x": 2048, "y": 2048},
            dtype="uint16",
            patch_url=sign,
            bbox=bounds
        )
        
        return self.data
    
    def process_and_save(self, lower_left, upper_right, width, height, 
                        output_pickle=None):
    
        # Set up the geographic transform
        gt = rasterio.transform.from_bounds(
            lower_left[1], lower_left[0],
            upper_right[1], upper_right[0],
            width, height
        )
         # Convert xarray data to numpy arrays for each band
        b01_data = self.data.B01.values
        b04_data = self.data.B04.values
        b06_data = self.data.B06.values
        b08_data = self.data.B08.values
        
        # Create DataFrame structure
        df_data = {
            'latitude': [],
            'longitude': [],
            'B01': [],
            'B04': [],
            'B06': [],
            'B08': []
        }
        
        # Extract coordinates
        lats = self.data.latitude.values
        lons = self.data.longitude.values
        
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lats): #lats is bigger than lon...
                df_data['latitude'].append(lat)
                df_data['longitude'].append(lon)
                df_data['B01'].append(int(b01_data[i, j]))
                df_data['B04'].append(int(b04_data[i, j]))
                df_data['B06'].append(int(b06_data[i, j]))
                df_data['B08'].append(int(b08_data[i, j]))
    
        
        # Create DataFrame
        df = pd.DataFrame(df_data)
    
        
        if output_pickle:
            with open(output_pickle, 'wb') as f:
                pickle.dump({
                    'data': df,
                    'metadata': {
                        'bounds': [lower_left, upper_right],
                        'width': width,
                        'height': height,
                        'crs': 'epsg:4326',
                        'transform': gt
                    }
                }, f)

            
        return df

def preprocess(filename):
        # Initialize the processor
    processor = SentinelDataProcessor()

    # Define your parameters
    lower_left = (40.75, -74.01)
    upper_right = (40.88, -73.86)
    time_window = "2021-06-01/2021-09-01"
    resolution = 10  # meters per pixel 
    scale = resolution / 111320.0
    
    bounds = [lower_left[1], lower_left[0], upper_right[1], upper_right[0]]

    # Fetch the data
    data = processor.fetch_data(bounds, time_window, scale=scale)
    height = len(data.latitude)
    width = len(data.longitude)
    print(f"Dimensions - H: {height}, W:{width}")
    
    # Process and save the data
    df = processor.process_and_save(
        lower_left=lower_left,
        upper_right=upper_right,
        width=width,
        height=height,
        output_pickle=f'{filename}.pkl'
    ) 
    
    return df

def main():
    # Prep raw data
    train_df = load_data('C:/Users/dotto/Portfolio/DS+ML/EY/Training_data_uhi_index_UHI2025-v2.csv')
    val_df = load_data('C:/Users/dotto/Portfolio/DS+ML/EY/Submission_template_UHI2025-v2.csv') 
    
    #prepped_train = preprocess('train_v1')
    #prepped_val = preprocess('val_v1')
    
    print(train_df.groupby(['Latitude']))


if __name__ == '__main__':
    main()
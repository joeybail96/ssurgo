import rasterio  # reading TIFF files
import numpy as np
import xarray as xr
from datetime import datetime
from pyproj import Transformer  # convert the coordinates from the projection used in the TIFF file to geographic coordinates
from osgeo import gdal # GDAL is a translator library for raster and vector geospatial data formats 
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xesmf as xe
import cartopy.feature as cartf
import matplotlib.ticker as mticker
from matplotlib.ticker import FixedLocator
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap
import rioxarray
from rasterio.enums import Resampling
from matplotlib.colors import LogNorm


def pl_USA_zoomed(self, sf, data_variable, title, clbar_label, output_png_path, 
                  bounding_box, clbar_vmin=None, clbar_vmax=None):
    """
    Plots a zoomed-in map of the specified data variable over a given bounding box area.
    
    Parameters:
    - sf: The source data (e.g., a DataFrame)
    - data_variable: The data variable to plot (e.g., 'PM10')
    - title: The title of the plot
    - clbar_label: The colorbar label
    - output_png_path: Path to save the output plot
    - bounding_box: A tuple (lon_min, lon_max, lat_min, lat_max) defining the map's extent
    - clbar_vmin: Minimum value for colorbar (optional)
    - clbar_vmax: Maximum value for colorbar (optional)
    """
    
    fig = plt.figure(figsize=(16, 12))
    data_crs = ccrs.PlateCarree()
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    lon_min, lon_max, lat_min, lat_max = bounding_box
    
    plt.title(title, fontsize=20, pad=10)
    ax.coastlines()
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 12}
    
    lon_offset = 0.625 / 2
    lat_offset = 0.5 / 2
    gl.xlocator = FixedLocator(np.arange(-180 + lon_offset, 180, 0.625))
    gl.ylocator = FixedLocator(np.arange(-90 + lat_offset, 90, 0.5))
    
    ax.add_feature(cartf.OCEAN)
    ax.add_feature(cartf.LAND, edgecolor='black')
    ax.add_feature(cartf.BORDERS)
    
    # Add state boundaries
    ax.add_feature(cartf.STATES, edgecolor='white', linestyle=':', linewidth=3)
    
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    # Add a red marker at a specific location (e.g., the "star")
    ax.scatter(-105.0, 40.05, color='red', edgecolor='black', s=500, marker='o',
               linewidth=2, transform=ccrs.PlateCarree(), zorder=10)
    
    data = sf[data_variable]
    lon2d, lat2d = np.meshgrid(sf.lon.values, sf.lat.values)
    
    # Setup log-scaled colorbar limits
    if clbar_vmin is None:
        clbar_vmin = float(np.nanmin(data.values))
    if clbar_vmax is None:
        clbar_vmax = float(np.nanmax(data.values))
    
    # Avoid zero values for log scaling
    safe_vmin = max(clbar_vmin, 1e-3)
    
    # Set colormap and normalization
    cmap = plt.get_cmap('viridis')
    norm = LogNorm(vmin=safe_vmin, vmax=clbar_vmax)
    
    mesh = ax.pcolormesh(lon2d, lat2d, data.values, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    
    # Add colorbar with logarithmic scale
    cbar = fig.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, aspect=50)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(clbar_label, fontsize=16)
    
    # Add text labels to each grid cell in the bounding box
    lon_centers = sf.lon.values
    lat_centers = sf.lat.values
    values = data.values
    
    for j in range(len(lat_centers)):
        for i in range(len(lon_centers)):
            if lon_min <= lon_centers[i] <= lon_max and lat_min <= lat_centers[j] <= lat_max:
                val = values[j, i]
                # Display values in the center of the grid with 3 decimal places
                txt = f"{val:.3f}" if not np.isnan(val) else "NaN"
                ax.text(
                    lon_centers[i], lat_centers[j], txt,
                    color="black", ha="center", va="center", fontsize=30, zorder=5
                )
    
    # Save and show plot
    plt.savefig(output_png_path)
    plt.show()
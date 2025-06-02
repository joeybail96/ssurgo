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



class Tiff_Regrid:
    def __init(self):
        self = None
    
    # convert tiff from EPSG 5070 -> 4326
    def convert4326(self, input_tiff_path, output_tiff_path):
        input_ds = gdal.Open(input_tiff_path)
        options = gdal.WarpOptions(
            format="GTiff",
            outputType = gdal.GDT_Float32,
            srcSRS = "EPSG:5070", # Use rios
            dstSRS ="EPSG:4326",
            resampleAlg = gdal.GRA_Average
        )
        gdal.Warp(output_tiff_path, input_ds, options = options)  # outout file to the output_tiff_path
        return output_tiff_path

    # convert tiff to nerCDF
    def tiff_CDF(self, output_cdf_path, target_variable, longname):
        dataset = gdal.Open(self.tiff_4326_path)
        geotransform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()
        # get lat and lon
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        x_indices = np.arange(width)
        y_indices = np.arange(height)
        lat = geotransform[3] + y_indices * geotransform[5]
        lon = geotransform[0] + x_indices * geotransform[1]
        # get data in array format
        band = dataset.GetRasterBand(1)
        data = band.ReadAsArray()
        # set minimum data to 0 # i should trigger error
        data[data == 99999] = np.nan
        # create the netCDF
        lat_array = xr.DataArray(data= lat, name = "lat", dims = "lat",
                                            attrs = {"long_name":"Latitude",
                                            "units" : "degrees_north",
                                            "axis" : "Y"})
        lon_array = xr.DataArray(data= lon, name = "lon", dims = "lon",
                                            attrs = {"long_name":"Longitude",
                                                    "units" : "degrees_east",
                                                    "axis" : "X"})
        coords_dict = {"lat" : lat_array,"lon" : lon_array}

        ec_array = xr.DataArray(data = data, name = target_variable, # set the variable name
                                                dims=["lat", "lon"],
                                                attrs = {'long_name':longname}) # describe this variable
        ec_ds = xr.Dataset(data_vars = {target_variable: ec_array},
                            coords = coords_dict)
        # save CDF
        ec_ds.to_netcdf(output_cdf_path)
        return ec_ds

    #    
    def playa_mask(self, output_cdf_path, target_variable, ec_var, longname, threshold):
    
        # Copy the dataset
        playa_mask_ds = self.ec_ncfile.copy()
        
        # Apply threshold: 1 if >= threshold, 0 otherwise
        playa_mask = (playa_mask_ds[ec_var] >= threshold).astype(float)
        
        # Create a DataArray for the target variable
        playa_mask_da = (playa_mask_ds[ec_var].copy() if isinstance(playa_mask_ds[ec_var], np.ndarray) 
                         else playa_mask_ds[ec_var].copy())
        
        # Update the values in the new DataArray with the thresholded mask
        playa_mask_da.values = playa_mask
        
        # Assign the modified DataArray to the target variable
        playa_mask_ds[target_variable] = playa_mask_da
        
        # Set the long_name attribute for the target variable
        playa_mask_ds[target_variable].attrs["long_name"] = longname
        
        # Drop the original ec_var from the dataset
        playa_mask_ds = playa_mask_ds.drop_vars(ec_var)
        
        # Save the modified dataset to a netCDF file
        playa_mask_ds.to_netcdf(output_cdf_path)
        
        # Store the result in the class attribute
        return playa_mask_ds
        


    def derive_cl(self, output_cdf_path, target_variable, og_var, longname):
        
        playa_cl_ds = self.regrid_playa_mask_ncfile.copy()
        
        playa_cl_ds[target_variable] = playa_cl_ds[og_var] * 0.0412
        
        playa_cl_ds[target_variable].attrs["long_name"] = longname
        
        playa_cl_ds = playa_cl_ds.drop_vars(og_var)
        
        playa_cl_ds.to_netcdf(output_cdf_path)
        
        return playa_cl_ds
        
        

    # Regrid netDF files       
    def regrid(self, source, template, output_regrid_path, regrid_method):
        """
        Regrid the source dataset to match the resolution and extent of the template grid,
        clipped to the source domain.
        """
    
        # Get bounds of the source data
        lat_min, lat_max = source.lat.min().item(), source.lat.max().item()
        lon_min, lon_max = source.lon.min().item(), source.lon.max().item()
    
        # Clip template grid to the bounds of the source
        clipped_template = template.sel(
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max)
        )
    
        # Build the target grid from the clipped template coordinates
        target_grid = xr.Dataset(
            {
                "lat": (["lat"], clipped_template.lat.data),
                "lon": (["lon"], clipped_template.lon.data),
            }
        )
    
        # Perform regridding
        regridder = xe.Regridder(source, target_grid, method=regrid_method, periodic=False, reuse_weights=False)
        ds_target = regridder(source)
    
        # Save regridded output
        ds_target.to_netcdf(output_regrid_path)
    
        return ds_target

    def regrid_dask(self, output_regrid_path, regrid_method, lat_grid, lon_grid):
        # Create the target grid as an xarray Dataset
        target_lon = np.arange(-130, -64, lon_grid)
        target_lat = np.arange(21, 53, lat_grid)
        
        target_grid = xr.Dataset(
            {
                "lon": ("lon", target_lon),
                "lat": ("lat", target_lat)
            }
        )
    
        # Convert the source dataset into a Dask array for more efficient chunking
        dask_data = self.playa_mask_ncfile.chunk({'lat': 3, 'lon': 3})  # Adjust chunk size based on your memory
    
        # Initialize the regridder with the Dask array
        regridder = xe.Regridder(dask_data, target_grid, method=regrid_method, reuse_weights=False, periodic=False, ignore_degenerate=True)
        
        # Perform the regridding on the chunked data
        ds_target = regridder(dask_data)
    
        # Compute the result (this will trigger the actual regridding process)
        ds_target.compute().to_netcdf(output_regrid_path)
        
        print(f"Regridded data saved to {output_regrid_path}")


    def chunk_and_regrid(self, output_regrid_path, regrid_method, lat_grid, lon_grid, chunk_size=500):

        # 1. Create the target grid
        target_lon = np.arange(-130, -64 + lon_grid, lon_grid)
        target_lat = np.arange(21, 53 + lat_grid, lat_grid)
        target_grid = xr.Dataset(
            {
                "lon": ("lon", target_lon),
                "lat": ("lat", target_lat)
            }
        )
    
        # 2. Prepare output container
        output_data = []
    
        # 3. Loop through chunks of the input lat dimension
        n_lats = self.playa_mask_ncfile.dims["lat"]
        for start in range(0, n_lats, chunk_size):
            end = min(start + chunk_size, n_lats)
            chunk = self.playa_mask_ncfile.isel(lat=slice(start, end))
    
            # 4. Regrid each chunk
            print(f"Regridding lat rows {start} to {end}...")
            regridder = xe.Regridder(chunk, target_grid, method=regrid_method, reuse_weights=False)
            chunk_regridded = regridder(chunk)
    
            output_data.append(chunk_regridded)
    
            # Clean up memory if needed
            del regridder 
            
        # 5. Combine all regridded chunks
        combined = xr.concat(output_data, dim="lat").sortby("lat")
        
        combined = combined.rename({'Playa_Mask': 'Playa_Mask_Regridded'})
    
        # 6. Save to netCDF
        combined.to_netcdf(output_regrid_path)
        print(f"Regridded data saved to {output_regrid_path}")
    
        self.regrid_playa_mask_ncfile = combined


    #
    def coarsen_nc_resolution(self, input_nc_path, output_nc_path, var_name='Playa_Mask', scale_factor=0.01):
        """
        Resamples the resolution of a NetCDF file using averaging and returns a dataset
        with lat/lon dimensions that match the structure of the original.
    
        Parameters:
        - input_nc_path: Path to input NetCDF file
        - output_nc_path: Path to save the downsampled NetCDF file
        - var_name: Name of the variable to downsample
        - scale_factor: Fractional resolution scaling (e.g., 0.01 = 1% resolution)
    
        Returns:
        - xarray.Dataset: The coarsened dataset
        """
        # Load and write CRS
        ds = xr.open_dataset(input_nc_path)
        data = ds[var_name].rio.write_crs("EPSG:4326", inplace=True)
    
        # Original dimensions
        orig_height, orig_width = data.shape
        new_height = int(orig_height * scale_factor)
        new_width = int(orig_width * scale_factor)
    
        # Resample using average
        resampled = data.rio.reproject(
            dst_crs="EPSG:4326",
            shape=(new_height, new_width),
            resampling=Resampling.average
        )
    
        # Rename dims from x/y to lon/lat
        resampled = resampled.rename({'x': 'lon', 'y': 'lat'})
    
        # Assign coordinate values
        resampled = resampled.assign_coords({
            'lat': resampled['lat'].values,
            'lon': resampled['lon'].values
        })
    
        # Repackage into Dataset and set attributes if needed
        downsampled_ds = resampled.to_dataset(name=var_name)
        downsampled_ds[var_name].attrs = ds[var_name].attrs  # copy variable attributes
    
        # Save and return
        downsampled_ds.to_netcdf(output_nc_path)
        print(f"Resampled dataset saved to: {output_nc_path}")
        return downsampled_ds



    # Plot CDF
    def pl_USA(self, sf, data_variable, title, clbar_label, output_png_path, colormap="viridis", clbar_vmin=None, clbar_vmax=None):
        """
        Plots a map of the specified data variable on the USA using Cartopy.
    
        Parameters:
        - sf (xarray.Dataset): The xarray dataset containing the variable.
        - data_variable (str): The name of the variable to plot (e.g., 'Playa_Mask').
        - title (str): The title of the plot.
        - clbar_label (str): The label for the colorbar.
        - output_png_path (str): The file path to save the output PNG image.
        - colormap (str): The colormap to use for the plot. Default is "viridis".
        - clbar_vmin (float or None): The lower bound for the colorbar. If None, it is automatically determined.
        - clbar_vmax (float or None): The upper bound for the colorbar. If None, it is automatically determined.
    
        Returns:
        - None: Displays and saves the plot.
        """
        fig = plt.figure(figsize=(21, 14))
        data_crs = ccrs.PlateCarree()
        glob_ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Set title and coastlines
        plt.title(title, fontsize=20, pad=10)
        glob_ax.coastlines()
        
        
        # Add land and ocean features
        glob_ax.add_feature(cartf.OCEAN)
        glob_ax.add_feature(cartf.LAND, edgecolor='black')
        glob_ax.add_feature(cartf.BORDERS)
        
        # Get the data variable
        data = sf[data_variable]
    
        # Create meshgrid of lon and lat
        lon2d, lat2d = np.meshgrid(sf.lon.values, sf.lat.values)
    
        # Plot with pcolormesh and specify the colorbar range if provided
        mesh = glob_ax.pcolormesh(lon2d, lat2d, data.values, cmap=colormap, vmin=clbar_vmin, vmax=clbar_vmax, transform=ccrs.PlateCarree())
    
        # Add colorbar with specified range
        cbar = fig.colorbar(mesh, location="right", shrink=0.6)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label(clbar_label, fontsize=18)
        
        # Save and show plot
        plt.savefig(output_png_path)
        plt.show()

        
        

    def pl_downscaled(self, sf, downscaled_sf, data_variable, title, clbar_label, output_png_path, 
                      bounding_box, clbar_vmin=None, clbar_vmax=None):
        
        fig = plt.figure(figsize=(16, 12))
        data_crs = ccrs.PlateCarree()
        ax = plt.axes(projection=data_crs)
    
        lon_min, lon_max, lat_min, lat_max = bounding_box
    
        plt.title(title, fontsize=20, pad=10)
        ax.coastlines()
    
        # Calculate spacing for sf grid
        lon_spacing = np.abs(sf.lon.values[1] - sf.lon.values[0])
        lat_spacing = np.abs(sf.lat.values[1] - sf.lat.values[0])
    
        # Define gridlines for each cell (sf grid)
        lon_edges = np.round(sf.lon.values - lon_spacing / 2, 6)
        lat_edges = np.round(sf.lat.values - lat_spacing / 2, 6)
    
        # Add fine gridlines that align with data resolution (sf gridlines)
        gl = ax.gridlines(draw_labels=True, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 12}
        gl.ylabel_style = {"size": 12}
        gl.xlocator = FixedLocator(lon_edges)
        gl.ylocator = FixedLocator(lat_edges)
    
        # Background features
        ax.add_feature(cartf.OCEAN)
        ax.add_feature(cartf.LAND, edgecolor='black')
        ax.add_feature(cartf.BORDERS)
        ax.add_feature(cartf.STATES, edgecolor='white', linestyle=':', linewidth=1)
    
        # Set extent and axes limits
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=data_crs)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
    
        # Add star marker
        ax.scatter(-105.0, 40.05, color='red', edgecolor='black', s=500, marker='o',
                   linewidth=2, transform=data_crs, zorder=10)
    
        # Plot data for sf dataset
        data = sf[data_variable]
        lon2d, lat2d = np.meshgrid(sf.lon.values, sf.lat.values)
    
        if clbar_vmin is None:
            clbar_vmin = float(np.nanmin(data.values))
        if clbar_vmax is None:
            clbar_vmax = float(np.nanmax(data.values)) + 0.1
    
        # Define discrete levels
        n_levels = 10
        levels = np.linspace(clbar_vmin, clbar_vmax, n_levels + 1)
        cmap = get_cmap('viridis', n_levels)
        norm = BoundaryNorm(boundaries=levels, ncolors=n_levels)
    
        mesh = ax.pcolormesh(lon2d, lat2d, data.values, cmap=cmap, norm=norm, transform=data_crs)
        
        # Compute grid spacing for downscaled data
        dlon = np.abs(downscaled_sf.lon.values[1] - downscaled_sf.lon.values[0])
        dlat = np.abs(downscaled_sf.lat.values[1] - downscaled_sf.lat.values[0])
        
        # Calculate edges from center points
        downscaled_lon_edges = np.round(np.append(
            downscaled_sf.lon.values - dlon / 2, 
            downscaled_sf.lon.values[-1] + dlon / 2
        ), 6)
        
        downscaled_lat_edges = np.round(np.append(
            downscaled_sf.lat.values - dlat / 2, 
            downscaled_sf.lat.values[-1] + dlat / 2
        ), 6)
        
        # Draw vertical red lines at each downscaled longitude edge
        for lon in downscaled_lon_edges:
            ax.plot([lon, lon], [lat_min, lat_max], color='red', linewidth=2, linestyle='-', transform=data_crs, zorder=6)
        
        # Draw horizontal red lines at each downscaled latitude edge
        for lat in downscaled_lat_edges:
            ax.plot([lon_min, lon_max], [lat, lat], color='red', linewidth=2, linestyle='-', transform=data_crs, zorder=6)

        # Colorbar
        cbar = fig.colorbar(mesh, ax=ax, orientation='horizontal',
                            pad=0.05, aspect=50,
                            ticks=levels, extend='neither')
    
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(clbar_label, fontsize=16)
    
        plt.savefig(output_png_path, bbox_inches='tight', dpi=300)
        plt.show()


    def pl_downscaled_only(self, sf, data_variable, title, clbar_label, output_png_path, 
                          bounding_box, clbar_vmin=0.001, clbar_vmax=1.0):
    
        fig = plt.figure(figsize=(16, 12))
        data_crs = ccrs.PlateCarree()
        ax = plt.axes(projection=ccrs.PlateCarree())
    
        lon_min, lon_max, lat_min, lat_max = bounding_box
    
        plt.title(title, fontsize=20, pad=10)
        ax.coastlines()
    
        # Calculate spacing
        lon_spacing = np.abs(sf.lon.values[1] - sf.lon.values[0])
        lat_spacing = np.abs(sf.lat.values[1] - sf.lat.values[0])
    
        # Define gridlines for each cell
        lon_edges = np.round(sf.lon.values - lon_spacing / 2, 6)
        lat_edges = np.round(sf.lat.values - lat_spacing / 2, 6)
    
        # Add fine gridlines
        gl = ax.gridlines(draw_labels=True, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 12}
        gl.xlocator = FixedLocator(lon_edges)
        gl.ylocator = FixedLocator(lat_edges)
    
        # Background features
        ax.add_feature(cartf.OCEAN)
        ax.add_feature(cartf.LAND, edgecolor='black')
        ax.add_feature(cartf.BORDERS)
        ax.add_feature(cartf.STATES, edgecolor='white', linestyle=':', linewidth=1)
    
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
        # Star marker
        ax.scatter(-105.0, 40.05, color='red', edgecolor='black', s=500, marker='o',
                   linewidth=2, transform=ccrs.PlateCarree(), zorder=10)
    
        # Plot data
        data = sf[data_variable]
        lon2d, lat2d = np.meshgrid(sf.lon.values, sf.lat.values)
    
        # Logarithmic color scale
        safe_vmin = max(clbar_vmin, 1e-3)
        cmap = plt.get_cmap('viridis')
        norm = LogNorm(vmin=safe_vmin, vmax=clbar_vmax)
    
        mesh = ax.pcolormesh(lon2d, lat2d, data.values, cmap=cmap, norm=norm,
                             transform=ccrs.PlateCarree())
    
        # Overlay value labels (only within bounding box)
        ny, nx = data.shape
        for j in range(ny):
            for i in range(nx):
                lon = sf.lon.values[i]
                lat = sf.lat.values[j]
                val = data.values[j, i]
    
                if (lon_min <= lon <= lon_max) and (lat_min <= lat <= lat_max):
                    if not np.isnan(val) and val > 0:
                        ax.text(
                            lon,
                            lat,
                            f"{val:.3f}",
                            transform=ccrs.PlateCarree(),
                            ha='center',
                            va='center',
                            fontsize=30,
                            color='black',
                            zorder=11
                        )
    
        # Colorbar
        cbar = fig.colorbar(mesh, ax=ax, orientation='horizontal',
                            pad=0.05, aspect=50, extend='both')
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(clbar_label, fontsize=16)
    
        # Save and show
        plt.savefig(output_png_path)
        plt.show()



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



    
    def plot_histogram_nonzeros(self, data, bins=100, x_tick_increment=0.05):
        """
        Plots a histogram of non-zero values from an xarray.DataArray.
    
        Parameters:
        - data (xarray.DataArray): The xarray DataArray containing the values.
        - bins (int): Number of bins for the histogram. Default is 100.
        - x_tick_increment (float): The increment for x-axis ticks. Default is 0.05.
    
        Returns:
        - None: Displays the histogram plot.
        """
        # Extract non-zero values
        non_zero_values = data.values[data.values != 0]
    
        # Plot histogram
        plt.figure(figsize=(10, 6))
        counts, bins_edges, patches = plt.hist(non_zero_values, bins=bins, color='skyblue', edgecolor='black')
    
        # Set x-ticks to increment by the specified value
        x_ticks = np.arange(min(bins_edges), max(bins_edges), x_tick_increment)
        plt.xticks(x_ticks, rotation=45)
    
        # Set y-axis to log scale
        plt.yscale('log')
    
        plt.title("Histogram of Non-Zero Values (Log Scale)")
        plt.xlabel("Value")
        plt.ylabel("Log(Frequency)")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()
import rasterio
import xarray as xr
import rioxarray
import numpy as np
import xesmf as xe
from affine import Affine
from rasterio.enums import Resampling
import xarray as xr
import rioxarray
from osgeo import gdal
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator
import cartopy.feature as cartf
import cartopy.feature as cfeature
from pyproj import Transformer
import pyproj
from matplotlib.ticker import MaxNLocator
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


class TifProcessor:
    def __init__(self):
        pass

    def tif_to_coards_netcdf(self, input_tif, output_nc, var_name, long_name, units):
        dataset = gdal.Open(input_tif)
        geotransform = dataset.GetGeoTransform()
    
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        x_indices = np.arange(width)
        y_indices = np.arange(height)
    
        # Projected coordinates (e.g., meters)
        x = geotransform[0] + x_indices * geotransform[1]
        y = geotransform[3] + y_indices * geotransform[5]
    
        # Read data
        band = dataset.GetRasterBand(1)
        data = band.ReadAsArray().astype(np.float32)
        data[data == 99999] = np.nan  # Handle NoData
    
        # Build coordinate arrays
        x_array = xr.DataArray(data=x, name="x", dims="x",
                               attrs={"long_name": "Easting", "units": "meters", "axis": "X"})
        y_array = xr.DataArray(data=y, name="y", dims="y",
                               attrs={"long_name": "Northing", "units": "meters", "axis": "Y"})
    
        # Main data variable
        data_array = xr.DataArray(data=data, name=var_name, dims=["y", "x"],
                                  attrs={"long_name": long_name, "units": units})
    
        # Combine
        ds = xr.Dataset({var_name: data_array}, coords={"x": x_array, "y": y_array})
        ds.rio.write_crs("EPSG:5070", inplace=True)
    
        ds.to_netcdf(output_nc)
        return ds
    
    
    
    def mask_by_threshold(self, ds, target_variable, ec_var, threshold, longname, output_cdf_path=None):
        masked_ds = ds.copy()
        mask = (masked_ds[ec_var] >= threshold).astype(float)
    
        mask_da = masked_ds[ec_var].copy()
        mask_da.values = mask
        mask_da.name = target_variable
        mask_da.attrs["long_name"] = longname
    
        masked_ds[target_variable] = mask_da
        masked_ds = masked_ds.drop_vars(ec_var)
    

        masked_ds.to_netcdf(output_cdf_path)
    
        return masked_ds
    
    
    
    def resample_dataset(self, ds, output_nc_path, var_name='Playa_Mask', scale_factor=0.01):
        data = ds[var_name].rio.write_crs("EPSG:5070", inplace=True)
    
        orig_height, orig_width = data.shape
        new_height = max(1, int(orig_height * scale_factor))
        new_width = max(1, int(orig_width * scale_factor))
    
        resampled = data.rio.reproject(
            dst_crs="EPSG:5070",  # Stay in projected space
            shape=(new_height, new_width),
            resampling=Resampling.average
        )
    
        # Keep projected coord names
        resampled = resampled.rename({'x': 'x', 'y': 'y'})
        resampled = resampled.assign_coords({
            'x': resampled['x'].values,
            'y': resampled['y'].values
        })
    
        downsampled_ds = resampled.to_dataset(name=var_name)
        downsampled_ds[var_name].attrs = ds[var_name].attrs
        downsampled_ds[var_name].attrs['scale_factor'] = scale_factor
    
        downsampled_ds.to_netcdf(output_nc_path)
        print(f"Resampled dataset saved to: {output_nc_path}")
        
        return downsampled_ds
    


    def coarsen_dataset(self, ds, output_nc_path, var_name='Playa_Mask', factor=10):
        # Extract the data array
        data = ds[var_name]
        
        # Get the original shape of the data
        original_shape = data.shape
        y_len, x_len = original_shape
        
        # Trim the data to be divisible by the factor
        y_trim = y_len - (y_len % factor)
        x_trim = x_len - (x_len % factor)
        
        # Trim the array to the new shape
        trimmed_data = data.isel(y=slice(0, y_trim), x=slice(0, x_trim))
        
        # Convert to numpy array for coarsening
        data_array = trimmed_data.values
        
        # Step 1: Reshape the array into blocks of size (factor, factor) and average
        coarsened_data = data_array.reshape(
            data_array.shape[0] // factor, factor, data_array.shape[1] // factor, factor
        ).mean(axis=(1, 3))  # Average across each block
        
        # Step 2: Convert back to an xarray DataArray with the new dimensions
        coarsened_da = xr.DataArray(coarsened_data, coords=[trimmed_data['y'][::factor], trimmed_data['x'][::factor]], dims=['y', 'x'])
    
        # Package the coarsened data into a new dataset
        coarsened_ds = coarsened_da.to_dataset(name=var_name)
        
        coarsened_ds.to_netcdf(output_nc_path)
        print(f"Coarsened dataset saved to: {output_nc_path}")
        
        return coarsened_ds    



    
    
    
    
    def reproject_to_4326(self, ds, output_nc, var_name):
        da = ds[var_name]
        da = da.rio.set_spatial_dims(x_dim='x', y_dim='y')
        da = da.rio.write_crs("EPSG:5070")
    
        reprojected = da.rio.reproject("EPSG:4326")
        reprojected = reprojected.rename({'x': 'lon', 'y': 'lat'})
        reprojected = reprojected.drop_vars('spatial_ref')
    
        reprojected.name = var_name
        ds_out = reprojected.to_dataset()
        ds_out.to_netcdf(output_nc)
    
        return ds_out





    def regrid_to_template(self, source_ds, template_ds, output_nc_path, method='conservative'):
        """
        Regrid the source dataset to match the resolution and extent of the template grid,
        clipped to the source domain.
        """
    
        # Extract source bounds
        lat_min, lat_max = source_ds.lat.min().item(), source_ds.lat.max().item()
        lon_min, lon_max = source_ds.lon.min().item(), source_ds.lon.max().item()
    
        # Clip template to source bounds
        clipped_template = template_ds.sel(
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max)
        )
    
        # Build target grid from clipped template coordinates
        target_grid = xr.Dataset(
            {
                "lat": (["lat"], clipped_template.lat.data),
                "lon": (["lon"], clipped_template.lon.data),
            }
        )
    
    
        # Perform regridding
        regridder = xe.Regridder(source_ds, target_grid, method=method, periodic=False, reuse_weights=False)
        ds_regridded = regridder(source_ds)
    
        # Save result
        ds_regridded.to_netcdf(output_nc_path)
    
        return ds_regridded


    def apply_chloride_fraction(self, mask_ds, output_nc_path, cl_fraction, output_name):
        var = list(mask_ds.data_vars)[0]
        da = mask_ds[var]
        result = da * cl_fraction
        result.name = output_name
        result.attrs["long_name"] = "Playa dust mass multiplied by chloride fraction"
        result.attrs["units"] = "kg/kg"
        result.to_netcdf(output_nc_path)

        return result.to_dataset()


    def trim_to_latlon_box(self, ds, bounding_box):
        """
        Trim an xarray dataset in EPSG:5070 using a bounding box in EPSG:4326.
    
        Parameters:
        - ds: xarray.Dataset with x/y coordinates in EPSG:5070
        - bounding_box: tuple of (min_lon, max_lon, min_lat, max_lat)
    
        Returns:
        - Trimmed xarray.Dataset
        """
        # Define the source (lat/lon) and target (EPSG:5070) CRS
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    
        # Extract and project the bounding box corners
        min_lon, max_lon, min_lat, max_lat = bounding_box
        x_min, y_min = transformer.transform(min_lon, min_lat)
        x_max, y_max = transformer.transform(max_lon, max_lat)
    
        # Sort in case projection flips axes
        x_min, x_max = sorted([x_min, x_max])
        y_min, y_max = sorted([y_min, y_max])
    
        # Trim the dataset using the projected bounding box
        ds_trimmed = ds.where(
            (ds.x >= x_min) & (ds.x <= x_max) &
            (ds.y >= y_min) & (ds.y <= y_max),
            drop=True
        )
    
        return ds_trimmed



    def plot_ds_trimmed(self, ds_trimmed, var_name, colormap, save_path=None, grid_thickness="0", show_colorbar=True, show_values=False, colorbar_limits=(0,1)):
             crs_proj = ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=23)
         
             fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': crs_proj})
         
             # Optional: map features
             ax.add_feature(cfeature.STATES, edgecolor='black', linewidth=2)
             ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=2)
             ax.coastlines(resolution='10m', color='black', linewidth=2)
         
             # Extract coordinates and data
             x = ds_trimmed['x'].values
             y = ds_trimmed['y'].values
             z = ds_trimmed[var_name].values
         
             # Adjust x and y to align with the edges of the grid cells
             x_edge = x[:-1] + np.diff(x) / 2  # Shift by half the cell width
             y_edge = y[:-1] + np.diff(y) / 2  # Shift by half the cell height
         
             # Create meshgrid for grid edges (edges are adjusted here)
             X, Y = np.meshgrid(x_edge, y_edge)
         
             # Ensure that Z has dimensions one smaller than X and Y for proper alignment
             Z = z[:-1, :-1]  # Adjust Z to be one less in both dimensions
             
             vmin, vmax = colorbar_limits
             
             # Plot with grayscale and outlined cells
             img = ax.pcolormesh(
                 X, Y, Z,
                 transform=crs_proj,
                 cmap=colormap,
                 shading='auto',  # Automatically handles shape mismatch
                 edgecolors='black',
                 linewidth=grid_thickness,
                 vmin=vmin,
                 vmax=vmax
             )
         
             if show_colorbar:
                 cbar = plt.colorbar(img, ax=ax, orientation='vertical', pad=0.05)
                 cbar.set_label(var_name)
         
             # Add gridlines corresponding to the data (fine resolution)
             ax.grid(True, which='both', color='black', linewidth=5)
         
             # Coarser gridlines (10 times coarser than the data)
             grid_resolution_x_coarse = (x[1] - x[0]) * 10  # 10 times the resolution in x
             grid_resolution_y_coarse = (y[1] - y[0]) * 10  # 10 times the resolution in y
         
             # Generate coarser gridlines
             coarse_grid_x = np.arange(x.min(), x.max(), grid_resolution_x_coarse)
             coarse_grid_y = np.arange(y.min(), y.max(), grid_resolution_y_coarse)
             
             if show_values:
                 # Loop through and add text at the center of each grid cell (using original coordinates)
                 for i in range(len(x) - 1):
                     for j in range(len(y) - 1):
                         # Calculate the center of each grid cell using original x, y coordinates
                         x_center = (x[i] + x[i + 1]) / 2
                         y_center = (y[j] + y[j + 1]) / 2
             
                         # Add text at the center of the grid cell
                         ax.text(
                             x_center, y_center, 
                             f'{z[j, i]:.2f}',  # Display the value with 2 decimal places
                             color='red', 
                             ha='center', 
                             va='center', 
                             fontsize=8
                         )
     
             plt.tight_layout()
         
             if save_path:
                 plt.savefig(save_path, dpi=300, bbox_inches='tight')
                 plt.close()
             else:
                 plt.show()  





    def plot_ds_together(self, ds1, ds2, var_name, colormap, save_path=None, show_colorbar=True, plot_ds2=False, alpha=0.3):
        crs_proj = ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=23)
    
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': crs_proj})
    
        # Optional: map features
        ax.add_feature(cfeature.STATES, edgecolor='black', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5)
        ax.coastlines(resolution='10m', color='black', linewidth=0.5)
    
        # Extract coordinates and data for ds1
        x1 = ds1['x'].values
        y1 = ds1['y'].values
        z1 = ds1[var_name].values
    
        # Adjust x and y to align with the edges of the grid cells
        x1_edge = x1[:-1] + np.diff(x1) / 2  # Shift by half the cell width
        y1_edge = y1[:-1] + np.diff(y1) / 2  # Shift by half the cell height
    
        # Create meshgrid for ds1 with adjusted edges
        X1, Y1 = np.meshgrid(x1_edge, y1_edge)
    
        # Ensure that Z1 has dimensions one smaller than X1 and Y1 for proper alignment
        Z1 = z1[:-1, :-1]  # Adjust Z1 to be one less in both dimensions
    
        # Plot ds1 (bottom layer)
        img1 = ax.pcolormesh(
            X1, Y1, Z1,
            transform=crs_proj,
            cmap=colormap,
            shading='auto',  # Automatically handles shape mismatch
            edgecolors='black',
            linewidth=0.002
        )
    
        if show_colorbar:
            cbar = plt.colorbar(img1, ax=ax, orientation='vertical', pad=0.05)  # Colorbar for ds1
            cbar.set_label(var_name)
    
        # Extract coordinates and data for ds2
        x2 = ds2['x'].values
        y2 = ds2['y'].values
        z2 = ds2[var_name].values
    
        # Adjust x and y to align with the edges of the grid cells for ds2
        x2_edge = x2[:-1] + np.diff(x2) / 2  # Shift by half the cell width
        y2_edge = y2[:-1] + np.diff(y2) / 2  # Shift by half the cell height
    
        # Create meshgrid for ds2 with adjusted edges
        X2, Y2 = np.meshgrid(x2_edge, y2_edge)
    
        # Ensure that Z2 has dimensions one smaller than X2 and Y2 for proper alignment
        Z2 = z2[:-1, :-1]  # Adjust Z2 to be one less in both dimensions
    
        if plot_ds2:
            # Plot ds2 (top layer) with alpha for transparency
            img2 = ax.pcolormesh(
                X2, Y2, Z2,
                transform=crs_proj,
                cmap=colormap,
                shading='auto',  # Automatically handles shape mismatch
                edgecolors='red',
                linewidth=2,
                alpha=alpha  # Add transparency (default 0.3)
            )
        else:
            img2 = ax.pcolormesh(
            X2, Y2, np.zeros_like(Z2),  # Dummy data, since we only want outlines
            transform=crs_proj,
            facecolors='none',          # No fill, just edges
            edgecolors='red',
            linewidth=2,
            alpha=1.0
) 
    
        if show_colorbar:
            cbar = plt.colorbar(img2, ax=ax, orientation='vertical', pad=0.05)  # Colorbar for ds2
            cbar.set_label(var_name)
    
        # Add gridlines corresponding to the data (fine resolution)
        ax.grid(True, which='both', color='black', linewidth=5)
    
        # Coarser gridlines (10 times coarser than the data)
        grid_resolution_x_coarse = (x1[1] - x1[0]) * 10  # 10 times the resolution in x
        grid_resolution_y_coarse = (y1[1] - y1[0]) * 10  # 10 times the resolution in y
    
        # Generate coarser gridlines
        coarse_grid_x = np.arange(x1.min(), x1.max(), grid_resolution_x_coarse)
        coarse_grid_y = np.arange(y1.min(), y1.max(), grid_resolution_y_coarse)
    
        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


    def plot_ds_epsg4326(
        self, ds, var_name, colormap,
        bounding_box=None,  # Optional (min_lon, max_lon, min_lat, max_lat)
        save_path=None, grid_thickness="0", show_colorbar=True,
        show_values=False, colorbar_limits=(0, 1)
    ):
        crs_proj = ccrs.PlateCarree()  # EPSG:4326 projection
    
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': crs_proj})
    
        # Map features
        ax.add_feature(cfeature.STATES, edgecolor='black', linewidth=2)
        ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=2)
        ax.coastlines(resolution='10m', color='black', linewidth=2)
    
        # Apply bounding box if provided
        if bounding_box:
            min_lon, max_lon, min_lat, max_lat = bounding_box
            ds = ds.where(
                (ds.lon >= min_lon) & (ds.lon <= max_lon) &
                (ds.lat >= min_lat) & (ds.lat <= max_lat),
                drop=True
            )
    
        # Extract data
        lon = ds['lon'].values
        lat = ds['lat'].values
        z = ds[var_name].values
    
        # Meshgrid for pcolormesh
        Lon, Lat = np.meshgrid(lon, lat)
    
        vmin, vmax = colorbar_limits
    
        img = ax.pcolormesh(
            Lon, Lat, z,
            transform=crs_proj,
            cmap=colormap,
            shading='auto',
            edgecolors='black',
            linewidth=grid_thickness,
            vmin=vmin,
            vmax=vmax
        )
    
        if show_colorbar:
            cbar = plt.colorbar(img, ax=ax, orientation='vertical', pad=0.05)
            cbar.set_label(var_name)
    
        if show_values:
            for i in range(len(lon)):
                for j in range(len(lat)):
                    ax.text(
                        lon[i], lat[j],
                        f'{z[j, i]:.2f}',
                        color='red',
                        ha='center',
                        va='center',
                        fontsize=8
                    )
    
        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()



    def plot_ds_together_4326(
        self, ds1, ds2, var_name, colormap,
        save_path=None, show_colorbar=True,
        plot_ds2=False, alpha=0.3,
        bounding_box=None  # Optional (min_lon, max_lon, min_lat, max_lat)
    ):
        crs_proj = ccrs.PlateCarree()
    
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': crs_proj})
    
        # Add map features
        ax.add_feature(cfeature.STATES, edgecolor='black', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5)
        ax.coastlines(resolution='10m', color='black', linewidth=0.5)
    
        # Apply optional bounding box
        if bounding_box:
            min_lon, max_lon, min_lat, max_lat = bounding_box
            ds1 = ds1.where((ds1.lon >= min_lon) & (ds1.lon <= max_lon) &
                            (ds1.lat >= min_lat) & (ds1.lat <= max_lat), drop=True)
            ds2 = ds2.where((ds2.lon >= min_lon) & (ds2.lon <= max_lon) &
                            (ds2.lat >= min_lat) & (ds2.lat <= max_lat), drop=True)
    
        # Extract and align ds1
        lon1 = ds1['lon'].values
        lat1 = ds1['lat'].values
        z1 = ds1[var_name].values
    
        Lon1, Lat1 = np.meshgrid(lon1, lat1)
        img1 = ax.pcolormesh(
            Lon1, Lat1, z1,
            transform=crs_proj,
            cmap=colormap,
            shading='auto',
            edgecolors='black',
            linewidth=0.002
        )
    
        if show_colorbar:
            cbar1 = plt.colorbar(img1, ax=ax, orientation='vertical', pad=0.05)
            cbar1.set_label(f'{var_name} (ds1)')
    
        # Extract and align ds2
        lon2 = ds2['lon'].values
        lat2 = ds2['lat'].values
        z2 = ds2[var_name].values
        Lon2, Lat2 = np.meshgrid(lon2, lat2)
    
        if plot_ds2:
            img2 = ax.pcolormesh(
                Lon2, Lat2, z2,
                transform=crs_proj,
                cmap=colormap,
                shading='auto',
                edgecolors='red',
                linewidth=2,
                alpha=alpha
            )
        else:
            img2 = ax.pcolormesh(
                Lon2, Lat2, np.zeros_like(z2),
                transform=crs_proj,
                facecolors='none',
                edgecolors='red',
                linewidth=2,
                alpha=1.0
            )
    
        if show_colorbar and plot_ds2:
            cbar2 = plt.colorbar(img2, ax=ax, orientation='vertical', pad=0.05)
            cbar2.set_label(f'{var_name} (ds2)')
    
        # Optional: gridlines
        ax.grid(True, which='both', color='black', linewidth=0.5)
    
        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

















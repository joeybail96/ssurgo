# utils.py

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


class TifProcessor:
    def __init__(self):
        pass

    def tif_to_coards_netcdf(self, input_tif, output_nc, var_name, long_name, units):
        # Open the TIFF using GDAL to get proper lat/lon structure
        dataset = gdal.Open(input_tif)
        geotransform = dataset.GetGeoTransform()
        
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        x_indices = np.arange(width)
        y_indices = np.arange(height)
    
        # Calculate lat/lon arrays
        lon = geotransform[0] + x_indices * geotransform[1]
        lat = geotransform[3] + y_indices * geotransform[5]
    
        # Read the raster band data
        band = dataset.GetRasterBand(1)
        data = band.ReadAsArray().astype(np.float32)
        data[data == 99999] = np.nan  # Handle nodata
    
        # Build coordinate arrays
        lat_array = xr.DataArray(data=lat, name="lat", dims="lat",
                                 attrs={"long_name": "Latitude", "units": "degrees_north", "axis": "Y"})
        lon_array = xr.DataArray(data=lon, name="lon", dims="lon",
                                 attrs={"long_name": "Longitude", "units": "degrees_east", "axis": "X"})
    
        # Create the main data variable
        data_array = xr.DataArray(data=data, name=var_name, dims=["lat", "lon"],
                                  attrs={"long_name": long_name, "units": units})
    
        # Combine into a dataset
        ds = xr.Dataset({var_name: data_array}, coords={"lat": lat_array, "lon": lon_array})
    
        # Save to NetCDF
        ds.to_netcdf(output_nc)
        return ds


    def mask_by_threshold(self, ds, target_variable, ec_var, threshold, longname, output_cdf_path=None):
        # Copy the dataset
        masked_ds = ds.copy()
    
        # Create the float binary mask
        mask = (masked_ds[ec_var] >= threshold).astype(float)
    
        # Copy structure from ec_var to the new target variable
        mask_da = masked_ds[ec_var].copy()
        mask_da.values = mask
        mask_da.name = target_variable
        mask_da.attrs["long_name"] = longname
    
        # Assign mask to target variable and drop original
        masked_ds[target_variable] = mask_da
        masked_ds = masked_ds.drop_vars(ec_var)
    
        # Optionally write to file
        if output_cdf_path:
            masked_ds.to_netcdf(output_cdf_path)
    
        return masked_ds

    
    def resample_dataset(self, ds, output_nc_path, var_name='Playa_Mask', scale_factor=0.01):
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

    
        # Write CRS if not already set
        data = ds[var_name].rio.write_crs("EPSG:5070", inplace=True)
    
        # Get original resolution
        orig_height, orig_width = data.shape
        new_height = max(1, int(orig_height * scale_factor))
        new_width = max(1, int(orig_width * scale_factor))
    
        # Downsample using rasterio's averaging
        resampled = data.rio.reproject(
            dst_crs="EPSG:5070",
            shape=(new_height, new_width),
            resampling=Resampling.average
        )
    
        # Rename dims from x/y to lon/lat
        resampled = resampled.rename({'x': 'lon', 'y': 'lat'})
    
        # Ensure coordinates are attached
        resampled = resampled.assign_coords({
            'lat': resampled['lat'].values,
            'lon': resampled['lon'].values
        })
    
        # Convert to dataset and keep attributes
        downsampled_ds = resampled.to_dataset(name=var_name)
        downsampled_ds[var_name].attrs = ds[var_name].attrs
        downsampled_ds[var_name].attrs['scale_factor'] = scale_factor
    
        # Save and return
        downsampled_ds.to_netcdf(output_nc_path)
        print(f"Resampled dataset saved to: {output_nc_path}")
        return downsampled_ds


    def reproject_to_4326(self, ds, output_nc, var_name):
        # Get the DataArray and set spatial metadata
        da = ds[var_name]
        da = da.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
        da = da.rio.write_crs("EPSG:5070")
    
        # Reproject to WGS84
        reprojected = da.rio.reproject("EPSG:4326")
        reprojected = reprojected.rename({'x': 'lon', 'y': 'lat'})
        
        reprojected = reprojected.drop_vars('spatial_ref')
    
        # Save directly to NetCDF — coordinates will be correct
        reprojected.name = var_name
        reprojected = reprojected.to_dataset()
        reprojected.to_netcdf(output_nc)
    
        return reprojected



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


    def geographic_to_5070_bbox(self, bounding_box):
        """
        Converts a geographic bounding box (lon_min, lon_max, lat_min, lat_max) in degrees
        to a projected bounding box in EPSG:5070 (in meters).
    
        Parameters:
            bounding_box (tuple): (lon_min, lon_max, lat_min, lat_max)
    
        Returns:
            tuple: (x_min, x_max, y_min, y_max) in EPSG:5070
        """
        lon_min, lon_max, lat_min, lat_max = bounding_box
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    
        x_min, y_min = transformer.transform(lon_min, lat_min)
        x_max, y_max = transformer.transform(lon_max, lat_max)
    
        return (x_min, x_max, y_min, y_max)
    
   
    def convert_5070_to_latlon(self, x_vals, y_vals):
       proj_5070 = pyproj.CRS("EPSG:5070")
       proj_latlon = pyproj.CRS("EPSG:4326")
       transformer = pyproj.Transformer.from_crs(proj_5070, proj_latlon, always_xy=True)
       
       lon_vals, lat_vals = transformer.transform(x_vals, y_vals)
       return lon_vals, lat_vals


    def plot_zoomed(self, ds, bbox, output_png_path):
        """
        Plots the Playa_Mask variable from an xarray.Dataset on the EPSG:5070 projection,
        zoomed into a user-specified bounding box, and overlays grid lines for each cell.
    
        Parameters:
            ds (xarray.Dataset): Dataset containing the Playa_Mask variable.
            bbox (tuple): Bounding box in the format (min_lon, max_lon, min_lat, max_lat)
                          with coordinates in geographic (EPSG:4326).
        """
    
        # Set up the projection (EPSG:5070 - NAD83 / Conus Albers)
        proj_5070 = ccrs.epsg(5070)
    
        # Extract data
        playa_mask = ds['Playa_Mask']
        lons = ds['lon'].values
        lats = ds['lat'].values
        lon_grid, lat_grid = np.meshgrid(lons, lats)
    
        # Calculate spacing
        lon_spacing = np.abs(lons[1] - lons[0])
        lat_spacing = np.abs(lats[1] - lats[0])
    
        # Define gridline positions (edges of each grid cell)
        lon_edges = lons - lon_spacing / 2
        lat_edges = lats - lat_spacing / 2
    
        # Convert gridline edges to geographic coordinates
        lon_edges_geo, _ = self.convert_5070_to_latlon(lon_edges, np.full_like(lon_edges, np.mean(lats)))
        _, lat_edges_geo = self.convert_5070_to_latlon(np.full_like(lat_edges, np.mean(lons)), lat_edges)
    
        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': proj_5070})
    
        # Plot Playa_Mask
        mask_plot = ax.pcolormesh(lon_grid, lat_grid, playa_mask, transform=proj_5070,
                                  cmap='gray_r', shading='auto')
    
        # Set extent using projected bbox
        bbox_proj = self.geographic_to_5070_bbox(bbox)
        ax.set_extent([bbox_proj[0], bbox_proj[1], bbox_proj[2], bbox_proj[3]], crs=proj_5070)
    
        # Add gridlines using geographic coords
        gl = ax.gridlines(draw_labels=False, alpha=1.0, color='black', linewidth=0.08, zorder=11)
        gl.xlocator = FixedLocator(np.round(lon_edges_geo, 4))
        gl.ylocator = FixedLocator(np.round(lat_edges_geo, 4))
    
        # Add features
        ax.add_feature(cfeature.STATES, edgecolor='black', linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.set_title('Playa Mask (Zoomed)', fontsize=14)
    
        # Add colorbar
        plt.colorbar(mask_plot, ax=ax, shrink=0.7, label='Mask Value')
    
        plt.savefig(output_png_path)
        #plt.show()




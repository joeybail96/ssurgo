from tif_utils_v3 import TifProcessor
import xarray as xr
import os

processor = TifProcessor()


base_path = "/uufs/chpc.utah.edu/common/home/haskins-group1/users/jbail/GEOSChem/SSURGO/Scripts/in_process_data"
coards_nc = os.path.join(base_path, "1US_EC_coards.nc")
mask_nc = os.path.join(base_path, "2US_EC_mask.nc")
resampled_nc = os.path.join(base_path, "3aUS_EC_resampled.nc")
coarsened_nc = os.path.join(base_path, "3bUS_EC_coarsened.nc")
reprojected_nc = os.path.join(base_path, "4bUS_EC_reprojected.nc")
regridded_nc = os.path.join(base_path, "5bUS_EC_regridded.nc")
chloride_nc = os.path.join(base_path, "6bUS_EC_cl_factored.nc")

#bounding_box = (-113, -112.5, 39.5, 40)
#bounding_box = (-114, -112, 38, 40)
bounding_box = (-125, -100, 30, 45)
#bounding_box = (-130, -64, 21, 53)
bounding_box_str = f"bb_{bounding_box[0]:.2f}_{bounding_box[1]:.2f}_{bounding_box[2]:.2f}_{bounding_box[3]:.2f}"

# plot raw data
nc2plot = coards_nc
ds = xr.open_dataset(nc2plot)
ds_trimmed = processor.trim_to_latlon_box(ds, bounding_box)
var2plot = 'EC'
colormap = 'viridis'
nc_base_name = os.path.basename(nc2plot).replace(".nc", "")
output_png_path = f"/uufs/chpc.utah.edu/common/home/haskins-group1/users/jbail/GEOSChem/SSURGO/Scripts/figures/{bounding_box_str}_{nc_base_name}_{var2plot}.png"
#processor.plot_ds_trimmed(ds_trimmed, var2plot, colormap, save_path = output_png_path, show_colorbar=False, show_values=False, colorbar_limits=(0,20))


# plot fine data
fine2plot = mask_nc
ds1 = xr.open_dataset(fine2plot)
ds1_trimmed = processor.trim_to_latlon_box(ds1, bounding_box)
var2plot = 'Playa_Mask'
colormap = 'Greys'
nc_base_name = os.path.basename(fine2plot).replace(".nc", "")
output_png_path = f"/uufs/chpc.utah.edu/common/home/haskins-group1/users/jbail/GEOSChem/SSURGO/Scripts/figures/{bounding_box_str}_{nc_base_name}_{var2plot}.png"
processor.plot_ds_trimmed(ds1_trimmed, var2plot, colormap, save_path = output_png_path, show_colorbar=False, show_values=False)


# plot coarse data
coarse2plot = coarsened_nc
ds2 = xr.open_dataset(coarse2plot)
ds2_trimmed = processor.trim_to_latlon_box(ds2, bounding_box)
var2plot = 'Playa_Mask'
colormap = 'Greys'
nc_base_name = os.path.basename(coarse2plot).replace(".nc", "")
output_png_path = f"/uufs/chpc.utah.edu/common/home/haskins-group1/users/jbail/GEOSChem/SSURGO/Scripts/figures/{bounding_box_str}_{nc_base_name}_{var2plot}.png"
processor.plot_ds_trimmed(ds2_trimmed, var2plot, colormap, save_path = output_png_path, show_colorbar=False, show_values=False)


# plot fine and coarse
nc_base_name = "fine_coarse"
var2plot = 'Playa_Mask'
colormap = 'Greys'
output_png_path = f"/uufs/chpc.utah.edu/common/home/haskins-group1/users/jbail/GEOSChem/SSURGO/Scripts/figures/{bounding_box_str}_3b{nc_base_name}_{var2plot}.png"
#processor.plot_ds_together(ds1_trimmed, ds2_trimmed, var2plot, colormap, save_path = output_png_path, show_colorbar=False)





nc2plot = reprojected_nc
ds1 = xr.open_dataset(nc2plot)
colormap = 'Greys'
nc_base_name = os.path.basename(nc2plot).replace(".nc","")
var2plot = 'Playa_Mask'
output_png_path = f"/uufs/chpc.utah.edu/common/home/haskins-group1/users/jbail/GEOSChem/SSURGO/Scripts/figures/{bounding_box_str}_{nc_base_name}_{var2plot}.png"
#processor.plot_ds_epsg4326(ds1, var2plot, colormap, bounding_box, save_path = output_png_path, show_colorbar=False, show_values=False)




nc2plot = regridded_nc
ds2 = xr.open_dataset(nc2plot)
colormap = 'Greys'
nc_base_name = os.path.basename(nc2plot).replace(".nc","")
var2plot = 'Playa_Mask'
output_png_path = f"/uufs/chpc.utah.edu/common/home/haskins-group1/users/jbail/GEOSChem/SSURGO/Scripts/figures/{bounding_box_str}_{nc_base_name}_{var2plot}.png"
#processor.plot_ds_epsg4326(ds2, var2plot, colormap, bounding_box, grid_thickness=0, save_path = output_png_path, show_colorbar=False, show_values=False)




# plot fine and coarse
nc_base_name = "reprojected_regridded"
var2plot = 'Playa_Mask'
colormap = 'Greys'
output_png_path = f"/uufs/chpc.utah.edu/common/home/haskins-group1/users/jbail/GEOSChem/SSURGO/Scripts/figures/{bounding_box_str}_5b{nc_base_name}_{var2plot}.png"
#processor.plot_ds_together_4326(ds1, ds2, var2plot, colormap, save_path = output_png_path, show_colorbar=False, bounding_box=bounding_box)




nc2plot = chloride_nc
colormap = 'viridis'
ds = xr.open_dataset(nc2plot)
var2plot = 'Playa_Cl'
output_png_path = f"/uufs/chpc.utah.edu/common/home/haskins-group1/users/jbail/GEOSChem/SSURGO/Scripts/figures/{bounding_box_str}_5b{nc_base_name}_{var2plot}.png"

processor.plot_ds_epsg4326(ds, var2plot, colormap, bounding_box, grid_thickness=0, save_path = output_png_path, show_colorbar=False, show_values=False)
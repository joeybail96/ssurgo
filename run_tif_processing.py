from tif_utils import Tiff_Regrid as tr
import xarray as xr
import os



class_tr = tr()


# Step 1: Convert TIFF to EPSG:4326
input_tif = "/uufs/chpc.utah.edu/common/home/haskins-group1/users/jbail/GEOSChem/SSURGO/QGIS/US_EC.tif"
tif_4326_path = "/uufs/chpc.utah.edu/common/home/haskins-group1/users/jbail/GEOSChem/SSURGO/Scripts/in_process_data/_archive/US_EC_4326.tif"
if not os.path.exists(tif_4326_path):
    print(f"Creating {tif_4326_path} ...")
    class_tr.tiff_4326_path = class_tr.convert4326(input_tif, tif_4326_path)
else:
    print(f"{tif_4326_path} already exists. Skipping conversion.")
    class_tr.tiff_4326_path = tif_4326_path

# Step 2: Convert TIFF to NetCDF (EC)
ec_nc_path = "/uufs/chpc.utah.edu/common/home/haskins-group1/users/jbail/GEOSChem/SSURGO/Scripts/in_process_data/_archive/US_EC_4326.nc"
if not os.path.exists(ec_nc_path):
    print(f"Creating {ec_nc_path} ...")
    class_tr.ec_ncfile = class_tr.tiff_CDF(ec_nc_path, "EC", "Electrical conductivity (dS/cm)")
else:
    print(f"{ec_nc_path} already exists. Loading...")
    class_tr.ec_ncfile = xr.open_dataset(ec_nc_path)

# Step 3: Create playa mask
playa_mask_nc_path = "/uufs/chpc.utah.edu/common/home/haskins-group1/users/jbail/GEOSChem/SSURGO/Scripts/in_process_data/_archive/US_playa_mask_4326.nc"
if not os.path.exists(playa_mask_nc_path):
    print(f"Creating playa mask at {playa_mask_nc_path} ...")
    class_tr.playa_mask_ncfile = class_tr.playa_mask(playa_mask_nc_path, "Playa_Mask", "EC", "Playa mask where EC >=8.1 dS/cm", 8.1)
else:
    print(f"{playa_mask_nc_path} already exists. Skipping playa mask generation.")
    class_tr.playa_mask_ncfile = xr.open_dataset(playa_mask_nc_path)

# Step 4: Downsample
low_res_path = "/uufs/chpc.utah.edu/common/home/haskins-group1/users/jbail/GEOSChem/SSURGO/Scripts/in_process_data/_archive/US_playa_mask_4326_01downsampled.nc"
if not os.path.exists(low_res_path):
    print(f"Downsampling and saving to {low_res_path} ...")
    class_tr.downsampled_playa_mask_ncfile = class_tr.coarsen_nc_resolution(playa_mask_nc_path, low_res_path, var_name='Playa_Mask', scale_factor=0.1)
else:
    print(f"{low_res_path} already exists. Loading...")
    class_tr.downsampled_playa_mask_ncfile = xr.open_dataset(low_res_path)

# Step 5: Regrid
dust_template = "/uufs/chpc.utah.edu/common/home/haskins-group1/users/jbail/GEOSChem/SSURGO/Scripts/dust_template/dust_emissions_05.20110201.nc"
regrid_path = "/uufs/chpc.utah.edu/common/home/haskins-group1/users/jbail/GEOSChem/SSURGO/Scripts/in_process_data/_archive/US_playa_mask_4326_01downsampled_conserv_05x0625.nc"
if not os.path.exists(regrid_path):
    print(f"Regridding and saving to {regrid_path} ...")
    dust_template_nc = xr.open_dataset(dust_template)
    class_tr.regrid_playa_mask_ncfile = class_tr.regrid(class_tr.downsampled_playa_mask_ncfile, dust_template_nc, regrid_path, 'conservative')
else:
    print(f"{regrid_path} already exists. Skipping regridding.")
    class_tr.regrid_playa_mask_ncfile = xr.open_dataset(regrid_path)

# Step 6: Factor in chloride content
cl_path = "/uufs/chpc.utah.edu/common/home/haskins-group1/users/jbail/GEOSChem/SSURGO/Scripts/in_process_data/_archive/US_cl_4326_conservative_05x0625.nc"
var_name = "f_playa_cl"
if not os.path.exists(cl_path):
    print(f"Factoring Cl and saving to {cl_path} ...")
    class_tr.playa_cl_ncfile = class_tr.derive_cl(cl_path, "Playa_Cl", "Playa_Mask", "Chloride fraction in f_playa (kg/kg)")
else:
    print(f"{cl_path} already exists. Skipping chloride derivation.")
    class_tr.playa_cl_ncfile = xr.open_dataset(cl_path)

print("Processing complete.")

# Plotting
output_png_path = "/uufs/chpc.utah.edu/common/home/haskins-group1/users/jbail/GEOSChem/SSURGO/Scripts/figures/zoomed_regridded_conservative_US_Cl_4326.png"
bounding_box = (-114, -112, 38, 40)
# bounding_box = (-125, -100, 30, 45)
# #bounding_box = (-130, -64, 21, 53)
#class_tr.pl_downscaled(class_tr.playa_mask_ncfile, class_tr.downsampled_playa_mask_ncfile, "Playa_Mask", "EC 4326", "", output_png_path, bounding_box, clbar_vmin=0, clbar_vmax=1.0)
#class_tr.pl_downscaled_only(class_tr.downsampled_playa_mask_ncfile, "Playa_Mask", "EC 4326", "", output_png_path, bounding_box, clbar_vmin=0, clbar_vmax=1.0)

#class_tr.pl_USA(class_tr.downsampled_playa_mask_ncfile, "Playa_Mask", "", "", output_png_path, colormap="viridis", clbar_vmin=None, clbar_vmax=None)

class_tr.pl_USA_zoomed(class_tr.playa_cl_ncfile, "Playa_Cl", "", "", output_png_path, bounding_box, clbar_vmin=0, clbar_vmax=1.0)


# #class_tr.plot_histogram_nonzeros(class_tr.playa_cl_ncfile['Playa_Cl'], bins=100, x_tick_increment=0.05)

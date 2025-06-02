import os
import xarray as xr
from tif_utils import TifProcessor


# Init processor
processor = TifProcessor()
base_path = "/uufs/chpc.utah.edu/common/home/haskins-group1/users/jbail/GEOSChem/SSURGO/Scripts/in_process_data"


# Step 1: Convert TIFF to COARDS-compliant NetCDF
coards_nc = os.path.join(base_path, "1US_EC_coards.nc")
input_tif = "/uufs/chpc.utah.edu/common/home/haskins-group1/users/jbail/GEOSChem/SSURGO/QGIS/US_EC.tif"
if not os.path.exists(coards_nc):
    print(f"Converting {input_tif} to COARDS-compliant NetCDF...")
    ds_coards = processor.tif_to_coards_netcdf(
        input_tif, coards_nc, var_name="EC",
        long_name="Electrical Conductivity", units="dS/m"
    )
else:
    print(f"{coards_nc} exists. Loading from file...")
    ds_coards = xr.open_dataset(coards_nc)



# Step 2: Mask by EC threshold
mask_nc = os.path.join(base_path, "2US_EC_mask.nc")
ec_threshold = 8.1
if not os.path.exists(mask_nc):
    print(f"Applying threshold mask: EC >= {ec_threshold}...")
    ds_mask = processor.mask_by_threshold(
        ds=ds_coards,
        target_variable="Playa_Mask",
        ec_var="EC",
        threshold=ec_threshold,
        longname="Binary mask for EC >= 8.1 dS/cm",
        output_cdf_path=mask_nc
    )
else:
    print(f"{mask_nc} exists. Loading from file...")
    ds_mask = xr.open_dataset(mask_nc)



# Step 3b: Coarsen the mask dataset
coarsened_nc = os.path.join(base_path, "3bUS_EC_coarsened.nc")
if not os.path.exists(coarsened_nc):
    print("Coarsening masked dataset...")
    # Apply the coarsening function, e.g., with factor 10
    ds_coarsened = processor.coarsen_dataset(ds_mask, coarsened_nc, var_name='Playa_Mask', factor=1.75)
else:
    print(f"{coarsened_nc} exists. Loading from file...")
    ds_coarsened = xr.open_dataset(coarsened_nc)


# Step 4: Reproject the resampled dataset
reprojected_nc = os.path.join(base_path, "4bUS_EC_reprojected.nc")
if not os.path.exists(reprojected_nc):
    print("Reprojecting resampled dataset from EPSG:5070 to EPSG:4326...")
    ds_reprojected = processor.reproject_to_4326(ds_coarsened, reprojected_nc, "Playa_Mask")
else:
    print(f"{reprojected_nc} exists. Loading from file...")
    ds_reprojected = xr.open_dataset(reprojected_nc)



# Step 5: Regrid to dust template
regridded_nc = os.path.join(base_path, "5bUS_EC_regridded.nc")
dust_template_path = "/uufs/chpc.utah.edu/common/home/haskins-group1/users/jbail/GEOSChem/SSURGO/Scripts/dust_template/dust_emissions_05.20110201.nc"
if not os.path.exists(regridded_nc):
    print("Regridding to dust emissions template grid...")
    template_ds = xr.open_dataset(dust_template_path)
    ds_regridded = processor.regrid_to_template(ds_reprojected, template_ds, regridded_nc, method="conservative")
else:
    print(f"{regridded_nc} exists. Loading from file...")
    ds_regridded = xr.open_dataset(regridded_nc)


# Step 6: Apply chloride fraction
cl_factored_nc = os.path.join(base_path, "6bUS_EC_cl_factored.nc")
chloride_fraction = 0.0412
if not os.path.exists(cl_factored_nc):
    print(f"Applying chloride fraction: {chloride_fraction}...")
    ds_cl = processor.apply_chloride_fraction(
        ds_regridded, cl_factored_nc, cl_fraction=chloride_fraction, output_name="Playa_Cl"
    )
else:
    print(f"{cl_factored_nc} exists. Loading from file...")
    ds_cl = xr.open_dataset(cl_factored_nc)

print("✅ Processing pipeline complete.")








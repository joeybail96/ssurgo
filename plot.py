from tif_utils import TifProcessor
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

# offline emissions
merra2_emissions_root = "/uufs/chpc.utah.edu/common/home/haskins-group1/data/ExtData/HEMCO/OFFLINE_DUST/v2021-08/0.5x0.625/2011"
goesit_emissions_root = "/uufs/chpc.utah.edu/common/home/haskins-group1/data/ExtData/HEMCO/OFFLINE_DUST/v2025-04/GEOSIT/0.5x0.625/2011"
playa_emissions_root = "/uufs/chpc.utah.edu/common/home/haskins-group1/data/ExtData/HEMCO/OFFLINE_PLAYA/v2021-08/0.5x0.625/2011"

bounding_box = (-125, -100, 30, 45)
bounding_box_str = f"bb_{bounding_box[0]:.2f}_{bounding_box[1]:.2f}_{bounding_box[2]:.2f}_{bounding_box[3]:.2f}"

def prompt_and_plot(prompt_text, plot_func):
    if input(f"{prompt_text} (y/n): ").strip().lower() == "y":
        plot_func()

# --- Raw data plot ---
def plot_raw():
    ds = xr.open_dataset(coards_nc)
    ds_trimmed = processor.trim_to_latlon_box(ds, bounding_box)
    output_png_path = f"figures/{bounding_box_str}_1US_EC_coards_EC.png"
    processor.plot_ds_trimmed(ds_trimmed, 'EC', 'viridis', save_path=output_png_path, show_colorbar=False, show_values=False, colorbar_limits=(0, 20))

#prompt_and_plot("Plot raw data?", plot_raw)

# --- Fine mask plot ---
def plot_fine():
    ds1 = xr.open_dataset(mask_nc)
    ds1_trimmed = processor.trim_to_latlon_box(ds1, bounding_box)
    output_png_path = f"figures/{bounding_box_str}_2US_EC_mask_Playa_Mask.png"
    processor.plot_ds_trimmed(ds1_trimmed, 'Playa_Mask', 'Greys', save_path=output_png_path, show_colorbar=False, show_values=False)

#prompt_and_plot("Plot fine resolution mask?", plot_fine)

# --- Coarse mask plot ---
def plot_coarse():
    ds2 = xr.open_dataset(coarsened_nc)
    ds2_trimmed = processor.trim_to_latlon_box(ds2, bounding_box)
    output_png_path = f"figures/{bounding_box_str}_3bUS_EC_coarsened_Playa_Mask.png"
    processor.plot_ds_trimmed(ds2_trimmed, 'Playa_Mask', 'Greys', save_path=output_png_path, show_colorbar=False, show_values=False)

#prompt_and_plot("Plot coarse resolution mask?", plot_coarse)

# --- Fine vs. coarse comparison plot ---
def plot_fine_coarse():
    ds1 = xr.open_dataset(mask_nc)
    ds1_trimmed = processor.trim_to_latlon_box(ds1, bounding_box)
    ds2 = xr.open_dataset(coarsened_nc)
    ds2_trimmed = processor.trim_to_latlon_box(ds2, bounding_box)
    output_png_path = f"figures/{bounding_box_str}_3b_fine_coarse_Playa_Mask.png"
    processor.plot_ds_together(ds1_trimmed, ds2_trimmed, 'Playa_Mask', 'Greys', save_path=output_png_path, show_colorbar=False)

#prompt_and_plot("Compare fine vs. coarse masks?", plot_fine_coarse)

# --- Reprojected plot ---
def plot_reprojected():
    ds1 = xr.open_dataset(reprojected_nc)
    output_png_path = f"figures/{bounding_box_str}_4bUS_EC_reprojected_Playa_Mask.png"
    processor.plot_ds_epsg4326(ds1, 'Playa_Mask', 'Greys', bounding_box, save_path=output_png_path, show_colorbar=False, show_values=False)

#prompt_and_plot("Plot reprojected data?", plot_reprojected)

# --- Regridded plot ---
def plot_regridded():
    ds2 = xr.open_dataset(regridded_nc)
    output_png_path = f"figures/{bounding_box_str}_5bUS_EC_regridded_Playa_Mask.png"
    processor.plot_ds_epsg4326(ds2, 'Playa_Mask', 'Greys', bounding_box, grid_thickness=0, save_path=output_png_path, show_colorbar=False, show_values=False)

#prompt_and_plot("Plot regridded data?", plot_regridded)

# --- Reprojected vs. regridded comparison plot ---
def plot_reprojected_regridded():
    ds1 = xr.open_dataset(reprojected_nc)
    ds2 = xr.open_dataset(regridded_nc)
    output_png_path = f"figures/{bounding_box_str}_5b_reprojected_regridded_Playa_Mask.png"
    processor.plot_ds_together_4326(ds1, ds2, 'Playa_Mask', 'Greys', save_path=output_png_path, show_colorbar=False, bounding_box=bounding_box)

#prompt_and_plot("Compare reprojected vs. regridded?", plot_reprojected_regridded)

# --- Chloride factored plot ---
def plot_chloride():
    ds = xr.open_dataset(chloride_nc)
    output_png_path = f"figures/{bounding_box_str}_5b_chloride_Playa_Cl.png"
    processor.plot_ds_epsg4326(ds, 'Playa_Cl', 'viridis', bounding_box, grid_thickness=0, save_path=None, show_colorbar=True, colorbar_limits=(0, 0.05))

#prompt_and_plot("Plot chloride-factored Playa Mask?", plot_chloride)



# --- Emissions plot ---
def plot_emissions():
    ds = xr.open_dataset(regridded_nc)
    output_png_path = f"figures/{bounding_box_str}_cl_emissions.png"
    processor.plot_emissions(ds, var_name='Playa_Mask', input_root=playa_emissions_root, 
                             time_range=('2011-02-01', '2011-03-15'), bounding_box=bounding_box, aerosol='playa')
    # processor.plot_emissions(ds, var_name='Playa_Mask', input_root=merra2_emissions_root, 
    #                          time_range=('2011-02-01', '2011-03-15'), bounding_box=bounding_box)
    # processor.plot_emissions_with_mask(ds, var_name='Playa_Mask', input_root=merra2_emissions_root, 
    #                          time_range=('2011-02-01', '2011-03-15'), bounding_box=bounding_box)
    # processor.plot_emissions_with_mask(ds, var_name='Playa_Mask', input_root=goesit_emissions_root, 
    #                          time_range=('2011-02-01', '2011-03-15'), bounding_box=bounding_box)
    # processor.plot_emissions(ds, var_name='Playa_Mask', input_root=goesit_emissions_root, 
    #                          time_range=('2011-02-01', '2011-03-15'), bounding_box=bounding_box)


# total_kg_merra2 = processor.sum_total_emissions(
#     input_root=merra2_emissions_root,
#     time_range=None,
#     bounding_box=None,
#     var_names=('EMIS_DST1', 'EMIS_DST2', 'EMIS_DST3', 'EMIS_DST4'))

# total_kg_goesit = processor.sum_total_emissions(
#     input_root=goesit_emissions_root,
#     time_range=None,
#     bounding_box=None,
#     var_names=('EMIS_DST1', 'EMIS_DST2', 'EMIS_DST3', 'EMIS_DST4'))


#prompt_and_plot("Plot emissions?", plot_emissions)
plot_emissions()




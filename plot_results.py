from tif_utils import Tiff_Regrid as tr
import xarray as xr


regrid_path = "/uufs/chpc.utah.edu/common/home/haskins-group1/users/jbail/GEOSChem/SSURGO/Scripts/in_process_data/US_playa_cl_4326_025x025.nc"

df = xr.open_dataset(regrid_path)
img_path = "/uufs/chpc.utah.edu/common/home/haskins-group1/users/jbail/GEOSChem/SSURGO/Scripts/figures/US_playa_cl_4326.png"

tr().pl_USA(df["Playa_Cl"], "Chloride in Playa", "kg/kg", img_path)


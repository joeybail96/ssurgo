import xesmf as xe
import numpy as np
import xarray as xr
import psutil
import os
import dask.array as da
import matplotlib.pyplot as plt

# ---- Memory Tracking Utility ----
def print_memory_usage(note=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)  # in MB
    print(f"[Memory] {note} RSS memory usage: {mem:.2f} MB")

# ---- Parameters ----
regrid_method = 'conservative'

# ---- Dask Version ----
def dask_regrid():
    # ---- Create Input Grid with Dask Chunks ----
    ds_in = xe.util.grid_global(0.5, 0.3)
    shape = ds_in["lon"].shape

    np.random.seed(0)
    data_np = np.random.randint(0, 2, size=shape).astype(float)

    # Using smaller chunks for better memory control
    data_dask = da.from_array(data_np, chunks=(2, 2))  # <-- Smaller chunks to manage memory

    ds_in["data"] = xr.DataArray(
        data_dask,
        dims=("y", "x"),
        coords={"lon": ds_in["lon"], "lat": ds_in["lat"]},
        name="data"
    )
    print_memory_usage("Dask - After creating input data")

    # ---- Create Coarse Output Grid ----
    ds_coarse = xe.util.grid_global(45, 45)

    # ---- Regridding ----
    regridder = xe.Regridder(
        ds_in, ds_coarse, regrid_method,
        filename='weights_conservative.nc',
        reuse_weights=False,
        periodic=False,
        ignore_degenerate=True
    )
    print_memory_usage("Dask - After regridder initialization")

    # Apply regridding (lazy Dask operation)
    ds_coarse[regrid_method] = regridder(ds_in["data"])

    # Explicitly clearing intermediate variables to avoid memory buildup
    del ds_in
    print_memory_usage("Dask - After regridding (before compute)")

    # Trigger computation (which materializes the Dask array)
    ds_coarse = ds_coarse.compute()
    print_memory_usage("Dask - After compute")

    # ---- Plotting ----
    ds_coarse[regrid_method].plot()
    plt.title("Dask Version - After Regridding")
    plt.show()

# ---- Non-Dask Version ----
def nondask_regrid():
    # ---- Generate input grid ----
    ds_in = xe.util.grid_global(0.5, 0.3)

    np.random.seed(0)
    shape = ds_in["lon"].shape
    ds_in["data"] = xr.DataArray(
        np.random.randint(0, 2, size=shape).astype(float),
        dims=("y", "x")
    )
    print_memory_usage("Non-Dask - After generating input grid")

    # ---- Regrid to coarse grid ----
    ds_coarse = xe.util.grid_global(45, 45)
    regridder = xe.Regridder(
        ds_in, ds_coarse, regrid_method,
        filename='weights_conservative.nc',
        reuse_weights=False,
        periodic=False,
        ignore_degenerate=True
    )
    print_memory_usage("Non-Dask - After creating regridder")

    ds_coarse[regrid_method] = regridder(ds_in["data"])
    print_memory_usage("Non-Dask - After applying regridding")

    # ---- Plotting ----
    ds_coarse[regrid_method].plot()
    plt.title("Non-Dask Version - After Regridding")
    plt.show()

# ---- Run and Compare Memory Usage ----
print("\n========== DASK VERSION ==========")
dask_regrid()

print("\n========== NON-DASK VERSION ==========")
nondask_regrid()

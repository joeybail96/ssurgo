import xesmf as xe
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

regrid_method = 'conservative'

# Generate input grid
ds_in = xe.util.grid_global(5, 3)

np.random.seed(0)
shape = ds_in["lon"].shape
ds_in["data"] = xr.DataArray(
    np.random.randint(0, 2, size=shape).astype(float),
    dims=("y", "x")
)



# Regrid to coarse grid
ds_coarse = xe.util.grid_global(45, 45)
regridder = xe.Regridder(ds_in, ds_coarse, regrid_method, filename = 'weights_conservative.nc', reuse_weights=False, periodic=False, ignore_degenerate=True)
ds_coarse[regrid_method] = regridder(ds_in["data"])

# Extract coarse grid edges
lon_edge_coarse = ds_coarse["lon_b"].values[0, :]
lat_edge_coarse = ds_coarse["lat_b"].values[:, 0]

# ---- Plot 1 & 2 Vertically (Grayscale, No Colorbars, Red Grid Overlay on Plot 1) ----
fig, axs = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)

# Plot 1: Original grid (with red coarse grid overlay)
axs[0].pcolormesh(
    ds_in["lon_b"],
    ds_in["lat_b"],
    ds_in["data"],
    edgecolors="black",
    linewidth=0.5,
    shading="flat",
    cmap="gray"
)
# Overlay red coarse grid
for lon in lon_edge_coarse:
    axs[0].plot([lon, lon], [lat_edge_coarse[0], lat_edge_coarse[-1]], color="red", linewidth=5)
for lat in lat_edge_coarse:
    axs[0].plot([lon_edge_coarse[0], lon_edge_coarse[-1]], [lat, lat], color="red", linewidth=5)

axs[0].set_xlabel("Longitude")
axs[0].set_ylabel("Latitude")
axs[0].set_aspect('equal', adjustable='box')

# Plot 2: Regridded with grayscale and values (no colorbar)
axs[1].pcolormesh(
    ds_coarse["lon_b"],
    ds_coarse["lat_b"],
    ds_coarse[regrid_method],
    edgecolors="red",
    linewidth=5,
    shading="flat",
    cmap="gray"
)

# Add value labels to each regridded cell
lon_centers = ds_coarse["lon"].values
lat_centers = ds_coarse["lat"].values
values = ds_coarse[regrid_method].values

ny, nx = values.shape
for j in range(ny):
    for i in range(nx):
        val = values[j, i]
        txt = f"{val:.2f}" if not np.isnan(val) else "NaN"
        axs[1].text(
            lon_centers[j, i],
            lat_centers[j, i],
            txt,
            color="red",
            ha="center",
            va="center",
            fontsize=24,
        )

axs[1].set_xlabel("Longitude")
axs[1].set_ylabel("Latitude")
axs[1].set_aspect('equal', adjustable='box')

import xarray as xr
ds = xr.open_dataset('weights_conservative.nc')
print(ds)

dst_index = 1
mask = ds['row'].values == dst_index

print("Destination cell:", dst_index)
print("Source cells:", ds['col'].values[mask])
print("Weights:", ds['S'].values[mask])
ds.close()
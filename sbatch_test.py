import sys

# Attempt to import the required modules
try:
    import tif_utils
    print("Successfully imported tif_utils")
except ImportError as e:
    print(f"Error importing tif_utils: {e}")
    sys.exit(1)

try:
    import xarray as xr
    print("Successfully imported xarray")
except ImportError as e:
    print(f"Error importing xarray: {e}")
    sys.exit(1)

import os
print("Successfully imported os")

# Print additional useful information
print("Test completed successfully!")

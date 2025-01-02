import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def plot_precipitation(ax, ds, precip_var, title, og_elev, letter, cmap='turbo'):
    """
    ax: Matplotlib axis
    ds: xarray Dataset
    precip_var: Precipitation variable name
    title: Title for the subplot
    elevation_data: Elevation data to overlay
    cmap: Colormap for precipitation
    show_grid: Whether to show lat/lon grid
    letter: Letter to place in the top left corner of the plot
    """
    # Extract data
    # precip = ds[precip_var].values
    lon = ds['lon'].values
    lat = ds['lat'].values
    
    # Interpolate elevation
    elev_int = og_elev.interp(lon=lon, lat=lat, method='linear')
    elevation_data = elev_int['elev'].values

    # Plot data
    # c = ax.contourf(lon, lat, precip, levels=np.linspace(0.25, 4.75, 19), cmap=cmap, extend='max', alpha=0.9, zorder=3)
    c = ax.contourf(lon, lat, elevation_data, levels=np.linspace(-750, 2000, 23), cmap='terrain', alpha=0.5, zorder=1) # step 125m
    ax.add_feature(cfeature.STATES, edgecolor='black', linewidth=0.7, zorder=2)
    ax.add_feature(cfeature.LAKES, alpha=0.7, zorder=2)
    ax.text(0.05, 0.95, letter, transform=ax.transAxes, fontsize=16, fontweight='bold', color='black', va='top', ha='left')
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, xpadding=-2, ypadding=-1.5, linewidth=0.5, color='gray', alpha=0.7) # xpadding=-6 rotate_labels=90
    gl.top_labels = True
    gl.left_labels = True
    gl.right_labels = False
    gl.bottom_labels = False
    gl.xformatter = LongitudeFormatter(number_format='.1f') # direction_label=False, degree_symbol='',
    gl.yformatter = LatitudeFormatter(number_format='.1f')
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    
    return c


# Load data
elev_ds = xr.open_dataset('/home/atyagi/les/elev/elev.nc')
l1 = 'results/erie_10/exp_latest/20160103_15e.nc'
l2 = 'results/michigan_10/exp_latest/20201225_12m.nc'
l3 = 'results/ontario_10/exp_latest/20231129_00o.nc'
l4 = 'results/superior_10/exp_latest/20240114_21s.nc'

erie = xr.open_dataset(l1)
michigan = xr.open_dataset(l2)
ontario = xr.open_dataset(l3)
superior = xr.open_dataset(l4)

# Set up figure and axes
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 15), subplot_kw={'projection': ccrs.PlateCarree()})
plt.subplots_adjust(wspace=0.025, hspace=0.025)
plt.rc('pdf', fonttype=42)

# Plot data
plot_precipitation(axes[1, 1], erie, '', '', elev_ds, 'Lake Erie')
plot_precipitation(axes[1, 0], michigan, '', '', elev_ds, 'Lake Michigan')
plot_precipitation(axes[0, 1], ontario, '', '', elev_ds, 'Lake Ontario')
c1 = plot_precipitation(axes[0, 0], superior, '', '', elev_ds, 'Lake Superior')

# Create colorbar
cbar = fig.colorbar(c1, ax=axes, orientation='horizontal', fraction=0.05, pad=0.025)
cbar.set_label('Elevation (m)', fontsize=16)
cbar.ax.tick_params(labelsize=14)

# Save and display
# plt.savefig('map.pdf', bbox_inches='tight')
# plt.savefig('map.png', bbox_inches='tight')
plt.show()

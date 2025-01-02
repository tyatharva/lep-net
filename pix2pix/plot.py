import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def plot_precipitation(ax, ds, precip_var, title, elevation_data, letter, cmap='turbo'):
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
    precip = ds[precip_var].values
    precip = np.transpose(precip)
    lon = ds['lon'].values
    lat = ds['lat'].values
    
    # Plot data
    c = ax.contourf(lon, lat, precip, levels=np.linspace(0.25, 4.75, 19), cmap=cmap, extend='max', alpha=0.9, zorder=3)
    ax.contourf(lon, lat, elevation_data, levels=np.linspace(-750, 2000, 23), cmap='terrain', alpha=0.5, zorder=1)
    ax.add_feature(cfeature.STATES, edgecolor='black', linewidth=0.7, zorder=2)
    ax.add_feature(cfeature.LAKES, alpha=0.7, zorder=2)
    ax.text(0.05, 0.95, letter, transform=ax.transAxes, fontsize=16, fontweight='bold', color='black', va='top', ha='left')
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, xpadding=-2, ypadding=-1.5, linewidth=0.5, color='gray', alpha=0.7)
    gl.top_labels = True
    gl.left_labels = True
    gl.right_labels = False
    gl.bottom_labels = False
    gl.xformatter = LongitudeFormatter(number_format='.1f')
    gl.yformatter = LatitudeFormatter(number_format='.1f')
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    
    return c


# Load elevation
elev_ds = xr.open_dataset('/home/atyagi/les/elev/elev.nc')
elev_lon = elev_ds['lon'].values
elev_lat = elev_ds['lat'].values
elev_data = elev_ds['elev'].values


# Load data
l1 = 'results/superior_10/exp_latest/20171212_03s.nc'
l2 = 'results/superior_10/exp_latest/20221218_03s.nc'
l3 = 'results/superior_10/exp_latest/20240114_21s.nc'

# l1 = 'results/michigan_10/exp_latest/20161214_06m.nc'
# l2 = 'results/michigan_10/exp_latest/20201225_12m.nc'
# l3 = 'results/michigan_10/exp_latest/20221120_03m.nc'

# l1 = 'results/erie_10/exp_latest/20160103_15e.nc'
# l2 = 'results/erie_10/exp_latest/20170128_15e.nc'
# l3 = 'results/erie_10/exp_latest/20220107_00e.nc'

# l1 = 'results/ontario_10/exp_latest/20170106_09o.nc'
# l2 = 'results/ontario_10/exp_latest/20171207_00o.nc'
# l3 = 'results/ontario_10/exp_latest/20231129_00o.nc'


hrrr_band = xr.open_dataset(l1)
hrrr_broad = xr.open_dataset(l2)
hrrr_hybrid = xr.open_dataset(l3)


# Interpolate elevation
lon = hrrr_band['lon'].values
lat = hrrr_band['lat'].values
elev_ds_interp = elev_ds.interp(lon=lon, lat=lat, method='linear')
elev_subset = elev_ds_interp['elev'].values


# Set up figure and axes
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(25, 15), subplot_kw={'projection': ccrs.PlateCarree()})
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.rc('pdf', fonttype=42)


# Create labels
cols = ['HRRR', 'Prediction', 'Target']
rows = [
        datetime.strptime(os.path.splitext(os.path.basename(l1))[0][:-1], '%Y%m%d_%H').strftime('%H%M UTC %d %b. %Y'),
        datetime.strptime(os.path.splitext(os.path.basename(l2))[0][:-1], '%Y%m%d_%H').strftime('%H%M UTC %d %b. %Y'),
        datetime.strptime(os.path.splitext(os.path.basename(l3))[0][:-1], '%Y%m%d_%H').strftime('%H%M UTC %d %b. %Y'),
        ]

for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1.05), xycoords='axes fraction', 
                ha='center', va='center', fontsize=18)

for ax, row in zip(axes[:, 0], rows):
    ax.annotate(row, xy=(-0.005, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                xycoords='axes fraction', textcoords='offset points',
                size=18, ha='center', va='center', rotation=90)


# Plot data
plot_precipitation(axes[0, 0], hrrr_band, 'HRRR', '', elev_subset, 'a')
plot_precipitation(axes[0, 1], hrrr_band, 'Prediction', '', elev_subset, 'b')
plot_precipitation(axes[0, 2], hrrr_band, 'Target', '', elev_subset, 'c')

plot_precipitation(axes[1, 0], hrrr_broad, 'HRRR', '', elev_subset, 'd')
plot_precipitation(axes[1, 1], hrrr_broad, 'Prediction', '', elev_subset, 'e')
plot_precipitation(axes[1, 2], hrrr_broad, 'Target', '', elev_subset, 'f')

plot_precipitation(axes[2, 0], hrrr_hybrid, 'HRRR', '', elev_subset, 'h')
plot_precipitation(axes[2, 1], hrrr_hybrid, 'Prediction', '', elev_subset, 'i')
c1 = plot_precipitation(axes[2, 2], hrrr_hybrid, 'Target', '', elev_subset, 'j')


# Create colorbar
cbar = fig.colorbar(c1, ax=axes, orientation='horizontal', fraction=0.05, pad=0.025)
cbar.set_label('Next-hour precipitation (mm)', fontsize=16) # (mm h$^{-1}$)
cbar.ax.tick_params(labelsize=14)
# cbar.set_ticks(np.arange(0.5, 5.0, 0.5))


# Save and display
# plt.savefig(l1[l1.index('/')+1:l1.index('_')] + '.pdf', bbox_inches='tight')
# plt.savefig(l1[l1.index('/')+1:l1.index('_')] + '.png', bbox_inches='tight')
plt.show()

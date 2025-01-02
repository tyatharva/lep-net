#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 18:40:34 2024

@author: atyagi
"""


import os
import re
import time
import gzip
import glob
import shutil
import random
import logging
import requests
import pywgrib2_s
import multiprocessing
import numpy as np
import pandas as pd
import xarray as xr
from cdo import *
from herbie import FastHerbie
from datetime import datetime, timedelta


def rand_set():
    sets = ['train', 'val', 'test']
    probs = [0.8, 0.1, 0.1]
    return random.choices(sets, probs)[0]


def create_dir(folder_name):
    backup_folder_path = f"./{folder_name}"
    os.makedirs(backup_folder_path, exist_ok=True)
    subfolders = ['hrrr', 'mrms']
    for subfolder in subfolders:
        os.makedirs(os.path.join(backup_folder_path, subfolder), exist_ok=True)


def mfilerdir_hrrr(directory):
    items = os.listdir(directory)
    for item in items:
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            for root, dirs, files in os.walk(item_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    new_file_name = file.split("__", 1)[-1]
                    original_folder_name = os.path.basename(item_path)
                    new_path = os.path.join(directory, original_folder_name + "_" + new_file_name)
                    shutil.move(file_path, new_path)
            shutil.rmtree(item_path)


def calculate_kuchera_ratio(ds):
    regex_pattern = r'TMP_(?:[5-9]\d{2}mb|1000mb|1013D2mb|2m)'
    temperature_vars = [var for var in ds.data_vars if re.match(regex_pattern, var)]
    if not temperature_vars: raise ValueError(f"No temperature variables found matching pattern: {regex_pattern}")
    temperature_data = [ds[var] for var in temperature_vars]
    
    combined_temps = xr.concat(temperature_data, dim='variable')
    tmax = combined_temps.max(dim='variable')
    kuchera_ratio = np.where(tmax > 271.16, 12 + 2 * (271.16 - tmax), 12 + (271.16 - tmax))
    kuchera_ratio = np.maximum(kuchera_ratio, 0)
    
    kuchera_ratio_da = xr.DataArray(
        kuchera_ratio,
        dims=ds[temperature_vars[0]].dims,
        coords=ds[temperature_vars[0]].coords,
    )
    ds['kuchera_ratio'] = kuchera_ratio_da
    
    keep_vars = ['TMP_925mb', 'TMP_850mb', 'TMP_700mb', 'TMP_2m']
    for var in temperature_vars:
        if var not in keep_vars:
            ds = ds.drop_vars(var)
    
    return ds


def get_hrrr(dirname, lake, base_time, hours_ahead, hrrr_run, thds): # dirname, htime, thds, lake
    DATES = pd.date_range(start=(base_time + timedelta(hours=hrrr_run)).strftime("%Y-%m-%d %H:00"), periods=1, freq="1H")
    
    data = FastHerbie(DATES, model="hrrr", product="prs", fxx=range(hours_ahead-hrrr_run, 1+hours_ahead-hrrr_run), max_threads=thds)
    data.download(
        searchString="((UGRD|VGRD):10 m)|((TMP|DPT):2 m)|(CAPE:surface)|(TMP:(?:[5-9]\d{2}|1000|1013.2|surface))|((DPT|UGRD|VGRD):(925|850|700))|(ABSV:500)|(ICEC)|(REFC)|(:APCP:.*:(0-1|1-2|2-3|3-4|4-5|5-6|6-7|7-8|8-9|9-10|10-11|11-12|12-13|13-14|14-15|15-16|16-17|17-18|18-19|19-20|20-21|21-22|22-23|23-24|24-25))",
        max_threads=thds,
        save_dir=f"./{dirname}/"
    )
    mfilerdir_hrrr(f"./{dirname}/hrrr/")

    grib_files = glob.glob(f"./{dirname}/hrrr/*.grib2")
    for grib_file in grib_files:
        pywgrib2_s.wgrib2([grib_file, "-netcdf", grib_file.replace(".grib2", ".nc")])

    nc_files = [file.replace(".grib2", ".nc") for file in grib_files]
    for file in nc_files:
        cdo.remapnn(f"../grids/{lake}", input=f"{file}", options="-f nc4", output=file.replace(".nc", "_1.nc"))
        file = file.replace(".nc", "_1.nc")
        cdo.merge(input=f"{file} ../elev/elev_{lake}.nc ../landsea/landsea_{lake}.nc", options="-f nc4", output=file.replace("_1.nc", "_2.nc"))

    nc_files = [file.replace(".nc", "_2.nc") for file in nc_files]
    cdo.mergetime(input=f"{' '.join(sorted(nc_files))}", options=f"-b F32 -P {thds} -f nc -r", output=f"./{dirname}/hrrr.nc")


def get_mrms(dirname, lake, base_time, hours_ahead, thds):
    date = base_time + timedelta(hours=hours_ahead)
    base = "https://mtarchive.geol.iastate.edu/"
    ext = "/mrms/ncep/MultiSensor_QPE_01H_Pass2/MultiSensor_QPE_01H_Pass2"

    url = f"{base}{date.strftime('%Y/%m/%d')}{ext}_00.00_{date.strftime('%Y%m%d')}-{date.strftime('%H0000')}.grib2.gz"
    response = requests.get(url, stream=True)
    grib_file = f"./{dirname}/mrms/{date.strftime('%Y%m%d')}_{date.strftime('%H')}.grib2.gz"

    with open(grib_file, 'wb') as f:
        f.write(response.content)

    with gzip.open(grib_file, 'rb') as f_in:
        with open(grib_file.replace('.gz', ''), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    pywgrib2_s.wgrib2([grib_file[:-3], "-netcdf", grib_file[:-3].replace(".grib2", ".nc")])
    grib_file = grib_file[:-3].replace(".grib2", ".nc")
    cdo.remapnn(f"../grids/{lake}", input=grib_file, options=f"-b F32 -P {thds} -f nc -r", output=f"./{dirname}/mrms.nc")



def merge(dirname, lake, split, get_target):
    ds1 = xr.open_dataset(f"./{dirname}/hrrr.nc")
    time_dim = ds1['TMP_surface'].coords['time']
    ds1['landsea'] = ds1['landsea'].expand_dims(time=time_dim, axis=0).broadcast_like(ds1['TMP_surface'])
    ds1['elev'] = ds1['elev'].expand_dims(time=time_dim, axis=0).broadcast_like(ds1['TMP_surface'])
    ds1['TMP_masked'] = ds1['TMP_surface'] * ds1['landsea']
    ds1['TMP_masked'].attrs['units'] = 'K'
    ds1['APCP_surface'].attrs['units'] = 'mm'
    ds1 = ds1.rename({'TMP_2maboveground': 'TMP_2m', 'DPT_2maboveground': 'DPT_2m', 'REFC_entireatmosphere': 'REFC', 'UGRD_10maboveground': 'UGRD_10m', 'VGRD_10maboveground': 'VGRD_10m'})
    ds1 = calculate_kuchera_ratio(ds1)
    if get_target:
        ds2 = xr.open_dataset(f"./{dirname}/mrms.nc")
        var_name = [var for var in ds2.data_vars if var not in ds2.dims][0]
        ds2 = ds2.rename({var_name: 'QPE_01H'})
        ds2['QPE_01H'].attrs = {'units': 'mm'}
        ds = xr.merge([ds1, ds2])
    else: ds = ds1
    ds1.close()
    if get_target: ds2.close()

    for var in ds.data_vars:
        ds[var].attrs = {}

    if lake == "m":
        ds = ds.rename({'time': 'z', 'lat': 'x', 'lon': 'y'})
        ds = ds.assign_coords(z=("z", range(len(ds.coords['z']))))
        ds = ds.assign_coords(x=("x", range(len(ds.coords['x']))))
        ds = ds.assign_coords(y=("y", range(len(ds.coords['y']))))

        for height in ['700mb', '850mb', '925mb', '10m']:
            old_u = ds[f'UGRD_{height}']
            old_v = ds[f'VGRD_{height}']
            ds[f'UGRD_{height}'] = old_v
            ds[f'VGRD_{height}'] = old_u

    else:
        ds = ds.rename({'time': 'z', 'lon': 'x', 'lat': 'y'})
        ds = ds.assign_coords(z=("z", range(len(ds.coords['z']))))
        ds = ds.assign_coords(x=("x", range(len(ds.coords['x']))))
        ds = ds.assign_coords(y=("y", range(len(ds.coords['y']))))

    ds = ds.transpose('z', 'x', 'y')
    ds = xr.Dataset(
        {var: (['z', 'x', 'y'], ds[var].values) for var in ds.data_vars},
        coords={
            'z': ds.coords['z'],
            'x': ds.coords['x'],
            'y': ds.coords['y'],
        }
    )
    
    # ds = ds.squeeze()
    if get_target:
        ins = ds.drop_vars("QPE_01H")
        out = ds["QPE_01H"]
        ins = ins.chunk({'z': 1, 'x': 600, 'y': 250})
        out = out.chunk({'z': 1, 'x': 600, 'y': 250})
        ins.to_zarr(f"./{split}A/{dirname}_input.zarr", mode='w', consolidated=True) # ins.to_netcdf(f"./{split}A/{dirname}_input.nc", format="NETCDF3_64BIT")
        out.to_zarr(f"./{split}B/{dirname}_target.zarr", mode='w', consolidated=True) # out.to_netcdf(f"./{split}B/{dirname}_target.nc", format="NETCDF3_64BIT")
    else: ds.to_zarr(f"./{split}A/{dirname}_input.zarr", mode='w', consolidated=True)
    ds.close()


def process_lake(lake, base_time, hours_ahead, hrrr_run, split, get_target, thds, retry_attempts=1):
    dirname = f"{(base_time+timedelta(hours=hours_ahead)).strftime('%Y%m%d_%H')}{lake}"

    for attempt in range(retry_attempts + 1):
        try:
            create_dir(dirname)
            
            hrrr_process = multiprocessing.Process(target=get_hrrr, args=(dirname, lake, base_time, hours_ahead, hrrr_run, thds // 2))
            if get_target: mrms_process = multiprocessing.Process(target=get_mrms, args=(dirname, lake, base_time, hours_ahead, thds // 2))
            hrrr_process.start()
            if get_target: mrms_process.start()
            hrrr_process.join()
            if get_target: mrms_process.join()

            merge(dirname, lake, split, get_target)
            shutil.rmtree(f"./{dirname}/")
            break

        except Exception as e:
            try: shutil.rmtree(f"./{dirname}/")
            except Exception as d: logging.warning(f"Couldn't remove {dirname} due to error: {d}")
            if attempt < retry_attempts: logging.warning(f"Retrying {lake} for date {base_time.strftime('%Y-%m-%d %H:%M:%S')} due to error: {e}")
            else: logging.error(f"Failed to process {lake} for date {base_time.strftime('%Y-%m-%d %H:%M:%S')} after {retry_attempts + 1} attempts due to error: {e}")


def get_data(base_time, hours_ahead, hrrr_run, split, get_target=False, thds=16):
    
    processes = []
    for lake in ['s', 'm', 'e', 'o']:
        p = multiprocessing.Process(target=process_lake, args=(lake, base_time, hours_ahead, hrrr_run, split, get_target, thds))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def main():
    st = time.time()
    split = "qtr"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    if os.path.exists(data_dir): shutil.rmtree(data_dir)
    os.makedirs(data_dir)
    os.makedirs(os.path.join(data_dir, f"{split}A"))
    os.makedirs(os.path.join(data_dir, f"{split}B"))
    os.chdir(data_dir)
    os.environ['REMAP_EXTRAPOLATE'] = 'off'
    try: os.remove('processing.log')
    except: pass
    logging.basicConfig(filename='processing.log', level=logging.ERROR)
    
    tim = datetime(year=2024, month=11, day=30, hour=00)
    for i in range(1, 7): get_data(tim, i, -1, split, get_target=True)
    
    et = time.time()
    ti = et - st
    print(f'\nDone in {ti} seconds')

if __name__ == "__main__":
    cdo = Cdo()
    main()

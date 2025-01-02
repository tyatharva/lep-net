#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 18:40:34 2024

@author: atyagi
"""


import os
import csv
import time
import gzip
import glob
import shutil
import random
import logging
import requests
import pywgrib2_s
import multiprocessing
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
    backup_folder_path = os.path.join('dataset/original', folder_name)
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


def get_hrrr(dirname, htime, thds, lake):
    DATES = pd.date_range(start=(htime - timedelta(hours=1)).strftime("%Y-%m-%d %H:00"), periods=1, freq="1H")
    data = FastHerbie(DATES, model="hrrr", product="prs", fxx=range(0,8), max_threads=thds)
    data.download(
        searchString="(TMP:surface)|((UGRD|VGRD):10 m)|((TMP|DPT):2 m)|(CAPE:surface)|((TMP|DPT|UGRD|VGRD):(925|850|700))|(ABSV:500)|(ICEC)|(REFC)|(:APCP:.*:(0-0|0-1|1-2|2-3|3-4|4-5|5-6|6-7))",
        max_threads=thds,
        save_dir=f"./dataset/original/{dirname}/"
    )
    mfilerdir_hrrr(f"./dataset/original/{dirname}/hrrr/")

    grib_files = glob.glob(f"./dataset/original/{dirname}/hrrr/*.grib2")
    for grib_file in grib_files:
        pywgrib2_s.wgrib2([grib_file, "-netcdf", grib_file.replace(".grib2", ".nc")])

    nc_files = [file.replace(".grib2", ".nc") for file in grib_files]
    for file in nc_files:
        cdo.remapnn(f"./grids/{lake}", input=f"{file}", options="-f nc4", output=file.replace(".nc", "_1.nc"))
        file = file.replace(".nc", "_1.nc")
        cdo.merge(input=f"{file} ./elev/elev_{lake}.nc ./landsea/landsea_{lake}.nc", options="-f nc4", output=file.replace("_1.nc", "_2.nc"))

    nc_files = [file.replace(".nc", "_2.nc") for file in nc_files]
    cdo.mergetime(input=f"{' '.join(sorted(nc_files))}", options=f"-b F32 -P {thds} -f nc -r", output=f"./dataset/original/{dirname}/hrrr.nc")


def get_mrms(dirname, htime, thds, lake):
    DATES = pd.date_range(start=(htime - timedelta(hours=1)).strftime("%Y-%m-%d %H:00"), periods=8, freq="1H")
    base = "https://mtarchive.geol.iastate.edu/"
    ext = "/mrms/ncep/GaugeCorr_QPE_01H/GaugeCorr_QPE_01H"
    if htime >= datetime(2020, 10, 14):
        ext = "/mrms/ncep/MultiSensor_QPE_01H_Pass2/MultiSensor_QPE_01H_Pass2"

    for date in DATES:
        url = f"{base}{date.strftime('%Y/%m/%d')}{ext}_00.00_{date.strftime('%Y%m%d')}-{date.strftime('%H0000')}.grib2.gz"
        response = requests.get(url, stream=True)
        grib_file = f"./dataset/original/{dirname}/mrms/{date.strftime('%Y%m%d')}_{date.strftime('%H')}.grib2.gz"

        with open(grib_file, 'wb') as f:
            f.write(response.content)

        with gzip.open(grib_file, 'rb') as f_in:
            with open(grib_file.replace('.gz', ''), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    grib_files = glob.glob(f"./dataset/original/{dirname}/mrms/*.grib2")
    for file in grib_files:
        pywgrib2_s.wgrib2([file, "-netcdf", file.replace(".grib2", ".nc")])
        file = file.replace(".grib2", ".nc")
        cdo.remapnn(f"./grids/{lake}", input=file, options="-f nc4", output=file.replace(".nc", "_1.nc"))

    nc_files = [file.replace(".grib2", "_1.nc") for file in grib_files]
    cdo.mergetime(input=f"{' '.join(sorted(nc_files))}", options=f"-b F32 -P {thds} -f nc -r", output=f"./dataset/original/{dirname}/mrms.nc")


def merge(dirname, lake, save):
    ds1 = xr.open_dataset(f"./dataset/original/{dirname}/hrrr.nc")
    time_dim = ds1['TMP_surface'].coords['time']
    ds1['landsea'] = ds1['landsea'].expand_dims(time=time_dim, axis=0).broadcast_like(ds1['TMP_surface'])
    ds1['elev'] = ds1['elev'].expand_dims(time=time_dim, axis=0).broadcast_like(ds1['TMP_surface'])
    ds1['TMP_masked'] = ds1['TMP_surface'] * ds1['landsea']
    ds1['TMP_masked'].attrs['units'] = 'K'
    ds1['APCP_surface'].attrs['units'] = 'mm'
    ds1 = ds1.rename({'TMP_2maboveground': 'TMP_2m', 'DPT_2maboveground': 'DPT_2m', 'REFC_entireatmosphere': 'REFC', 'UGRD_10maboveground': 'UGRD_10m', 'VGRD_10maboveground': 'VGRD_10m'})
    ds2 = xr.open_dataset(f"./dataset/original/{dirname}/mrms.nc")
    var_name = [var for var in ds2.data_vars if var not in ds2.dims][0]
    ds2 = ds2.rename({var_name: 'QPE_01H'})
    ds2['QPE_01H'].attrs = {'units': 'mm'}
    ds = xr.merge([ds1, ds2])
    ds1.close()
    ds2.close()

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

    ins = ds.drop_vars("QPE_01H")
    out = ds["QPE_01H"].isel(z=slice(2, 8))
    out = out.assign_coords(z=("z", range(len(out.coords['z']))))
    ins = ins.chunk({'z': 1, 'x': 600, 'y': 250})
    out = out.chunk({'z': 1, 'x': 600, 'y': 250})
    ins.to_zarr(f"./dataset/{save}/{dirname}_input.zarr", mode='w', consolidated=True) # ins.to_netcdf(f"./dataset/{save}/{dirname}_input.nc", format="NETCDF3_64BIT")
    out.to_zarr(f"./dataset/{save}/{dirname}_target.zarr", mode='w', consolidated=True) # out.to_netcdf(f"./dataset/{save}/{dirname}_target.nc", format="NETCDF3_64BIT")
    ds.close()

def process_day(date, thds, lake, save):
    dirname = f"{date.strftime('%Y%m%d_%H')}{lake}"
    try:
        create_dir(dirname)
        hrrr_process = multiprocessing.Process(target=get_hrrr, args=(dirname, date, thds//2, lake))
        mrms_process = multiprocessing.Process(target=get_mrms, args=(dirname, date, thds//2, lake))
        hrrr_process.start()
        mrms_process.start()
        hrrr_process.join()
        mrms_process.join()
        merge(dirname, lake, save)
        shutil.rmtree(f"./dataset/original/{dirname}/")
    except Exception as e:
        logging.error(f"Error processing {dirname}: {e}")
        raise


def process_lake(lake, dates, thds, retry_attempts=1):
    for date in dates:
        save = rand_set()
        for hour in range(24):
            date_hour = date.replace(hour=hour)
            for attempt in range(retry_attempts + 1):
                try:
                    process_day(date_hour, thds, lake, save)
                    break
                except Exception as e:
                    try: shutil.rmtree(f"./dataset/original/{date_hour.strftime('%Y%m%d_%H')}{lake}/")
                    except: logging.warning(f"Couldn't remove {date_hour.strftime('%Y%m%d_%H')}{lake} after error")
                    if attempt < retry_attempts:
                        logging.warning(f"Retrying {lake} for date {date_hour.strftime('%Y-%m-%d %H:%M:%S')} due to error: {e}")
                    else:
                        logging.error(f"Failed to process {lake} for date {date_hour.strftime('%Y-%m-%d %H:%M:%S')} after {retry_attempts + 1} attempts")


def main(thds=32):
    st = time.time()
    os.environ['REMAP_EXTRAPOLATE'] = 'off'
    try: os.remove('processing.log')
    except: pass
    logging.basicConfig(filename='processing.log', level=logging.ERROR)

    with open('dates.csv', 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        lake_dates = {lake: [] for lake in headers}

        for row in reader:
            for lake, date_str in zip(headers, row):
                lake_dates[lake].append(datetime.strptime(date_str, '%m/%d/%Y'))

    processes = []
    for lake in lake_dates:
        p = multiprocessing.Process(target=process_lake, args=(lake, lake_dates[lake], thds))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    et = time.time()
    ti = et - st
    print(f'\nDone in {ti} seconds')

if __name__ == "__main__":
    cdo = Cdo()
    main()

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
from pythia_datasets import DATASETS

#filepath = DATASETS.fetch('CESM2_sst_data.nc')
filepath='./model/Tools/test_datas/CESM2_sst_data.nc'
data = xr.open_dataset(filepath)
#filepath2 = DATASETS.fetch('CESM2_grid_variables.nc')
filepath2='./model/Tools/test_datas/CESM2_grid_variables.nc'
areacello = xr.open_dataset(filepath2).areacello

ds = xr.merge([data, areacello])
print(ds)

fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=180))
ax.coastlines()
ax.gridlines()
ds.tos.isel(time=0).plot(
    ax=ax, transform=ccrs.PlateCarree(), vmin=-2, vmax=30, cmap='coolwarm'
)

tos_nino34 = ds.sel(lat=slice(-5, 5), lon=slice(190, 240))
print(tos_nino34)

tos_nino34 = ds.where(
    (ds.lat < 5) & (ds.lat > -5) & (ds.lon > 190) & (ds.lon < 240), drop=True
)
print(tos_nino34)

fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=180))
ax.coastlines()
ax.gridlines()
tos_nino34.tos.isel(time=0).plot(
    ax=ax, transform=ccrs.PlateCarree(), vmin=-2, vmax=30, cmap='coolwarm'
)
ax.set_extent((120, 300, 10, -10))

gb = tos_nino34.tos.groupby('time.month')
mean=gb.mean(dim='time')
tos_nino34_anom = gb - gb.mean(dim='time')
index_nino34 = tos_nino34_anom.weighted(tos_nino34.areacello).mean(dim=['lat', 'lon'])

index_nino34_rolling_mean = index_nino34.rolling(time=5, center=True).mean()

index_nino34.plot(size=8)
index_nino34_rolling_mean.plot()
plt.legend(['anomaly', '5-month running mean anomaly'])
plt.title('SST anomaly over the Niño 3.4 region');

std_dev = tos_nino34.tos.std()
print(std_dev)

normalized_index_nino34_rolling_mean = index_nino34_rolling_mean / std_dev

fig = plt.figure(figsize=(12, 6))

time=normalized_index_nino34_rolling_mean.time.data
data=normalized_index_nino34_rolling_mean.where(
     normalized_index_nino34_rolling_mean >= 0.4
    ).data
plt.fill_between(
    time,
    data,
    0.4,
    color='red',
    alpha=0.9,
)
plt.fill_between(
    time,
    data,
    -0.4,
    color='blue',
    alpha=0.9,
)

normalized_index_nino34_rolling_mean.plot(color='black')
plt.axhline(0, color='black', lw=0.5)
plt.axhline(0.4, color='black', linewidth=0.5, linestyle='dotted')
plt.axhline(-0.4, color='black', linewidth=0.5, linestyle='dotted')
plt.title('Niño 3.4 Index');

print("end")




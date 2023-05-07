import pygrib

msgs = pygrib.open('./pgrbanl_mean_2014_UGRD_sigma.grib')

msg1 = msgs[1]

print(msg1)

print(msg1.shortName)

lats, lons = msg1.latlons()

print(lats)

print(lons)

array = msg1.values

print(array)

msgs.close()
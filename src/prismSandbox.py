import certifi
import os
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

from datetime import date
from epymorph.kit import *
from epymorph.adrio import us_tiger, prism
import platformdirs

# Show where epymorph is caching data
cache_dir = platformdirs.user_cache_dir("epymorph")
print(f"Epymorph cache location: {cache_dir}\n")

coconino_scope = CountyScope.in_counties(["04005"], year=2020)
time_frame = TimeFrame.range(date(2020, 1, 1), date(2020, 12, 31))

mean_temp_adrio = prism.Temperature("Mean")
precip_adrio = prism.Precipitation()

with sim_messaging(live=False):
    temperature = mean_temp_adrio.with_context(
        scope=coconino_scope,
        time_frame=time_frame,
        params={"centroid": us_tiger.GeometricCentroid()},
    ).evaluate()

    precipitation = precip_adrio.with_context(
        scope=coconino_scope,
        time_frame=time_frame,
        params={"centroid": us_tiger.GeometricCentroid()},
    ).evaluate()

print(f"County: {coconino_scope.node_ids}")
print(f"\nTemperature in Celsius (first 10 days):\n{temperature[:10]}")
print(f"Min: {temperature.min():.1f}°C, Max: {temperature.max():.1f}°C")
print(f"\nPrecipitation in mm (first 10 days):\n{precipitation[:10]}")
print(f"Min: {precipitation.min():.1f}mm, Max: {precipitation.max():.1f}mm")

print(f"\nPrecipitation data: {precipitation}")
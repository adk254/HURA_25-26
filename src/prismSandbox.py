import certifi
import os
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

from datetime import date
from epymorph.kit import *
from epymorph.adrio import acs5, us_tiger, prism

coconino_scope = CountyScope.in_counties(["04005"], year=2020)
time_frame = TimeFrame.range(date(2020, 1, 1), date(2020, 12, 31))
mean_temp_adrio = prism.Temperature("Mean")

rume = SingleStrataRUME.build(
    ipm.No(),
    mm.No(),
    init.NoInfection(),
    scope=coconino_scope,
    time_frame=time_frame,
    params={
        "temperature": mean_temp_adrio,
        "population": acs5.Population(),
        "centroid": us_tiger.GeometricCentroid(),
    },
)

rume.estimate_data()

with sim_messaging(live=False):
    temperature = mean_temp_adrio.with_context(
        scope=rume.scope,
        time_frame=rume.time_frame,
        params={
            "centroid": us_tiger.GeometricCentroid(),
        },
    ).evaluate()

print(f"County: {coconino_scope.node_ids}")
print(f"Temperature in Celsius (first 10 days):\n{temperature[:10]}")
print(f"Min: {temperature.min():.1f}°C, Max: {temperature.max():.1f}°C")
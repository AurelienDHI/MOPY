import numpy as np
import pandas as pd
import mikeoperationspy as mopy

app = mopy.Application(mopy.ConnectionInfo(
    flavour     = "PostgreSQL",
    host        = "localhost",
    port        = 5436,
    user        = "admin",
    password    = "dssadmin",
    database    = "ELSA-MOPY",
    workspace   = "workspace1",
))
app

ds = (
    app.time_series_manager
       .get("/Keirunga")
       .to_mikeio()
)
ds

result = ds[0].max().values

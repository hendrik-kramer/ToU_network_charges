# use linopti environment
# conda activate linopti
# in Spyder: right click on tab main_script.py --> set console working directory

# %matplotlib qt

import sys
print(sys.executable)

import pandas as pd
import numpy as np
import xarray as xr

import functions_tariff_network_charge_study.load_functions as f_load




# Load network charges (regular and reduced)
parameter_filepath_dsos = r"Z:\10_Paper\13_Alleinautorenpaper\Aufgabe_Hendrik_v4.xlsx"
timesteps = f_load.load_timesteps(2024)
network_charges_xr, _, _, _,  xr_ht_charge, xr_st_charge, xr_nt_charge = f_load.load_network_charges(parameter_filepath_dsos, timesteps) # dimension: Time x DSO region x scenario (red, reg)
network_charges_red_xr = network_charges_xr.sel(s="red")

xr_nt_charge_new = xr_st_charge / 10

nt_xr = (network_charges_red_xr == network_charges_red_xr.min("t"))
ht_xr = (network_charges_red_xr == network_charges_red_xr.max("t"))
st_xr =  (network_charges_red_xr != network_charges_red_xr.min("t")) & (network_charges_red_xr != network_charges_red_xr.max("t"))


change_possible = (xr_nt_charge > 0.101 * xr_st_charge )
change_possible.mean()

# load SLP
filepath_slp = r"Z:\10_Paper\13_Alleinautorenpaper\SLP\Lastprofil_Haushalt_H0_2025.xlsx"
slp = pd.read_excel(filepath_slp, sheet_name="Jahreszeitreihe_2024")[["Zeitstempel", "Ergebnis"]].iloc[0:34944,:] # discard dec 30, 31
slp["Zeitstempel"] = network_charges_xr["t"].to_pandas().index
slp = slp.set_index("Zeitstempel")
slp.index.name = "t"
slp["Ergebnis"] = 2500/sum(slp["Ergebnis"]) * slp["Ergebnis"] # scale to 2500 kWh
slp_xr = xr.DataArray(slp["Ergebnis"])


nt_sum = (nt_xr * slp_xr).sum(dim="t")
st_sum = (st_xr * slp_xr).sum(dim="t")
ht_sum = (ht_xr * slp_xr).sum(dim="t")

nt_rel = nt_sum / (nt_sum + st_sum + ht_sum)
st_rel = st_sum / (nt_sum + st_sum + ht_sum)
ht_rel = ht_sum / (nt_sum + st_sum + ht_sum)

xr_ht_charge_new = ((xr_nt_charge - xr_nt_charge_new) * nt_sum) / ht_sum
xr_ht_charge_new = np.maximum(xr_ht_charge_new, 0)
xr_ht_charge_new = xr_ht_charge_new + xr_ht_charge

xr_ht_charge_new.mean()
xr_ht_charge_new.sum() / change_possible.sum()

pd_new_charges = pd.concat([xr_nt_charge_new.to_pandas(), xr_ht_charge_new.to_pandas()], axis=1).rename(columns={0:"NT", 1:"HT"})
pd_new_charges.to_csv(r"Z:\10_Paper\13_Alleinautorenpaper\VNB\new_network_charges_sensitivity.csv")

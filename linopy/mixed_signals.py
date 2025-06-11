# use linopti environment
# conda activate linopti
# in Spyder: right click on tab main_script.py --> set console working directory

# %matplotlib qt

import sys
print(sys.executable)

import pandas as pd
from linopy import Model
import matplotlib.pyplot as plt 
from datetime import date, timedelta, datetime
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.colors as mcolors
import glob
import os
import warnings
import re


import functions_tariff_network_charge_study.load_functions as f_load


# === LOAD SPOT PRICES =====

print("load 15h auction data")
files = glob.glob(os.path.join(r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_input\preise", "id_auktion_15_uhr", '*.csv'))
print(files)
all_prices = pd.DataFrame()

for ct_file in files:
    
    ct_csv = pd.read_csv(ct_file, skiprows=1)
    warnings.filterwarnings("ignore")
    ct_csv = ct_csv.set_index(pd.to_datetime(ct_csv["Delivery day"], format="%d/%m/%Y")).drop(columns="Delivery day")
    warnings.filterwarnings("default")
    ct_csv = ct_csv.drop(ct_csv.columns[ct_csv.columns.str.contains("Hour") == False], axis=1)
    ct_csv.columns = ct_csv.columns.str.replace("Hour ","").str.replace("A","").str.replace("Q1","00:00").str.replace("Q2","15:00").str.replace("Q3","30:00").str.replace("Q4","45:00").str.replace(" ",":")
    col_hours = [re.split(r'[:B]+', ct_col)[0] for ct_col in ct_csv.columns]
    col_b = [re.findall(r'B', ct_col) for ct_col in ct_csv.columns]
    col_b = [ct_col[0] if len(ct_col) > 0 else "" for ct_col in col_b]
    col_rest = ct_csv.columns.to_series().str.split(":",n=1).str[1].to_list()
    ct_csv.columns = pd.Series([str(int(ct_col)-1) for ct_col in col_hours]).astype(str) + col_b + [":"]*len(col_rest) + col_rest
    ct_csv_stack = ct_csv.stack(level=-1, dropna=True).reset_index().rename(columns={"level_1":"time", 0:"spot_price"})
    ct_csv_stack["helper_hour"] = ct_csv_stack["time"].str.split(':').str[0].str.replace("B",".5").astype(float)
    ct_csv_stack = ct_csv_stack.sort_values(["Delivery day", "helper_hour"])
    
    all_prices = pd.concat([all_prices, ct_csv_stack], axis=0)
    # cannot be sorted given 2x hour B
all_prices = all_prices.reset_index()


# === BOX PLOT ====

# deduce mean of each day (15pm until 15pm next day)
all_prices["spot_price_shifted_by_60_idx"] = all_prices["spot_price"].shift(-15*4)
mean_prices = all_prices.groupby(by=all_prices["Delivery day"])["spot_price_shifted_by_60_idx"].mean()
all_prices["daily_mean_price"] = all_prices["Delivery day"].map(mean_prices)
all_prices["daily_spot_signal"] = all_prices["spot_price_shifted_by_60_idx"] - all_prices["daily_mean_price"] 
all_prices["spot_signal_EUR_MWh"] = all_prices["daily_spot_signal"].shift(+15*4)
all_prices["spot_signal_ct_kWh"] = all_prices["spot_signal_EUR_MWh"]/10


all_prices_pivot = all_prices.pivot(index="Delivery day", columns="time", values="spot_signal_ct_kWh")
all_prices_pivot_sorted = pd.concat([all_prices_pivot.iloc[:,0:4],  all_prices_pivot.iloc[:,44:48],  all_prices_pivot.iloc[:,64:68], all_prices_pivot.iloc[:,68:], all_prices_pivot.iloc[:,4:44], all_prices_pivot.iloc[:,48:64]], axis=1)
all_prices_pivot_sorted.columns = all_prices_pivot_sorted.columns.str[0:-3]

all_prices_pivot_sorted_wo_timeshift = pd.concat([all_prices_pivot_sorted.iloc[:,0:12], all_prices_pivot_sorted.iloc[:,16:]], axis=1)

fig_signal, ax_signal = plt.subplots(layout='constrained')
fig_signal.set_figheight(5)
fig_signal.set_figwidth(18)
ax_signal = all_prices_pivot_sorted_wo_timeshift.plot(ax=ax_signal, kind="box", color=dict(boxes='black', whiskers='black', medians='r', caps='black'), showfliers=False, grid=False)
ax_signal.set_ylim(-12,12)
ax_signal.set_xticklabels(all_prices_pivot_sorted_wo_timeshift.columns, rotation=90)
ax_signal.set_yticks([-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12])
ax_signal.grid(which='major', axis='y', linestyle='--', color="lightgray")
ax_signal.set_ylabel("Price minus daily mean price")
ax_signal.set_xlabel("Time of the day")
ax_signal.set_title("IDA(1) prices from 2018 through 2024 -- auction price minus mean of prices between 15:00 and 14:45 D+1")

ax_signal.hlines(y=[0], xmin=-1, xmax=98, colors=['lightgray'], linestyles=['-'], linewidth=2, zorder=0)
ax_signal.set_xlim(0,97)
fig_signal.savefig(r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results\hourly_spot_signal.svg", format="svg")



# ===== LOAD NETWORK CHARGES ====
timesteps_all = pd.DataFrame()
for ct_year in [2018, 2019, 2020, 2021, 2022, 2023, 2024]:
    timesteps_ct = f_load.load_timesteps(ct_year)
    timesteps_all = pd.concat([timesteps_all, timesteps_ct], axis=0)
timesteps_all_years = timesteps_all.drop_duplicates()

network_charges_xr_all_years = f_load.load_network_charges(parameter_filepath_dsos, timesteps_all_years) # dimension: Time x DSO region x scenario (red, reg)
network_charges_pandas_all_years = network_charges_xr_all_years.sel(s="red").drop("s").to_pandas()
network_charges_pandas_all_years_unique = network_charges_pandas_all_years[~network_charges_pandas_all_years.index.duplicated(keep='first')]
network_charges_pandas_all_years_unique_no_2025 = network_charges_pandas_all_years_unique[network_charges_pandas_all_years_unique.index.year<=2024]

network_charges_signal = network_charges_pandas_all_years_unique_no_2025 - network_charges_pandas_all_years_unique_no_2025.median()


# ===== SCATTER PLOT =====

fig_signal_scatter, ax_ssc = plt.subplots(layout='constrained')
fig_signal_scatter.set_figheight(3)
fig_signal_scatter.set_figwidth(15)
   
x_vals = []
y_vals = []

for ct_dsos in network_charges_signal.columns:
    htnt = (network_charges_signal[ct_dsos] != 0)
    ax_ssc.scatter(all_prices["spot_signal_ct_kWh"][htnt.values], network_charges_signal[ct_dsos][htnt.values], alpha=0.005, facecolor="blue")
    
    y_vals.extend(list( network_charges_signal[ct_dsos][htnt.values]))
    x_vals.extend(list( all_prices["spot_signal_ct_kWh"][htnt.values]))

    
ax_ssc.set_xlim(-100,100)    
ax_ssc.set_ylim(-12,12)    

ax_ssc.hlines(y=[0], xmin=-100, xmax=100, colors=['k'], linestyles=['--'], linewidth=2, zorder=2)
ax_ssc.vlines(x=[0], ymin=-12, ymax=12, colors=['k'], linestyles=['--'], linewidth=2, zorder=2)
ax_ssc.axline([-100, -100], [100, 100], color="k", linestyle="--", linewidth=2)

ax_ssc.grid(True, color="lightgray")

ax_ssc.set_xlabel("energy price signal in ct/kWh")
ax_ssc.set_ylabel("network charge signal in ct/kWh")
ax_ssc.set_ylabel("2018 through 2024")

x_vec  = np.array(x_vals).round(4)[~np.isnan(x_vals)]
y_vec  = np.array(y_vals).round(4)[~np.isnan(x_vals)]

x_vec  = np.array(x_vec)[~np.isnan(y_vec)]
y_vec  = np.array(y_vec)[~np.isnan(y_vec)]

# get linear regression of all vals
coeffs  = np.polyfit(x_vec, y_vec, 1)
# r-squared
p = np.poly1d(coeffs)
yhat = p(x_vec)                         # or [p(z) for z in x]
ybar = np.sum(y_vec)/len(y_vec)          # or sum(y)/len(y)
ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot = np.sum((y_vec - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
det = ssreg / sstot
 
x_linsp = np.linspace(-100,100)
ax_ssc.plot(x_linsp, coeffs[0]*x_linsp + coeffs[1], color="red" ) 




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
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

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
fig_signal.set_figheight(6)
fig_signal.set_figwidth(18)

# put evening hours up front
all_prices_pivot_sorted_wo_timeshift_eve = pd.concat([all_prices_pivot_sorted_wo_timeshift.iloc[:,61:], all_prices_pivot_sorted_wo_timeshift.iloc[:,0:60]], axis=1)

ax_signal = all_prices_pivot_sorted_wo_timeshift_eve.plot(ax=ax_signal, kind="box", whis=(10, 90), patch_artist=True, color=dict(boxes='black', whiskers='black', medians='darkred', caps='black'), boxprops=dict(facecolor="lightgray"), showfliers=False, fontsize=20)


q95 = all_prices_pivot_sorted_wo_timeshift_eve.quantile(q=0.95).to_frame().reset_index().reset_index()
q95["index"] = q95["index"].shift(-1)
q05 = all_prices_pivot_sorted_wo_timeshift_eve.quantile(q=0.05).to_frame().reset_index().reset_index()
q05["index"] = q05["index"].shift(-1)
q05["hour_str"] = q05["time"].str.split(":").str[0].astype(str)



ax_singal2= q95.plot(ax=ax_signal, x="index", y=0.95, kind="scatter", marker="_", color="k", zorder=2)
ax_singal2.xaxis.set_visible(False)
ax_singal3 = q05.plot(ax=ax_signal, x="index", y=0.05, kind="scatter", marker="_", color="k", zorder=2)
ax_singal3.set_xticklabels(q05["hour_str"])

ax_singal3.xaxis.set_visible(True)

#ax_singal3.set_xticklabels(list(q05["hour_str"]))

for i, label in enumerate(ax_singal3.get_xticklabels()):
    if i % 4 != 0:
        label.set_visible(False)  # Hide labels not multiple of 4
    

ax_signal.grid(which='major', axis='y', linestyle='--', color="lightgray")
ax_signal.set_ylabel("Electricity price difference in ct/kWh", fontsize=20)
ax_signal.set_xlabel("Hour of the day", fontsize=20)
ax_signal.set_title("End-consumer retail price minus mean retail price of planning horizon", fontsize=20)
ax_signal.set_yticks([-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12])
ax_signal.set_ylim(-12,14)

ax_signal.hlines(y=[0], xmin=-1, xmax=98, colors=['lightgray'], linestyles=['-'], linewidth=2, zorder=0)
ax_signal.vlines(x=np.arange(1, 97, 4), ymin=-20, ymax=20, colors=['lightgray'], linestyles=['--'], linewidth=1, zorder=0)

ax_signal.set_xlim(0,97)
fig_signal.savefig(r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results\hourly_spot_signal.svg", format="svg")






# ===== LOAD NETWORK CHARGES ====

# run main script until loop before to get variables

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
fig_signal_scatter.set_figheight(7)
fig_signal_scatter.set_figwidth(21)
   
x_vals = []
y_vals = []

for ct_dsos in network_charges_signal.columns:
    htnt = (network_charges_signal[ct_dsos] != 0)
    #ax_ssc.scatter(all_prices["spot_signal_ct_kWh"][htnt.values], network_charges_signal[ct_dsos][htnt.values], alpha=0.002, facecolor="blue", zorder=1)  
    y_vals.extend(list( network_charges_signal[ct_dsos][htnt.values]))
    x_vals.extend(list( all_prices["spot_signal_ct_kWh"][htnt.values]))

pd_all_data = pd.DataFrame({'xvals':x_vals, 'yvals': y_vals}).dropna(subset = ['xvals', 'yvals'])
pd_all_data_selected = pd_all_data[(pd_all_data.xvals>=-45) & (pd_all_data.xvals<=45)]

len(pd_all_data_selected) / len(pd_all_data) # data not in plot

ax_ssc.axis('equal')

xx = 40
yy = 40/3

heatmap, xedges, yedges = np.histogram2d(pd_all_data_selected.xvals, pd_all_data_selected.yvals, bins=[600,200])
heatmap = np.where(heatmap==0, np.nan, heatmap)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
hist_signal = ax_ssc.imshow(heatmap.T, extent=extent, origin='lower', cmap="viridis_r", aspect="auto" )
fig_signal_scatter.colorbar(hist_signal, label="Data observations", orientation="vertical")  

ax_ssc.set_xlim(xmin=-xx, xmax=xx)
ax_ssc.set_ylim(-yy,yy)

ax_ssc.hlines(y=[0], xmin=-xx, xmax=xx, colors=['k'], linestyles=['-'], linewidth=1, zorder=2)
ax_ssc.vlines(x=[0], ymin=-yy, ymax=yy, colors=['k'], linestyles=['-'], linewidth=1, zorder=2)
ax_ssc.axline([-yy, -yy], [yy, yy], color="k", linestyle="-", linewidth=1)


# Change major ticks to show every 20.
ax_ssc.xaxis.set_major_locator(MultipleLocator(5))
ax_ssc.yaxis.set_major_locator(MultipleLocator(5))


ax_ssc.grid(True, color="gray", linestyle="--", linewidth=1, zorder=0)

ax_ssc.set_xlabel("energy price signal (IDA1) in ct/kWh \n \xa0 \n below  ← | → above \n the mean electricity price in planning period")
ax_ssc.set_ylabel("network charge signal in ct/kWh \n \xa0 \n Standard minus low  ← | → High minus standard")
ax_ssc.set_title("Relationship between energy and network signal")



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
ax_ssc.plot(x_linsp, coeffs[0]*x_linsp + coeffs[1], color="lightblue", linestyle=":") 

q1_spot_stronger = np.mean((np.array(x_vals) > 0) & (np.array(y_vals) > 0) & (np.array(x_vals) > np.array(y_vals)))
q1_nc_stronger = np.mean((np.array(x_vals) > 0) & (np.array(y_vals) > 0) & (np.array(x_vals) < np.array(y_vals)))

q2 = np.mean((np.array(x_vals) < 0) & (np.array(y_vals) > 0))

q3_nc_stronger = np.mean((np.array(x_vals) < 0) & (np.array(y_vals) < 0) & (np.array(x_vals) > np.array(y_vals)))
q3_spot_stronger = np.mean((np.array(x_vals) < 0) & (np.array(y_vals) < 0) & (np.array(x_vals) < np.array(y_vals)))

q4 = np.mean((np.array(x_vals) > 0) & (np.array(y_vals) < 0))


#fig_signal_scatter.savefig(r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results\pos_neg_signals.svg", format="svg")
fig_signal_scatter.savefig(r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results\pos_neg_signals.svg", format="svg")



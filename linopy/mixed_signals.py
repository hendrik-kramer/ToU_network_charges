# use linopti environment
# conda activate linopti
# use spyder ide in linopti environement (click on windows --> type spyder --> see search results --> open "Sypder (linopt)")
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
#import matplotlib.colors as (mcolors, ListedColormap)
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
import matplotlib.cm as cm

import glob
import os
import warnings
import re

import functions_tariff_network_charge_study.load_functions as f_load



charge_strategy = "smart"  # "smart", "scheduled", "immediate"
weight_lookup = {"smart":{"weight_time_preference":0, "weight_only_low_segment":0},
                 "scheduled":{"weight_time_preference":1, "weight_only_low_segment":9999},
                 "immediate":{"weight_time_preference":1, "weight_only_low_segment":0}}
parameters_opti = {
    "prices": "spot", # "spot", "mean"
    "year":2024,
    "dso_subset" : range(0,100), # excel read in only consideres 100 rows!
    "emob_subset" : range(0,50),
    "settings_setup": "only_EV", # "only_EV", # "prosumage"
    "network_charges_sensisitity_study": False, # change settings  in the load function possible (works only for smart and spot)
    "auction": "da_auction_hourly_12_uhr_cubic",  # "da_auction_hourly_12_uhr_linInterpol", "da_auction_hourly_12_uhr_stairs", "da_auction_quarterly_12_uhr", id_auktion_15_uhr"
    "quarter" : "all", # "Q1", "Q2, ...
    # relevant after STRISE sconferece
    "penalty_no_charge_before_arrival": 9999,
    "penalty_no_st_ht": 999,
    "weight_no_charge_before_arrival" : 1, #999999,  # prevent charging at noon when new information is revealed  
    "weight_only_low_segment" : weight_lookup[charge_strategy]["weight_only_low_segment"], #999,
    "weight_time_preference" : weight_lookup[charge_strategy]["weight_time_preference"], #99,
    }





# ====================================
# === BOX PLOT INPUT ST and annual reduction ====
# ====================================


filename_dsos = r"Z:\10_Paper\13_Alleinautorenpaper\Aufgabe_Hendrik_v4.xlsx"
all_prices = pd.read_excel(filename_dsos, sheet_name="Entgelte")


st_values = all_prices["AP_ST_ct/kWh"].iloc[0:99]
st_values.name = None
modul1_values = all_prices["Modul_1_GP"].iloc[0:99]
modul1_values.name = None

meanpointprops = dict(marker='x', markeredgecolor='black', markerfacecolor='black', markersize=4) #firebrick


if (False):
    # Figure und Subplots (vertikal)
    fig_input_boxplot, axs_input_boxplots = plt.subplots(2, 1, figsize=(6, 3))
    
    # Erster Boxplot, horizontal
    st_values.plot(ax=axs_input_boxplots[0], kind="box", vert=False, patch_artist=True, widths=0.7, notch=True, showmeans=True, meanprops=meanpointprops, color=dict(boxes='black', whiskers='black', medians='black', caps='black'), boxprops=dict(facecolor="lightgray"), showfliers=False, fontsize=20)
    axs_input_boxplots[0].set_title('Standard network charge in ct/kWh', fontsize=16)
    axs_input_boxplots[0].grid(which='major', axis='x', linestyle='--', color="lightgray")
    
    # Zweiter Boxplot, horizontal
    modul1_values.plot(ax=axs_input_boxplots[1], kind="box", vert=False, patch_artist=True, widths=0.7, notch=True, showmeans=True, meanprops=meanpointprops, color=dict(boxes='black', whiskers='black', medians='black', caps='black'), boxprops=dict(facecolor="lightgray"), showfliers=False, fontsize=20)
    axs_input_boxplots[1].set_title('Annual network charge reduction in € (after tax)', fontsize=16)
    axs_input_boxplots[1].grid(which='major', axis='x', linestyle='--', color="lightgray")
    
    plt.tight_layout()
    plt.show()
    
    fig_input_boxplot.savefig(r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results\input_boxplots.svg", format="svg")
    






my_fontsize = 20
filename_prices = r"Z:\10_Paper\13_Alleinautorenpaper\daten_input\preise\da_auktion_12_uhr_hourly\energy-charts_DAM_hourly_2018_2025.csv"



# ====================================
# === BOX PLOT INPUT SPOT PRICES ====
# ====================================
if (False):
    all_prices = pd.read_csv(filename_prices, skiprows=1).rename(columns={"Unnamed: 0":"time_utc", "Preis (EUR/MWh, EUR/tCO2)":"spot_signal_EUR_MWh"})
    all_prices["time_utc"] = pd.to_datetime(all_prices["time_utc"], utc=True).dt.tz_convert("Europe/Berlin")
    all_prices["spot_signal_ct_kWh"] = all_prices["spot_signal_EUR_MWh"]/10
    all_prices['Time_DE'] = pd.to_datetime(all_prices["time_utc"]).dt.tz_convert('Europe/Berlin')
    all_prices['iso_year'] = all_prices['Time_DE'].dt.isocalendar().year
    all_prices['iso_week'] = all_prices['Time_DE'].dt.isocalendar().week
    all_prices["Delivery day"] = pd.to_datetime(all_prices['Time_DE'].dt.date).astype(str)
    all_prices["time"] = all_prices['Time_DE'].dt.time.astype(str)
    
    all_prices_2024 = all_prices[all_prices["Time_DE"].dt.year==2024]
    all_prices_2018_2024 = all_prices[(all_prices["Time_DE"].dt.year>=2018) & (all_prices["Time_DE"].dt.year<=2024)]
        
    all_prices_pivot_2024 = all_prices_2024.pivot_table(index="Delivery day", columns="time", values="spot_signal_ct_kWh", aggfunc='mean')  # if hour is duplicate (time shift) use mean:
    all_prices_pivot_2018_2024  = all_prices_2018_2024.pivot_table(index="Delivery day", columns="time", values="spot_signal_ct_kWh", aggfunc='mean')  # if hour is duplicate (time shift) use mean:
    
    # put evening hours up front
    # deduce mean of planning period 35 h
    all_prices_pivot_shifted_2024 = pd.concat([all_prices_pivot_2024.iloc[:,13:24], all_prices_pivot_2024.iloc[:,0:24].shift(-1), ], axis=1)
    all_prices_pivot_shifted_2018_2024 = pd.concat([all_prices_pivot_2018_2024.iloc[:,13:24], all_prices_pivot_2018_2024.iloc[:,0:24].shift(-1), ], axis=1)
    
    all_prices_pivot_shifted_daily_mean_2024 = all_prices_pivot_shifted_2024.mean(axis=1)
    all_prices_pivot_shifted_daily_mean_2018_2024 = all_prices_pivot_shifted_2018_2024.mean(axis=1)
    
    all_prices_pivot_minus_mean_2024 = all_prices_pivot_2024.sub(all_prices_pivot_shifted_daily_mean_2024, axis="rows")
    all_prices_pivot_minus_mean_2018_2024 = all_prices_pivot_2018_2024.sub(all_prices_pivot_shifted_daily_mean_2018_2024, axis="rows")
    
    all_prices_pivot_minus_mean_2024 = all_prices_pivot_minus_mean_2024.fillna(0) # quickfix timeshift none -> zero, otherwise no violin
    all_prices_pivot_minus_mean_2018_2024 = all_prices_pivot_minus_mean_2018_2024.fillna(0)  # quickfix timeshift none -> zero, otherwise no violin
    
    #meanpointprops = dict(marker='x', markeredgecolor='black', markerfacecolor='black', markersize=4) #firebrick
    #ax_signal = all_prices_pivot_minus_mean.plot(ax=ax_signal, kind="box", whis=(10, 90), patch_artist=True, widths=0.8, notch=True, showmeans=False, meanprops=meanpointprops, color=dict(boxes='black', whiskers='black', medians='black', caps='black'), boxprops=dict(facecolor="lightgray"), showfliers=False, fontsize=20)
    
     
    def add_label(violin, label): # to make dummy legend for violin plots
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))
    
    
    fig_signal, ax_signal = plt.subplots(nrows=1, ncols=1, figsize=(16, 5))
    
    
    parts_2024  = ax_signal.violinplot(all_prices_pivot_minus_mean_2024, showextrema=False, showmedians=True, side='low', widths=0.8)  # left side only 2024
    parts_2018_2024  = ax_signal.violinplot(all_prices_pivot_minus_mean_2018_2024, showextrema=False, showmedians=True, side='high', widths=0.8)# left side all years 2018-2024
    
    
    
    # layout violin plots
    for pc in parts_2024['bodies']:
        pc.set_facecolor('darkgray')
        #pc.set_edgecolor('black')
        #pc.set_linewidth(0.8)
        pc.set_alpha(1)
        
    parts_2024['cmedians'].set_edgecolor('black')
    parts_2024['cmedians'].set_linewidth(1)
    parts_2024['cmedians'].set_alpha(1)
        
    for pc in parts_2018_2024['bodies']:
        pc.set_facecolor('lightgray')
        #pc.set_edgecolor('black')
        #pc.set_linewidth(0.8)
        pc.set_alpha(1)
    
    parts_2018_2024['cmedians'].set_edgecolor('black')
    parts_2018_2024['cmedians'].set_linewidth(1)
    parts_2018_2024['cmedians'].set_alpha(1)
    
    #ax_singal2 = q95.plot(ax=ax_signal, x="index", y=0.95, kind="scatter", marker="_", color="k", zorder=2, s=150)
    #ax_singal2.xaxis.set_visible(False)
    #ax_singal3 = q05.plot(ax=ax_signal, x="index", y=0.05, kind="scatter", marker="_", color="k", zorder=2, s=150)
    #ax_singal3.set_xticklabels(q05["hour_str"])
    
    ax_signal.xaxis.set_visible(True)
    #ax_signal.set_xticklabels(list(q05["hour_str"]))
    
    ax_signal.grid(which='major', axis='y', linestyle='--', color="darkgray")
    ax_signal.set_ylabel("Price difference in ct/kWh", fontsize=my_fontsize)
    ax_signal.set_xlabel("Hour of the day", fontsize=my_fontsize)
    
    ax_signal.set_yticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    ax_signal.set_ylim(-20,20)
    ax_signal.set_yticklabels(ax_signal.get_yticklabels(), fontsize=my_fontsize)
    
    ax_signal.set_xticks(np.arange(1, 25))
    ax_signal.set_xlim(0.5,24.5)
    ax_signal.set_xticklabels(ax_signal.get_xticklabels(), fontsize=my_fontsize)
    
    # horizontal zero line
    ax_signal.hlines(y=[0], xmin=-1, xmax=24, colors=['lightgray'], linestyles=['-'], linewidth=2, zorder=0)
    
    
    labels = [] # reset
    add_label(parts_2024, "2024")    
    add_label(parts_2018_2024, "2018 - 2024")    
    
    plt.legend(*zip(*labels), loc="upper left", ncols=2, fontsize=my_fontsize)
    
    fig_signal.tight_layout()    
    
    fig_signal.savefig(r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results\hourly_da_spot_signal_2018_2024_violin.svg", format="svg")




# ==============================================
# ===== MIXED SIGNALS HEATMAP =====  // SET SENSITIVITY STUDY = FALSE
# ==============================================

# load all 
all_prices = pd.read_csv(filename_prices, skiprows=1).rename(columns={"Unnamed: 0":"time_utc", "Preis (EUR/MWh, EUR/tCO2)":"spot_signal_EUR_MWh"})
all_prices["time_utc"] = pd.to_datetime(all_prices["time_utc"], utc=True).dt.tz_convert("Europe/Berlin")
all_prices["spot_signal_ct_kWh"] = all_prices["spot_signal_EUR_MWh"]/10
all_prices['Time_DE'] = pd.to_datetime(all_prices["time_utc"]).dt.tz_convert('Europe/Berlin')
all_prices['iso_year'] = all_prices['Time_DE'].dt.isocalendar().year
all_prices['iso_week'] = all_prices['Time_DE'].dt.isocalendar().week
all_prices["Delivery day"] = pd.to_datetime(all_prices['Time_DE'].dt.date).astype(str)
all_prices["time"] = all_prices['Time_DE'].dt.time.astype(str)
all_prices = all_prices[(all_prices["Time_DE"].dt.isocalendar().year>=2018) & (all_prices["Time_DE"].dt.isocalendar().year<=2024)] # always all years
all_prices_pivot = all_prices.pivot_table(index="Delivery day", columns="time", values="spot_signal_ct_kWh", aggfunc='mean')  # if hour is duplicate (time shift) use mean:

# put evening hours up front
# deduce mean of planning period 35 h
all_prices_pivot_shifted = pd.concat([all_prices_pivot.iloc[:,13:24], all_prices_pivot.iloc[:,0:24].shift(-1), ], axis=1)
all_prices_pivot_shifted_daily_mean = all_prices_pivot_shifted.mean(axis=1)
all_prices_pivot_minus_mean = all_prices_pivot.sub(all_prices_pivot_shifted_daily_mean, axis="rows")

# unpivot to get timeseries again
all_prices_minus_mean = all_prices_pivot_minus_mean.transpose().unstack().reset_index().rename(columns={0:"spot_signal_ct_kWh"})
all_prices_minus_mean["datetime"] = pd.to_datetime(all_prices_minus_mean["Delivery day"] + " " + all_prices_minus_mean["time"])
all_prices_minus_mean = all_prices_minus_mean.drop(columns=["Delivery day", "time"]).set_index("datetime")

# retime to quarter hours
all_prices_minus_mean_15min = all_prices_minus_mean.resample("15min").mean().ffill()
all_prices_minus_mean_15min.loc[pd.to_datetime("2024-12-31 23:15:00"),:] = all_prices_minus_mean_15min.iloc[-1]["spot_signal_ct_kWh"]
all_prices_minus_mean_15min.loc[pd.to_datetime("2024-12-31 23:30:00"),:] = all_prices_minus_mean_15min.iloc[-1]["spot_signal_ct_kWh"]
all_prices_minus_mean_15min.loc[pd.to_datetime("2024-12-31 23:45:00"),:] = all_prices_minus_mean_15min.iloc[-1]["spot_signal_ct_kWh"]
all_prices_minus_mean_15min.index.name = ""

idx_iso_2024 = (all_prices_minus_mean_15min.index.isocalendar().year==2024)
all_prices_minus_mean_15min_iso_2024 = all_prices_minus_mean_15min[idx_iso_2024 
                                                                   | (all_prices_minus_mean_15min.index==pd.to_datetime("2024-12-31 23:15:00"))
                                                                   | (all_prices_minus_mean_15min.index==pd.to_datetime("2024-12-31 23:30:00"))
                                                                   | (all_prices_minus_mean_15min.index==pd.to_datetime("2024-12-31 23:45:00"))]



# ===== LOAD NETWORK CHARGES ====  

# run main script until loop before to get variables

timesteps_all = pd.DataFrame()
for ct_year in [2018, 2019, 2020, 2021, 2022, 2023, 2024]:
    timesteps_ct = f_load.load_timesteps(ct_year)
    timesteps_all = pd.concat([timesteps_all, timesteps_ct], axis=0)
timesteps_all_years = timesteps_all.drop_duplicates()

network_charges_xr_all_years, _ , _ , _ , _ , _ , _ , _   = f_load.load_network_charges(filename_dsos, timesteps_all_years, parameters_opti) # dimension: Time x DSO region x scenario (red, reg)
network_charges_pandas_all_years = network_charges_xr_all_years.sel(s="red").drop("s").to_pandas()
network_charges_pandas_all_years_unique = network_charges_pandas_all_years[~network_charges_pandas_all_years.index.duplicated(keep='first')]
network_charges_pandas_all_years_unique_no_2025 = network_charges_pandas_all_years_unique[network_charges_pandas_all_years_unique.index.year<=2024]

network_charges_signal = network_charges_pandas_all_years_unique_no_2025 - network_charges_pandas_all_years_unique_no_2025.median()
network_charges_signal_iso_2024 = network_charges_signal[network_charges_signal.index.isocalendar().year==2024]

# === export subset of critical prices
critical_steps = pd.DataFrame(columns=network_charges_signal_iso_2024.columns, index=network_charges_signal_iso_2024.index)
for ct_col in critical_steps.columns:
    critical_steps.loc[:,ct_col] = (1*(network_charges_signal_iso_2024[ct_col] > 0).to_numpy() & 1*(all_prices_minus_mean_15min_iso_2024.to_numpy().T < 0) ) #&
                                 #  1*(np.absolute(all_prices_minus_mean_15min_iso_2024.to_numpy().T) > np.absolute(network_charges_signal_iso_2024[ct_col].to_numpy( ) ) ) )
critical_steps.to_csv(r"Z:\10_Paper\13_Alleinautorenpaper\critical_timesteps_iso2024.csv")


# ===== HEATMAP PLOT, Exemplary for 2024 ===== 



# ===== years 2018 - 2024 =====
x_vals = []
y_vals = []
for ct_dsos in network_charges_signal.columns:
    #htnt = (network_charges_signal[ct_dsos] != 0)
    #y_vals.extend(list( network_charges_signal[ct_dsos][htnt.values]))
    #x_vals.extend(list( all_prices_minus_mean_15min["spot_signal_ct_kWh"][htnt.values]))

    y_vals.extend(list( network_charges_signal[ct_dsos][:]))
    x_vals.extend(list( all_prices_minus_mean_15min["spot_signal_ct_kWh"][:]))

pd_all_data = pd.DataFrame({'xvals':x_vals, 'yvals': y_vals}).dropna(subset = ['xvals', 'yvals'])


# ===== only 2024 =====
this_year = 2024
all_years = True

def get_shares(this_year, network_charges_signal, all_prices_minus_mean_15min):
    
    x_vals = []
    y_vals = []
    
   
    network_charges_signal_one_year = network_charges_signal[network_charges_signal.index.year==this_year]
    all_prices_one_year = all_prices_minus_mean_15min[all_prices_minus_mean_15min.index.year == this_year]
        
    for ct_dsos in network_charges_signal.columns:
        
        htnt = (network_charges_signal_one_year[ct_dsos] != 0)
        y_vals.extend(list( network_charges_signal_one_year[ct_dsos][htnt.values]))
        x_vals.extend(list( all_prices_one_year["spot_signal_ct_kWh"][htnt.values]))
    
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    
    pd_all_data_one_year = pd.DataFrame({'xvals':x_vals, 'yvals': y_vals}).dropna(subset = ['xvals', 'yvals'])
    
    q1_market = int(100*np.round(np.mean((x_vals > 0) & (y_vals > 0) & (x_vals > y_vals)),2))
    q1_network = int(100*np.round(np.mean((x_vals > 0) & (y_vals > 0) & (x_vals < y_vals)),2))
    q2_network = int(100*np.round(np.mean((x_vals < 0) & (y_vals > 0) & (-x_vals < y_vals)),2))
    q2_market = int(100*np.round(np.mean((x_vals < 0) & (y_vals > 0) & (-x_vals > y_vals)),2))
    q3_market = int(100*np.round(np.mean((x_vals < 0) & (y_vals < 0) & (x_vals < y_vals)),2))
    q3_network = int(100*np.round(np.mean((x_vals < 0) & (y_vals < 0) & (x_vals > y_vals)),2))
    q4_network = int(100*np.round(np.mean((x_vals > 0) & (y_vals < 0) & (x_vals < -y_vals)),2))
    q4_market = int(100*np.round(np.mean((x_vals > 0) & (y_vals < 0) & (x_vals > -y_vals)),2))
    
    return [q1_market, q1_network, q2_network, q2_market, q3_market, q3_network, q4_network, q4_market]


res_vals = get_shares(this_year, network_charges_signal, all_prices_minus_mean_15min)

q1_market = res_vals[0]
q1_network = res_vals[1]
q2_network = res_vals[2]
q2_market = res_vals[3]
q3_market = res_vals[4]
q3_network = res_vals[5]
q4_network = res_vals[6]
q4_market = res_vals[7] 


# large heatmap of mixed signals
fig_signal_scatter, ax_ssc = plt.subplots(layout='constrained')
fig_signal_scatter.set_figwidth(15)
fig_signal_scatter.set_figheight(7) # irrelevant due to aspect = equal


#len(pd_all_data_selected) / len(pd_all_data) # data not in plot

ax_ssc.axis('equal')

xx = 30
yy = 15

x_bins = int(np.round(2*xx))
y_bins = int(np.round(2*yy))
#ax_ssc.set_xlim(xmin=-xx, xmax=xx)
#ax_ssc.set_ylim(-yy,yy)

# create gridded heatmap
heatmap, xedges, yedges = np.histogram2d(pd_all_data.xvals, pd_all_data.yvals, range=[(-xx,xx),(-yy,yy)], bins=[x_bins,y_bins])
heatmap = np.where(heatmap==0, np.nan, heatmap).T
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

heatmap[np.isnan(heatmap)] = 0 # set values to gray outiside of data range

greens = cm.get_cmap('Greens', 100)
newcolors = greens(np.linspace(0, 1, 100))
newcolors[:1, :] = (0.75, 0.75, 0.75, 0.5)
cmap_newgreens = ListedColormap(newcolors)


hist_signal = ax_ssc.imshow(heatmap, extent=extent, cmap=cmap_newgreens, origin="lower") #viridis_r  #origin='lower', 


ax_ssc.set_aspect('equal',adjustable='box')
ax_ssc.tick_params(axis='x', labelsize=my_fontsize)
ax_ssc.tick_params(axis='y', labelsize=my_fontsize)


# add colorbar
cbar = fig_signal_scatter.colorbar(hist_signal, orientation="vertical")
cbar.set_label("Data pairs per bin", fontsize=my_fontsize)
ticklabs = cbar.ax.get_yticklabels()
cbar.ax.set_yticklabels(ticklabs, fontsize=my_fontsize) 
cbar.ax.yaxis.set_ticks([0, 50e3, 100e3, 150e3, 200e3, 250e3])
cbar.ax.set_yticklabels(['0', '50k', '100k', '150k', '200k', '250k'])

cbar.ax.tick_params(labelsize=my_fontsize)

ax_ssc.hlines(y=[0], xmin=-xx, xmax=xx, colors=['gray'], linestyles=['--'], linewidth=2, zorder=2)
ax_ssc.vlines(x=[0], ymin=-yy, ymax=yy, colors=['gray'], linestyles=['--'], linewidth=2, zorder=2)
ax_ssc.axline([-yy, -yy], [yy, yy], color="gray", linestyle="--", linewidth=2)
ax_ssc.axline([yy, -yy], [-yy, yy], color="gray", linestyle="--", linewidth=2)

ax_ssc.text(-12.5, 7.5, r'$\text{II}_{market}$' + "\n" + str(q2_market) + " %", fontsize=16, ha='center', va='center')
ax_ssc.text(-7.5, 12.5, r'$\text{II}_{network}$' + "\n" + str(q2_network) + " %", fontsize=16, ha='center', va='center')
ax_ssc.text(7.5, 12.5, r'$\text{I}_{network}$' + "\n" + str(q1_network) + " %", fontsize=16, ha='center', va='center')
ax_ssc.text(12.5, 7.5, r'$\text{I}_{market}$' + "\n" + str(q1_market) + " %",fontsize=16, ha='center', va='center')
ax_ssc.text(-17.5, -7.5, r'$\text{III}_{market}$' + "\n" + str(q3_market) + " %",fontsize=16, ha='center', va='center')
ax_ssc.text(-7.5, -12.5, r'$\text{III}_{network}$' + "\n" + str(q3_network) + " %", fontsize=16, ha='center', va='center')
ax_ssc.text(12.5, -7.5, r'$\text{IV}_{market}$' + "\n" + str(q4_market) + " %", fontsize=16, ha='center', va='center')
ax_ssc.text(7.5, -12.5, r'$\text{IV}_{network}$' + "\n" + str(q4_network) + " %", fontsize=16, ha='center', va='center')


# Change major ticks to show every 20.
ax_ssc.xaxis.set_major_locator(MultipleLocator(5))
ax_ssc.yaxis.set_major_locator(MultipleLocator(5))

ax_ssc.grid(True, color="gray", linestyle="--", linewidth=1, zorder=0)

ax_ssc.set_xlabel("Market signal in ct/kWh \n \xa0 \n lower  ←|→ higher \n than the mean electricity price of the daily planning period", fontsize=my_fontsize)
ax_ssc.set_ylabel("Low – Standard  ←|→ High – Standard \n \xa0 \n   Network signal in ct/kWh",  fontsize=my_fontsize)

ax_ssc.set_xlim(xmin=-xx, xmax=xx)
ax_ssc.set_ylim(-yy,yy)

from scipy.stats import pearsonr
corr_coef, _ = pearsonr(pd_all_data.xvals, pd_all_data.yvals)

fig_signal_scatter.savefig(r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results\pos_neg_da_signals.svg", format="svg")



# ===== HEATMAP PLOT, 2018 - 2023 ===== 


from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


my_fontsize = 20
arr = np.empty((6, 8))

years = [2018, 2019, 2020, 2021, 2022, 2023]
for i, yr in enumerate(years):
    print(yr)
    res_vals = get_shares(yr, network_charges_signal, all_prices_minus_mean_15min)
    arr[i, :] = res_vals

arr = arr/100

vertices = np.array([ [[0, 0], [1, 0], [1, 1]],
                     [[0, 0], [1, 1], [0, 1]], 
                     [[0, 0], [0, 1], [-1, 1]], 
                     [[0, 0], [-1, 1], [-1, 0]], 
                     [[0, 0], [-1, 0], [-1, -1]], 
                     [[0, 0], [-1, -1], [0, -1]],
                     [[0, 0], [0, -1], [1, -1]], 
                     [[0, 0], [1, -1], [1, 0]] ])

layout = [ ["2018", "2019", "2020", "colorbar"],
           ["2021", "2022", "2023", "colorbar"] ]



fig_seperate_years, axd = plt.subplot_mosaic(layout, figsize=(12, 6), constrained_layout=True, gridspec_kw={
        'width_ratios': [1, 1, 1, 0.3],  # rechts schmaler
        'wspace': 0.0,                   # kein horizontaler Zwischenraum
        'hspace': 0.0
    })

# Links: Jahrestitel in Layout-Positionen A..F (hier durch Jahre ersetzt)
left_keys = ["2018", "2019", "2020", "2021", "2022", "2023"]

# Dreiecke in jedem linken Subplot zeichnen
for idx, key in enumerate(left_keys):
    ax = axd[key]
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(left_keys[idx], fontsize=my_fontsize)
    ax.set_aspect('equal', adjustable='box')


    # Diagonalen und Achsenlinien (optional)
    ax.plot([-1, 1], [-1, 1], color='gray', linestyle='--', linewidth=1)
    ax.plot([-1, 1], [ 1, -1], color='gray', linestyle='--', linewidth=1)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    # 8 Dreiecke mit grünem Farbton; Alpha aus arr[idx, j]
    for j in range(8):
        tri = vertices[j]
        alpha = float(arr[idx, j])
        color = (0.0, 0.5, 0.0, alpha)  # RGBA: Grün mit Alpha aus arr
        poly = Polygon(tri, closed=True, facecolor=color, edgecolor=None)
        ax.add_patch(poly)
        
        # Textannotation mittig im Dreieck: arr[idx, j] als Prozent
        cx = tri[:, 0].mean()
        cy = tri[:, 1].mean()
        perc = int(round(arr[idx, j] * 100))
        ax.text(cx, cy, f"{perc}%", fontsize=16, ha='center', va='center', color='black')

    # red line
    extra_tri = np.array([[-0.2, 0.1], [-0.9, 0.8], [-0.9, 0.1]])
    poly_extra = Polygon(extra_tri, closed=True, facecolor='none', edgecolor='darkred', linestyle='--', linewidth=1)
    ax.add_patch(poly_extra)


# Rechte Spalte (colorbar) bleibt leer
axR = axd["colorbar"]
axR.axis('off')

# Greens-Colorbar von weiß (0%) bis grün (100%)
greens = plt.cm.Greens(np.linspace(0, 0.5, 256))[:, :3]
white = np.array([[1.0, 1.0, 1.0]])
cmap_colors = np.vstack((white, greens))
custom_cmap = ListedColormap(cmap_colors)

norm = Normalize(vmin=0.0, vmax=0.5)
sm = ScalarMappable(cmap=custom_cmap, norm=norm)
sm.set_array([])

cbar = plt.colorbar(sm, ax=axR, orientation='vertical')
cbar.set_ticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
cbar.set_ticklabels(['0%', '10%', '20%', '30%', '40%', '50%'])
cbar.ax.tick_params(labelsize=my_fontsize)
cbar.set_label('Data pairs (relative)', fontsize=my_fontsize)

plt.show()

fig_seperate_years.savefig(r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results\signals_six_years.svg", format="svg")

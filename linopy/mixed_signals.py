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

#from statistics import linear_regression

import functions_tariff_network_charge_study.load_functions as f_load




# ====================================
# === BOX PLOT INPUT ST and annual reduction ====
# ====================================


filename = r"Z:\10_Paper\13_Alleinautorenpaper\Aufgabe_Hendrik_v4.xlsx"
all_prices = pd.read_excel(filename, sheet_name="Entgelte")


st_values = all_prices["AP_ST_ct/kWh"].iloc[0:99]
st_values.name = None
modul1_values = all_prices["Modul_1_GP"].iloc[0:99]
modul1_values.name = None

meanpointprops = dict(marker='x', markeredgecolor='black', markerfacecolor='black', markersize=4) #firebrick


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
filename = r"Z:\10_Paper\13_Alleinautorenpaper\daten_input\preise\da_auktion_12_uhr_hourly\energy-charts_DAM_hourly_2018_2025.csv"



# ====================================
# === BOX PLOT INPUT SPOT PRICES ====
# ====================================

all_years_not_only_2024 = True

all_prices = pd.read_csv(filename, skiprows=1).rename(columns={"Unnamed: 0":"time_utc", "Preis (EUR/MWh, EUR/tCO2)":"spot_signal_EUR_MWh"})
all_prices["time_utc"] = pd.to_datetime(all_prices["time_utc"], utc=True).dt.tz_convert("Europe/Berlin")
all_prices["spot_signal_ct_kWh"] = all_prices["spot_signal_EUR_MWh"]/10
all_prices['Time_DE'] = pd.to_datetime(all_prices["time_utc"]).dt.tz_convert('Europe/Berlin')
all_prices['iso_year'] = all_prices['Time_DE'].dt.isocalendar().year
all_prices['iso_week'] = all_prices['Time_DE'].dt.isocalendar().week
all_prices["Delivery day"] = pd.to_datetime(all_prices['Time_DE'].dt.date).astype(str)
all_prices["time"] = all_prices['Time_DE'].dt.time.astype(str)

if not(all_years_not_only_2024):
    all_prices = all_prices[all_prices["Time_DE"].dt.year==2024]
else:
    all_prices = all_prices[(all_prices["Time_DE"].dt.year>=2018) & (all_prices["Time_DE"].dt.year<=2024)]
all_prices_pivot = all_prices.pivot_table(index="Delivery day", columns="time", values="spot_signal_ct_kWh", aggfunc='mean')  # if hour is duplicate (time shift) use mean:

# put evening hours up front
# deduce mean of planning period 35 h
all_prices_pivot_shifted = pd.concat([all_prices_pivot.iloc[:,13:24], all_prices_pivot.iloc[:,0:24].shift(-1), ], axis=1)
all_prices_pivot_shifted_daily_mean = all_prices_pivot_shifted.mean(axis=1)
all_prices_pivot_minus_mean = all_prices_pivot.sub(all_prices_pivot_shifted_daily_mean, axis="rows")

# calculate 5% 95% quantile values
q95 = all_prices_pivot_minus_mean.quantile(q=0.95).to_frame().reset_index().reset_index()
q95["index"] = q95["index"].shift(-1)
q95.loc[23, "index"] = 24.0
q05 = all_prices_pivot_minus_mean.quantile(q=0.05).to_frame().reset_index().reset_index()
q05["index"] = q05["index"].shift(-1)
q05.loc[23, "index"] = 24.0
q05["hour_str"] = q05["time"].str.split(":").str[0].astype(str)

# start with plot
fig_signal, ax_signal = plt.subplots(layout='constrained')
fig_signal.set_figwidth(16)
fig_signal.set_figheight(5)

meanpointprops = dict(marker='x', markeredgecolor='black', markerfacecolor='black', markersize=4) #firebrick
ax_signal = all_prices_pivot_minus_mean.plot(ax=ax_signal, kind="box", whis=(10, 90), patch_artist=True, widths=0.8, notch=True, showmeans=True, meanprops=meanpointprops, color=dict(boxes='black', whiskers='black', medians='black', caps='black'), boxprops=dict(facecolor="lightgray"), showfliers=False, fontsize=20)

ax_singal2 = q95.plot(ax=ax_signal, x="index", y=0.95, kind="scatter", marker="_", color="k", zorder=2, s=150)
ax_singal2.xaxis.set_visible(False)
ax_singal3 = q05.plot(ax=ax_signal, x="index", y=0.05, kind="scatter", marker="_", color="k", zorder=2, s=150)
ax_singal3.set_xticklabels(q05["hour_str"])

ax_singal3.xaxis.set_visible(True)
ax_singal3.set_xticklabels(list(q05["hour_str"]))

ax_signal.grid(which='major', axis='y', linestyle='--', color="lightgray")
ax_signal.set_ylabel("Price difference in ct/kWh", fontsize=my_fontsize)
ax_signal.set_xlabel("Hour of the day", fontsize=my_fontsize)

ax_signal.set_yticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12])
ax_signal.set_ylim(-10,12)

ax_signal.hlines(y=[0], xmin=-1, xmax=24, colors=['lightgray'], linestyles=['-'], linewidth=2, zorder=0)
ax_signal.set_xlim(0.5,24.5)

if all_years_not_only_2024:
    ax_signal.set_title("2018 – 2024", fontsize=my_fontsize)
    fig_signal.savefig(r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results\hourly_da_spot_signal_2018_2024.svg", format="svg")
else:
    ax_signal.set_title("2024", fontsize=my_fontsize)
    fig_signal.savefig(r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results\hourly_da_spot_signal_2024.svg", format="svg")




# ==============================================
# ===== MIXED SIGNALS HEATMAP + CENTROIDS =====  // SET SENSITIVITY STUDY = FALSE
# ==============================================

# load all 
all_prices = pd.read_csv(filename, skiprows=1).rename(columns={"Unnamed: 0":"time_utc", "Preis (EUR/MWh, EUR/tCO2)":"spot_signal_EUR_MWh"})
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

#all_prices_minus_mean_15min_xr = xr.DataArray(all_prices_minus_mean_15min["spot_signal_ct_kWh"], dims="t")



# ===== LOAD NETWORK CHARGES ====  

# run main script until loop before to get variables

timesteps_all = pd.DataFrame()
for ct_year in [2018, 2019, 2020, 2021, 2022, 2023, 2024]:
    timesteps_ct = f_load.load_timesteps(ct_year)
    timesteps_all = pd.concat([timesteps_all, timesteps_ct], axis=0)
timesteps_all_years = timesteps_all.drop_duplicates()

network_charges_xr_all_years, _ , _ , _ , _ , _ , _ , _   = f_load.load_network_charges(parameter_filepath_dsos, timesteps_all_years, parameters_opti) # dimension: Time x DSO region x scenario (red, reg)
network_charges_pandas_all_years = network_charges_xr_all_years.sel(s="red").drop("s").to_pandas()
network_charges_pandas_all_years_unique = network_charges_pandas_all_years[~network_charges_pandas_all_years.index.duplicated(keep='first')]
network_charges_pandas_all_years_unique_no_2025 = network_charges_pandas_all_years_unique[network_charges_pandas_all_years_unique.index.year<=2024]

network_charges_signal = network_charges_pandas_all_years_unique_no_2025 - network_charges_pandas_all_years_unique_no_2025.median()



# ===== HEATMAP PLOT ===== 



# ===== years 2018 - 2024 =====
x_vals = []
y_vals = []
for ct_dsos in network_charges_signal.columns:
    htnt = (network_charges_signal[ct_dsos] != 0)
    y_vals.extend(list( network_charges_signal[ct_dsos][htnt.values]))
    x_vals.extend(list( all_prices_minus_mean_15min["spot_signal_ct_kWh"][htnt.values]))

pd_all_data = pd.DataFrame({'xvals':x_vals, 'yvals': y_vals}).dropna(subset = ['xvals', 'yvals'])


# ===== only 2024 =====
x_vals_2024 = []
y_vals_2024 = []
network_charges_signal_2024 = network_charges_signal[network_charges_signal.index.year==2024]
all_prices_2024 = all_prices_minus_mean_15min[all_prices_minus_mean_15min.index.year == 2024]
for ct_dsos in network_charges_signal_2024.columns:
    htnt = (network_charges_signal_2024[ct_dsos] != 0)
    y_vals_2024.extend(list( network_charges_signal_2024[ct_dsos][htnt.values]))
    x_vals_2024.extend(list( all_prices_2024["spot_signal_ct_kWh"][htnt.values]))

pd_all_data_2024 = pd.DataFrame({'xvals':x_vals, 'yvals': y_vals}).dropna(subset = ['xvals', 'yvals'])




fig_signal_scatter, ax_ssc = plt.subplots(layout='constrained')
fig_signal_scatter.set_figwidth(18)
fig_signal_scatter.set_figheight(7) # irrelevant due to aspect = equal


#len(pd_all_data_selected) / len(pd_all_data) # data not in plot

ax_ssc.axis('equal')

xx = 40
yy = 40/3

#ax_ssc.set_xlim(xmin=-xx, xmax=xx)
#ax_ssc.set_ylim(-yy,yy)

# create gridded heatmap
heatmap, xedges, yedges = np.histogram2d(pd_all_data.xvals, pd_all_data.yvals, bins=[600,200])
heatmap = np.where(heatmap==0, np.nan, heatmap).T
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

hist_signal = ax_ssc.imshow(heatmap, extent=extent, origin='lower', cmap="viridis_r")


# deduce median heatmap
if (False):
    heatmap_zero = heatmap
    heatmap_zero[np.isnan(heatmap_zero)] = 0
    rowsum = heatmap_zero.sum(axis=1)
    
    idx_median = np.zeros(len(rowsum))
    for ct_row in range(0, len(rowsum)):
        if rowsum[ct_row] > 0:  
            #idx_median[ct_row] = np.argwhere(heatmap_zero[ct_row,:] == np.percentile(heatmap_zero[ct_row,:], 50, interpolation='nearest'))
            #idx_median[ct_row] = heatmap_zero[ct_row,:].index(np.percentile(heatmap_zero[ct_row,:],50,interpolation='nearest'))
            idx_median[ct_row] = np.argsort(heatmap_zero[ct_row,:])[int((len(heatmap_zero[ct_row,:]) - 1) * 0.5)]
        else:
            idx_median[ct_row] = np.nan
    
    # remove implausible values
    rowsum[(idx_median>550) | (idx_median<50)] = 0

    heatmap_red = np.zeros((200,600,))
    rows = np.linspace(0,199,200, dtype=int)
    for ct_row in rows[rowsum>0]:
        heatmap_red[ct_row, idx_median[ct_row].astype(np.int64)] = 1        
    heatmap_red = np.where(heatmap_red==0, 0, heatmap_red)
    
    cmap_red = mcolors.ListedColormap(['none', 'red'])
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap_red.N)
    
    ax_ssc.imshow(heatmap_red, extent=extent, origin='lower', cmap=cmap_red, zorder=2)


ax_ssc.set_aspect('equal',adjustable='box')
ax_ssc.tick_params(axis='x', labelsize=my_fontsize)
ax_ssc.tick_params(axis='y', labelsize=my_fontsize)



# add colorbar
cbar = fig_signal_scatter.colorbar(hist_signal, orientation="vertical")
cbar.set_label("Data observations in each bin \n(Heatmap covers 2018-2024)", fontsize=my_fontsize)
ticklabs = cbar.ax.get_yticklabels()
cbar.ax.set_yticklabels(ticklabs, fontsize=my_fontsize)  


cbar.ax.tick_params(labelsize=my_fontsize)

ax_ssc.hlines(y=[0], xmin=-xx, xmax=xx, colors=['gray'], linestyles=['--'], linewidth=2, zorder=2)
ax_ssc.vlines(x=[0], ymin=-yy, ymax=yy, colors=['gray'], linestyles=['--'], linewidth=2, zorder=2)
ax_ssc.axline([-yy, -yy], [yy, yy], color="gray", linestyle="--", linewidth=2)
ax_ssc.axline([yy, -yy], [-yy, yy], color="gray", linestyle="--", linewidth=2)

ax_ssc.annotate(r'$II_{market}$',(-24.9, 11.5), fontsize=16)
ax_ssc.annotate(r'$II_{network}$',(-9.9, 11.5), fontsize=16)
ax_ssc.annotate(r'$I_{network}$',(5.1, 11.5), fontsize=16)
ax_ssc.annotate(r'$I_{market}$',(20.1, 11.5), fontsize=16)
ax_ssc.annotate(r'$III_{market}$',(-19.9, -12), fontsize=16)
ax_ssc.annotate(r'$III_{network}$',(-9.9, -12), fontsize=16)
ax_ssc.annotate(r'$IV_{network}$',(0.1, -12), fontsize=16)
ax_ssc.annotate(r'$IV_{market}$',(15.1, -12), fontsize=16)


# Change major ticks to show every 20.
ax_ssc.xaxis.set_major_locator(MultipleLocator(5))
ax_ssc.yaxis.set_major_locator(MultipleLocator(5))


ax_ssc.grid(True, color="gray", linestyle="--", linewidth=1, zorder=0)

ax_ssc.set_xlabel("Day ahead price signal (DA) in ct/kWh \n \xa0 \n lower  ← | → higher \n than the mean electricity price of the daily planning period", fontsize=my_fontsize)
ax_ssc.set_ylabel("Network charge signal in ct/kWh \n \xa0 \n Low – Standard  ← | → High – Standard",  fontsize=my_fontsize)


# deduce regression points for 2018 - 2014
x_vec  = np.array(x_vals).round(4)[~np.isnan(x_vals)]
y_vec  = np.array(y_vals).round(4)[~np.isnan(x_vals)]
x_vec_nonan = np.array(x_vec)[~np.isnan(y_vec)]
y_vec_nonan  = np.array(y_vec)[~np.isnan(y_vec)]

x_vec_nonan_pos = x_vec_nonan[y_vec_nonan>0]
y_vec_nonan_pos = y_vec_nonan[y_vec_nonan>0]
x_vec_nonan_neg = x_vec_nonan[y_vec_nonan<0]
y_vec_nonan_neg = y_vec_nonan[y_vec_nonan<0]

# deduce regression points for 2014
x_vec_2024  = np.array(x_vals_2024).round(4)[~np.isnan(x_vals_2024)]
y_vec_2024  = np.array(y_vals_2024).round(4)[~np.isnan(x_vals_2024)]
x_vec_nonan_2024 = np.array(x_vec_2024)[~np.isnan(y_vec_2024)]
y_vec_nonan_2024  = np.array(y_vec_2024)[~np.isnan(y_vec_2024)]

x_vec_nonan_pos_2024 = x_vec_nonan_2024[y_vec_nonan_2024>0]
y_vec_nonan_pos_2024 = y_vec_nonan_2024[y_vec_nonan_2024>0]
x_vec_nonan_neg_2024 = x_vec_nonan_2024[y_vec_nonan_2024<0]
y_vec_nonan_neg_2024 = y_vec_nonan_2024[y_vec_nonan_2024<0]

x_vec_nonan_pos_mean = x_vec_nonan_pos.mean()
y_vec_nonan_pos_mean = y_vec_nonan_pos.mean()
x_vec_nonan_neg_mean = x_vec_nonan_neg.mean()
y_vec_nonan_neg_mean = y_vec_nonan_neg.mean()
x_vec_nonan_pos_2024_mean = x_vec_nonan_pos_2024.mean()
y_vec_nonan_pos_2024_mean = y_vec_nonan_pos_2024.mean()
x_vec_nonan_neg_2024_mean = x_vec_nonan_neg_2024.mean()
y_vec_nonan_neg_2024_mean = y_vec_nonan_neg_2024.mean()

# do regressions
a_reg_pos, residuals_pos, _, _ = np.linalg.lstsq(np.vstack([x_vec_nonan_pos]).T, y_vec_nonan_pos.T, rcond=None)
a_reg_neg, residuals_neg, _, _ = np.linalg.lstsq(np.vstack([x_vec_nonan_neg]).T, y_vec_nonan_neg.T, rcond=None)
a_reg_pos_2024, residuals_pos_2024, _, _ = np.linalg.lstsq(np.vstack([x_vec_nonan_pos_2024]).T, y_vec_nonan_pos_2024.T, rcond=None)
a_reg_neg_2024, residuals_neg_2024, _, _ = np.linalg.lstsq(np.vstack([x_vec_nonan_neg_2024]).T, y_vec_nonan_neg_2024.T, rcond=None)

x_pos_vals = np.linspace(0, 70)
x_neg_vals = np.linspace(-70,0)

#ax_ssc.plot(x_pos_vals, a_reg_pos*x_pos_vals, color="blue", linestyle="-", linewidth=2, label="2018-2024") # \n (m_pos=" + "{:.4f}".format(a_reg_pos[0]) + ", m_neg=" + "{:.4f}".format(a_reg_neg[0]) + ")"  )
#ax_ssc.plot(x_neg_vals, a_reg_neg*x_neg_vals, color="blue", linestyle="-", linewidth=2, label=None)
#ax_ssc.plot(x_pos_vals, a_reg_pos_2024*x_pos_vals, color="lightblue", linestyle="-", linewidth=2, label="only 2024") #" \n (m_pos=" + "{:.4f}".format(a_reg_pos_2024[0]) + ", m_neg=" + "{:.4f}".format(a_reg_neg_2024[0]) + ")"  )
#ax_ssc.plot(x_neg_vals, a_reg_neg_2024*x_neg_vals, color="lightblue", linestyle="-", linewidth=2, label=None)

ax_ssc.scatter(x_vec_nonan_pos_mean, y_vec_nonan_pos_mean, color="blue", label="2018-2024", facecolors='none', linewidth=2, s=100) # \n (m_pos=" + "{:.4f}".format(a_reg_pos[0]) + ", m_neg=" + "{:.4f}".format(a_reg_neg[0]) + ")"  )
ax_ssc.scatter(x_vec_nonan_neg_mean, y_vec_nonan_neg_mean, color="blue", label=None, facecolors='none', linewidth=2, s=100)
ax_ssc.scatter(x_vec_nonan_pos_2024_mean, y_vec_nonan_pos_2024_mean, color="darkred", label="only 2024", s=100, marker="x", linewidth=2,) # \n (m_pos=" + "{:.4f}".format(a_reg_pos[0]) + ", m_neg=" + "{:.4f}".format(a_reg_neg[0]) + ")"  )
ax_ssc.scatter(x_vec_nonan_neg_2024_mean, y_vec_nonan_neg_2024_mean, color="darkred", label=None, s=100, marker="x", linewidth=2)

#plt.legend(loc="lower right", title="    mean regression \n   through origin for \nupper/lower half-plane", fontsize=16, title_fontsize=16, alignment="center")
plt.legend(loc="lower right", title="        Centroids for \n upper/lower half-plane", fontsize=16, title_fontsize=16, alignment="center")

ax_ssc.set_xlim(xmin=-xx, xmax=xx)
ax_ssc.set_ylim(-yy,yy)


if (False):
    fig_test, ax_test = plt.subplots(layout='constrained')
    ax_test.scatter(x_vec_nonan_pos_2024, y_vec_nonan_pos_2024, alpha=0.01)
    ax_test.plot(x_pos_vals, a_reg_pos_2024*x_pos_vals, color="red", linestyle="-", linewidth=2, label="only 2024") #" \n (m_pos=" + "{:.4f}".format(a_reg_pos_2024[0]) + ", m_neg=" + "{:.4f}".format(a_reg_neg_2024[0]) + ")"  )


# get amount of points per region

q1_spot = np.mean((np.array(x_vals) > 0) & (np.array(y_vals) > 0) & (np.array(x_vals) > np.array(y_vals)))
q1_nc = np.mean((np.array(x_vals) > 0) & (np.array(y_vals) > 0) & (np.array(x_vals) < np.array(y_vals)))

q2_network = np.mean((np.array(x_vals) < 0) & (np.array(y_vals) > 0) & (-np.array(x_vals) < np.array(y_vals)))
q2_spot = np.mean((np.array(x_vals) < 0) & (np.array(y_vals) > 0) & (-np.array(x_vals) > np.array(y_vals)))

q3_nc = np.mean((np.array(x_vals) < 0) & (np.array(y_vals) < 0) & (np.array(x_vals) > np.array(y_vals)))
q3_spot = np.mean((np.array(x_vals) < 0) & (np.array(y_vals) < 0) & (np.array(x_vals) < np.array(y_vals)))

q4_nc = np.mean((np.array(x_vals) > 0) & (np.array(y_vals) < 0) & (np.array(x_vals) < -np.array(y_vals)))
q4_market = np.mean((np.array(x_vals) > 0) & (np.array(y_vals) < 0) & (np.array(x_vals) > -np.array(y_vals)))




fig_signal_scatter.savefig(r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results\pos_neg_da_signals.svg", format="svg")



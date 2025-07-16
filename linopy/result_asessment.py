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
import matplotlib.cm as cm


# read  results
folder_name = "2025-07-14_04-01_all_spot_scheduled_charging_only_EV_r100_v50"


folder_path = Path("../daten_results") / folder_name

result_C_OP_ALL_eur = xr.open_dataarray(folder_path / "C_OP_ALL.nc")
result_SOC_EV = xr.open_dataarray(folder_path / "SOC_EV.nc")
result_P_BUY = xr.open_dataarray(folder_path / "P_BUY.nc")
result_P_EV_NOT_HOME = xr.open_dataarray(folder_path / "P_EV_NOT_HOME.nc")
#result_SOC_MISSING = xr.open_dataarray(folder_path / "SOC_MISSING.nc")

# reconvert seconds to datetime
epoch_time = datetime(1970, 1, 1)
dti = pd.DatetimeIndex(epoch_time + pd.to_timedelta(result_SOC_EV["t"], unit='s')).tz_localize("UTC").tz_convert("Europe/Berlin")

result_SOC_EV["t"] = dti
result_P_BUY["t"] = dti
result_P_EV_NOT_HOME["t"] = dti
#result_SOC_MISSING["t"] = dti


# ===== plotting ====


if (False):  # BAR PLOT MEAN COST SAVINGS PER DSO
      
    scenarios = result_C_OP_ALL_eur["r"].to_pandas().to_list()
    dso_means = {'regular network charges': result_C_OP_ALL_eur.sel(s='reg').mean(dim=["v"]),
                    'reduced network charges': result_C_OP_ALL_eur.sel(s='red').mean(dim=["v"]) }
    
    x = np.arange(len(result_C_OP_ALL_eur.mean(dim="v")))  # the label locations
    width = 0.25  # the width of the bars
    colors_plot = ["#D04119", "#004c93"]
    hatch_plot = ["","//"]
    ct = 0
    
    fig, ax = plt.subplots(layout='constrained')
    fig.set_figheight(5)
    fig.set_figwidth(15)

    for attribute, measurement in dso_means.items():
        offset = width * ct
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors_plot[ct], hatch=hatch_plot[ct])
        #ax.bar_label(rects, padding=3)
        ct += 1

    ax.set_ylabel('Procurement and Network Cost in Euro')
    str_v = str(len(result_C_OP_ALL_eur["v"]))
    ax.set_title('Charging at home, averaging over ' + str_v + ' different mobility use cases')
    ax.set_xticks(x + width)
    ax.set_xticklabels(scenarios, rotation=90)
    ax.legend(loc='lower center', ncols=2)
    ax.grid(color='lightgray', linestyle='--', linewidth=1, axis="y")
    ax.set_axisbelow(True)

    fig.savefig(folder_path / "dso_cost_barlot.svg")



if (False): # LINKED BUY AND PRICE TIMESERIS
    axx1 = plt.subplot(2,1,1)
    result_P_BUY.sel(r='Westnetz').isel(v=0).to_pandas().plot(ax=axx1)
    
    axx2 = plt.subplot(2,1,2, sharex=axx1)
    (network_charges_xr + spot_prices_xr).sel(r="Westnetz").to_pandas().plot(ax=axx2)


if (False): # HEATMAP SAVINGS
    
    savings = (result_C_OP_ALL_eur.sel(s="reg") - result_C_OP_ALL_eur.sel(s="red")).to_pandas()
    savings["row_sum"] = savings.sum(axis=1) 
    savings_sorted = savings.sort_values("row_sum", ascending=False)
    savings_sorted = savings_sorted.drop(columns=["row_sum"]).transpose()
    savings_sorted_twice = savings_sorted.sort_values(savings_sorted.columns[0], axis="index", ascending=False)

    # get specific colorbar
    max_val = savings_sorted.max().max()    
    min_val = savings_sorted.min().min()    
    zero_fraction = np.abs(min_val) / (np.abs(min_val) + np.abs(max_val))

    #mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', [(0,"red"),(zero_fraction, "white"), (1, "blue")], 256)
    mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', [(0,"red"),(0.33, "white"), (1, "blue")], 256)



    ude_colors = ['#efe4bf','#004c93']
    cmap_ude = mcolors.LinearSegmentedColormap.from_list('ude', ude_colors)

    fig_hm, ax_hm = plt.subplots(figsize=(15, 6)) 
    heatmap = ax_hm.imshow(savings_sorted_twice.to_numpy(), cmap=mycmap, interpolation='nearest', aspect=2, vmin=-50, vmax=100)
    plt.xlabel('DSOs (columns sorted by column sum)', fontsize=20)
    plt.ylabel('Mobility patterns\n(rows sorted by \n values in first column)', fontsize=20)
    plt.title("Savings when switching from regular to reduced network charges in €", fontsize=20)
    ax_hm.tick_params(axis='both', which='major', labelsize=20)
    cbar = fig_hm.colorbar(heatmap, ax=ax_hm,  orientation="vertical")
    cbar.ax.tick_params(size=20, labelsize=20)
    
    fig_hm.savefig(folder_path / "heatmap_savings.svg")



    
    
if (False): # PEAK REDUCTION HEATMAP
    
    time_index = result_P_BUY["t"].to_pandas().index.hour
    time_index_evening = ((time_index >= 17) & (time_index <= 21))

    peak_reduction = (result_P_BUY.sel(s="reg", t=time_index_evening).max("t")-result_P_BUY.sel(s="red", t=time_index_evening).max("t")).to_pandas()
    peak_reduction["row_sum"] = peak_reduction.sum(axis=1) 
    peak_reduction_sorted = peak_reduction.sort_values("row_sum", ascending=False)
    peak_reduction_sorted = peak_reduction_sorted.drop(columns=["row_sum"]).transpose()
    peak_reduction_sorted_twice = peak_reduction_sorted.sort_values(peak_reduction_sorted.columns[0], axis="index", ascending=False)
    
    ude_colors = ['#8b2d0d', 'white', '#004c93'] # rot weiß blau   #sand  #efe4bf'
    cmap_ude = mcolors.LinearSegmentedColormap.from_list('ude', ude_colors)

    fig_kw_reduction, ax_hm = plt.subplots(figsize=(15, 6)) 
    heatmap_power = ax_hm.imshow(peak_reduction_sorted_twice.to_numpy(), cmap=cmap_ude, vmin=-11, vmax=11, interpolation='nearest', aspect=2)
    plt.xlabel('DSOs (columns sorted by column sum)', fontsize=20)
    plt.ylabel('Mobility patterns\n(rows sorted by values \n in first column)', fontsize=20)
    plt.title("Peak reduction in kW when switching from regular to reduced network charges in kW", fontsize=20)
    cbar = fig_kw_reduction.colorbar(heatmap_power, ax=ax_hm,  orientation="vertical")
    cbar.ax.tick_params(size=20, labelsize=20)

    plt.tight_layout()

    fig_kw_reduction.savefig(folder_path / "peak_reduction_savings.svg")
    
    
    
if (False): # Total Cost for scheduled and smart charging // Annual cost savings end-consumer
    
    # pfade scheduled
    # network charge _ elec
    folder_str = r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results" + r"\\"
    
    # scheduled charge EV
    immediate_spot_only_charge = folder_str + r"2025-07-10_14-54_all_spot_immediate_charging_only_EV_r50_v10" + r"\\"
    scheduled_spot_only_charge = folder_str + r"2025-07-10_11-32_all_spot_scheduled_charging_only_EV_r50_v10" + r"\\"
    smart_spot_only_charge = folder_str + r"2025-07-10_11-37_all_spot_smart_charging_only_EV_r50_v10" + r"\\"
    immediate_mean_only_charge = folder_str + r"2025-07-09_22-11_all_mean_immediate_charging_only_EV_r50_v10" + r"\\"
    scheduled_mean_only_charge = folder_str + r"2025-07-10_00-59_all_mean_scheduled_charging_only_EV_r50_v10" + r"\\"
    smart_mean_only_charge = folder_str + r"2025-07-10_01-00_all_mean_smart_charging_only_EV_r50_v10" + r"\\"

    # data preparation
    dso_x_ev = 500
    immediate_spot_static = xr.open_dataarray(immediate_spot_only_charge + "C_OP_ALL.nc").sel(s="reg").to_numpy().reshape(dso_x_ev)
    immediate_spot_ToU = xr.open_dataarray(immediate_spot_only_charge + "C_OP_ALL.nc").sel(s="red").to_numpy().reshape(dso_x_ev)
    scheduled_spot_static = xr.open_dataarray(scheduled_spot_only_charge + "C_OP_ALL.nc").sel(s="reg").to_numpy().reshape(dso_x_ev)
    scheduled_spot_ToU = xr.open_dataarray(scheduled_spot_only_charge + "C_OP_ALL.nc").sel(s="red").to_numpy().reshape(dso_x_ev)
    smart_spot_static = xr.open_dataarray(smart_spot_only_charge + "C_OP_ALL.nc").sel(s="reg").to_numpy().reshape(dso_x_ev)
    smart_spot_ToU = xr.open_dataarray(smart_spot_only_charge + "C_OP_ALL.nc").sel(s="red").to_numpy().reshape(dso_x_ev)

    immediate_mean_static = xr.open_dataarray(immediate_mean_only_charge + "C_OP_ALL.nc").sel(s="reg").to_numpy().reshape(dso_x_ev)
    immediate_mean_ToU = xr.open_dataarray(immediate_mean_only_charge + "C_OP_ALL.nc").sel(s="red").to_numpy().reshape(dso_x_ev)
    scheduled_mean_static = xr.open_dataarray(scheduled_mean_only_charge + "C_OP_ALL.nc").sel(s="reg").to_numpy().reshape(dso_x_ev)
    scheduled_mean_ToU = xr.open_dataarray(scheduled_mean_only_charge + "C_OP_ALL.nc").sel(s="red").to_numpy().reshape(dso_x_ev)
    smart_mean_static = xr.open_dataarray(smart_mean_only_charge + "C_OP_ALL.nc").sel(s="reg").to_numpy().reshape(dso_x_ev)
    smart_mean_ToU = xr.open_dataarray(smart_mean_only_charge + "C_OP_ALL.nc").sel(s="red").to_numpy().reshape(dso_x_ev)

    pd_ToU_static_spot = pd.DataFrame({'immediate':immediate_spot_static, 'scheduled':scheduled_spot_static, 'smart':smart_spot_static})
    pd_ToU_dynamic_spot = pd.DataFrame({'immediate': immediate_spot_ToU, 'scheduled':scheduled_spot_ToU, 'smart':smart_spot_ToU})

    pd_ToU_static_mean = pd.DataFrame({'immediate':immediate_mean_static, 'scheduled':scheduled_mean_static, 'smart':smart_mean_static})
    pd_ToU_dynamic_mean = pd.DataFrame({'immediate': immediate_mean_ToU, 'scheduled':scheduled_mean_ToU, 'smart':smart_mean_ToU})

    x = np.linspace(0, 2 * np.pi, 400)
    y = np.sin(x ** 2)

    fig_grouped_boxplots_cost_savings, axs = plt.subplots(2, 2, figsize=(15, 8))
    #fig_grouped_boxplots_cost_savings.suptitle("Scenario: EV only")
    
    # https://matplotlib.org/stable/gallery/statistics/boxplot.html
    meanpointprops = dict(marker='x', markeredgecolor='black', markerfacecolor='black') #firebrick
    flierprops = dict(marker='o', markerfacecolor=(0,0,0,0.1), markersize=6, markeredgecolor=(0,0,0,1))
    
    pd_ToU_static_mean.plot(ax = axs[0, 0],  kind="box", widths=0.7, patch_artist=True, notch=True, showmeans=True, meanprops=meanpointprops,  flierprops=flierprops, color=dict(boxes='black', whiskers='black', medians='black', caps='black'), boxprops=dict(facecolor="lightgray"))
    pd_ToU_dynamic_mean.plot(ax = axs[0, 1], kind="box", widths=0.7, patch_artist=True, notch=True, showmeans=True, meanprops=meanpointprops,  flierprops=flierprops, color=dict(boxes='black', whiskers='black', medians='black', caps='black'), boxprops=dict(facecolor="lightgray"))
    pd_ToU_static_spot.plot(ax = axs[1, 0],  kind="box", widths=0.7, patch_artist=True, notch=True, showmeans=True, meanprops=meanpointprops,  flierprops=flierprops, color=dict(boxes='black', whiskers='black', medians='black', caps='black'), boxprops=dict(facecolor="lightgray"))
    pd_ToU_dynamic_spot.plot(ax = axs[1, 1], kind="box", widths=0.7, patch_artist=True, notch=True, showmeans=True, meanprops=meanpointprops,  flierprops=flierprops, color=dict(boxes='black', whiskers='black', medians='black', caps='black'), boxprops=dict(facecolor="lightgray"))
    
    axs[1, 0].set_xlabel("Static", fontsize=20)
    axs[1, 1].set_xlabel("Dynamic", fontsize=20)
    axs[0, 0].set_ylabel("Standard", fontsize=20)
    axs[1, 0].set_ylabel("Time of Use", fontsize=20)
    
    axs[1, 0].tick_params(axis='both', labelsize=20)
    axs[1, 1].tick_params(axis='both', labelsize=20)
    axs[0, 0].tick_params(axis='both', labelsize=20)
    axs[0, 1].tick_params(axis='both', labelsize=20)

    axs[0, 0].set_ylim(0,1000)
    axs[0, 1].set_ylim(0,1000)
    axs[1, 0].set_ylim(0,1000)
    axs[1, 1].set_ylim(0,1000)

    ytickvals = np.array(range(0,11))*100

    axs[0,0].set_yticks(ytickvals)
    axs[0,0].set_yticklabels([str(y) if y%200==0 else " " for y in ytickvals], fontsize=20)
    axs[0,1].set_yticks(ytickvals)
    axs[0,1].set_yticklabels([str(y) if y%200==0 else " " for y in ytickvals], fontsize=20)
    axs[1,0].set_yticks(ytickvals)
    axs[1,0].set_yticklabels([str(y) if y%200==0 else " " for y in ytickvals], fontsize=20)
    axs[1,1].set_yticks(ytickvals)
    axs[1,1].set_yticklabels([str(y) if y%200==0 else " " for y in ytickvals], fontsize=20)

    
    fig_grouped_boxplots_cost_savings.supxlabel("Electricity price", fontsize=20)
    fig_grouped_boxplots_cost_savings.supylabel("Network charge", fontsize=20)

    for ax in axs.flat:
        ax.xaxis.grid(False)
        ax.yaxis.grid(True, linestyle="--", color="lightgray")
        
    #fig_grouped_boxplots_cost_savings.savefig(folder_path / "grouped_boxplot_cost.svg")
    fig_grouped_boxplots_cost_savings.savefig(r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results\annual_cost_savings_end_consumer.svg")

    
    
if (False): # PRICE COMPARISON

    # Dynamic price minus Average price of rolling period (15 pm to 15pm next day) 
    
    dso_col = "EWE NETZ"
    
        
    ude_colors_inv = ['#004c93', 'white', '#8b2d0d'] # blau weiß rot   #sand  #efe4bf'
    cmap_ude_inv = mcolors.LinearSegmentedColormap.from_list('ude_inv', ude_colors_inv)
    
    time_index = spot_prices_xr["t"].to_pandas().index.hour
    #time_index_evening = ((time_index >= 17) & (time_index <= 21))

    signal = pd.DataFrame([spot_prices_xr.to_pandas()]).transpose().rename(columns={0:"value"})
    signal["Date"] = signal.index.strftime("%Y-%m-%d")
    signal["Time"] = signal.index.strftime("%H_%M")

    signal_spot_pivot = pd.pivot_table(signal, index=signal.Date, columns=signal.Time, values="value")
    col_names  = [a + "+1" for a in signal_pivot.columns[0:15*4]]

    signal_spot_15_15 = pd.merge(signal_pivot, pd.DataFrame(signal_spot_pivot.iloc[:,0:15*4].shift(-1).to_numpy(), columns=col_names, index=signal_spot_pivot.index), left_index=True, right_index=True)
    signal_spot_15_15 = signal_spot_15_15.iloc[:,15*4:]

    daily_mean_spot_price = signal_spot_15_15.mean(axis=1)
    rel_signal_spot_15_15 = signal_spot_15_15.sub(daily_mean_spot_price, axis=0)
    
    # networrk charges
    network_charge_reduction = (network_charges_xr.sel(s="red")-network_charges_xr.sel(s="reg")).to_pandas()
    network_charge_reduction["Date"] = signal.index.strftime("%Y-%m-%d")
    network_charge_reduction["Time"] = signal.index.strftime("%H_%M")
    
    signal_nc_pivot = pd.pivot_table(network_charge_reduction, index=network_charge_reduction.Date, columns=network_charge_reduction.Time, values=dso_col)
    signal_nc_15_15 = pd.merge(signal_nc_pivot, pd.DataFrame(signal_nc_pivot.iloc[:,0:15*4].shift(-1).to_numpy(), columns=col_names, index=signal_nc_pivot.index), left_index=True, right_index=True)
    signal_nc_15_15 = signal_nc_15_15.iloc[:,15*4:]
   
    fig_signal, axes_signal = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

    heatmap_spot = axes_signal[0].imshow(rel_signal_15_15, cmap=cmap_ude_inv, aspect=2, vmin=-15, vmax=15)
    axes_signal[0].set_title("Spot price minus mean daily spot price")
    plt.xlabel('Hour')
    plt.ylabel('Day')
    axes_signal[0].set_xticks(range(0,96,4), rel_signal_15_15.columns[range(0,96,4)], rotation=90)
    axes_signal[0].set_yticks(range(0,len(rel_signal_15_15)), rel_signal_15_15.index, rotation=0)
    fig_kw_reduction.colorbar(heatmap_spot, ax=axes_signal[0],  orientation="vertical", extend="both")

    heatmap_nc = axes_signal[1].imshow(signal_nc_15_15, cmap=cmap_ude_inv, aspect=2, vmin=-15, vmax=15)
    axes_signal[1].set_title("+ ToU network charge minus regular network charge")
    plt.xlabel('Hour')
    plt.ylabel('Day')
    axes_signal[1].set_xticks(range(0,96,4), signal_nc_15_15.columns[range(0,96,4)], rotation=90)
    axes_signal[1].set_yticks(range(0,len(signal_nc_15_15)), signal_nc_15_15.index, rotation=0)
    fig_kw_reduction.colorbar(heatmap_nc, ax=axes_signal[1],  orientation="vertical", extend="both")


    heatmap_signal = axes_signal[2].imshow(signal_nc_15_15+rel_signal_15_15, cmap=cmap_ude_inv, aspect=2, vmin=-15, vmax=15)
    axes_signal[2].set_title("= Network charge and spot signal added")
    plt.xlabel('Hour')
    plt.ylabel('Day')
    axes_signal[2].set_xticks(range(0,96,4), signal_nc_15_15.columns[range(0,96,4)], rotation=90)
    axes_signal[2].set_yticks(range(0,len(signal_nc_15_15)), signal_nc_15_15.index, rotation=0)
    fig_kw_reduction.colorbar(heatmap_signal, ax=axes_signal[2],  orientation="vertical", extend="both")



    
# =============================================================================
# if (False): # CHARGE POWER 
# 
# 
#     def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot",  H=".", **kwargs):
#         """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
#     labels is a list of the names of the dataframe, used for the legend
#     title is a string for the title of the plot
#     H is the hatch used for identification of the different dataframe"""
#         import matplotlib as mpl
#         from cycler import cycler
#         mpl.rcParams["axes.prop_cycle"] = cycler('color', ["#efe4bf", "#004c93"])
#     
#         n_df = len(dfall)
#         n_col = len(dfall[0].columns) 
#         n_ind = len(dfall[0].index)
#         axe = plt.subplot(111)
#         plt.subplots_adjust(bottom=0.476,left=0.057,right=0.917,top=0.9)
#         
#         for df in dfall : # for each data frame
#             axe = df.plot(kind="bar",
#                           edgecolor="black",
#                           linewidth=1,
#                           stacked=True,
#                           ax=axe,
#                           legend=False,
#                           grid=False,
#                           **kwargs)  # make bar plots
#     
#         h,l = axe.get_legend_handles_labels() # get the handles we want to modify
#         for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
#             for j, pa in enumerate(h[i:i+n_col]):
#                 for rect in pa.patches: # for each index
#                     rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
#                     #rect.set_color(colors_plot[j,int(np.floor(i/n_df))])
#                     #print(int(i / n_col))
#                     rect.set_hatch(H * int(i / n_col)) #edited part     
#                     rect.set_width(1 / float(n_df + 1))
#                     #print(int(np.floor(i/n_df)),j)
#     
#         axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
#         axe.set_xticklabels(df.index, rotation = 90)
#         axe.set_title(title)
#     
#         # Add invisible data to add another legend
#         n=[]        
#         for i in range(n_df):
#             n.append(axe.bar(0, 0, color="#eeeeee", hatch=H * i))
#     
#         l1 = axe.legend(h[:n_col], l[:n_col], loc="upper left", ncol=2) #[1.01, 0.5]
#         if labels is not None:
#             l2 = plt.legend(n, labels,  loc="upper right", ncol=2) # loc=[1.01, 0.1],
#         axe.add_artist(l1)
#         
#         # background grid lines
#         axe.grid(color='lightgray', linestyle='--', linewidth=1, axis="y")
#         axe.set_axisbelow(True)
#         axe.set_ylim([0, 1.3*max(df_reg.max().max(), df_red.max().max())])
#         #axe.subplots_adjust(bottom=0.2)
#         return axe
# 
#     
#     df_reg = pd.DataFrame( {"P_BUY":result_P_BUY.sel(s="reg").sum("t").mean("v"), "P_EV_NOT_HOME":result_P_EV_NOT_HOME.sel(s="reg").sum("t").mean("v")}, index=result_P_BUY["r"] )
#     df_red = pd.DataFrame( {"P_BUY":result_P_BUY.sel(s="red").sum("t").mean("v"), "P_EV_NOT_HOME":result_P_EV_NOT_HOME.sel(s="red").sum("t").mean("v")}, index=result_P_BUY["r"] )
# 
#     plot_clustered_stacked([df_reg, df_red],["regular network charges", "reduced network charges"], title="Energy consumed in kWh")
# 
#     fig.savefig(folder_path / "dso_energy_barlot.svg")
# =============================================================================
    
    


folder_str = r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results" + r"\\"
    
# files: immediate, scheduled, smart
spot_only_charge_list = [folder_str + x for x in [r"2025-07-10_14-54_all_spot_immediate_charging_only_EV_r50_v10" + r"\\",
                                            r"2025-07-10_11-32_all_spot_scheduled_charging_only_EV_r50_v10" + r"\\",
                                            r"2025-07-10_11-37_all_spot_smart_charging_only_EV_r50_v10" + r"\\"  ]  ]
                                 
mean_only_charge_list = [folder_str + x for x in [r"2025-07-09_22-11_all_mean_immediate_charging_only_EV_r50_v10" + r"\\",
                                                          r"2025-07-10_00-59_all_mean_scheduled_charging_only_EV_r50_v10" + r"\\",
                                                          r"2025-07-10_01-00_all_mean_smart_charging_only_EV_r50_v10" + r"\\"  ]  ]

charge_mode = ["immediate", "scheduled", "smart"]

fig_kW_reduction, axs_kW_reduction = plt.subplots(ncols=3, figsize=(15, 4))  # 3 subplots horizontally  
y_min, y_max = -0.4, 0.4


pd_ct = pd.DataFrame()
                  
for ct in range(0, len(mean_only_charge_list)):
    spot_only_charge = spot_only_charge_list[ct]
    mean_only_charge = mean_only_charge_list[ct]

    # data preparation
    mean_static = xr.open_dataarray(mean_only_charge + "P_BUY.nc").sel(s="reg").mean(["v","r"]).to_pandas()
    mean_ToU = xr.open_dataarray(mean_only_charge + "P_BUY.nc").sel(s="red").mean(["v","r"]).to_pandas()
    spot_static = xr.open_dataarray(spot_only_charge + "P_BUY.nc").sel(s="reg").mean(["v","r"]).to_pandas()
    spot_ToU = xr.open_dataarray(spot_only_charge + "P_BUY.nc").sel(s="red").mean(["v","r"]).to_pandas()
 
    pd_ct[charge_mode[ct] + "_mean_static_standard"] = mean_static
    pd_ct[charge_mode[ct] + "_mean_ToU_standard"] = mean_ToU
    pd_ct[charge_mode[ct] + "_spot_static_standard"] = spot_static
    pd_ct[charge_mode[ct] + "_spot_ToU_standard"] = spot_ToU
    
dti = pd.DatetimeIndex(epoch_time + pd.to_timedelta(xr.open_dataarray(spot_only_charge + "P_BUY.nc")["t"], unit='s')).tz_localize("UTC").tz_convert("Europe/Berlin")
pd_ct = pd_ct.set_index(dti)
pd_ct["hour decimal"] = pd_ct.index.hour + pd_ct.index.minute/60

pd_day = pd_ct.groupby(["hour decimal"]).mean()






# kW reduction plots
if (False):

    fig_kw_savings, axs_kw_savings = plt.subplots(ncols=2, figsize=(15, 6))   

    # linker plot
    pd_charge_mode = pd.concat([ pd_day["immediate_mean_static_standard"] - pd_day["immediate_mean_static_standard"],
                                    pd_day["scheduled_mean_static_standard"] - pd_day["immediate_mean_static_standard"],
                                    pd_day["smart_mean_static_standard"] - pd_day["immediate_mean_static_standard"],
                                    pd_day["immediate_spot_static_standard"] - pd_day["immediate_spot_static_standard"],
                                    pd_day["scheduled_spot_static_standard"] - pd_day["immediate_spot_static_standard"],
                                    pd_day["smart_spot_static_standard"] - pd_day["immediate_spot_static_standard"] ],
                                    
                                    axis = 1).rename(columns={"immediate_mean_static_standard":"reference static immediate", 0:"switch to scheduled (static)", 1:"switch to smart (static)", 
                                                              "immediate_spot_static_standard":"reference:\ndynamic immediate", 2:"switch to scheduled (dynamic)", 3:"switch to smart (dynamic)" })



    axs_kw_savings[0].plot(pd_charge_mode["reference static immediate"], linestyle="--", color="k", linewidth=1, zorder=2, label="reference: immediate \n(static, dynamic)")
                                                              
    #axs_kw_savings[0].plot(pd_charge_mode["reference static immediate"], linestyle=":", color="#8b3003", linewidth=1, zorder=2, label="reference: \nstatic immediate")
    axs_kw_savings[0].plot(pd_charge_mode["switch to scheduled (static)"], linestyle="-", color="k", linewidth=1, zorder=2, label="switch to scheduled \n(static, dynamic)")
    axs_kw_savings[0].plot(pd_charge_mode["switch to smart (static)"], linestyle="-", color="#c13f1a", linewidth=1, zorder=1, label="switch to smart (static)")

    #axs_kw_savings[0].plot(pd_charge_mode["switch to scheduled (dynamic)"], linestyle="--", color="#00386c", linewidth=1, zorder=1, label="switch to scheduled (dynamic)")
    axs_kw_savings[0].plot(pd_charge_mode["switch to smart (dynamic)"], linestyle="-", color="#0087ff", linewidth=1, zorder=1, label="switch to smart (dynamic)")

    axs_kw_savings[0].legend(fontsize=16, loc="upper right")
    axs_kw_savings[0].grid(color='lightgray', linestyle='--', linewidth=1, axis="both")
    axs_kw_savings[0].set_xticks(np.array([0, 3, 6, 9, 12, 15, 18, 21, 24]))
    axs_kw_savings[0].set_xticklabels([0, 3, 6, 9, 12, 15, 18, 21, 24], fontsize=20)
    axs_kw_savings[0].set_ylabel("Mean Power reduction \n through charge strategy in kW", fontsize=20)
    axs_kw_savings[0].tick_params(axis='y', labelsize=20)
    axs_kw_savings[0].set_xlabel("Time in hours", fontsize=20)



    # rechter plot
    pd_smart = pd.concat([ pd_day["smart_mean_static_standard"] - pd_day["smart_mean_static_standard"],
                                    pd_day["smart_mean_ToU_standard"] - pd_day["smart_mean_static_standard"],
                                    pd_day["smart_spot_static_standard"] - pd_day["smart_mean_static_standard"],
                                    pd_day["smart_spot_ToU_standard"] - pd_day["smart_mean_static_standard"] ],
                                    axis = 1).rename(columns={"smart_mean_static_standard":"reference: static standard", 0:"static ToU", 1:"dynamic standard", 2:"dynamic ToU"})
    
    linestyle_list = ['--', '-', '--', '-']  # same length as columns
    color_list = ["#8b3003", "#c13f1a", "#00386c", "#0087ff"]
    
    #axs_kw_savings[1].fill_between(np.array(range(0,96))/4, pd_smart["static ToU"], linestyle=linestyle_list[1], color=color_list[1], alpha=0.5, zorder=0)
    axs_kw_savings[1].plot(pd_smart["reference: static standard"], linestyle=linestyle_list[0], color=color_list[0], alpha=1, linewidth=1,  zorder=2, label="reference: smart \nstatic standard")
    axs_kw_savings[1].plot(pd_smart["static ToU"], linestyle=linestyle_list[1], color=color_list[1], alpha=1, zorder=0, linewidth=1, label="static ToU")

    #axs_kw_savings[1].fill_between(np.array(range(0,96))/4, pd_smart["dynamic standard"], pd_smart["dynamic ToU"], linestyle=linestyle_list[2], color=color_list[2], alpha=0.5, zorder=0)
    axs_kw_savings[1].plot(pd_smart["dynamic standard"], linestyle=linestyle_list[2], color=color_list[2], alpha=1, zorder=1, linewidth=1, label="dynamic standard")
    axs_kw_savings[1].plot(pd_smart["dynamic ToU"], linestyle=linestyle_list[3], color=color_list[3], alpha=1, linewidth=1,  zorder=3, label="dynamic ToU")

    axs_kw_savings[1].legend(fontsize=16)
        
    axs_kw_savings[1].set_ylim(y_min, y_max)
    axs_kw_savings[1].set_xlabel("Time in hours", fontsize=20)
    axs_kw_savings[1].set_ylabel("Mean Power reduction \n for smart charging tariffs in kW", fontsize=20)
    axs_kw_savings[1].set_ylim(-0.2, 0.6)
    axs_kw_savings[1].tick_params(axis='y', labelsize=20)
    axs_kw_savings[1].set_xlim(0, 24)
    axs_kw_savings[1].set_xticks(np.array([0, 3, 6, 9, 12, 15, 18, 21, 24]))
    axs_kw_savings[1].set_xticklabels([0, 3, 6, 9, 12, 15, 18, 21, 24], fontsize=20)
    axs_kw_savings[1].grid(color='lightgray', linestyle='--', linewidth=1, axis="both")

    plt.tight_layout()
    plt.show()

    fig_kw_savings.savefig(r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results\fig_kw_savings.svg")









if (True): # EV SOC
    pd_res = result_SOC_EV.isel(v=1, r=1).to_pandas()
    plt.figure()
    plt.plot(pd_res)
    plt.legend(pd_res.columns)
    plt.ylabel("SOC EV in kWh")
    plt.show()


if (True) and parameters_model["settings_setup"] == "prosumage": # P_PV
    plt.figure()
    plt.plot(result_P_PV)
    plt.legend(result_P_PV.columns)
    plt.ylabel("P PV in kW")
    plt.show()

    
if (True) and parameters_model["settings_setup"] == "prosumage": # BESS SOC
    plt.figure()
    plt.plot(result_SOC_BESS.isel(v=1,r=1))
    plt.legend(result_SOC_BESS.columns)
    plt.ylabel("SOC BESS in kWh")
    plt.show()


if (False): # P_BUY
    plt.figure()
    plt.plot(result_P_BUY)
    plt.legend(result_P_BUY.columns)
    plt.show()


if (False):
    # quantile plot of EV over all regions
    fig1, ax1 = plt.subplots()
    for quantile in [0.0, 0.1, 0.2, 0.3, 0.4]:
        ax1.fill_between(result_SOC_EV.isel(v=1,s=1).to_pandas().index,
                        result_SOC_EV.isel(v=1,s=1).to_pandas().reset_index().set_index("t").quantile(axis=1, q=quantile),
                        result_SOC_EV.isel(v=1,s=1).to_pandas().reset_index().set_index("t").quantile(axis=1, q=1-quantile),
                        color='b', alpha= quantile + 0.3, edgecolor=None)
    ax1.plot(result_SOC_EV.isel(v=1,s=1).to_pandas().reset_index().set_index("t").median(axis=1), color="k", linewidth=1)
    plt.show()



    buy_scatter = m["P_BUY"].solution.isel(r=1, v=1).to_pandas().rename(columns={"red":"red_buy", "reg":"reg_buy"}).reset_index()
    cost_pd = cost_xr.isel(r=1).to_pandas().reset_index().rename(columns={"red":"red_spot", "reg":"reg_spot"}).reset_index()
    
    merge_result = buy_scatter.merge(cost_pd)
    merge_result["time_of_day"] = merge_result.t.dt.hour + merge_result.t.dt.minute/60
    merge_result = merge_result[merge_result["reg_buy"] > 0 ]
    merge_result["marker_size"] = 25 * merge_result["reg_buy"]
    

    fig, ax = plt.subplots(1, 1)
    merge_result.plot.scatter(x='time_of_day',y='reg_spot', s="marker_size", alpha=0.3, xlabel="Time of the day", ylabel="charge price", ax=ax, c="blue", legend="regular")
    merge_result.plot.scatter(x='time_of_day',y='red_spot', s="marker_size", alpha=0.3, xlabel="Time of the day", ylabel="charge price", ax=ax, c="orange", legend="reduced")
    ax.legend(["regular", "reduced"])





# ====== DEDUCE HEAT PUMP DEMAND FROM TEMPERATURE ======


#alpha = 0.0025  # W/(m^2*K)
#surface = 200 #m^2
#limit_temp = 15 # °C

#heat_demand = alpha * surface * np.maximum(limit_temp-temperature_cut, 0) 
#heat_demand_xr = xr.DataArray(heat_demand, dims='t')


#if (False):
#    plt.plot(heat_demand)



# ====== Heat pump parameters =====
#e_max = 20  # kWh
#p_hp = 4 # kW
#cop = 3 # [-]
#timesteplength = 1 # h

# ===== optimization model =====


#m_perf_foresight_det = model_perf_forsight.build_hp_model(prices, dsos, prices_xr, e_max, p_hp, cop, heat_demand_xr, penalty, timesteplength)

#e_init_percent = 0.6
#e_min_end_percent = 0.8




# plots
#if (False):
#    result_P_HP.iloc[1:671,:].plot(style=["-","--"], color=["r","k"], ylabel="power in kW", xlabel="time")


#if (False):
#    result_E_HStor.iloc[1:671,:].plot(style=["-","--"], color=["r","k"], ylabel="energy in kWh", xlabel="time")


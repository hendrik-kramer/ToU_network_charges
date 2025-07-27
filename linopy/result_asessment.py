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
    
    
# =============================================================================
# Total Cost for scheduled and smart charging // Annual cost for end-consumer
# =============================================================================
   
    
if (False): 
    
    # pfade scheduled
    # network charge _ elec
    folder_str = r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results" + r"\\"
    
    # scheduled charge EV
    immediate_mean_only_charge = folder_str + r"2025-07-25_01-09_all_mean_partfill_immediate_charging_only_EV_r30_v20" + r"\\"
    immediate_spot_only_charge = folder_str + r"2025-07-25_00-51_all_spot_partfill_immediate_charging_only_EV_r30_v20" + r"\\"
    scheduled_mean_only_charge = folder_str + r"2025-07-24_14-22_all_mean_partfill_scheduled_charging_only_EV_r30_v20" + r"\\"
    scheduled_spot_only_charge = folder_str + r"2025-07-25_10-43_all_spot_partfill_scheduled_charging_only_EV_r30_v20" + r"\\" 
    smart_mean_only_charge = folder_str +     r"2025-07-25_10-21_all_mean_smart_charging_only_EV_r30_v20" + r"\\"
    smart_spot_only_charge = folder_str +     r"2025-07-26_11-38_all_spot_smart_charging_only_EV_r30_v20" + r"\\"

   


    # Test section
    if (False):
        p_immediate_mean_standard = xr.open_dataarray(immediate_mean_only_charge + "P_BUY.nc").sel(s="reg").mean(["r","v"]).sum("t").to_pandas()
        p_immediate_mean_ToU = xr.open_dataarray(immediate_mean_only_charge + "P_BUY.nc").sel(s="red").mean(["r","v"]).sum("t").to_pandas()
        p_scheduled_mean_standard = xr.open_dataarray(scheduled_mean_only_charge + "P_BUY.nc").sel(s="reg").mean(["r","v"]).sum("t").to_pandas()
        p_scheduled_mean_ToU = xr.open_dataarray(scheduled_mean_only_charge + "P_BUY.nc").sel(s="red").mean(["r","v"]).sum("t").to_pandas()
        p_smart_mean_standard = xr.open_dataarray(smart_mean_only_charge + "P_BUY.nc").sel(s="reg").mean(["r","v"]).sum("t").to_pandas()
        p_smart_mean_ToU = xr.open_dataarray(smart_mean_only_charge + "P_BUY.nc").sel(s="red").mean(["r","v"]).sum("t").to_pandas()
   
        print("===== B_BUY ====")
        print("immediate, standard: \t" + str(p_immediate_mean_standard.sum()))
        print("scheduled, standard: \t" + str(p_scheduled_mean_standard.sum()))
        print("smart, standard: \t \t" + str(p_smart_mean_standard.sum()))
        print("immediate, ToU: \t \t" + str(p_immediate_mean_ToU.sum()))
        print("scheduled, ToU: \t \t" + str(p_scheduled_mean_ToU.sum()))
        print("smart, ToU: \t \t \t" + str(p_smart_mean_ToU.sum()))
   
        plt.plot(p_immediate_mean_standard, linewidth=4, label="immediate static standard", linestyle="-")
        plt.plot(p_scheduled_mean_standard, linewidth=3, label="scheduled static standard", linestyle="--")
       
      
        plt.legend()



    # data preparation
    cost_type_file = "C_OP_ALL.nc"  # "C_OP_ALL.nc", "C_OP_HOME.nc" (no price spikes for scheduled charging due to public pole charging)
    result_shape = xr.open_dataarray(immediate_mean_only_charge + cost_type_file).sel(s="reg").shape
    dso_x_ev = result_shape[0] * result_shape[1]
    immediate_spot_standard = xr.open_dataarray(immediate_spot_only_charge + cost_type_file).sel(s="reg").to_numpy().reshape(dso_x_ev)
    immediate_spot_ToU = xr.open_dataarray(immediate_spot_only_charge + cost_type_file).sel(s="red").to_numpy().reshape(dso_x_ev)
    scheduled_spot_standard = xr.open_dataarray(scheduled_spot_only_charge + cost_type_file).sel(s="reg").to_numpy().reshape(dso_x_ev)
    scheduled_spot_ToU = xr.open_dataarray(scheduled_spot_only_charge + cost_type_file).sel(s="red").to_numpy().reshape(dso_x_ev)
    smart_spot_standard = xr.open_dataarray(smart_spot_only_charge + cost_type_file).sel(s="reg").to_numpy().reshape(dso_x_ev)
    smart_spot_ToU = xr.open_dataarray(smart_spot_only_charge + cost_type_file).sel(s="red").to_numpy().reshape(dso_x_ev)

    immediate_mean_standard = xr.open_dataarray(immediate_mean_only_charge + cost_type_file).sel(s="reg").to_numpy().reshape(dso_x_ev)
    immediate_mean_ToU = xr.open_dataarray(immediate_mean_only_charge + cost_type_file).sel(s="red").to_numpy().reshape(dso_x_ev)
    scheduled_mean_standard = xr.open_dataarray(scheduled_mean_only_charge + cost_type_file).sel(s="reg").to_numpy().reshape(dso_x_ev)
    scheduled_mean_ToU = xr.open_dataarray(scheduled_mean_only_charge + cost_type_file).sel(s="red").to_numpy().reshape(dso_x_ev)
    smart_mean_standard = xr.open_dataarray(smart_mean_only_charge + cost_type_file).sel(s="reg").to_numpy().reshape(dso_x_ev)
    smart_mean_ToU = xr.open_dataarray(smart_mean_only_charge + cost_type_file).sel(s="red").to_numpy().reshape(dso_x_ev)

    pd_standard_static = pd.DataFrame({'immediate':immediate_mean_standard, 'scheduled':scheduled_mean_standard, 'smart':smart_mean_standard})
    pd_standard_dynamic = pd.DataFrame({'immediate': immediate_spot_standard, 'scheduled':scheduled_spot_standard, 'smart':smart_spot_standard})

    pd_ToU_static = pd.DataFrame({'immediate':immediate_mean_ToU, 'scheduled':scheduled_mean_ToU, 'smart':smart_mean_ToU})
    pd_ToU_dynamic = pd.DataFrame({'immediate': immediate_spot_ToU, 'scheduled':scheduled_spot_ToU, 'smart':smart_spot_ToU})


    x = np.linspace(0, 2 * np.pi, 400)
    y = np.sin(x ** 2)

    fig_grouped_boxplots_cost_savings, axs = plt.subplots(2, 2, figsize=(15, 8))
    #fig_grouped_boxplots_cost_savings.suptitle("Scenario: EV only")
    
    # https://matplotlib.org/stable/gallery/statistics/boxplot.html
    meanpointprops = dict(marker='x', markeredgecolor='black', markerfacecolor='black') #firebrick
    flierprops = dict(marker='o', markerfacecolor=(0,0,0,0.1), markersize=6, markeredgecolor=(0,0,0,1))
    
    pd_standard_static.plot(ax = axs[0, 0],  kind="box", widths=0.7, patch_artist=True, notch=True, showmeans=True, meanprops=meanpointprops,  flierprops=flierprops, color=dict(boxes='black', whiskers='black', medians='black', caps='black'), boxprops=dict(facecolor="lightgray"))
    pd_standard_dynamic.plot(ax = axs[0, 1], kind="box", widths=0.7, patch_artist=True, notch=True, showmeans=True, meanprops=meanpointprops,  flierprops=flierprops, color=dict(boxes='black', whiskers='black', medians='black', caps='black'), boxprops=dict(facecolor="lightgray"))
    pd_ToU_static.plot(ax = axs[1, 0],  kind="box", widths=0.7, patch_artist=True, notch=True, showmeans=True, meanprops=meanpointprops,  flierprops=flierprops, color=dict(boxes='black', whiskers='black', medians='black', caps='black'), boxprops=dict(facecolor="lightgray"))
    pd_ToU_dynamic.plot(ax = axs[1, 1], kind="box", widths=0.7, patch_artist=True, notch=True, showmeans=True, meanprops=meanpointprops,  flierprops=flierprops, color=dict(boxes='black', whiskers='black', medians='black', caps='black'), boxprops=dict(facecolor="lightgray"))
    
    axs[1, 0].set_xlabel("Static", fontsize=20, fontweight='bold')
    axs[1, 1].set_xlabel("Dynamic", fontsize=20, fontweight='bold')
    axs[0, 0].set_ylabel("Standard", fontsize=20, fontweight='bold')
    axs[1, 0].set_ylabel("Time of Use", fontsize=20, fontweight='bold')
    
    axs[1, 0].tick_params(axis='both', labelsize=20)
    axs[1, 1].tick_params(axis='both', labelsize=20)
    axs[0, 0].tick_params(axis='both', labelsize=20)
    axs[0, 1].tick_params(axis='both', labelsize=20)

    axs[0, 0].set_ylim(0,600)
    axs[0, 1].set_ylim(0,600)
    axs[1, 0].set_ylim(0,600)
    axs[1, 1].set_ylim(0,600)

    ytickvals = np.array(range(0,13))*50

    axs[0,0].set_yticks(ytickvals)
    axs[0,0].set_yticklabels([str(y)+"€" if y%100==0 else " " for y in ytickvals], fontsize=20)
    axs[0,1].set_yticks(ytickvals)
    axs[0,1].set_yticklabels([str(y)+"€"  if y%100==0 else " " for y in ytickvals], fontsize=20)
    axs[1,0].set_yticks(ytickvals)
    axs[1,0].set_yticklabels([str(y)+"€"  if y%100==0 else " " for y in ytickvals], fontsize=20)
    axs[1,1].set_yticks(ytickvals)
    axs[1,1].set_yticklabels([str(y)+"€"  if y%100==0 else " " for y in ytickvals], fontsize=20)

    
    fig_grouped_boxplots_cost_savings.supxlabel("Electricity price", fontsize=20, fontweight='bold')
    fig_grouped_boxplots_cost_savings.supylabel("Network charge", fontsize=20, fontweight='bold')

    for ax in axs.flat:
        ax.xaxis.grid(False)
        ax.yaxis.grid(True, linestyle="--", color="lightgray")
        
    #fig_grouped_boxplots_cost_savings.savefig(folder_path / "grouped_boxplot_cost.svg")
    fig_grouped_boxplots_cost_savings.savefig(r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results\annual_cost_end_consumer.svg")

    
    


    
# =============================================================================
# CHARGE POWER 
# =============================================================================


epoch_time = datetime(1970, 1, 1)

folder_str = r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results" + r"\\"
    
# files: immediate, scheduled, smart

# only ev
spot_only_charge_list = [folder_str + x for x in [r"2025-07-25_00-51_all_spot_partfill_immediate_charging_only_EV_r30_v20" + r"\\",
                                                  r"2025-07-25_10-43_all_spot_partfill_scheduled_charging_only_EV_r30_v20" + r"\\",
                                                  r"2025-07-26_11-38_all_spot_smart_charging_only_EV_r30_v20" + r"\\"  ]  ]
                                 
mean_only_charge_list = [folder_str + x for x in [r"2025-07-25_01-09_all_mean_partfill_immediate_charging_only_EV_r30_v20" + r"\\",
                                                  r"2025-07-24_14-22_all_mean_partfill_scheduled_charging_only_EV_r30_v20" + r"\\",
                                                  r"2025-07-25_10-21_all_mean_smart_charging_only_EV_r30_v20" + r"\\"  ]  ]



charge_mode = ["immediate", "scheduled", "smart"]



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
    
dti = pd.DatetimeIndex(epoch_time + pd.to_timedelta(xr.open_dataarray(mean_only_charge + "P_BUY.nc")["t"], unit='s')).tz_localize("UTC").tz_convert("Europe/Berlin")
pd_ct = pd_ct.set_index(dti)
pd_ct["hour decimal"] = pd_ct.index.hour + pd_ct.index.minute/60

pd_day = pd_ct.groupby(["hour decimal"]).mean()






# kW reduction plots
if (False):

    fig_kw_savings, axs_kw_savings = plt.subplots(ncols=2, figsize=(15, 6))   

    # linker plot
    pd_charge_mode = pd.concat([ pd_day["immediate_mean_static_standard"], # alle immediate (spot, mean, ToU, standard) müssten gleich sein !!!!!
                                    pd_day["scheduled_mean_static_standard"],
                                    pd_day["smart_mean_static_standard"],
                                    pd_day["smart_mean_ToU_standard"],
                                    pd_day["smart_spot_static_standard"],
                                    pd_day["smart_spot_ToU_standard"] ],
                                    
                                    axis = 1).rename(columns={"immediate_mean_static_standard":"immediate",
                                                              "scheduled_mean_static_standard":"scheduled", 
                                                              "smart_mean_static_standard":"smart (static, standard)", 
                                                              "smart_mean_ToU_standard":"smart (static, ToU)",
                                                              "smart_spot_static_standard":"smart (dynamic, standard)",
                                                              "smart_spot_ToU_standard":"smart (dynamic, ToU)" })



    axs_kw_savings[0].plot(pd_charge_mode["immediate"], linestyle="-", color="k", linewidth=1, zorder=2, label="immediate")
    axs_kw_savings[0].plot(pd_charge_mode["smart (static, standard)"], linestyle="-", color="#8b3003", linewidth=1, zorder=1, label="smart (static, standard)")
    axs_kw_savings[0].plot(pd_charge_mode["smart (static, ToU)"], linestyle="--", color="#c13f1a", linewidth=1, zorder=1, label= "smart (static, ToU)")

    axs_kw_savings[0].plot(pd_charge_mode["scheduled"], linestyle="-.", color="gray", linewidth=1, zorder=2, label=  "scheduled")
    axs_kw_savings[0].plot(pd_charge_mode["smart (dynamic, standard)"], linestyle="-", color="#00386c", linewidth=1, zorder=1, label=  "smart (dynamic, standard)")
    axs_kw_savings[0].plot(pd_charge_mode["smart (dynamic, ToU)"], linestyle="--", color="#0087ff", linewidth=1, zorder=1, label= "smart (dynamic, ToU)")



    #axs_kw_savings[0].plot(pd_charge_mode["switch to scheduled (dynamic)"], linestyle="--", color="#00386c", linewidth=1, zorder=1, label="switch to scheduled (dynamic)")
    #axs_kw_savings[0].plot(pd_charge_mode["switch to smart (dynamic)"], linestyle="-", color="#0087ff", linewidth=1, zorder=1, label= r'$\rightarrow$' + " smart (dynamic)")

    axs_kw_savings[0].legend(fontsize=11, ncols=2, loc="upper center", title="Impact of charge strategy", title_fontsize=16)
    axs_kw_savings[0].grid(color='lightgray', linestyle='--', linewidth=1, axis="both")
    axs_kw_savings[0].set_xticks(np.array([0, 3, 6, 9, 12, 15, 18, 21, 24]))
    axs_kw_savings[0].set_xticklabels([0, 3, 6, 9, 12, 15, 18, 21, 24], fontsize=20)
    axs_kw_savings[0].set_ylim(-.1, 0.6)
    axs_kw_savings[0].set_ylabel("Mean Power consumption in kW", fontsize=20)
    axs_kw_savings[0].tick_params(axis='y', labelsize=20)
    axs_kw_savings[0].set_xlabel("Time in hours", fontsize=20)



    # rechter plot
    pd_smart = pd.concat([ pd_day["smart_mean_static_standard"] - pd_day["smart_mean_static_standard"] ,
                                    pd_day["smart_mean_ToU_standard"] - pd_day["smart_mean_static_standard"] ,
                                    pd_day["smart_spot_static_standard"] - pd_day["smart_mean_static_standard"] ,
                                    pd_day["smart_spot_ToU_standard"] - pd_day["smart_mean_static_standard"] ],
                                    axis = 1).rename(columns={"smart_mean_static_standard":"reference: static standard", 0:"static ToU", 1:"dynamic standard", 2:"dynamic ToU"})
    
    linestyle_list = ['-', '--', '-', '--']  # same length as columns
    color_list = ["#8b3003", "#c13f1a", "#00386c", "#0087ff"]
    
    #axs_kw_savings[1].fill_between(np.array(range(0,96))/4, pd_smart["static ToU"], linestyle=linestyle_list[1], color=color_list[1], alpha=0.5, zorder=0)
    axs_kw_savings[1].plot(pd_smart["reference: static standard"], linestyle=linestyle_list[0], color=color_list[0], alpha=1, linewidth=1,  zorder=2, label="reference: \nstatic standard")
    axs_kw_savings[1].plot(pd_smart["static ToU"], linestyle=linestyle_list[1], color=color_list[1], alpha=1, zorder=0, linewidth=1, label="static ToU")

    #axs_kw_savings[1].fill_between(np.array(range(0,96))/4, pd_smart["dynamic standard"], pd_smart["dynamic ToU"], linestyle=linestyle_list[2], color=color_list[2], alpha=0.5, zorder=0)
    axs_kw_savings[1].plot(pd_smart["dynamic standard"], linestyle=linestyle_list[2], color=color_list[2], alpha=1, zorder=1, linewidth=1, label="dynamic standard")
    axs_kw_savings[1].plot(pd_smart["dynamic ToU"], linestyle=linestyle_list[3], color=color_list[3], alpha=1, linewidth=1,  zorder=3, label="dynamic ToU")

    axs_kw_savings[1].legend(fontsize=11, ncols=2, title="Impact of tariff components \nfor smart charging ", title_fontsize=16, loc="upper right")

        
    axs_kw_savings[1].set_ylim(-1, 1)
    axs_kw_savings[1].set_xlabel("Time in hours", fontsize=20)
    axs_kw_savings[1].set_ylabel("Mean Power change in kW", fontsize=20)
    axs_kw_savings[1].set_ylim(-.2, .4)
    axs_kw_savings[1].tick_params(axis='y', labelsize=20)
    axs_kw_savings[1].set_xlim(0, 24)
    axs_kw_savings[1].set_xticks(np.array([0, 3, 6, 9, 12, 15, 18, 21, 24]))
    axs_kw_savings[1].set_xticklabels([0, 3, 6, 9, 12, 15, 18, 21, 24], fontsize=20)
    axs_kw_savings[1].grid(color='lightgray', linestyle='--', linewidth=1, axis="both")

    plt.tight_layout()
    plt.show()

    fig_kw_savings.savefig(r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results\power_consumption.svg")





    
# =============================================================================
# CHARGE POWER CHANGE -> SENSITIVITY ANALYSIS
# =============================================================================


epoch_time = datetime(1970, 1, 1)

folder_str = r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results" + r"\\"
    

# set sensi in parameters opti to true
_, _, _, _, _, _, _, sensi_different = f_load.load_network_charges(parameter_filepath_dsos, timesteps, parameters_opti)
sensi_different_list = list(sensi_different)
sensi_different_list = [sensi_different_list[ct] for ct in list(parameters_opti["dso_subset"])]

# files: immediate, scheduled, smart

# regular
spot_only_charge_list = [folder_str + x for x in [r"2025-07-25_00-51_all_spot_partfill_immediate_charging_only_EV_r30_v20" + r"\\",
                                                  r"2025-07-25_10-43_all_spot_partfill_scheduled_charging_only_EV_r30_v20" + r"\\",
                                                  r"2025-07-26_11-38_all_spot_smart_charging_only_EV_r30_v20" + r"\\"  ]  ]
                                 
mean_only_charge_list = [folder_str + x for x in [r"2025-07-25_01-09_all_mean_partfill_immediate_charging_only_EV_r30_v20" + r"\\",
                                                  r"2025-07-24_14-22_all_mean_partfill_scheduled_charging_only_EV_r30_v20" + r"\\",
                                                  r"2025-07-25_10-21_all_mean_smart_charging_only_EV_r30_v20" + r"\\"  ]  ]

# sensitivity
spot_only_charge_list_sensi = [folder_str + x for x in [r"2025-07-26_22-15_all_spot_partfill_immediate_charging_only_EV_r30_v20_sensi" + r"\\",
                                                  r"2025-07-26_18-48_all_spot_partfill_scheduled_charging_only_EV_r30_v20_sensi" + r"\\",
                                                  r"2025-07-27_02-50_all_spot_smart_charging_only_EV_r30_v20_sensi" + r"\\"  ]  ]
                                 
mean_only_charge_list_sensi = [folder_str + x for x in [r"2025-07-26_23-18_all_mean_partfill_immediate_charging_only_EV_r30_v20_sensi" + r"\\",
                                                  r"2025-07-26_19-50_all_mean_partfill_scheduled_charging_only_EV_r30_v20_sensi" + r"\\",
                                                  r"2025-07-27_03-48_all_mean_smart_charging_only_EV_r30_v20_sensi" + r"\\"  ]  ]




charge_mode = ["immediate", "scheduled", "smart"]





pd_ct = pd.DataFrame()

                  
for ct in range(0, len(mean_only_charge_list)):
    
    spot_only_charge = spot_only_charge_list[ct]
    mean_only_charge = mean_only_charge_list[ct]

    # data preparation
    mean_static = xr.open_dataarray(mean_only_charge + "P_BUY.nc").sel(s="reg", r=sensi_different_list).mean(["v", "r"]).to_pandas()
    mean_ToU = xr.open_dataarray(mean_only_charge + "P_BUY.nc").sel(s="red", r=sensi_different_list).mean(["v", "r"]).to_pandas()
    spot_static = xr.open_dataarray(spot_only_charge + "P_BUY.nc").sel(s="reg", r=sensi_different_list).mean(["v", "r"]).to_pandas()
    spot_ToU = xr.open_dataarray(spot_only_charge + "P_BUY.nc").sel(s="red", r=sensi_different_list).mean(["v", "r"]).to_pandas()
    
    spot_only_charge_sensi = spot_only_charge_list_sensi[ct]
    mean_only_charge_sensi = mean_only_charge_list_sensi[ct]
    
    # data preparation
    mean_static_sensi = xr.open_dataarray(mean_only_charge_sensi + "P_BUY.nc").sel(s="reg", r=sensi_different_list).mean(["v", "r"]).to_pandas()
    mean_ToU_sensi = xr.open_dataarray(mean_only_charge_sensi + "P_BUY.nc").sel(s="red", r=sensi_different_list).mean(["v", "r"]).to_pandas()
    spot_static_sensi = xr.open_dataarray(spot_only_charge_sensi + "P_BUY.nc").sel(s="reg", r=sensi_different_list).mean(["v", "r"]).to_pandas()
    spot_ToU_sensi = xr.open_dataarray(spot_only_charge_sensi + "P_BUY.nc").sel(s="red", r=sensi_different_list).mean(["v", "r"]).to_pandas()
    
    pd_ct[charge_mode[ct] + "_mean_static_standard"] = mean_static - mean_static_sensi
    pd_ct[charge_mode[ct] + "_mean_ToU_standard"] = mean_ToU - mean_ToU_sensi
    pd_ct[charge_mode[ct] + "_spot_static_standard"] = spot_static - spot_static_sensi
    pd_ct[charge_mode[ct] + "_spot_ToU_standard"] = spot_ToU - spot_ToU_sensi
    
dti = pd.DatetimeIndex(epoch_time + pd.to_timedelta(xr.open_dataarray(mean_only_charge + "P_BUY.nc")["t"], unit='s')).tz_localize("UTC").tz_convert("Europe/Berlin")
pd_ct = pd_ct.set_index(dti)
pd_ct["hour decimal"] = pd_ct.index.hour + pd_ct.index.minute/60

pd_day = pd_ct.groupby(["hour decimal"]).mean()






# kW reduction plots
if (False):

    fig_kw_sensi, axs_kw_sensi = plt.subplots(ncols=2, figsize=(15, 6))   

    # linker plot
    pd_charge_mode = pd.concat([ pd_day["immediate_mean_static_standard"], # alle immediate (spot, mean, ToU, standard) müssten gleich sein !!!!!
                                    pd_day["scheduled_mean_static_standard"],
                                    pd_day["smart_mean_static_standard"],
                                    pd_day["smart_mean_ToU_standard"],
                                    pd_day["smart_spot_static_standard"],
                                    pd_day["smart_spot_ToU_standard"] ],
                                    
                                    axis = 1).rename(columns={"immediate_mean_static_standard":"immediate",
                                                              "scheduled_mean_static_standard":"scheduled", 
                                                              "smart_mean_static_standard":"smart (static, standard)", 
                                                              "smart_mean_ToU_standard":"smart (static, ToU)",
                                                              "smart_spot_static_standard":"smart (dynamic, standard)",
                                                              "smart_spot_ToU_standard":"smart (dynamic, ToU)" })



    axs_kw_sensi[0].plot(pd_charge_mode["immediate"], linestyle="-", color="k", linewidth=1, zorder=2, label="immediate")
    axs_kw_sensi[0].plot(pd_charge_mode["smart (static, standard)"], linestyle="-", color="#8b3003", linewidth=1, zorder=1, label="smart (static, standard)")
    axs_kw_sensi[0].plot(pd_charge_mode["smart (static, ToU)"], linestyle="--", color="#c13f1a", linewidth=1, zorder=1, label= "smart (static, ToU)")

    axs_kw_sensi[0].plot(pd_charge_mode["scheduled"], linestyle="-.", color="gray", linewidth=1, zorder=2, label=  "scheduled")
    axs_kw_sensi[0].plot(pd_charge_mode["smart (dynamic, standard)"], linestyle="-", color="#00386c", linewidth=1, zorder=1, label=  "smart (dynamic, standard)")
    axs_kw_sensi[0].plot(pd_charge_mode["smart (dynamic, ToU)"], linestyle="--", color="#0087ff", linewidth=1, zorder=1, label= "smart (dynamic, ToU)")



    #axs_kw_sensi[0].plot(pd_charge_mode["switch to scheduled (dynamic)"], linestyle="--", color="#00386c", linewidth=1, zorder=1, label="switch to scheduled (dynamic)")
    #axs_kw_sensi[0].plot(pd_charge_mode["switch to smart (dynamic)"], linestyle="-", color="#0087ff", linewidth=1, zorder=1, label= r'$\rightarrow$' + " smart (dynamic)")

    axs_kw_sensi[0].legend(fontsize=11, ncols=2, loc="upper center", title="Impact of charge strategy", title_fontsize=16)
    axs_kw_sensi[0].grid(color='lightgray', linestyle='--', linewidth=1, axis="both")
    axs_kw_sensi[0].set_xticks(np.array([0, 3, 6, 9, 12, 15, 18, 21, 24]))
    axs_kw_sensi[0].set_xticklabels([0, 3, 6, 9, 12, 15, 18, 21, 24], fontsize=20)
    axs_kw_sensi[0].set_ylim(-.01, .01)
    axs_kw_sensi[0].set_ylabel("Mean Power consumption in kW", fontsize=20)
    axs_kw_sensi[0].tick_params(axis='y', labelsize=20)
    axs_kw_sensi[0].set_xlabel("Time in hours", fontsize=20)



    # rechter plot
    pd_smart = pd.concat([ pd_day["smart_mean_static_standard"] - pd_day["smart_mean_static_standard"] ,
                                    pd_day["smart_mean_ToU_standard"] - pd_day["smart_mean_static_standard"] ,
                                    pd_day["smart_spot_static_standard"] - pd_day["smart_mean_static_standard"] ,
                                    pd_day["smart_spot_ToU_standard"] - pd_day["smart_mean_static_standard"] ],
                                    axis = 1).rename(columns={"smart_mean_static_standard":"reference: static standard", 0:"static ToU", 1:"dynamic standard", 2:"dynamic ToU"})
    
    linestyle_list = ['-', '--', '-', '--']  # same length as columns
    color_list = ["#8b3003", "#c13f1a", "#00386c", "#0087ff"]
    
    #axs_kw_sensi[1].fill_between(np.array(range(0,96))/4, pd_smart["static ToU"], linestyle=linestyle_list[1], color=color_list[1], alpha=0.5, zorder=0)
    axs_kw_sensi[1].plot(pd_smart["reference: static standard"], linestyle=linestyle_list[0], color=color_list[0], alpha=1, linewidth=1,  zorder=2, label="reference: \nstatic standard")
    axs_kw_sensi[1].plot(pd_smart["static ToU"], linestyle=linestyle_list[1], color=color_list[1], alpha=1, zorder=0, linewidth=1, label="static ToU")

    #axs_kw_sensi[1].fill_between(np.array(range(0,96))/4, pd_smart["dynamic standard"], pd_smart["dynamic ToU"], linestyle=linestyle_list[2], color=color_list[2], alpha=0.5, zorder=0)
    axs_kw_sensi[1].plot(pd_smart["dynamic standard"], linestyle=linestyle_list[2], color=color_list[2], alpha=1, zorder=1, linewidth=1, label="dynamic standard")
    axs_kw_sensi[1].plot(pd_smart["dynamic ToU"], linestyle=linestyle_list[3], color=color_list[3], alpha=1, linewidth=1,  zorder=3, label="dynamic ToU")

    axs_kw_sensi[1].legend(fontsize=11, ncols=2, title="Impact of tariff components \nfor smart charging ", title_fontsize=16, loc="upper right")

        
    axs_kw_sensi[1].set_ylim(-1, 1)
    axs_kw_sensi[1].set_xlabel("Time in hours", fontsize=20)
    axs_kw_sensi[1].set_ylabel("Mean Power change in kW", fontsize=20)
    axs_kw_sensi[1].set_ylim(-.01, 0.01)
    axs_kw_sensi[1].tick_params(axis='y', labelsize=20)
    axs_kw_sensi[1].set_xlim(0, 24)
    axs_kw_sensi[1].set_xticks(np.array([0, 3, 6, 9, 12, 15, 18, 21, 24]))
    axs_kw_sensi[1].set_xticklabels([0, 3, 6, 9, 12, 15, 18, 21, 24], fontsize=20)
    axs_kw_sensi[1].grid(color='lightgray', linestyle='--', linewidth=1, axis="both")

    plt.tight_layout()
    plt.show()

    fig_kw_sensi.savefig(r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results\power_consumption_reduction_sensi.svg")






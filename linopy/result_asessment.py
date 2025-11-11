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
folder_name = "2025-11-09_02-46_mean_immediate_only_EV_r20_v50"


folder_path = Path("../daten_results") / folder_name

result_C_ALL_eur = xr.open_dataarray(folder_path / "C_ALL.nc")
result_SOC_EV = xr.open_dataarray(folder_path / "SOC_EV.nc")
result_P_HOME = xr.open_dataarray(folder_path / "P_HOME.nc")
result_P_PUBLIC = xr.open_dataarray(folder_path / "P_PUBLIC.nc")
#result_SOC_MISSING = xr.open_dataarray(folder_path / "SOC_MISSING.nc")

# reconvert seconds to datetime
epoch_time = datetime(1970, 1, 1)
dti = pd.DatetimeIndex(epoch_time + pd.to_timedelta(result_SOC_EV["t"], unit='s')).tz_localize("UTC").tz_convert("Europe/Berlin")

dti_dt = result_P_HOME["t"].to_pandas().index

result_SOC_EV["t"] = dti
result_P_HOME["t"] = dti
result_P_PUBLIC["t"] = dti

# ax1 = result_P_HOME.mean("r").mean("v").to_pandas().groupby(dti_dt.hour).mean().plot()
# result_P_PUBLIC.mean("r").mean("v").to_pandas().groupby(dti_dt.hour).mean().plot(ax=ax1, linestyle="--")
# result_SOC_EV.mean("r").mean("v").to_pandas().groupby(dti_dt.hour).mean().plot()
# m.variables.SOC_EV.solution.sum("r").sum("v").to_pandas().plot()

# ===== plotting ====


if (False):  # BAR PLOT MEAN COST SAVINGS PER DSO
      
    scenarios = result_C_ALL_eur["r"].to_pandas().to_list()
    dso_means = {'regular network charges': result_C_ALL_eur.sel(s='reg').mean(dim=["v"]),
                    'reduced network charges': result_C_ALL_eur.sel(s='red').mean(dim=["v"]) }
    
    x = np.arange(len(result_C_ALL_eur.mean(dim="v")))  # the label locations
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
    str_v = str(len(result_C_ALL_eur["v"]))
    ax.set_title('Charging at home, averaging over ' + str_v + ' different mobility use cases')
    ax.set_xticks(x + width)
    ax.set_xticklabels(scenarios, rotation=90)
    ax.legend(loc='lower center', ncols=2)
    ax.grid(color='lightgray', linestyle='--', linewidth=1, axis="y")
    ax.set_axisbelow(True)

    fig.savefig(folder_path / "dso_cost_barlot.svg")



    
    
if (False): # PEAK REDUCTION HEATMAP
    
    time_index = result_P_HOME["t"].to_pandas().index.hour
    time_index_evening = ((time_index >= 17) & (time_index <= 21))

    peak_reduction = (result_P_HOME.sel(s="reg", t=time_index_evening).max("t")-result_P_HOME.sel(s="red", t=time_index_evening).max("t")).to_pandas()
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
    immediate_mean_only_charge_str = r"2025-11-10_18-56_mean_immediate_only_EV_r10_v10" + r"\\"
    immediate_spot_only_charge_str = r"2025-11-10_22-25_spot_immediate_only_EV_r10_v10" + r"\\" 
    scheduled_mean_only_charge_str = r"2025-11-10_20-58_mean_scheduled_only_EV_r10_v10" + r"\\"
    scheduled_spot_only_charge_str = r"2025-11-10_21-44_spot_scheduled_only_EV_r10_v10" + r"\\"  
    smart_mean_only_charge_str =   r"2025-11-11_15-20_mean_smart_only_EV_r10_v10" + r"\\"
    smart_spot_only_charge_str =  r"2025-11-11_10-21_spot_smart_only_EV_r10_v10" + r"\\"
    
    immediate_mean_only_charge = folder_str + immediate_mean_only_charge_str
    immediate_spot_only_charge = folder_str + immediate_spot_only_charge_str
    scheduled_mean_only_charge = folder_str + scheduled_mean_only_charge_str
    scheduled_spot_only_charge = folder_str + scheduled_spot_only_charge_str
    smart_mean_only_charge = folder_str +   smart_mean_only_charge_str
    smart_spot_only_charge = folder_str +   smart_spot_only_charge_str

   


    # Test section
    if (False):
        p_immediate_mean_standard = xr.open_dataarray(immediate_mean_only_charge + "P_HOME.nc").sel(s="reg").mean(["r","v"]).sum("t").to_pandas()
        p_immediate_mean_ToU = xr.open_dataarray(immediate_mean_only_charge + "P_HOME.nc").sel(s="red").mean(["r","v"]).sum("t").to_pandas()
        p_scheduled_mean_standard = xr.open_dataarray(scheduled_mean_only_charge + "P_HOME.nc").sel(s="reg").mean(["r","v"]).sum("t").to_pandas()
        p_scheduled_mean_ToU = xr.open_dataarray(scheduled_mean_only_charge + "P_HOME.nc").sel(s="red").mean(["r","v"]).sum("t").to_pandas()
        p_smart_mean_standard = xr.open_dataarray(smart_mean_only_charge + "P_HOME.nc").sel(s="reg").mean(["r","v"]).sum("t").to_pandas()
        p_smart_mean_ToU = xr.open_dataarray(smart_mean_only_charge + "P_HOME.nc").sel(s="red").mean(["r","v"]).sum("t").to_pandas()
   
        print("===== P_HOME ====")
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
    cost_type_file = "C_ALL.nc"  # "C_OP_ALL.nc", "C_OP_HOME.nc" (no price spikes for scheduled charging due to public pole charging)
    axis_y_max = 200
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
    flierprops = dict(marker='o', markerfacecolor=(0,0,0,0), markersize=6, markeredgecolor=(0,0,0,0))  # set to transparent
    
    pd_standard_static.plot(ax = axs[0, 0],  kind="box", widths=0.7, patch_artist=True, notch=True, showmeans=False, meanprops=meanpointprops,  flierprops=flierprops, color=dict(boxes='black', whiskers='black', medians='black', caps='black'), boxprops=dict(facecolor="lightgray"))
    pd_standard_dynamic.plot(ax = axs[0, 1], kind="box", widths=0.7, patch_artist=True, notch=True, showmeans=False, meanprops=meanpointprops,  flierprops=flierprops, color=dict(boxes='black', whiskers='black', medians='black', caps='black'), boxprops=dict(facecolor="lightgray"))
    pd_ToU_static.plot(ax = axs[1, 0],  kind="box", widths=0.7, patch_artist=True, notch=True, showmeans=False, meanprops=meanpointprops,  flierprops=flierprops, color=dict(boxes='black', whiskers='black', medians='black', caps='black'), boxprops=dict(facecolor="lightgray"))
    pd_ToU_dynamic.plot(ax = axs[1, 1], kind="box", widths=0.7, patch_artist=True, notch=True, showmeans=False, meanprops=meanpointprops,  flierprops=flierprops, color=dict(boxes='black', whiskers='black', medians='black', caps='black'), boxprops=dict(facecolor="lightgray"))
    
    axs[1, 0].set_xlabel("Static", fontsize=20, fontweight='bold')
    axs[1, 1].set_xlabel("Dynamic", fontsize=20, fontweight='bold')
    axs[0, 0].set_ylabel("Standard", fontsize=20, fontweight='bold')
    axs[1, 0].set_ylabel("Time of Use", fontsize=20, fontweight='bold')
    
    axs[1, 0].tick_params(axis='both', labelsize=20)
    axs[1, 1].tick_params(axis='both', labelsize=20)
    axs[0, 0].tick_params(axis='both', labelsize=20)
    axs[0, 1].tick_params(axis='both', labelsize=20)

    axs[0, 0].set_ylim(0,axis_y_max)
    axs[0, 1].set_ylim(0,axis_y_max)
    axs[1, 0].set_ylim(0,axis_y_max)
    axs[1, 1].set_ylim(0,axis_y_max)

    ytickvals = np.linspace(0,int(axis_y_max/25)*25,int(axis_y_max/25+1)).astype(int)

    axs[0,0].set_yticks(ytickvals)
    axs[0,0].set_yticklabels([str(y)+"€" if y%50==0 else " " for y in ytickvals], fontsize=20)
    axs[0,1].set_yticks(ytickvals)
    axs[0,1].set_yticklabels([str(y)+"€"  if y%50==0 else " " for y in ytickvals], fontsize=20)
    axs[1,0].set_yticks(ytickvals)
    axs[1,0].set_yticklabels([str(y)+"€"  if y%50==0 else " " for y in ytickvals], fontsize=20)
    axs[1,1].set_yticks(ytickvals)
    axs[1,1].set_yticklabels([str(y)+"€"  if y%50==0 else " " for y in ytickvals], fontsize=20)

    
    fig_grouped_boxplots_cost_savings.supxlabel("Electricity price", fontsize=20, fontweight='bold')
    fig_grouped_boxplots_cost_savings.supylabel("Network charge", fontsize=20, fontweight='bold')

    for ax in axs.flat:
        ax.xaxis.grid(False)
        ax.yaxis.grid(True, linestyle="--", color="lightgray")
        
    fig_grouped_boxplots_cost_savings.savefig(r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results\annual_cost_end_consumer.svg")

    
    


    
# =============================================================================
# CHARGE POWER  
# kW reduction plots
# =============================================================================

if (False):
    
    variable_file = "P_HOME.nc"  # "P_PUBLIC.nc"
    
    epoch_time = datetime(1970, 1, 1)

    folder_str = r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results" + r"\\"
        
    # files: immediate, scheduled, smart
    mean_only_charge_list = [folder_str + x for x in [immediate_mean_only_charge_str,
                                                      scheduled_mean_only_charge_str,
                                                      smart_mean_only_charge_str  ]  ]
    
    spot_only_charge_list = [folder_str + x for x in [immediate_spot_only_charge_str,
                                                      scheduled_spot_only_charge_str,
                                                      smart_spot_only_charge_str ]  ]
                                     
    
    charge_mode = ["immediate", "scheduled", "smart"]
    
    pd_ct = pd.DataFrame()
                      
    for ct in range(0, len(mean_only_charge_list)):
        spot_only_charge = spot_only_charge_list[ct]
        mean_only_charge = mean_only_charge_list[ct]
    
        # data preparation
        mean_static = xr.open_dataarray(mean_only_charge + variable_file).sel(s="reg").mean(["v","r"]).to_pandas()
        mean_ToU = xr.open_dataarray(mean_only_charge + variable_file).sel(s="red").mean(["v","r"]).to_pandas()
        spot_static = xr.open_dataarray(spot_only_charge + variable_file).sel(s="reg").mean(["v","r"]).to_pandas()
        spot_ToU = xr.open_dataarray(spot_only_charge + variable_file).sel(s="red").mean(["v","r"]).to_pandas()
     
        pd_ct[charge_mode[ct] + "_mean_static_standard"] = mean_static
        pd_ct[charge_mode[ct] + "_mean_ToU_standard"] = mean_ToU
        pd_ct[charge_mode[ct] + "_spot_static_standard"] = spot_static
        pd_ct[charge_mode[ct] + "_spot_ToU_standard"] = spot_ToU
        
    dti = pd.DatetimeIndex(epoch_time + pd.to_timedelta(xr.open_dataarray(mean_only_charge + variable_file)["t"], unit='s')).tz_localize("UTC").tz_convert("Europe/Berlin")
    pd_ct = pd_ct.set_index(dti)
    pd_ct["hour decimal"] = pd_ct.index.hour + pd_ct.index.minute/60
    
    pd_day = pd_ct.groupby(["hour decimal"]).mean()


    fig_kw_savings, axs_kw_savings = plt.subplots(ncols=2, figsize=(15, 6))   

    # linker plot
    pd_charge_mode = pd.concat([ pd_day["immediate_mean_static_standard"], 
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
    axs_kw_savings[0].plot(pd_charge_mode["smart (static, standard)"], linestyle="-", color="#c13f1a", linewidth=1, zorder=1, label="smart (static, standard)")
    axs_kw_savings[0].plot(pd_charge_mode["smart (static, ToU)"], linestyle="--", color="#c13f1a", linewidth=1, zorder=1, label= "smart (static, ToU)")

    axs_kw_savings[0].plot(pd_charge_mode["scheduled"], linestyle="-.", color="k", linewidth=1, zorder=2, label=  "scheduled")
    axs_kw_savings[0].plot(pd_charge_mode["smart (dynamic, standard)"], linestyle="-", color="#0087ff", linewidth=1, zorder=1, label=  "smart (dynamic, standard)")
    axs_kw_savings[0].plot(pd_charge_mode["smart (dynamic, ToU)"], linestyle="--", color="#0087ff", linewidth=1, zorder=1, label= "smart (dynamic, ToU)")



    axs_kw_savings[0].legend(fontsize=11, ncols=2, loc="upper center", title="Impact of charge strategy (all timesteps)", title_fontsize=16)
    axs_kw_savings[0].grid(color='lightgray', linestyle='--', linewidth=1, axis="both")
    axs_kw_savings[0].set_xticks(np.array([0, 3, 6, 9, 12, 15, 18, 21, 24]))
    axs_kw_savings[0].set_xticklabels([0, 3, 6, 9, 12, 15, 18, 21, 24], fontsize=20)
    axs_kw_savings[0].set_ylim(-.1, 0.6)
    axs_kw_savings[0].set_ylabel("Mean Power consumption in kW", fontsize=20)
    axs_kw_savings[0].tick_params(axis='y', labelsize=20)
    axs_kw_savings[0].set_xlabel("Time in hours", fontsize=20)

    # ===========================================
    # identify critical timesteps
    # ============================================
    dti = pd.DatetimeIndex(epoch_time + pd.to_timedelta(xr.open_dataarray(mean_only_charge + variable_file)["t"], unit='s')).tz_localize("UTC").tz_convert("Europe/Berlin")

    
    critical_steps = pd.read_csv(r"Z:\10_Paper\13_Alleinautorenpaper\critical_timesteps_iso2024.csv", index_col="t")
    critical_steps[critical_steps==0] = np.nan
    critical_steps = critical_steps.iloc[:, range(0,30)]
    critical_steps_xr = xr.DataArray(critical_steps, dims=["t","r"]) 
    critical_steps_xr["t"] = dti
    pd_ct = pd.DataFrame()
           
    for ct in range(0, len(mean_only_charge_list)):
        spot_only_charge = spot_only_charge_list[ct]
        mean_only_charge = mean_only_charge_list[ct]
    
        # data preparation
        mean_static = xr.open_dataarray(mean_only_charge + variable_file).sel(s="reg") 
        mean_static["t"] = dti
        mean_static = (mean_static * critical_steps_xr).mean(["v","r"]).to_pandas()
        
        mean_ToU = xr.open_dataarray(mean_only_charge + variable_file).sel(s="red")
        mean_ToU["t"] = dti
        mean_ToU = (mean_ToU * critical_steps_xr).mean(["v","r"]).to_pandas()

        spot_static = xr.open_dataarray(spot_only_charge + variable_file).sel(s="reg")
        spot_static["t"] = dti
        spot_static = (spot_static * critical_steps_xr).mean(["v","r"]).to_pandas()

        spot_ToU = xr.open_dataarray(spot_only_charge + variable_file).sel(s="red")
        spot_ToU["t"] = dti
        spot_ToU = (spot_ToU * critical_steps_xr).mean(["v","r"]).to_pandas()

     
        pd_ct[charge_mode[ct] + "_mean_static_standard"] = mean_static
        pd_ct[charge_mode[ct] + "_mean_ToU_standard"] = mean_ToU
        pd_ct[charge_mode[ct] + "_spot_static_standard"] = spot_static
        pd_ct[charge_mode[ct] + "_spot_ToU_standard"] = spot_ToU
        
    pd_ct = pd_ct.set_index(dti)
    pd_ct["hour decimal"] = pd_ct.index.hour + pd_ct.index.minute/60
    
    pd_day_critical = pd_ct.groupby(["hour decimal"]).mean()



    # rechter plot
    # linker plot
    pd_charge_mode = pd.concat([ pd_day_critical["immediate_mean_static_standard"], 
                                    pd_day_critical["scheduled_mean_static_standard"],
                                    pd_day_critical["smart_mean_static_standard"],
                                    pd_day_critical["smart_mean_ToU_standard"],
                                    pd_day_critical["smart_spot_static_standard"],
                                    pd_day_critical["smart_spot_ToU_standard"] ],
                                    
                                    axis = 1).rename(columns={"immediate_mean_static_standard":"immediate",
                                                              "scheduled_mean_static_standard":"scheduled", 
                                                              "smart_mean_static_standard":"smart (static, standard)", 
                                                              "smart_mean_ToU_standard":"smart (static, ToU)",
                                                              "smart_spot_static_standard":"smart (dynamic, standard)",
                                                              "smart_spot_ToU_standard":"smart (dynamic, ToU)" })



    axs_kw_savings[1].plot(pd_charge_mode["immediate"], linestyle="-", color="k", linewidth=1, zorder=2, label="immediate")
    axs_kw_savings[1].plot(pd_charge_mode["smart (static, standard)"], linestyle="-", color="#c13f1a", linewidth=1, zorder=1, label="smart (static, standard)")
    axs_kw_savings[1].plot(pd_charge_mode["smart (static, ToU)"], linestyle="--", color="#c13f1a", linewidth=1, zorder=1, label= "smart (static, ToU)")

    axs_kw_savings[1].plot(pd_charge_mode["scheduled"], linestyle="-.", color="k", linewidth=1, zorder=2, label=  "scheduled")
    axs_kw_savings[1].plot(pd_charge_mode["smart (dynamic, standard)"], linestyle="-", color="#0087ff", linewidth=1, zorder=1, label=  "smart (dynamic, standard)")
    axs_kw_savings[1].plot(pd_charge_mode["smart (dynamic, ToU)"], linestyle="--", color="#0087ff", linewidth=1, zorder=1, label= "smart (dynamic, ToU)")



    axs_kw_savings[1].legend(fontsize=11, ncols=2, loc="upper center", title="Impact of charge strategy (critical timesteps)", title_fontsize=16)
    axs_kw_savings[1].grid(color='lightgray', linestyle='--', linewidth=1, axis="both")
    axs_kw_savings[1].set_xticks(np.array([0, 3, 6, 9, 12, 15, 18, 21, 24]))
    axs_kw_savings[1].set_xticklabels([0, 3, 6, 9, 12, 15, 18, 21, 24], fontsize=20)
    axs_kw_savings[1].set_ylim(-.1, 0.6)
    axs_kw_savings[1].set_ylabel("Mean Power consumption in kW", fontsize=20)
    axs_kw_savings[1].tick_params(axis='y', labelsize=20)
    axs_kw_savings[1].set_xlabel("Time in hours", fontsize=20)
    plt.tight_layout()
    plt.show()

    fig_kw_savings.savefig(r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results\power_consumption.svg")





    
# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================


epoch_time = datetime(1970, 1, 1)

folder_str = r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results" + r"\\"
    

# set sensi in parameters opti to true
_, _, _, _, _, _, _, sensi_different = f_load.load_network_charges(parameter_filepath_dsos, timesteps, parameters_opti)
sensi_different_list = list(sensi_different)
sensi_different_list = [sensi_different_list[ct] for ct in list(parameters_opti["dso_subset"])] # Dimension stimmt nach Lauf nicht, dann nochmal die ersten Zeilen von main_script ausführen

# regular
spot_only_smart = r"2025-08-02_05-01_all_spot_smart_charging_only_EV_r100_v50_orig" + r"\\"                 
mean_only_smart = r"2025-08-01_21-28_all_mean_smart_charging_only_EV_r100_v50_orig" + r"\\"
# sensitivity
spot_only_smart_sensi = r"2025-08-06_13-46_all_spot_smart_charging_only_EV_r100_v50_sensi" + r"\\"                 
mean_only_smart_sensi = r"2025-08-07_08-53_all_mean_smart_charging_only_EV_r100_v50_sensi" + r"\\" 

# cost loading
dso_x_ev = xr.open_dataarray(folder_str + mean_only_smart + "C_OP_ALL.nc").sel(s="red", r=sensi_different_list).size
mean_ToU_c = xr.open_dataarray(folder_str + mean_only_smart + "C_OP_ALL.nc").sel(s="red", r=sensi_different_list).to_pandas().to_numpy().reshape(dso_x_ev)
spot_ToU_c = xr.open_dataarray(folder_str + spot_only_smart + "C_OP_ALL.nc").sel(s="red", r=sensi_different_list).to_pandas().to_numpy().reshape(dso_x_ev)
mean_ToU_sensi_c = xr.open_dataarray(folder_str + mean_only_smart_sensi + "C_OP_ALL.nc").sel(s="red", r=sensi_different_list).to_pandas().to_numpy().reshape(dso_x_ev)
spot_ToU_sensi_c = xr.open_dataarray(folder_str + spot_only_smart_sensi + "C_OP_ALL.nc").sel(s="red", r=sensi_different_list).to_pandas().to_numpy().reshape(dso_x_ev)
cost_sensi = pd.DataFrame({'static base case':mean_ToU_c, 'static sensitivity':mean_ToU_sensi_c, 'dynamic base case':spot_ToU_c, 'dynamic sensitivity': spot_ToU_sensi_c})
# no linebreak space between "base" and "case"

# power consumption loading
mean_ToU = xr.open_dataarray(folder_str + mean_only_smart + "P_BUY.nc").sel(s="red", r=sensi_different_list).sum(["v"]).mean(["r"]).to_pandas()
spot_ToU = xr.open_dataarray(folder_str + spot_only_smart + "P_BUY.nc").sel(s="red", r=sensi_different_list).sum(["v"]).mean(["r"]).to_pandas()
mean_ToU_sensi = xr.open_dataarray(folder_str + mean_only_smart_sensi + "P_BUY.nc").sel(s="red", r=sensi_different_list).sum(["v"]).mean(["r"]).to_pandas()
spot_ToU_sensi = xr.open_dataarray(folder_str + spot_only_smart_sensi + "P_BUY.nc").sel(s="red", r=sensi_different_list).sum(["v"]).mean(["r"]).to_pandas()


pd_ct = pd.DataFrame()

pd_ct["smart ToU static original"] = mean_ToU 
pd_ct["smart ToU static new"] = mean_ToU_sensi
pd_ct["smart ToU dynamic original"] = spot_ToU 
pd_ct["smart ToU dynamic new"] = spot_ToU_sensi
    
dti = pd.DatetimeIndex(epoch_time + pd.to_timedelta(xr.open_dataarray(folder_str + mean_only_smart + "P_BUY.nc")["t"], unit='s')).tz_localize("UTC").tz_convert("Europe/Berlin")
pd_ct = pd_ct.set_index(dti)
pd_ct["hour decimal"] = pd_ct.index.hour + pd_ct.index.minute/60

pd_day = pd_ct.groupby(["hour decimal"]).mean()






# kW reduction plots
if (False):

    fig_sensi, axs_sensi = plt.subplots(ncols=2, figsize=(15, 6))   


    # LINKER PLOT
    meanpointprops = dict(marker='x', markeredgecolor='black', markerfacecolor='black') #firebrick
    flierprops = dict(marker='o', markerfacecolor=(0,0,0,0), markersize=6, markeredgecolor=(0,0,0,0))  # set to transparent
    cost_sensi.plot(ax = axs_sensi[0],  kind="box", widths=0.7, patch_artist=True, notch=True, showmeans=True, meanprops=meanpointprops,  flierprops=flierprops, color=dict(boxes='black', whiskers='black', medians='black', caps='black'), boxprops=dict(facecolor="lightgray"))
    axs_sensi[0].set_ylabel("Cost for end-consumers in €", fontsize=20)
    axs_sensi[0].set_xticklabels(cost_sensi.columns.str.replace(" ","\n"), fontsize=20)
    axs_sensi[0].set_ylim(50, 450)


    axs_sensi[0].grid(color='lightgray', linestyle='--', linewidth=1, axis="both")
    axs_sensi[0].tick_params(axis='both', labelsize=20)


    #  RECHTER PLOT
    linestyle_list =  ['-', '--', '-', '--']  # same length as columns
    color_list = ["#8b3003", "#c13f1a", "#00386c", "#0087ff"]
    axs_sensi[1].plot(pd_day["smart ToU static new"] - pd_day["smart ToU static original"], linestyle=linestyle_list[0], color=color_list[0], alpha=1, linewidth=1,  zorder=2, label="static electricity prices \n(sensitivity minus base case)")
    axs_sensi[1].plot(pd_day["smart ToU dynamic new"] - pd_day["smart ToU dynamic original"], linestyle=linestyle_list[2], color=color_list[2], alpha=1, zorder=1, linewidth=1, label="dynamic electricity prices \n(sensitivity minus base case)")

    axs_sensi[1].legend(fontsize=16, ncols=1, loc="upper right")

        
    axs_sensi[1].set_ylim(-1, 1)
    axs_sensi[1].set_xlabel("Time in hours", fontsize=20)
    axs_sensi[1].set_ylabel("Mean power change \n of 50 EV in kW", fontsize=20)
    axs_sensi[1].set_ylim(-0.35, 0.35)
    axs_sensi[1].tick_params(axis='both', labelsize=20)
    axs_sensi[1].set_xlim(0, 24)
    axs_sensi[1].set_xticks(np.array([0, 3, 6, 9, 12, 15, 18, 21, 24]))
    axs_sensi[1].set_xticklabels([0, 3, 6, 9, 12, 15, 18, 21, 24], fontsize=20)
    axs_sensi[1].grid(color='lightgray', linestyle='--', linewidth=1, axis="both")

    plt.tight_layout()
    plt.show()

    fig_sensi.savefig(r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results\sensitivity_test.svg")






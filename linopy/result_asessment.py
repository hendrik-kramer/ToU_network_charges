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
folder_name = "2025-11-20_02-00_spot_smart_only_EV_r100_v50_poly"


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


    
    
# =============================================================================
# Total Cost for scheduled and smart charging // Annual cost for end-consumer
# =============================================================================
   
    
if (False): 
    
    # pfade scheduled
    # network charge _ elec
    folder_str = r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results" + r"\\"
    
    # scheduled charge EV
    immediate_mean_only_charge_str = r"2025-11-16_03-54_mean_immediate_only_EV_r100_v50" + r"\\"
    immediate_spot_only_charge_str = r"2025-11-21_07-57_spot_immediate_only_EV_r100_v50_poly" + r"\\" 
    scheduled_mean_only_charge_str = r"2025-11-17_02-54_mean_scheduled_only_EV_r100_v50" + r"\\"
    scheduled_spot_only_charge_str = r"2025-11-22_08-29_spot_scheduled_only_EV_r100_v50_poly" + r"\\"  
    smart_mean_only_charge_str =   r"2025-11-15_08-30_mean_smart_only_EV_r100_v50" + r"\\"
    smart_spot_only_charge_str =  r"2025-11-21_00-39_spot_smart_only_EV_r100_v50_poly" + r"\\" #r"2025-11-15_15-33_spot_smart_only_EV_r100_v50" + r"\\"
    
    immediate_mean_only_charge = folder_str + immediate_mean_only_charge_str
    immediate_spot_only_charge = folder_str + immediate_spot_only_charge_str
    scheduled_mean_only_charge = folder_str + scheduled_mean_only_charge_str
    scheduled_spot_only_charge = folder_str + scheduled_spot_only_charge_str
    smart_mean_only_charge = folder_str +   smart_mean_only_charge_str
    smart_spot_only_charge = folder_str +   smart_spot_only_charge_str

   


    # Test section
    if (False):
        p_immediate_mean_standard = xr.open_dataarray(immediate_mean_only_charge + "C_ALL.nc")
        p_immediate_mean_ToU = xr.open_dataarray(immediate_mean_only_charge + "C_HOME.nc").sel(s="red").mean(["r","v"]).to_pandas()
        p_scheduled_mean_standard = xr.open_dataarray(scheduled_mean_only_charge + "C_HOME.nc").sel(s="reg").mean(["r","v"]).to_pandas()
        p_scheduled_mean_ToU = xr.open_dataarray(scheduled_mean_only_charge + "C_HOME.nc").sel(s="red").mean(["r","v"]).to_pandas()
        p_smart_mean_standard = xr.open_dataarray(smart_mean_only_charge + "C_HOME.nc").sel(s="reg").mean(["r","v"]).to_pandas()
        p_smart_mean_ToU = xr.open_dataarray(smart_mean_only_charge + "C_HOME.nc").sel(s="red").mean(["r","v"]).to_pandas()
   
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
    immediate_spot_standard = xr.open_dataarray(immediate_spot_only_charge + cost_type_file).sel(s="reg").to_numpy().reshape(dso_x_ev) / 100 
    immediate_spot_ToU = xr.open_dataarray(immediate_spot_only_charge + cost_type_file).sel(s="red").to_numpy().reshape(dso_x_ev) / 100
    scheduled_spot_standard = xr.open_dataarray(scheduled_spot_only_charge + cost_type_file).sel(s="reg").to_numpy().reshape(dso_x_ev) / 100
    scheduled_spot_ToU = xr.open_dataarray(scheduled_spot_only_charge + cost_type_file).sel(s="red").to_numpy().reshape(dso_x_ev) /100
    smart_spot_standard = xr.open_dataarray(smart_spot_only_charge + cost_type_file).sel(s="reg").to_numpy().reshape(dso_x_ev)  / 100
    smart_spot_ToU = xr.open_dataarray(smart_spot_only_charge + cost_type_file).sel(s="red").to_numpy().reshape(dso_x_ev)  / 100

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
    
    bla, props = pd_standard_static.plot(ax = axs[0, 0],  kind="box", widths=0.4, patch_artist=True, return_type='both', notch=False, showmeans=True, meanprops=meanpointprops, showfliers=False, flierprops=flierprops, color=dict(boxes='black', whiskers='black', medians='black', caps='black'), boxprops=dict(facecolor="lightgray")) 
    props['boxes'][0].set_facecolor("dimgrey")
    props['boxes'][2].set_facecolor("#8b3003") # dunkelrot

    bla, props = pd_standard_dynamic.plot(ax = axs[0, 1], kind="box", widths=0.4, patch_artist=True, return_type='both', notch=False, showmeans=True, meanprops=meanpointprops, showfliers=False, flierprops=flierprops, color=dict(boxes='black', whiskers='black', medians='black', caps='black'), boxprops=dict(facecolor="lightgray"))
    props['boxes'][0].set_facecolor("dimgrey")
    props['boxes'][2].set_facecolor("#004c93") # dunkelblau
    
    bla, props = pd_ToU_static.plot(ax = axs[1, 0],  kind="box", widths=0.4, patch_artist=True, notch=False, return_type='both',  showmeans=True, meanprops=meanpointprops, showfliers=False, flierprops=flierprops, color=dict(boxes='black', whiskers='black', medians='black', caps='black'), boxprops=dict(facecolor="lightgray"))
    props['boxes'][0].set_facecolor("dimgrey")
    props['boxes'][2].set_facecolor("#c13f1a") # hellrot

    bla, props = pd_ToU_dynamic.plot(ax = axs[1, 1], kind="box", widths=0.4, patch_artist=True, return_type='both', notch=False, showmeans=True, meanprops=meanpointprops, showfliers=False, flierprops=flierprops, color=dict(boxes='black', whiskers='black', medians='black', caps='black'), boxprops=dict(facecolor="lightgray"))
    props['boxes'][0].set_facecolor("dimgrey")
    props['boxes'][2].set_facecolor("#0087ff") # hellblau

    
    benchmark_upper_left = pd_standard_static.median().median()
    benchmark_upper_right = pd_standard_dynamic["immediate"].median()
    benchmark_lower_left = pd_ToU_static["immediate"].median()
    benchmark_lower_right = pd_ToU_dynamic["immediate"].median()
    
    # upper left line (red)
    axs[0, 0].axhline(benchmark_upper_left, color="k", linestyle="--", zorder=0)
    axs[0, 1].axhline(benchmark_upper_right, color="k",  linestyle="--", zorder=0)
    axs[1, 0].axhline(benchmark_lower_left, color="k",  linestyle="--", zorder=0)    
    axs[1, 1].axhline(benchmark_lower_right, color="k", linestyle="--", zorder=0)   

    axs[0, 0].text( 1.24, benchmark_upper_left+3, u"\u2199 " + "{:.0f}".format(benchmark_upper_right)+"€", color="black", fontsize="16")
    axs[0, 1].text( 1.24, benchmark_upper_right+3, u"\u2199 " + "{:.0f}".format(benchmark_upper_right)+"€", color="black", fontsize="16") #, backgroundcolor="w")
    axs[1, 0].text( 1.24, benchmark_lower_left+3, u"\u2199 " + "{:.0f}".format(benchmark_lower_left)+"€", color="black", fontsize="16") #, backgroundcolor="w")
    axs[1, 1].text( 1.24, benchmark_lower_right+3, u"\u2199 "  + "{:.0f}".format(benchmark_lower_right)+"€", color="black", fontsize="16") #, backgroundcolor="w")


    
    axs[1, 0].set_xlabel("Static", fontsize=20, fontweight='bold', color="#a6380f") # mean red
    axs[1, 1].set_xlabel("Dynamic", fontsize=20, fontweight='bold', color="#006ac9") # mean blue
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
    
    axs[0, 0].set_xlim(0.7, 3.7)
    axs[0, 1].set_xlim(0.7, 3.7)
    axs[1, 0].set_xlim(0.7, 3.7)
    axs[1, 1].set_xlim(0.7, 3.7)
    
    head_length=5
    
    axs[0, 1].arrow( 2.30, benchmark_upper_right, 0.0, -(benchmark_upper_right-pd_standard_dynamic["scheduled"].median())+head_length, fc="gray", ec="gray", head_width=0.05, head_length=head_length )
    axs[0, 1].text( 2.32, benchmark_upper_right-15, "-"+"{:.0f}".format(benchmark_upper_right-pd_standard_dynamic["scheduled"].median())+"€", color="gray", fontsize="16")
    axs[0, 1].arrow( 3.30, benchmark_upper_right, 0.0, -(benchmark_upper_right-pd_standard_dynamic["smart"].median())+head_length, fc="#004c93", ec="#004c93", head_width=0.05, head_length=head_length )
    axs[0, 1].text( 3.32, benchmark_upper_right-15, "-"+"{:.0f}".format(benchmark_upper_right-pd_standard_dynamic["smart"].median())+"€", color="#004c93", fontsize="16")

    
    axs[1, 0].arrow( 2.30, benchmark_lower_left, 0.0, -(benchmark_lower_left-pd_ToU_static["scheduled"].median())+head_length, fc="gray", ec="gray", head_width=0.05, head_length=head_length )
    axs[1, 0].text( 2.32, benchmark_lower_left-(benchmark_lower_left-pd_ToU_static["scheduled"].median())/2, "-"+"{:.0f}".format(benchmark_lower_left-pd_ToU_static["scheduled"].median())+"€", color="gray", fontsize="16")
    axs[1, 0].arrow( 3.30, benchmark_lower_left, 0.0, -(benchmark_lower_left-pd_ToU_static["smart"].median())+head_length, fc="#c13f1a", ec="#c13f1a", head_width=0.05, head_length=head_length )
    axs[1, 0].text( 3.32, benchmark_lower_left-(benchmark_lower_left-pd_ToU_static["smart"].median())/2, "-"+"{:.0f}".format(benchmark_lower_left-pd_ToU_static["smart"].median())+"€", color="#c13f1a", fontsize="16")

    
    axs[1, 1].arrow( 2.30, benchmark_lower_right, 0.0, -(benchmark_lower_right-pd_ToU_dynamic["scheduled"].median())+head_length, fc="gray", ec="gray", head_width=0.05, head_length=head_length )
    axs[1, 1].text( 2.32, benchmark_lower_right-(benchmark_lower_right-pd_ToU_dynamic["scheduled"].median())/2, "-"+"{:.0f}".format(benchmark_lower_right-pd_ToU_dynamic["scheduled"].median())+"€", color="gray", fontsize="16")
    axs[1, 1].arrow( 3.30, benchmark_lower_right, 0.0, -(benchmark_lower_right-pd_ToU_dynamic["smart"].median())+head_length, fc="#0087ff", ec="#0087ff", head_width=0.05, head_length=head_length )
    axs[1, 1].text( 3.32, benchmark_lower_right-(benchmark_lower_right-pd_ToU_dynamic["smart"].median())/2, "-"+"{:.0f}".format(benchmark_lower_right-pd_ToU_dynamic["smart"].median())+"€", color="#0087ff", fontsize="16")

    


    ytickvals = np.linspace(0,int(axis_y_max/25)*25,int(axis_y_max/25+1)).astype(int)

    axs[0,0].set_yticks(ytickvals)
    axs[0,0].set_yticklabels([str(y)+"€" if y%50==0 else " " for y in ytickvals], fontsize=20)
    axs[0,1].set_yticks(ytickvals)
    axs[0,1].set_yticklabels([str(y)+"€"  if y%50==0 else " " for y in ytickvals], fontsize=20)
    axs[1,0].set_yticks(ytickvals)
    axs[1,0].set_yticklabels([str(y)+"€"  if y%50==0 else " " for y in ytickvals], fontsize=20)
    axs[1,1].set_yticks(ytickvals)
    axs[1,1].set_yticklabels([str(y)+"€"  if y%50==0 else " " for y in ytickvals], fontsize=20)

    str_xticks = ["\xa0       immediate", "\xa0      scheduled", "\xa0      smart"]
    axs[0,0].set_xticklabels(str_xticks, fontsize=20) # fontdict={'horizontalalignment':"left"}
    axs[0,1].set_xticklabels(str_xticks, fontsize=20) # fontdict={'horizontalalignment':"left"}
    axs[1,0].set_xticklabels(str_xticks, fontsize=20) # fontdict={'horizontalalignment':"left"}
    axs[1,1].set_xticklabels(str_xticks, fontsize=20) # fontdict={'horizontalalignment':"left"}





    fig_grouped_boxplots_cost_savings.supxlabel("(b) Electricity price", fontsize=20, fontweight='bold')
    fig_grouped_boxplots_cost_savings.supylabel("(a) Network charge", fontsize=20, fontweight='bold')

    for ax in axs.flat:
        ax.xaxis.grid(False)
        ax.yaxis.grid(True, linestyle="--", color="lightgray", zorder=0)
       
    plt.tight_layout()
    plt.show()
    
       
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
    
    #mosaic_layout = [[3, "Winter"],  [1, 2]]
    #fig_kw_savings, axs_kw_savings = plt.subplot_mosaic(mosaic_layout, figsize=(15, 8))   
    fig_kw_savings, axs_kw_savings = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={'height_ratios': [1, 1]})
    #fig_kw_savings.suptitle("Impact of EV Charge Strategy on Power Consumption", fontsize=20)

    
    # pd_day = pd_ct.groupby(["hour decimal"]).mean()
    id_season = (pd_ct.index.month>=1) & (pd_ct.index.month<=3)
    pd_day = pd_ct.loc[id_season,:].groupby(["hour decimal"]).mean()

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



    axs_kw_savings[0,0].plot(pd_charge_mode["immediate"], linestyle="-", color="dimgrey", linewidth=1, zorder=2, label="immediate")
    axs_kw_savings[0,0].plot(pd_charge_mode["scheduled"], linestyle="--", color="gray", linewidth=1, zorder=2, label=  "scheduled")

    axs_kw_savings[0,0].plot(pd_charge_mode["smart (static, standard)"], linestyle="-", color="#8b3003", linewidth=1, zorder=1, label="smart (static, standard)")
    axs_kw_savings[0,0].plot(pd_charge_mode["smart (static, ToU)"], linestyle="--", color="#c13f1a", linewidth=1, zorder=1, label= "smart (static, ToU)")

    axs_kw_savings[0,0].plot(pd_charge_mode["smart (dynamic, standard)"], linestyle="-", color="#004c93", linewidth=1, zorder=1, label=  "smart (dynamic, standard)")
    axs_kw_savings[0,0].plot(pd_charge_mode["smart (dynamic, ToU)"], linestyle="--", color="#0087ff", linewidth=1, zorder=1, label= "smart (dynamic, ToU)")


    axs_kw_savings[0,0].set_title("(a) January, February, March", fontsize=20)
    #axs_kw_savings["Winter"].legend(fontsize=16, ncols=1, loc="upper right", bbox_to_anchor=(0.9,0.98))
    axs_kw_savings[0,0].grid(color='lightgray', linestyle='--', linewidth=1, axis="both", zorder=0)
    axs_kw_savings[0,0].set_xticks(np.array([0, 3, 6, 9, 12, 15, 18, 21, 24]))
    axs_kw_savings[0,0].set_xticklabels([0, 3, 6, 9, 12, 15, 18, 21, 24], fontsize=20)

    axs_kw_savings[0,0].set_yticks(np.array([0, 0.4, 0.8]))
    axs_kw_savings[0,0].set_yticklabels([0, 0.4, 0.8], fontsize=20)
    axs_kw_savings[0,0].set_ylim(-.1, 1)
    
    axs_kw_savings[0,0].set_ylabel("Mean power in kW", fontsize=20)
    axs_kw_savings[0,0].tick_params(axis='y', labelsize=20)
    axs_kw_savings[0,0].set_xlabel("Hour of the day", fontsize=20)







    
    
    
    # ==== SPRING =====
    id_season = (pd_ct.index.month>=4) & (pd_ct.index.month<=6)
    pd_day = pd_ct.loc[id_season,:].groupby(["hour decimal"]).mean()
    
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
    
    axs_kw_savings[0,1].plot(pd_charge_mode["immediate"], linestyle="-", color="dimgrey", linewidth=1, zorder=2, label="immediate")
    axs_kw_savings[0,1].plot(pd_charge_mode["scheduled"], linestyle="--", color="gray", linewidth=1, zorder=2, label=  "scheduled")
    
    axs_kw_savings[0,1].plot(pd_charge_mode["smart (static, standard)"], linestyle="-", color="#8b3003", linewidth=1, zorder=1, label="smart (static, standard)")
    axs_kw_savings[0,1].plot(pd_charge_mode["smart (static, ToU)"], linestyle="--", color="#c13f1a", linewidth=1, zorder=1, label= "smart (static, ToU)")
    
    axs_kw_savings[0,1].plot(pd_charge_mode["smart (dynamic, standard)"], linestyle="-", color="#004c93", linewidth=1, zorder=1, label=  "smart (dynamic, standard)")
    axs_kw_savings[0,1].plot(pd_charge_mode["smart (dynamic, ToU)"], linestyle="--", color="#0087ff", linewidth=1, zorder=1, label= "smart (dynamic, ToU)")
    
    
    axs_kw_savings[0,1].set_title("(b) April, May, June", fontsize=20)
    #axs_kw_savings[1].legend(fontsize=16, ncols=1, loc="upper right", bbox_to_anchor=(0.9,0.98))
    axs_kw_savings[0,1].grid(color='lightgray', linestyle='--', linewidth=1, axis="both", zorder=0)
    axs_kw_savings[0,1].set_xticks(np.array([0, 3, 6, 9, 12, 15, 18, 21, 24]))
    axs_kw_savings[0,1].set_xticklabels([0, 3, 6, 9, 12, 15, 18, 21, 24], fontsize=20)

    axs_kw_savings[0,1].set_yticks(np.array([0, 0.4, 0.8]))
    axs_kw_savings[0,1].set_yticklabels([0, 0.4, 0.8], fontsize=20)
    axs_kw_savings[0,1].set_ylim(-.1, 1)    

    axs_kw_savings[0,1].set_ylabel("Mean power in kW", fontsize=20)
    axs_kw_savings[0,1].tick_params(axis='y', labelsize=20)
    axs_kw_savings[0,1].set_xlabel("Hour of the day", fontsize=20)


    
    # ==== Summer =====
    id_season = (pd_ct.index.month>=7) & (pd_ct.index.month<=9)
    pd_day = pd_ct.loc[id_season,:].groupby(["hour decimal"]).mean()
    
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
    
    axs_kw_savings[1,0].plot(pd_charge_mode["immediate"], linestyle="-", color="dimgrey", linewidth=1, zorder=2, label="immediate")
    axs_kw_savings[1,0].plot(pd_charge_mode["scheduled"], linestyle="--", color="gray", linewidth=1, zorder=2, label=  "scheduled")
    
    axs_kw_savings[1,0].plot(pd_charge_mode["smart (static, standard)"], linestyle="-", color="#8b3003", linewidth=1, zorder=1, label="smart (static, standard)")
    axs_kw_savings[1,0].plot(pd_charge_mode["smart (static, ToU)"], linestyle="--", color="#c13f1a", linewidth=1, zorder=1, label= "smart (static, ToU)")

    axs_kw_savings[1,0].plot(pd_charge_mode["smart (dynamic, standard)"], linestyle="-", color="#004c93", linewidth=1, zorder=1, label=  "smart (dynamic, standard)")
    axs_kw_savings[1,0].plot(pd_charge_mode["smart (dynamic, ToU)"], linestyle="--", color="#0087ff", linewidth=1, zorder=1, label= "smart (dynamic, ToU)")
    
    
    axs_kw_savings[1,0].set_title("(c) July, August, September", fontsize=20)
    #axs_kw_savings[2].legend(fontsize=16, ncols=1, loc="upper right", bbox_to_anchor=(0.9,0.98))
    axs_kw_savings[1,0].grid(color='lightgray', linestyle='--', linewidth=1, axis="both", zorder=0)
    axs_kw_savings[1,0].set_xticks(np.array([0, 3, 6, 9, 12, 15, 18, 21, 24]))
    axs_kw_savings[1,0].set_xticklabels([0, 3, 6, 9, 12, 15, 18, 21, 24], fontsize=20)
    
    axs_kw_savings[1,0].set_yticks(np.array([0, 0.4, 0.8]))
    axs_kw_savings[1,0].set_yticklabels([0, 0.4, 0.8], fontsize=20)
    axs_kw_savings[1,0].set_ylim(-.1, 1)
    
    axs_kw_savings[1,0].set_ylabel("Mean power in kW", fontsize=20)
    axs_kw_savings[1,0].tick_params(axis='y', labelsize=20)
    axs_kw_savings[1,0].set_xlabel("Hour of the day", fontsize=20)


    
    # ==== Autumn =====
    id_season = (pd_ct.index.month>=10) & (pd_ct.index.month<=12)
    pd_day = pd_ct.loc[id_season,:].groupby(["hour decimal"]).mean()
    
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
    
    axs_kw_savings[1,1].plot(pd_charge_mode["immediate"], linestyle="-", color="dimgrey", linewidth=1, zorder=2, label="immediate")
    axs_kw_savings[1,1].plot(pd_charge_mode["scheduled"], linestyle="--", color="gray", linewidth=1, zorder=2, label=  "scheduled")
    
    axs_kw_savings[1,1].plot(pd_charge_mode["smart (static, standard)"], linestyle="-", color="#8b3003", linewidth=1, zorder=1, label="smart (static, standard)")
    axs_kw_savings[1,1].plot(pd_charge_mode["smart (static, ToU)"], linestyle="--", color="#c13f1a", linewidth=1, zorder=1, label= "smart (static, ToU)")
    
    axs_kw_savings[1,1].plot(pd_charge_mode["smart (dynamic, standard)"], linestyle="-", color="#004c93", linewidth=1, zorder=1, label=  "smart (dynamic, standard)")
    axs_kw_savings[1,1].plot(pd_charge_mode["smart (dynamic, ToU)"], linestyle="--", color="#0087ff", linewidth=1, zorder=1, label= "smart (dynamic, ToU)")
    
    
    axs_kw_savings[1,1].set_title("(d) October, November, December", fontsize=20)
    #axs_kw_savings[3].legend(fontsize=16, ncols=1, loc="upper right", bbox_to_anchor=(0.9,0.98))
    axs_kw_savings[1,1].grid(color='lightgray', linestyle='--', linewidth=1, axis="both", zorder=0)
    axs_kw_savings[1,1].set_xticks(np.array([0, 3, 6, 9, 12, 15, 18, 21, 24]))
    axs_kw_savings[1,1].set_xticklabels([0, 3, 6, 9, 12, 15, 18, 21, 24], fontsize=20)
    
    axs_kw_savings[1,1].set_yticks(np.array([0, 0.4, 0.8, 1.2]))
    axs_kw_savings[1,1].set_yticklabels([0, 0.4, 0.8, 1.2], fontsize=20)
    axs_kw_savings[1,1].set_ylim(-.1, 1.3)
    
    axs_kw_savings[1,1].set_ylabel("Mean power in kW", fontsize=20)
    axs_kw_savings[1,1].tick_params(axis='y', labelsize=20)
    axs_kw_savings[1,1].set_xlabel("Hour of the day", fontsize=20)

    #fig_kw_savings.legend(axs_kw_savings["Winter"].get_legend_handles_labels())
    lines = axs_kw_savings[0,0].get_legend_handles_labels()
    #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    axbox = axs_kw_savings[1,1].get_position()

    

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    plt.legend(lines[0], lines[1], loc = 'lower center', fontsize=16, ncols=3, bbox_to_anchor=[0, axbox.y0-0.1,1,1], bbox_transform=fig_kw_savings.transFigure)

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


# ========================
# Quantile Plot
# ========================

bla = xr.open_dataarray(spot_only_charge + variable_file).sel(s="red")
bla = bla.drop_vars('s')
bla2 = bla.sum("v") / (50*11)
#bla2 = bla.stack(new_dim=('r', 'v'))
#bla2 = bla2.drop_vars('v')

multi_index = pd.MultiIndex.from_arrays(
        [
            dti.date,
            dti.hour + dti.minute / 60,
        ],
        names=["d", "qh"],
    )
bla2["t"] = multi_index
bla3 = bla2.drop_duplicates('t').unstack("t")



bla3 = bla3.isel(r=1) # nur westnetz
pd_bla_quantile = bla3.quantile(q=[1, 0.99, 0.95, 0.9, 0.8], dim="d").to_pandas().transpose()

# Simultanes Laden
#bla4 = bla3.to_pandas()
#bla5 = 1*(bla4 > 0)
#pd_bla_quantile = bla5.quantile(q=[1, 0.99, 0.95, 0.9, 0.8]).transpose()


df = pd_bla_quantile  # Form: Zeilen = qh, Spalten = Quantile (z.B. 0.90, 0.95, 0.98, 0.99)

# Spalten nach ihrem Quantilwert aufsteigend sortieren: niedrigstes Quantil unten, höchstes oben
cols = sorted(df.columns, key=float)
x = df.index.values  # qh-Werte


fig, ax = plt.subplots()

# Alpha-Werte: unten voll deckend, nach oben zunehmend transparenter
n = len(cols)
alphas = np.linspace(1.0, 0.2, n)  # z.B. linear von 1.0 (unten) zu 0.2 (oben)

y_prev = np.zeros_like(x, dtype=float)
for k, col in enumerate(cols):
    y = df[col].values
    # Fläche zwischen vorheriger und aktueller Linie
    ax.fill_between(x, y_prev, y, color='blue', alpha=alphas[k], linewidth=0)
    # Schwarze Linie mit 0.5 Breite
    ax.plot(x, y, color='black', linewidth=0.5)
    y_prev = y

ax.set_xlabel('qh')
ax.set_ylabel('solution')
from matplotlib.patches import Patch
r, g, b = mcolors.to_rgb('blue')
handles = [
    Patch(facecolor=(r, g, b, alphas[k]), edgecolor='black', linewidth=0.5, label=f"q = {cols[k]:g}")
    for k in range(n)
]
handles = handles[::-1]
ax.legend(handles=handles, title='Quantile', loc='best')

plt.show()

# ====================================
# Plots Simultane Nutzung
# ===================================


import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# Daten öffnen und price-Dimension hinzufügen
spot_da = xr.open_dataarray(spot_only_charge + variable_file)
mean_da = xr.open_dataarray(mean_only_charge + variable_file)

charges = xr.concat(
    [mean_da.expand_dims(price=["mean"]),
     spot_da.expand_dims(price=["spot"])],
    dim="price",
)

# Summe über v und Normierung
charges2 = charges.sum("v") / (50 * 11)

# Zeit in (d, qh) aufspalten
# dti muss ein pandas.DatetimeIndex sein, passend zu charges2["t"]
# Beispiel: dti = pd.to_datetime(charges2["t"].to_pandas())
multi_index = pd.MultiIndex.from_arrays(
    [dti.date, dti.hour + dti.minute / 60.0],
    names=["d", "qh"],
)
charges2 = charges2.assign_coords(t=("t", multi_index))
charges3 = charges2.drop_duplicates("t").unstack("t")

# Netz auswählen (wie zuvor)
charges4 = charges3.isel(r=1)

# Quantile über d berechnen (pro s und price separat)
quant = charges4.quantile(q=[1, 0.99, 0.98, 0.95, 0.9, 0.8], dim="d")  # dims: quantile, qh, s, price

# Plot-Hilfsfunktion: Schattierung + schwarze Linien + Legende mit Kästchen
def plot_quantile_panel(ax, da_panel, facecolor_hex):
    # da_panel: DataArray mit Dimensionen quantile, qh (eine s, eine price)
    df = da_panel.to_pandas().transpose()  # Zeilen = qh, Spalten = Quantile
    cols = sorted(df.columns, key=float)   # unten niedrigstes, oben höchstes
    x = df.index.values

    # gewünschte X-Ticks (0,3,...,24)
    ticks = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24])
    ax.set_xlim(0, 24)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=20)

    ticks = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax.set_yticks(ticks)
    ax.set_yticklabels(["0%", "", "20%", "", "40%", "", "60%"], fontsize=20)

    # Grid
    ax.grid(color='lightgray', linestyle='--', linewidth=1, axis='both')

    # Alphas: unten deckend, nach oben transparenter
    n = len(cols)
    alphas = np.linspace(1.0, 0.2, n)

    # Schattierung + schwarze Linien
    y_prev = np.zeros_like(x, dtype=float)
    for k, col in enumerate(cols):
        y = df[col].values
        ax.fill_between(x, y_prev, y, color=facecolor_hex, alpha=alphas[k], linewidth=0)
        ax.plot(x, y, color="black", linewidth=0.5)
        y_prev = y

    # Legende: Kästchen mit gleicher Panel-Farbe, Rand schwarz ohne Transparenz
    r, g, b = mcolors.to_rgb(facecolor_hex)
    legend_cols = list(reversed(cols))
    legend_alphas = list(reversed(alphas))
    handles = [
        Patch(facecolor=(r, g, b, legend_alphas[i]),
              edgecolor="black", linewidth=0.5,
              label=f"q = {legend_cols[i]:g}")
        for i in range(n)
    ]
    ax.legend(handles=handles, title="Quantile", loc="upper center", ncol=2, fontsize=14, title_fontsize=14)
    ax.set_xlabel("Time in hours", fontsize=20)
    ax.set_ylabel("Energy (relative)", fontsize=20)




# 2×2 Subplot: Zeilen = s (reg oben, red unten), Spalten = price (mean links, spot rechts)
fig_parallel, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 8)) #sharex=True, sharey=True

# Farben pro Panel
panel_colors = {
    (0, 0): "#8b3003",  # oben links (reg, mean)
    (0, 1): "#004c93",  # oben rechts (reg, spot)
    (1, 0): "#c13f1a",  # unten links (red, mean)
    (1, 1): "#0087ff",  # unten rechts (red, spot)
}

# Panels zeichnen
plot_quantile_panel(axs[0, 0], quant.sel(price="mean", s="reg"), panel_colors[(0, 0)])
axs[0, 0].legend_.set_bbox_to_anchor((0.35, 1.0))
plot_quantile_panel(axs[0, 1], quant.sel(price="spot", s="reg"), panel_colors[(0, 1)])
axs[0, 1].legend_.set_bbox_to_anchor((0.62, 1.0))
plot_quantile_panel(axs[1, 0], quant.sel(price="mean", s="red"), panel_colors[(1, 0)])
axs[1, 0].legend_.set_bbox_to_anchor((0.59, 1.0))
plot_quantile_panel(axs[1, 1], quant.sel(price="spot", s="red"), panel_colors[(1, 1)])
axs[1, 1].legend_.set_bbox_to_anchor((0.59, 1.0))

fig_parallel.text(0.04, 0.76, "Standard", fontsize=20, fontweight='bold', rotation=90, va='center')
fig_parallel.text(0.04, 0.33, "Time of use", fontsize=20, fontweight='bold', rotation=90, va='center')

fig_parallel.text(0.33, 0.06, "Static", fontsize=20, fontweight='bold', ha='center', color="#a6380f")
fig_parallel.text(0.79, 0.06, "Dynamic", fontsize=20, fontweight='bold', ha='center',  color="#006ac9")

# Gemeinsamer Titel mit Superscript und griechischem Delta
fig_parallel.suptitle(r"Energy consumption relative to all vehicles fully charging", fontsize=20)
fig_parallel.supxlabel("(b) Electricity price", fontsize=20, fontweight='bold')
fig_parallel.supylabel("(a) Network charge", fontsize=20, fontweight='bold')
fig_parallel.tight_layout(rect=[0.03, 0.02, 1, 1])
fig_parallel.show()
fig_parallel.savefig(r"C:\Users\Hendrik.Kramer\Documents\GitHub\ToU_network_charges\daten_results\simultenous_power_quantile.svg")

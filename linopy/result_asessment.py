# use linopti environment
# conda activate linopti
# in Spyder: right click on tab main_script.py --> set console working directory

# %matplotlib qt

# load first section of "main_script.py"
# to obtain variable "parameters_opti"

import sys
print(sys.executable)

import xarray as xr
import pandas as pd
import numpy as np
import xarray as xr

from datetime import date, timedelta, datetime
from pathlib import Path
from linopy import Model

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerPatch
import matplotlib.lines as mlines

import functions_tariff_network_charge_study.load_functions as f_load



# read  results
folder_name = "2026-09-12_15-39_mean_immediate_only_EV"


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
    folder_str = r"C:\Users\Hendrik.Kramer\Documents\Repos\ToU_network_charges\daten_results" + r"\\"
    
    # scheduled charge EV
    immediate_mean_only_charge_str = r"2026-09-16_22-19_mean_immediate_only_EV" + r"\\"
    immediate_spot_only_charge_str = r"2026-09-17_21-32_spot_immediate_only_EV" + r"\\" 
    scheduled_mean_only_charge_str = r"2026-09-17_20-39_mean_scheduled_only_EV" + r"\\"
    scheduled_spot_only_charge_str = r"2026-09-18_09-41_spot_scheduled_only_EV" + r"\\"  
    smart_mean_only_charge_str =   r"2026-09-16_13-11_mean_smart_only_EV" + r"\\"
    smart_spot_only_charge_str =  r"2026-09-17_07-17_spot_smart_only_EV" + r"\\" #r"2025-11-15_15-33_spot_smart_only_EV_r100_v50" + r"\\"
    
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
    axis_y_max = 725
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

    
    benchmark_upper_left = pd_standard_static["immediate"].median()
    benchmark_upper_right = pd_standard_dynamic["immediate"].median()
    benchmark_lower_left = pd_ToU_static["immediate"].median()
    benchmark_lower_right = pd_ToU_dynamic["immediate"].median()
    
    # upper left line (red)
    axs[0, 0].axhline(benchmark_upper_left, color="k", linestyle="--", zorder=0)
    axs[0, 1].axhline(benchmark_upper_right, color="k",  linestyle="--", zorder=0)
    axs[1, 0].axhline(benchmark_lower_left, color="k",  linestyle="--", zorder=0)    
    axs[1, 1].axhline(benchmark_lower_right, color="k", linestyle="--", zorder=0)   

    axs[0, 0].text( 1.22, benchmark_upper_left+15, u"\u2199 " + "{:.0f}".format(benchmark_upper_left)+"€", color="black", fontsize="16")
    axs[0, 1].text( 1.22, benchmark_upper_right+15, u"\u2199 " + "{:.0f}".format(benchmark_upper_right)+"€", color="black", fontsize="16") #, backgroundcolor="w")
    axs[1, 0].text( 1.22, benchmark_lower_left+15, u"\u2199 " + "{:.0f}".format(benchmark_lower_left)+"€", color="black", fontsize="16") #, backgroundcolor="w")
    axs[1, 1].text( 1.22, benchmark_lower_right+15, u"\u2199 "  + "{:.0f}".format(benchmark_lower_right)+"€", color="black", fontsize="16") #, backgroundcolor="w")


    
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
    
    head_length=12
    
    axs[0, 1].arrow( 2.30, benchmark_upper_right, 0.0, -(benchmark_upper_right-pd_standard_dynamic["scheduled"].median())+head_length, fc="gray", ec="gray", head_width=0.05, head_length=head_length, zorder=3  )
    axs[0, 1].text( 2.32, benchmark_upper_right-60, "-"+"{:.0f}".format(benchmark_upper_right-pd_standard_dynamic["scheduled"].median())+"€", color="gray", fontsize="16")
    axs[0, 1].arrow( 3.30, benchmark_upper_right, 0.0, -(benchmark_upper_right-pd_standard_dynamic["smart"].median())+head_length, fc="#004c93", ec="#004c93", head_width=0.05, head_length=head_length, zorder=3  )
    axs[0, 1].text( 3.32, benchmark_upper_right-60, "-"+"{:.0f}".format(benchmark_upper_right-pd_standard_dynamic["smart"].median())+"€", color="#004c93", fontsize="16")

    
    axs[1, 0].arrow( 2.30, benchmark_lower_left, 0.0, -(benchmark_lower_left-pd_ToU_static["scheduled"].median())+head_length, fc="gray", ec="gray", head_width=0.05, head_length=head_length, zorder=3  )
    axs[1, 0].text( 2.32, benchmark_lower_left-(benchmark_lower_left-pd_ToU_static["scheduled"].median())/2, "-"+"{:.0f}".format(benchmark_lower_left-pd_ToU_static["scheduled"].median())+"€", color="gray", fontsize="16")
    axs[1, 0].arrow( 3.30, benchmark_lower_left, 0.0, -(benchmark_lower_left-pd_ToU_static["smart"].median())+head_length, fc="#c13f1a", ec="#c13f1a", head_width=0.05, head_length=head_length, zorder=3  )
    axs[1, 0].text( 3.32, benchmark_lower_left-(benchmark_lower_left-pd_ToU_static["smart"].median())/2, "-"+"{:.0f}".format(benchmark_lower_left-pd_ToU_static["smart"].median())+"€", color="#c13f1a", fontsize="16")

    
    axs[1, 1].arrow( 2.30, benchmark_lower_right, 0.0, -(benchmark_lower_right-pd_ToU_dynamic["scheduled"].median())+head_length, fc="gray", ec="gray", head_width=0.05, head_length=head_length, zorder=3  )
    axs[1, 1].text( 2.32, benchmark_lower_right-(benchmark_lower_right-pd_ToU_dynamic["scheduled"].median())/2, "-"+"{:.0f}".format(benchmark_lower_right-pd_ToU_dynamic["scheduled"].median())+"€", color="gray", fontsize="16")
    axs[1, 1].arrow( 3.30, benchmark_lower_right, 0.0, -(benchmark_lower_right-pd_ToU_dynamic["smart"].median())+head_length, fc="#0087ff", ec="#0087ff", head_width=0.05, head_length=head_length, zorder=3 )
    axs[1, 1].text( 3.32, benchmark_lower_right-(benchmark_lower_right-pd_ToU_dynamic["smart"].median())/2, "-"+"{:.0f}".format(benchmark_lower_right-pd_ToU_dynamic["smart"].median())+"€", color="#0087ff", fontsize="16")

    


    ytickvals = np.linspace(0,int(axis_y_max/50)*50,int(axis_y_max/50+1)).astype(int)

    axs[0,0].set_yticks(ytickvals)
    axs[0,0].set_yticklabels([str(y)+"€" if y%100==0 else " " for y in ytickvals], fontsize=20)
    axs[0,1].set_yticks(ytickvals)
    axs[0,1].set_yticklabels([str(y)+"€"  if y%100==0 else " " for y in ytickvals], fontsize=20)
    axs[1,0].set_yticks(ytickvals)
    axs[1,0].set_yticklabels([str(y)+"€"  if y%100==0 else " " for y in ytickvals], fontsize=20)
    axs[1,1].set_yticks(ytickvals)
    axs[1,1].set_yticklabels([str(y)+"€"  if y%100==0 else " " for y in ytickvals], fontsize=20)

    str_xticks = ["\xa0       Immediate", "\xa0      Scheduled", "\xa0      Smart"]
    axs[0,0].set_xticklabels(str_xticks, fontsize=20) # fontdict={'horizontalalignment':"left"}
    axs[0,1].set_xticklabels(str_xticks, fontsize=20) # fontdict={'horizontalalignment':"left"}
    axs[1,0].set_xticklabels(str_xticks, fontsize=20) # fontdict={'horizontalalignment':"left"}
    axs[1,1].set_xticklabels(str_xticks, fontsize=20) # fontdict={'horizontalalignment':"left"}





    fig_grouped_boxplots_cost_savings.supxlabel("Electricity Price", fontsize=20, fontweight='bold')
    fig_grouped_boxplots_cost_savings.supylabel("Network Charge", fontsize=20, fontweight='bold')

    for ax in axs.flat:
        ax.xaxis.grid(False)
        ax.yaxis.grid(True, linestyle="--", color="lightgray", zorder=0)
       
    plt.tight_layout()
    plt.show()
    
       
    fig_grouped_boxplots_cost_savings.savefig(r"C:\Users\Hendrik.Kramer\Documents\Repos\ToU_network_charges\daten_results\annual_cost_end_consumer.svg")

    
    


    
# =============================================================================
# CHARGE POWER  
# kW reduction plots
# =============================================================================

if (False):
    
    variable_file = "P_HOME.nc"  # "P_PUBLIC.nc"
    
    epoch_time = datetime(1970, 1, 1)

    folder_str = r"C:\Users\Hendrik.Kramer\Documents\Repos\ToU_network_charges\daten_results" + r"\\"
        
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
    
    
    fig_kw_savings, (axs_kw_savings_winter, axs_kw_savings_summer) = plt.subplots(1, 2, figsize=(15, 6)) #gridspec_kw={'height_ratios': [1, 1]}
    
    # === WINTER MONTH === october - march (both included)
    id_season = (pd_ct.index.month>=10) | (pd_ct.index.month<=3)
    pd_day = pd_ct.loc[id_season,:].groupby(["hour decimal"]).mean()

    # linker plot
    pd_charge_mode = pd.concat([ pd_day["immediate_mean_static_standard"], 
                                    pd_day["scheduled_mean_static_standard"],
                                    pd_day["smart_mean_static_standard"],
                                    pd_day["smart_mean_ToU_standard"],
                                    pd_day["smart_spot_static_standard"],
                                    pd_day["smart_spot_ToU_standard"] ],
                                    
                                    axis = 1).rename(columns={"immediate_mean_static_standard":"Immediate",
                                                              "scheduled_mean_static_standard":"Scheduled", 
                                                              "smart_mean_static_standard":"Smart (static, standard)", 
                                                              "smart_mean_ToU_standard":"Smart (static, ToU)",
                                                              "smart_spot_static_standard":"Smart (dynamic, standard)",
                                                              "smart_spot_ToU_standard":"Smart (dynamic, ToU)" })



    axs_kw_savings_winter.plot(pd_charge_mode["Immediate"], linestyle="-", color="dimgrey", linewidth=1, zorder=2, label="Immediate")
    axs_kw_savings_winter.plot(pd_charge_mode["Scheduled"], linestyle="--", color="gray", linewidth=1, zorder=2, label=  "Scheduled")

    axs_kw_savings_winter.plot(pd_charge_mode["Smart (static, standard)"], linestyle="-", color="#8b3003", linewidth=1, zorder=1, label="Smart (Static, Standard)")
    axs_kw_savings_winter.plot(pd_charge_mode["Smart (static, ToU)"], linestyle="--", color="#c13f1a", linewidth=1, zorder=1, label= "Smart (Static, ToU)")

    axs_kw_savings_winter.plot(pd_charge_mode["Smart (dynamic, standard)"], linestyle="-", color="#004c93", linewidth=1, zorder=1, label=  "Smart (Dynamic, Standard)")
    axs_kw_savings_winter.plot(pd_charge_mode["Smart (dynamic, ToU)"], linestyle="--", color="#0087ff", linewidth=1, zorder=1, label= "Smart (Dynamic, ToU)")


    axs_kw_savings_winter.set_title("Winter (October-March)", fontsize=20)
    #axs_kw_savings["Winter"].legend(fontsize=16, ncols=1, loc="upper right", bbox_to_anchor=(0.9,0.98))
    axs_kw_savings_winter.grid(color='lightgray', linestyle='--', linewidth=1, axis="both", zorder=0)
    axs_kw_savings_winter.set_xticks(np.array([0, 3, 6, 9, 12, 15, 18, 21, 24]))
    axs_kw_savings_winter.set_xticklabels([0, 3, 6, 9, 12, 15, 18, 21, 24], fontsize=20)
    axs_kw_savings_winter.set_xlim(-0.3, 24.3)

    yticks = np.arange(0, 3.5, 0.25)
    axs_kw_savings_winter.set_yticks(yticks)
    axs_kw_savings_winter.set_yticklabels( [f'{t:.1f}' if i % 2 == 0 else '' for i, t in enumerate(yticks)], fontsize=20)
    axs_kw_savings_winter.set_ylim(-.1, 3.35)
    
    axs_kw_savings_winter.set_ylabel("Mean Power in kW", fontsize=20)
    axs_kw_savings_winter.tick_params(axis='y', labelsize=20)
    axs_kw_savings_winter.set_xlabel("Hour of the Day", fontsize=20)


    
    
    # ==== SUMMER =====
    id_season = (pd_ct.index.month>=4) & (pd_ct.index.month<=9)
    pd_day = pd_ct.loc[id_season,:].groupby(["hour decimal"]).mean()
    
    pd_charge_mode = pd.concat([ pd_day["immediate_mean_static_standard"], 
                                    pd_day["scheduled_mean_static_standard"],
                                    pd_day["smart_mean_static_standard"],
                                    pd_day["smart_mean_ToU_standard"],
                                    pd_day["smart_spot_static_standard"],
                                    pd_day["smart_spot_ToU_standard"] ],
                                    
                                    axis = 1).rename(columns={"immediate_mean_static_standard":"Immediate",
                                                              "scheduled_mean_static_standard":"Scheduled", 
                                                              "smart_mean_static_standard":"Smart (static, standard)", 
                                                              "smart_mean_ToU_standard":"Smart (static, ToU)",
                                                              "smart_spot_static_standard":"Smart (dynamic, standard)",
                                                              "smart_spot_ToU_standard":"Smart (dynamic, ToU)" })
    
    axs_kw_savings_summer.plot(pd_charge_mode["Immediate"], linestyle="-", color="dimgrey", linewidth=1, zorder=2, label="Immediate")
    axs_kw_savings_summer.plot(pd_charge_mode["Scheduled"], linestyle="--", color="gray", linewidth=1, zorder=2, label=  "Scheduled")
    
    axs_kw_savings_summer.plot(pd_charge_mode["Smart (static, standard)"], linestyle="-", color="#8b3003", linewidth=1, zorder=1, label="Smart (Static, Standard)")
    axs_kw_savings_summer.plot(pd_charge_mode["Smart (static, ToU)"], linestyle="--", color="#c13f1a", linewidth=1, zorder=1, label= "Smart (Static, ToU)")
    
    axs_kw_savings_summer.plot(pd_charge_mode["Smart (dynamic, standard)"], linestyle="-", color="#004c93", linewidth=1, zorder=1, label=  "Smart (Dynamic, Standard)")
    axs_kw_savings_summer.plot(pd_charge_mode["Smart (dynamic, ToU)"], linestyle="--", color="#0087ff", linewidth=1, zorder=1, label= "Smart (Dynamic, ToU)")
    
    
    axs_kw_savings_summer.set_title("Summer (April-September)", fontsize=20)
    #axs_kw_savings[1].legend(fontsize=16, ncols=1, loc="upper right", bbox_to_anchor=(0.9,0.98))
    axs_kw_savings_summer.grid(color='lightgray', linestyle='--', linewidth=1, axis="both", zorder=0)
    axs_kw_savings_summer.set_xticks(np.array([0, 3, 6, 9, 12, 15, 18, 21, 24]))
    axs_kw_savings_summer.set_xticklabels([0, 3, 6, 9, 12, 15, 18, 21, 24], fontsize=20)
    axs_kw_savings_summer.set_xlim(-0.3, 24.3)

    yticks = np.arange(0, 3.5, 0.25)
    axs_kw_savings_summer.set_yticks(yticks)
    axs_kw_savings_summer.set_yticklabels( [f'{t:.1f}' if i % 2 == 0 else '' for i, t in enumerate(yticks)], fontsize=20)
    axs_kw_savings_summer.set_ylim(-.1, 3.35)   

    #axs_kw_savings_summer.set_ylabel("Mean Power in kW", fontsize=20)
    axs_kw_savings_summer.tick_params(axis='y', labelsize=20)
    axs_kw_savings_summer.set_xlabel("Hour of the Day", fontsize=20)


    axbox = axs_kw_savings_summer.get_position()


    plt.tight_layout(rect=[0, 0.12, 1, 1])
    
    lines = axs_kw_savings_winter.get_legend_handles_labels()
    plt.legend(lines[0], lines[1], loc = 'lower center', fontsize=16, ncols=3, bbox_to_anchor=[0, axbox.y0-0.12,1,1], bbox_transform=fig_kw_savings.transFigure)

    plt.show()
    
    
    
    fig_kw_savings.savefig(r"C:\Users\Hendrik.Kramer\Documents\Repos\ToU_network_charges\daten_results\power_consumption_winter_summer.svg")





# ====================================
# Plots Simultane Nutzung, Quantile Plot
# ===================================

class HandlerTopLinePatch(HandlerPatch):
    """Legend handler: filled rect with only a black line on top."""

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # Filled rectangle (no edge)
        rect = plt.Rectangle(
            (-xdescent, -ydescent), width, height,
            facecolor=orig_handle.get_facecolor(),
            edgecolor="none",
            transform=trans,
        )
        # Black line across the top of the rectangle
        top_y = height - ydescent
        line = mlines.Line2D(
            [-xdescent, width - xdescent], [top_y, top_y],
            color="black", linewidth=0.5,
            transform=trans,
        )
        return [rect, line]



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

N_DAYS = charges4.sizes["d"]

# Quantile über d berechnen (pro s und price separat)
exceed_days = [0, 1, 2, 6, 13, 29]   # number of days exceeding
quantile_values = [1 - k / N_DAYS for k in exceed_days]
#quantile_values = [1, 0.99, 0.997, 0.995, 0.993, 0.991]
quant = charges4.quantile(q=quantile_values, dim="d")  # dims: quantile, qh, s, price

# Plot-Hilfsfunktion: Schattierung + schwarze Linien + Legende mit Kästchen
def plot_quantile_panel(ax, da_panel, facecolor_hex, n_days=N_DAYS):
    # da_panel: DataArray mit Dimensionen quantile, qh (eine s, eine price)
    df = da_panel.to_pandas().transpose()  # Zeilen = qh, Spalten = Quantile
    cols = sorted(df.columns, key=float)   # unten niedrigstes, oben höchstes
    x = df.index.values

    # gewünschte X-Ticks (0,3,...,24)
    ticks = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24])
    ax.set_xlim(0, 24)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=20)

    ticks = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.set_yticks(ticks)
    ax.set_yticklabels(["0.0", "", "0.2", "", "0.4", "", "0.6", "", "0.8", "", "1.0"], fontsize=20)
    ax.set_ylim(0, 1)

    # Grid
    ax.grid(color='lightgray', linestyle='--', linewidth=1, axis='both')

    n = len(cols)

    # Alphas: reversed so that the TOP band (q=1) is darkest (alpha=1.0)
    # and the BOTTOM band (q=0.99) is lightest (alpha=0.2)
    # cols goes from lowest quantile (0.99) to highest (1.0)
    # band k fills from cols[k-1] to cols[k], so band 0 is the bottom-most
    # We want band 0 (lowest, near q=0.99) to be lightest → alpha low
    # and band n-1 (highest, near q=1) to be darkest → alpha high
    alphas = np.linspace(0.2, 1.0, n)  # bottom band lightest, top band darkest

    # Schattierung + schwarze Linien
    y_prev = np.zeros_like(x, dtype=float)
    for k, col in enumerate(cols):
        y = df[col].values
        ax.fill_between(x, y_prev, y, color=facecolor_hex, alpha=alphas[k], linewidth=0)
        ax.plot(x, y, color="black", linewidth=0.5)
        y_prev = y

    # Legende: Kästchen mit gleicher Panel-Farbe, Rand schwarz ohne Transparenz
    r, g, b = mcolors.to_rgb(facecolor_hex)
    legend_cols = list(reversed(cols))    # highest q first → darkest first
    legend_alphas = list(reversed(alphas))
    
    def label_for_quantile(q, n_days):
        """Convert quantile to human-readable exceedance label."""
        k = round((1 - q) * n_days)
        if k == 0:
            return "1 day"
        elif k > 0:
            return f"{k+1} days"
    
    handles = [
        Patch(
            facecolor=(r, g, b, legend_alphas[i]),
            edgecolor="none",
            linewidth=0,
            label=label_for_quantile(legend_cols[i], n_days),
        )
        for i in range(n)
    ]
    
    ax.legend(
        handles=handles,
        handler_map={Patch: HandlerTopLinePatch()},
        loc="upper center",
        ncol=2,          # one column works better with longer labels
        title="Simultaneity per Year:",
        fontsize=16,
        title_fontsize=16
    )
    ax.set_xlabel("Hour of the Day", fontsize=20)
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
axs[0, 0].legend_.set_bbox_to_anchor((0.62, 1))
plot_quantile_panel(axs[0, 1], quant.sel(price="spot", s="reg"), panel_colors[(0, 1)])
axs[0, 1].legend_.set_bbox_to_anchor((0.62, 1))
plot_quantile_panel(axs[1, 0], quant.sel(price="mean", s="red"), panel_colors[(1, 0)])
axs[1, 0].legend_.set_bbox_to_anchor((0.62, 1))
plot_quantile_panel(axs[1, 1], quant.sel(price="spot", s="red"), panel_colors[(1, 1)])
axs[1, 1].legend_.set_bbox_to_anchor((0.62, 1))

fig_parallel.text(0.04, 0.76, "Standard", fontsize=20, fontweight='bold', rotation=90, va='center')
fig_parallel.text(0.04, 0.33, "Time of Use", fontsize=20, fontweight='bold', rotation=90, va='center')

fig_parallel.text(0.33, 0.06, "Static", fontsize=20, fontweight='bold', ha='center', color="#a6380f")
fig_parallel.text(0.79, 0.06, "Dynamic", fontsize=20, fontweight='bold', ha='center',  color="#006ac9")

# Gemeinsamer Titel mit Superscript und griechischem Delta
#fig_parallel.suptitle(r"Energy Consumption relative to all EVs Fully charging", fontsize=20)
fig_parallel.supxlabel("Electricity Price", fontsize=20, fontweight='bold')
fig_parallel.supylabel("Network Charge", fontsize=20, fontweight='bold')
fig_parallel.tight_layout(rect=[0.03, 0.02, 1, 1])
fig_parallel.show()

fig_parallel.savefig(r"C:\Users\Hendrik.Kramer\Documents\Repos\ToU_network_charges\daten_results\simultenous_power_quantile_2026.svg")




# =============================================================================
# Critical hours
# =============================================================================

# Load prices and network charges
#year_input = 2024
#timesteps = f_load.load_timesteps(year_input)
#parameter_folderpath_prices = r"..\daten_input\preise" + "\\"
#spot_prices_xr = f_load.load_spot_prices(year_input, parameter_folderpath_prices, "da_auction_hourly_12_uhr_cubic", timesteps)  # in ct/kWh
#tariff_static_price = f_load.get_annual_static_tariff_prices(spot_prices_xr) 
#parameter_filepath_dsos = r"..\daten_input\network_charges\Aufgabe_Hendrik_v4.xlsx"
#network_charges_xr, xr_dso_quarters_sum, xr_ht_length, xr_nt_length, xr_ht_charge, xr_st_charge, xr_nt_charge, sensi_different = f_load.load_network_charges(parameter_filepath_dsos, timesteps, False) # dimension: Time x DSO region x scenario (red, reg)

#network_charges_xr_red = network_charges_xr.sel(s="red").drop_vars("s")

def load_mask_as_xr_multiyear(filepath, name, charges_ref):
    """Load a CSV mask and reshape to xr.DataArray with dims (y, r, d, qh)."""
    pd_mask = pd.read_csv(filepath, index_col="t")
    t_index = pd.to_datetime(pd_mask.index.astype(str), utc=True).tz_convert('Europe/Berlin').tz_localize(None)

    # drop leap year day
    not_leap_day = ~((t_index.month == 2) & (t_index.day == 29))
    t_index  = t_index[not_leap_day]
    pd_mask  = pd_mask[not_leap_day]

    days      = t_index.normalize()
    qh_values = t_index.hour + t_index.minute / 60
    cal_years = t_index.year.to_numpy()

    all_years = sorted(np.unique(cal_years))
    qh_coords = charges_ref.qh.values
    r_coords  = charges_ref.r.values
    n_qh      = len(qh_coords)
    n_r       = len(r_coords)
    n_d_max   = 365  # max days after dropping Feb 29

    data_all_years = np.full((len(all_years), n_r, n_d_max, n_qh), np.nan)

    for i_y, yr in enumerate(all_years):
        mask_yr   = (cal_years == yr)
        t_idx_yr  = t_index[mask_yr]
        days_yr   = t_idx_yr.normalize()
        qh_yr     = t_idx_yr.hour + t_idx_yr.minute / 60
        multi_idx = pd.MultiIndex.from_arrays([days_yr, qh_yr], names=['d', 'qh'])

        df_yr = pd.DataFrame(pd_mask.values[mask_yr], index=multi_idx, columns=pd_mask.columns)
        df_yr = df_yr[~df_yr.index.duplicated(keep='first')]
        df_yr = df_yr.unstack(level='qh')
        df_yr.columns.names = ['r', 'qh']

        n_d_yr = df_yr.shape[0]
        data_all_years[i_y, :, :n_d_yr, :] = df_yr.values.reshape(n_d_yr, n_r, n_qh).transpose(1, 0, 2)

    xr_out = xr.DataArray(
        data=data_all_years,
        coords={
            'y':  all_years,
            'r':  r_coords,
            'd':  np.arange(n_d_max),
            'qh': qh_coords
        },
        dims=['y', 'r', 'd', 'qh'],
        name=name
    )
    return xr_out


charges_critical = charges3.sel(price="spot", s="red").drop_vars(["price", "s"])

# === Load all four masks ===
mask_critical_red     = load_mask_as_xr_multiyear(r"Z:\10_Paper\13_Alleinautorenpaper\critical_timesteps_red.csv",    "critical_red",     charges_critical) == 1
mask_critical_blue    = load_mask_as_xr_multiyear(r"Z:\10_Paper\13_Alleinautorenpaper\critical_timesteps_blue.csv",   "critical_blue",    charges_critical) == 1
mask_no_critical_red  = load_mask_as_xr_multiyear(r"Z:\10_Paper\13_Alleinautorenpaper\no_critical_timesteps_red.csv", "no_critical_red",  charges_critical) == 1
mask_no_critical_blue = load_mask_as_xr_multiyear(r"Z:\10_Paper\13_Alleinautorenpaper\no_critical_timesteps_blue.csv","no_critical_blue", charges_critical) == 1

# === Apply masks (broadcast charges_critical over year dimension) ===
charges_critical_red      = charges_critical.where(mask_critical_red)
charges_no_critical_red   = charges_critical.where(mask_no_critical_red)
charges_critical_blue     = charges_critical.where(mask_critical_blue)
charges_no_critical_blue  = charges_critical.where(mask_no_critical_blue)


first_day = charges_critical.isel(d=0)
last_day  = charges_critical.isel(d=-1)

# check how many days are missing at start and end
n_days_current = len(charges_critical.d.values)  # 364
n_days_target  = 365

n_missing = n_days_target - n_days_current  # typically 1 (Dec 30 or 31) or 2

# append missing days at the end (Dec 30, Dec 31)
extra_days = xr.concat([last_day] * n_missing, dim='d')
charges_critical_365 = xr.concat([charges_critical, extra_days], dim='d')
charges_critical_365 = charges_critical_365.assign_coords(d=np.arange(n_days_target))

# === Apply masks ===
charges_critical_red      = charges_critical_365.where(mask_critical_red)
charges_no_critical_red   = charges_critical_365.where(mask_no_critical_red)
charges_critical_blue     = charges_critical_365.where(mask_critical_blue)
charges_no_critical_blue  = charges_critical_365.where(mask_no_critical_blue)

# === Compute max per quarter hour over all days, per (y, r) ===
max_critical_red     = charges_critical_red.max(dim='d',     skipna=True)
max_no_critical_red  = charges_no_critical_red.max(dim='d',  skipna=True)
max_critical_blue    = charges_critical_blue.max(dim='d',    skipna=True)
max_no_critical_blue = charges_no_critical_blue.max(dim='d', skipna=True)

# === Compute deltas ===
diff_red  = max_critical_red  - max_no_critical_red
diff_blue = max_no_critical_blue - max_critical_blue


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax1 = axes[0]
ax2 = axes[1]

# ── Left subfigure (red) ──────────────────────────────────────────────────────
for r in diff_red.r.values:
    for yr in diff_red.y.values:
        ax1.plot(
            diff_red.qh.values,
            diff_red.sel(r=r, y=yr).values,
            alpha=0.05,
            color='#8b3003'
        )
ax1.axhline(0, color='darkgray', linewidth=2, linestyle='--')
ax1.grid(True, linestyle='--', color='lightgray', zorder=0)
ax1.set_xlim(0, 24)
ax1.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
ax1.set_xticklabels([0, 3, 6, 9, 12, 15, 18, 21, 24], fontsize=20, zorder=0)
ax1.set_ylim(-0.25, 0.25)
yticks1 = [round(v * 0.05, 2) for v in range(-5, 6)]
ax1.set_yticks(yticks1)
ax1.set_yticklabels([str(v) for v in yticks1], fontsize=20)
ax1.set_xlabel('Hour of the Day', fontsize=20)
ax1.set_ylabel('Relative Shift in Max. Consumption', fontsize=20)
ax1.set_title('Low Market Signal & High Network Signal', fontsize=18)

# ── Right subfigure (blue) ────────────────────────────────────────────────────
for r in diff_blue.r.values:
    for yr in diff_blue.y.values:
        ax2.plot(
            diff_blue.qh.values,
            diff_blue.sel(r=r, y=yr).values,
            alpha=0.05,
            color='#004c93'
        )
ax2.axhline(0, color='darkgray', linewidth=2, linestyle='--', zorder=0)
ax2.grid(True, linestyle='--', color='lightgray', zorder=0)
ax2.set_xlim(0, 24)
ax2.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
ax2.set_xticklabels([0, 3, 6, 9, 12, 15, 18, 21, 24], fontsize=20)
yticks = [round(v * 0.1, 1) for v in range(-10, 11)]
ax2.set_yticks(yticks)
ax2.set_yticklabels([str(v) if i % 2 == 0 else '' for i, v in enumerate(yticks)], fontsize=20)
ax2.set_ylim(-1, 1)
ax2.set_xlabel('Hour of the Day', fontsize=20)
ax2.set_ylabel('Relative Shift in Max. Consumption', fontsize=20)
ax2.set_title('Low Market Signal & Low Network Signal', fontsize=18)

plt.tight_layout()
plt.show()

fig_parallel.savefig(r"C:\Users\Hendrik.Kramer\Documents\Repos\ToU_network_charges\daten_results\simultenous_power_ht_nt_2026.svg")


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

if (False):
    epoch_time = datetime(1970, 1, 1)
    
    folder_str = r"C:\Users\Hendrik.Kramer\Documents\Repos\ToU_network_charges\daten_results" + r"\\"
    parameter_filepath_dsos = r"C:\Users\Hendrik.Kramer\Documents\Repos\ToU_network_charges\daten_input\network_charges\Aufgabe_Hendrik_v4.xlsx"
    timesteps = f_load.load_timesteps(parameters_opti["year"])
    
    
    
    # regular
    spot_only_smart = r"2025-11-21_00-39_spot_smart_only_EV_r100_v50_poly" + r"\\"                 
    # sensitivity regulatory
    spot_only_smart_sensi = r"2025-11-25_04-25_spot_smart_only_EV_r100_v50_sensi_regulatory" + r"\\"                 
    # sensitivity double
    spot_only_smart_sensi2 = r"2025-11-25_11-33_spot_smart_only_EV_r100_v50_sensi_double" + r"\\"                 
    
    # cost loading
    dso_x_ev = xr.open_dataarray(folder_str + spot_only_smart + "C_ALL.nc").sel(s="red").size
    spot_ToU_c = xr.open_dataarray(folder_str + spot_only_smart + "C_ALL.nc").sel(s="red").to_pandas().to_numpy().reshape(dso_x_ev)
    spot_ToU_sensi_c = xr.open_dataarray(folder_str + spot_only_smart_sensi + "C_ALL.nc").sel(s="red").to_pandas().to_numpy().reshape(dso_x_ev)
    spot_ToU_sensi_c2 = xr.open_dataarray(folder_str + spot_only_smart_sensi2 + "C_ALL.nc").sel(s="red").to_pandas().to_numpy().reshape(dso_x_ev)
    
    cost_sensi = pd.DataFrame({'base case':spot_ToU_c, 'regulatory limit': spot_ToU_sensi_c,'Half and double':spot_ToU_sensi_c2}) / 100  # ct --> Euro
    # no linebreak space between "base" and "case"
    
    # power consumption loading
    spot_ToU = xr.open_dataarray(folder_str + spot_only_smart + "P_HOME.nc").sel(s="red").mean(["v"]).mean(["r"]).to_pandas()
    spot_ToU_sensi = xr.open_dataarray(folder_str + spot_only_smart_sensi + "P_HOME.nc").sel(s="red").mean(["v"]).mean(["r"]).to_pandas()
    spot_ToU_sensi2 = xr.open_dataarray(folder_str + spot_only_smart_sensi2 + "P_HOME.nc").sel(s="red").mean(["v"]).mean(["r"]).to_pandas()
    
    pd_ct = pd.DataFrame()
    
    pd_ct["Base case"] = spot_ToU 
    pd_ct["Regulatory limit"] = spot_ToU_sensi
    pd_ct["Half and double"] = spot_ToU_sensi2
    
    dti = pd.DatetimeIndex(epoch_time + pd.to_timedelta(xr.open_dataarray(folder_str + spot_only_smart + "P_HOME.nc")["t"], unit='s')).tz_localize("UTC").tz_convert("Europe/Berlin")
    pd_ct = pd_ct.set_index(dti)
    pd_ct["hour decimal"] = pd_ct.index.hour + pd_ct.index.minute/60
    
    pd_day = pd_ct.groupby(["hour decimal"]).mean()






# kW reduction plots
if (False):

    fig_sensi, axs_sensi = plt.subplots(ncols=2, figsize=(15, 6), gridspec_kw={'width_ratios': [0.6, 0.4]})   


    # RECHTER PLOT
    meanpointprops = dict(marker='x', markeredgecolor='black', markerfacecolor='black') #firebrick
    flierprops = dict(marker='o', markerfacecolor=(0,0,0,0), markersize=6, markeredgecolor=(0,0,0,0))  # set to transparent
    cost_sensi.plot(ax = axs_sensi[1],  kind="box", widths=0.7, patch_artist=True, notch=True, showmeans=True, meanprops=meanpointprops,  flierprops=flierprops, color=dict(boxes='black', whiskers='black', medians='black', caps='black'), boxprops=dict(facecolor="lightgray"))
    axs_sensi[1].set_ylabel("Cost in €", fontsize=20)
    axs_sensi[1].set_xticklabels(cost_sensi.columns.str.replace(" ","\n"), fontsize=20)
    axs_sensi[1].set_ylim(-3, 125)


    axs_sensi[1].grid(color='lightgray', linestyle='--', linewidth=1, axis="both")
    axs_sensi[1].tick_params(axis='both', labelsize=20)
    axs_sensi[1].set_title("(b) Cost for end-consumers", fontsize=20)



    #  LINKER PLOT
    axs_sensi[0].plot(pd_day["Base case"], linestyle="-", alpha=1, linewidth=5,  zorder=0, color="lightgray", label="Base case")
    axs_sensi[0].plot(pd_day["Regulatory limit"], linestyle="-", alpha=1, linewidth=3, color="darkgray", zorder=1, label="Regulatory limit")
    axs_sensi[0].plot(pd_day["Half and double"], linestyle="-", alpha=1, zorder=2, linewidth=1, color="dimgrey", label= "Half and double")

    axs_sensi[0].legend(fontsize=16, ncols=1, loc="upper right")

    axs_sensi[0].set_title("(a) Mean charge power during all seasons", fontsize=20)

    axs_sensi[0].set_ylim(-1, 1)
    axs_sensi[0].set_xlabel("Time in hours", fontsize=20)
    axs_sensi[0].set_ylabel("Power in kW", fontsize=20)
    axs_sensi[0].set_ylim(-0.05, 0.63)
    axs_sensi[0].tick_params(axis='both', labelsize=20)
    axs_sensi[0].set_xlim(0, 24)
    axs_sensi[0].set_xticks(np.array([0, 3, 6, 9, 12, 15, 18, 21, 24]))
    axs_sensi[0].set_xticklabels([0, 3, 6, 9, 12, 15, 18, 21, 24], fontsize=20)
    axs_sensi[0].grid(color='lightgray', linestyle='--', linewidth=1, axis="both")

    plt.tight_layout()
    plt.show()

    fig_sensi.savefig(r"C:\Users\Hendrik.Kramer\Documents\Repos\ToU_network_charges\daten_results\sensitivity_test.svg")



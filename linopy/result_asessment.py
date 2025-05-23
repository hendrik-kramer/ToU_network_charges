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



# read  results
folder_name = "2025-05-20_08-52_Q2_smart_charging_only_EV_r50_v10"


folder_path = Path("../daten_results") / folder_name

result_C_OP_NO_PENALTY_eur = xr.open_dataarray(folder_path / "C_OP_NO_PENALTY.nc")
result_SOC_EV = xr.open_dataarray(folder_path / "SOC_EV.nc")
result_P_BUY = xr.open_dataarray(folder_path / "P_BUY.nc")
result_P_EV_NOT_HOME = xr.open_dataarray(folder_path / "P_EV_NOT_HOME.nc")
result_SOC_MISSING = xr.open_dataarray(folder_path / "SOC_MISSING.nc")

# reconvert seconds to datetime
epoch_time = datetime(1970, 1, 1)
dti = pd.DatetimeIndex(epoch_time + pd.to_timedelta(result_SOC_EV["t"], unit='s')).tz_localize("UTC").tz_convert("Europe/Berlin")

result_SOC_EV["t"] = dti
result_P_BUY["t"] = dti
result_P_EV_NOT_HOME["t"] = dti
result_SOC_MISSING["t"] = dti


# ===== plotting ====


if (False):  # BAR PLOT MEAN COST SAVINGS PER DSO
      
    scenarios = result_C_OP_NO_PENALTY_eur["r"].to_pandas().to_list()
    dso_means = {'regular network charges': result_C_OP_NO_PENALTY_eur.sel(s='reg').mean(dim=["v"]),
                    'reduced network charges': result_C_OP_NO_PENALTY_eur.sel(s='red').mean(dim=["v"]) }
    
    x = np.arange(len(result_C_OP_NO_PENALTY_eur.mean(dim="v")))  # the label locations
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
    str_v = str(len(result_C_OP_NO_PENALTY_eur["v"]))
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
    
    savings = (result_C_OP_NO_PENALTY_eur.sel(s="reg") - result_C_OP_NO_PENALTY_eur.sel(s="red")).to_pandas()
    savings["row_sum"] = savings.sum(axis=1) 
    savings_sorted = savings.sort_values("row_sum", ascending=False)
    savings_sorted = savings_sorted.drop(columns=["row_sum"]).transpose()
    savings_sorted_twice = savings_sorted.sort_values(savings_sorted.columns[0], axis="index", ascending=False)

    

    ude_colors = ['#efe4bf','#004c93']
    cmap_ude = mcolors.LinearSegmentedColormap.from_list('ude', ude_colors)

    fig_hm, ax_hm = plt.subplots(figsize=(15, 5)) 
    heatmap = ax_hm.imshow(savings_sorted_twice.to_numpy(), cmap=cmap_ude, interpolation='nearest', aspect=1)
    plt.xlabel('DSOs\n(columns sorted by column sum)')
    plt.ylabel('Mobility patterns\n(rows sorted by values in first column)')
    fig_hm.colorbar(heatmap, ax=ax_hm,  orientation="vertical", label="Savings when switching from regular \n to reduced network charges in €")

    fig_hm.savefig(folder_path / "heatmap_savings.svg")



if (False): # SAVINGS DEPENDENT ON MILEAGE BEHAVIOR (SCATTER + LIN REG)
    

    emob_demand_quarter = np.repeat(emob_demand_xr.sum("t").to_numpy(), result_C_OP_NO_PENALTY_eur["r"].size)
    savings_scatter = (result_C_OP_NO_PENALTY_eur.sel(s="reg") - result_C_OP_NO_PENALTY_eur.sel(s="red") ).to_numpy().reshape([result_C_OP_NO_PENALTY_eur.sel(s="reg").size, ])
    
    savings_true = (savings_scatter != 0)
    emob_demand_quarter_pos = emob_demand_quarter[savings_true]
    savings_scatter_pos = savings_scatter[savings_true]
    
    
    fig_km, ax_km = plt.subplots(figsize=(15, 5)) 
    scatter = ax_km.scatter(emob_demand_quarter_pos, savings_scatter_pos, alpha=0.3)
    
    coef = np.polyfit(emob_demand_quarter_pos,savings_scatter_pos,1)
    poly1d_fn = np.poly1d(coef) 
    str(poly1d_fn)
    
    x = np.arange(1000)
    y = poly1d_fn(x)
    plt.plot(x, y, linestyle="--", color="k", label=str("y = " + str(poly1d_fn) ))
    plt.legend()
    
    plt.xlim([300,700])
    plt.xlabel("Energy consumption in kWh")
    plt.ylabel("Savings in €")
    plt.show()
    
    fig_km.savefig(folder_path / "savings_per_energy_consumed_milage.svg")
    
    
    
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

    fig_kw_reduction, ax_hm = plt.subplots(figsize=(15, 5)) 
    heatmap_power = ax_hm.imshow(peak_reduction_sorted_twice.to_numpy(), cmap=cmap_ude, vmin=-11, vmax=11, interpolation='nearest', aspect=2)
    plt.xlabel('DSOs\n(columns sorted by column sum)')
    plt.ylabel('Mobility patterns\n(rows sorted by values in first column)')
    fig_kw_reduction.colorbar(heatmap_power, ax=ax_hm,  orientation="vertical", label="Peak reduction in kW when switching from regular \n to reduced network charges in kW")
    plt.tight_layout()

    fig_kw_reduction.savefig(folder_path / "hpeak_reduction_savings.svg")
    
    
    
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
    signal_spot_15_15 = signal_15_15.iloc[:,15*4:]

    daily_mean_spot_price = signal_spot_15_15.mean(axis=1)
    rel_signal_spot_15_15 = signal_spot_15_15.sub(daily_mean_spot_price, axis=0)
    
    # netwrork charges
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



    
if (False): # CHARGE POWER 


    def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot",  H=".", **kwargs):
        """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
    labels is a list of the names of the dataframe, used for the legend
    title is a string for the title of the plot
    H is the hatch used for identification of the different dataframe"""
        import matplotlib as mpl
        from cycler import cycler
        mpl.rcParams["axes.prop_cycle"] = cycler('color', ["#efe4bf", "#004c93"])
    
        n_df = len(dfall)
        n_col = len(dfall[0].columns) 
        n_ind = len(dfall[0].index)
        axe = plt.subplot(111)
        plt.subplots_adjust(bottom=0.476,left=0.057,right=0.917,top=0.9)
        
        for df in dfall : # for each data frame
            axe = df.plot(kind="bar",
                          edgecolor="black",
                          linewidth=1,
                          stacked=True,
                          ax=axe,
                          legend=False,
                          grid=False,
                          **kwargs)  # make bar plots
    
        h,l = axe.get_legend_handles_labels() # get the handles we want to modify
        for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
            for j, pa in enumerate(h[i:i+n_col]):
                for rect in pa.patches: # for each index
                    rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                    #rect.set_color(colors_plot[j,int(np.floor(i/n_df))])
                    #print(int(i / n_col))
                    rect.set_hatch(H * int(i / n_col)) #edited part     
                    rect.set_width(1 / float(n_df + 1))
                    #print(int(np.floor(i/n_df)),j)
    
        axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
        axe.set_xticklabels(df.index, rotation = 90)
        axe.set_title(title)
    
        # Add invisible data to add another legend
        n=[]        
        for i in range(n_df):
            n.append(axe.bar(0, 0, color="#eeeeee", hatch=H * i))
    
        l1 = axe.legend(h[:n_col], l[:n_col], loc="upper left", ncol=2) #[1.01, 0.5]
        if labels is not None:
            l2 = plt.legend(n, labels,  loc="upper right", ncol=2) # loc=[1.01, 0.1],
        axe.add_artist(l1)
        
        # background grid lines
        axe.grid(color='lightgray', linestyle='--', linewidth=1, axis="y")
        axe.set_axisbelow(True)
        axe.set_ylim([0, 1.3*max(df_reg.max().max(), df_red.max().max())])
        #axe.subplots_adjust(bottom=0.2)
        return axe

    
    df_reg = pd.DataFrame( {"P_BUY":result_P_BUY.sel(s="reg").sum("t").mean("v"), "P_EV_NOT_HOME":result_P_EV_NOT_HOME.sel(s="reg").sum("t").mean("v")}, index=result_P_BUY["r"] )
    df_red = pd.DataFrame( {"P_BUY":result_P_BUY.sel(s="red").sum("t").mean("v"), "P_EV_NOT_HOME":result_P_EV_NOT_HOME.sel(s="red").sum("t").mean("v")}, index=result_P_BUY["r"] )

    plot_clustered_stacked([df_reg, df_red],["regular network charges", "reduced network charges"], title="Energy consumed in kWh")

    fig.savefig(folder_path / "dso_energy_barlot.svg")
    
    


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


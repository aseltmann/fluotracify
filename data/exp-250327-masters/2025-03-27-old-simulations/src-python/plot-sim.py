import datetime
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
from IPython.display import display

os.chdir("/home/lea/Programs/drmed-git")
FLUOTRACIFY_PATH = "./src/"
sys.path.append(FLUOTRACIFY_PATH)

from fluotracify.simulations import (import_simulation_from_csv as isfc,
                                     analyze_simulations as ans)

sns.set_theme(style="whitegrid", font_scale=2, palette='colorblind',
              context='paper')

def get_tt(dr):
    dr = dr.removesuffix('-3000.0').removesuffix('-7000.0').removesuffix('-1000.0')
    dr = float(dr)
    tt, _ = ans.convert_diffcoeff_to_transittimes(dr, 250)
    return f'\nsimulated trace\n$\\tau_{{sim}}={tt:.2f}ms$'

def save_plot(filename, txt):
    plot_file = f'{filename}{txt}'.replace(' ', '_').replace(
        '\n', '-').replace('"', '').replace('{', '').replace(
        '}', '').replace('$', '').replace('=', '-').replace('\\', '')
    plt.savefig(f'{plot_file}.pdf', bbox_inches='tight', dpi=300)
    os.system(f'pdf2svg {plot_file}.pdf {plot_file}.svg')
    os.system(f'rm {plot_file}.pdf')

today = datetime.date.today()

# ----------------------- Detector Dropout ---------------------------------
folder = (
    "/run/media/lea/bluey/20_Forschung, Lehre, Studium/Forschung/DOKTOR"
    "/drmed-collections/drmed-simexps/secondartefact_Aug2019_rand"
)
files = list(Path(folder).rglob("*.csv"))
nfiles = len(files)
exp_pars = pd.DataFrame()

for idx, myfile in enumerate(files):
    # save some parameters of the experiment from csv file
    exp_par = pd.read_csv(
        myfile, sep=",", nrows=9, index_col=0, usecols=[0, 1], engine="python"
    ).squeeze("columns")
    exp_pars = pd.concat(
        [exp_pars, exp_par], axis=1, ignore_index=True, sort=False
    )

exp_pars.loc["path and file name"] = exp_pars.loc["path and file name"].apply(
    lambda x: Path(x).name
)
exp_pars_num = (
    exp_pars
    .loc["extent of the PSF":"height of the simulation"]
    .apply(pd.to_numeric)
)
for i in ["number of fast molecules", "diffusion rate of molecules"]:
    exp_pars_num.loc[i].plot(kind="hist", title=f"{i}")
    plt.show()

print(exp_pars.loc["diffusion rate of molecules"].value_counts())

for mol in sorted(exp_pars_num.loc["diffusion rate of molecules"].unique()):
    myidx = exp_pars_num.loc["diffusion rate of molecules"].eq(mol)
    myfiles = exp_pars.loc["path and file name", myidx]
    display(f"{mol}", (
        exp_pars
        .loc[["path and file name", "number of fast molecules"],
             exp_pars.loc["path and file name"].isin(myfiles)]
        .T
    ))

sim_path = Path("../drmed-collections/2019-08-sim-detector-dropout")
col_per_example = 2
lab_thresh = 0

sim, _, nsamples, sim_params = isfc.import_from_csv(
    folder=sim_path,
    header=10,
    frac_train=1,
    col_per_example=col_per_example,
    dropindex=None,
    dropcolumns=None)

sim = sim.drop("Unnamed: 200", axis="columns")
diffrates = sim_params.loc["diffusion rate of molecules"].astype(np.float32)
nmols = sim_params.loc["number of fast molecules"].astype(np.float32)
sim_columns = [f"{d:.4}" for d in np.repeat(diffrates, nsamples[0])]
sim_sep = isfc.separate_data_and_labels(array=sim,
                                        nsamples=nsamples,
                                        col_per_example=col_per_example)
sim_dirty = sim_sep['0']
sim_dirty.columns = sim_columns

sim_labels = sim_sep['1']
sim_labels.columns = sim_columns
sim_labbool = sim_labels < lab_thresh
sim_labbool.columns = sim_columns

filename = (f"./data/exp-250327-masters/{today}-old-simulations/"
            "jupyter/{today}-detdrop")
plot_index = ["0.5", "5.0"]
plot_traceno = [0, 0]
for i, (idx, t) in enumerate(zip(plot_index, plot_traceno)):
    fig = plt.figure()
    ax = plt.subplot(111)
    txt = get_tt(idx)
    ax.set_prop_cycle(color=[sns.color_palette()[4]])
    sim_labbool_scaled = sim_dirty.loc[:, idx].iloc[
        :, t].max() * sim_labbool.loc[:, idx].iloc[:, t]
    sns.lineplot(data=sim_labbool_scaled, alpha=0.5)
    plt.fill_between(x=sim_labbool.loc[:, idx].iloc[:, t].index,
                     y1=sim_labbool_scaled,
                     y2=0, alpha=0.5, label='label:\ndetector dropout')

    ax.set_prop_cycle(color=[sns.color_palette()[2]])
    sim_invbool_scaled = sim_dirty.loc[:, idx].iloc[
        :, t].max() * ~sim_labbool.loc[:, idx].iloc[:, t]
    plt.fill_between(x=sim_labbool.loc[:, idx].iloc[:, t].index,
                     y1=sim_invbool_scaled,
                     y2=0, alpha=0.5, label='\nlabel:\nno artifacts')
    ax.set_prop_cycle(color=[sns.color_palette()[0]])
    sns.lineplot(data=sim_dirty.loc[:, idx].iloc[:, t], label=txt)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.setp(ax, xlabel=r'Time [$ms$]', ylabel=r'Intensity [a.u.]', title='')
    save_plot(filename, f'{txt}-{i}')

# ------------------------------ Photobleaching ----------------------------

folder = (
    "/run/media/lea/bluey/20_Forschung, Lehre, Studium/Forschung/DOKTOR"
    "/drmed-collections/drmed-simexps/thirdartefact_Sep2019"
)
files = list(Path(folder).rglob("*.csv"))
nfiles = len(files)
exp_pars = pd.DataFrame()

for idx, myfile in enumerate(files):
    # save some parameters of the experiment from csv file
    exp_par = pd.read_csv(
        myfile, sep=",", nrows=10, index_col=0, usecols=[0, 1], engine="python"
    ).squeeze("columns")
    exp_pars = pd.concat(
        [exp_pars, exp_par], axis=1, ignore_index=True, sort=False
    )
exp_pars.loc["path and file name"] = exp_pars.loc["path and file name"].apply(
    lambda x: Path(x).name
)
exp_pars_num = (
    exp_pars
    .loc["extent of the PSF":"height of the simulation"]
    .apply(pd.to_numeric)
)
for i in ["number of fast molecules", "number of bleached molecules",
          "diffusion rate of molecules"]:
    exp_pars_num.loc[i].plot(kind="hist", title=f"{i}")
    plt.show()

print(exp_pars.loc["diffusion rate of molecules"].value_counts())

for mol in sorted(exp_pars_num.loc["diffusion rate of molecules"].unique()):
    myidx = exp_pars_num.loc["diffusion rate of molecules"].eq(mol)
    myfiles = exp_pars.loc["path and file name", myidx]
    display(f"{mol}", (
        exp_pars.loc[["path and file name", "number of fast molecules",
                      "number of bleached molecules"],
                     exp_pars.loc["path and file name"].isin(myfiles)]
        .T
    ))

sim_path = Path("../drmed-collections/2019-09-sim-photobleaching")
col_per_example = 2
lab_thresh = 1

sim, _, nsamples, sim_params = isfc.import_from_csv(
    folder=sim_path,
    header=11,
    frac_train=1,
    col_per_example=col_per_example,
    dropindex=None,
    dropcolumns=None)

sim = sim.drop("Unnamed: 200", axis="columns")
diffrates = sim_params.loc["diffusion rate of molecules"].astype(np.float32)
nmols = sim_params.loc["number of fast molecules"].astype(np.float32)
sim_columns = [f"{d:.4}-{n}" for d, n in zip(
    np.repeat(diffrates, nsamples[0]), np.repeat(nmols, nsamples[0])
)]
sim_sep = isfc.separate_data_and_labels(array=sim,
                                        nsamples=nsamples,
                                        col_per_example=col_per_example)
sim_dirty = sim_sep['0']
sim_dirty.columns = sim_columns

sim_labels = sim_sep['1']
sim_labels.columns = sim_columns
sim_labbool = sim_labels > lab_thresh
sim_labbool.columns = sim_columns

filename = (f"./data/exp-250327-masters/{today}-old-simulations/"
            "jupyter/{datetime.date.today()}-bleach")
plot_index = ["0.5-3000.0", "0.5-7000.0", "5.0-1000.0", "5.0-7000.0"]
plot_traceno = [3, 1, 0, 2]
for i, (idx, t) in enumerate(zip(plot_index, plot_traceno)):
    fig = plt.figure()
    ax = plt.subplot(111)
    txt = get_tt(idx)
    ax.set_prop_cycle(color=[sns.color_palette()[4]])
    sim_labbool_scaled = sim_dirty.loc[:, idx].iloc[
        :, t].max() * sim_labbool.loc[:, idx].iloc[:, t]
    sns.lineplot(data=sim_labbool_scaled, alpha=0.5)
    plt.fill_between(x=sim_labbool.loc[:, idx].iloc[:, t].index,
                     y1=sim_labbool_scaled,
                     y2=0, alpha=0.5, label='label:\nphotobleaching')

    ax.set_prop_cycle(color=[sns.color_palette()[2]])
    sim_invbool_scaled = sim_dirty.loc[:, idx].iloc[
        :, t].max() * ~sim_labbool.loc[:, idx].iloc[:, t]
    plt.fill_between(x=sim_labbool.loc[:, idx].iloc[:, t].index,
                     y1=sim_invbool_scaled,
                     y2=0, alpha=0.5, label='\nlabel:\nno artifacts')
    ax.set_prop_cycle(color=[sns.color_palette()[0]])
    sns.lineplot(data=sim_dirty.loc[:, idx].iloc[:, t], label=txt)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.setp(ax, xlabel=r'Time [$ms$]', ylabel=r'Intensity [a.u.]', title='')
    save_plot(filename, f'{txt}-{idx.lstrip("0.5-").lstrip("5.0-")}-{i}')

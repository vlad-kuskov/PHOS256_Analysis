
import ROOT as pyroot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from itertools import product

class PHOSAnalyzer:

    def __init__(self, module, args_list, run_list):
        # lists to keep run specifiers and corresponding runs and ADC channels (256 in total)
        if isinstance(args_list, list):
            self.args_list = args_list
        else:
            print("args_list is not iterable!!!\nSetting an empty list to args_list.\nYou can change args_list by method set_run_args().")
            self.args_list = []
        if isinstance(run_list, list):
            self.run_list = run_list
        else:
            print("run_list is not iterable!!!\nSetting an empty list to run_list.\nYou can change run_list by method set_run_list().")
            self.run_list = []
        # PHOS256 consists of 16x16 channels in x and z directions
        self.module = module[1]
        self.sru    = module[-1]
        self.z_range = range(30,46,1) # we use DTC ports 2-9 in SRU, which corresponds for the upper part of a module
        self.x_range = {"0": range(0,16,1), "1": range(16,32,1), "2": range(32,48,1), "3": range(48,64,1)}.get(module[-1])
        self.channels = [f"x{ix}_z{iz}" for ix, iz in product(self.x_range, self.z_range)]
        # lists to keep  ADC channels (256 in total)
        self.ch_wo_signal = []
        # Data frame to keep values of ADC channels (mean and StdDev) for every run
        self.df_pos = pd.DataFrame(index=self.channels, columns=self.args_list)
        self.df_rms = pd.DataFrame(index=self.channels, columns=self.args_list)
        self.df_ped_hg = pd.DataFrame(index=self.channels, columns=["(mean, edges)"])
        self.df_ped_lg = pd.DataFrame(index=self.channels, columns=["(mean, edges)"])
        self.hg2lg = []
        # Base path for a directory with data files
        self.base_path = "/mnt/d/Projects/QML/PHOS256/data/"


    def set_module(self, module: str):

        # if (module.find('M') == 0 or module.find('_') == 0):
        #     print("The input module type should be in the format Mn_m, where:")
        #     print("\t n - number of PHOS module,")
        #     print("\t m - number of SRU in module")
        #     print("Setting x and z ranges as for M1_3")
        #     set_module("M1_3")
        # else:

        self.module = module[1]
        self.sru = module[-1]
        self.z_range = range(30,46,1) # we use DTC ports 2-9 in SRU, which corresponds for the upper part of a module
        self.x_range = {"0": range(0,16,1), "1": range(16,32,1), "2": range(32,48,1), "3": range(48,64,1)}.get(module[-1])
        self.channels = [f"x{ix}_z{iz}" for ix, iz in product(self.x_range, self.z_range)]

    def set_run_list(self, run_list):

        if isinstance(run_list, list):
            self.run_list = run_list
        else:
            print("run_list is not a list!!!\nSetting an empty list to run_list.\nYou can change run_list by method set_run_list().")
            self.run_list = []


    def set_run_args(self, args_list):

        if isinstance(args_list, list):
            self.args_list = args_list
        else:
            print("args_list is not a list!!!\nSetting an empty list to args_list.\nYou can change args_list by method set_run_args().")
            self.args_list = []


    def set_base_path(self, base_path):
        self.base_path = base_path


    def get_df_pos(self):
        return self.df_pos
    
    
    def get_df_rms(self):
        return self.df_rms
    
    def get_module(self):
        return self.module
    
    def load_pedestals(self, ped_file_name):

        ped_file = pyroot.TFile(self.base_path + ped_file_name, "read")
        if (not ped_file):
            return
        
        for ich in self.channels:
            hist_ped_lg = ped_file.Get(f"hPed_g0_m{self.module}_{ich}")

            mean, rms = hist_ped_lg.GetMean(), hist_ped_lg.GetStdDev()
            nbins = hist_ped_lg.GetNbinsX()
            amps  = np.array([hist_ped_lg.GetBinContent(i+1) for i in range(nbins)])
            max_amp = hist_ped_lg.GetMaximum()

            _, peak_pos = find_peaks(amps,
                                     width=1.2,
                                     height=max_amp,
                                     prominence=max_amp
                                    )
            
            if (len(peak_pos["left_bases"]) > 0):
                left_edge, right_edge = peak_pos["left_bases"][0], peak_pos["right_bases"][0]+1
            else:
                left_edge, right_edge = mean - 3*rms, mean + 3*rms

            self.df_ped_lg.at[ich, "(mean, edges)"] = [mean, left_edge, right_edge]

            hist_ped_hg = ped_file.Get(f"hPed_g1_m{self.module}_{ich}")
            mean, rms = hist_ped_hg.GetMean(), hist_ped_hg.GetStdDev()
            nbins = hist_ped_hg.GetNbinsX()
            amps  = np.array([hist_ped_hg.GetBinContent(i+1) for i in range(nbins)])
            max_amp = hist_ped_hg.GetMaximum()

            _, peak_pos = find_peaks(amps,
                                     width=2.,
                                     height=max_amp,
                                     prominence=max_amp
                                    )
            
            if (len(peak_pos["left_bases"]) > 0):
                left_edge, right_edge = peak_pos["left_bases"][0], peak_pos["right_bases"][0]+1
            else:
                left_edge, right_edge = mean - rms, mean + rms

            self.df_ped_hg.at[ich, "(mean, edges)"] = [mean, left_edge, right_edge]

            hist_ped_lg.Delete()
            hist_ped_hg.Delete()

        ped_file.Close()


    def load_hg2lg(self, hg2lg):
        self.hg2lg = hg2lg


    def fill_df_from_TH1(self, do_ped_sub=True, gain="HG", use_lg=False):

        gain_idx = {"LG": 0, "HG": 1}.get(gain)
        ped_width = {"LG": 1.5, "HG": 2.5}.get(gain)

        for run, label in zip(self.run_list, self.args_list):
            data_tmp = pyroot.TFile(self.base_path + f"led{run}.root", "read")
            for i, ich in enumerate(self.channels):
                hist = data_tmp.Get(f"hLED_g{gain_idx}_m{self.module}_{ich}")
                if (not hist) or (not isinstance(hist, pyroot.TH1F)):
                    continue
                mean, rms = hist.GetMean(), hist.GetStdDev()

                if (do_ped_sub):
                    mean, rms = subtract_ped_from_TH1(hist, ped_width)

                if (use_lg and (mean > 900. or mean == 0.)):
                    hist_lg = data_tmp.Get(f"hLED_g0_m{self.module}_{ich}")
                    if (not hist_lg) or (not isinstance(hist_lg, pyroot.TH1F)):
                        continue
                    mean_lg, rms_lg = hist_lg.GetMean(), hist_lg.GetStdDev()
                    if (do_ped_sub):
                        mean_lg, rms_lg = subtract_ped_from_TH1(hist_lg, ped_width)
                    if (len(self.hg2lg) == 0):
                        print("HG/LG is not load!!! Peak from HG, LG is not used\n")
                    else:
                        mean = mean_lg * self.hg2lg[i]
                        rms  = rms_lg
                    hist_lg.Delete()

                self.df_pos.at[ich, label] = mean
                self.df_rms.at[ich, label] = rms

                hist.Delete()
            
            data_tmp.Close()


    def df_to_grid(self, df, label):
        grid = np.empty((len(self.x_range), len(self.z_range)))
        grid[:] = np.nan
        for ix, x in enumerate(self.x_range):
            for iz, z in enumerate(self.z_range):
                chan = f"x{x}_z{z}"
                grid[iz, ix] = df.at[chan, label]

        return grid
    

    def get_fec_csp(self, channel="x0_z30"):
        (x, z) = (int(part[1:]) for part in channel.split("_"))
        fec = z % self.z_range[0] // 2
        row = z % 2

        # CSPs are arrenged as in PHOS modules 1, 2, 3
        csp_arr = [ [ 0,  1,  2,  3,  4,  5,  6,  7, 24, 25, 26, 27, 28, 29, 30, 31],   # lower row of CSPs (at z=0)
                    [16, 17, 18, 19, 20, 21, 22, 23,  8,  9, 10, 11, 12, 13, 14, 15] ]  # upper row of CSPs (at z=1)

        csp = {(ix, iz): csp_arr[row][ix - self.x_range[0]] for ix, iz in product(self.x_range, self.z_range)}.get((x, z))

        return fec, csp

    # CSP id for APD settings
    def get_csp_id(self, csp):
        if(csp < 0 or 31 < csp):
            return -1
        else:
            hvid = -1
            if (csp < 16):
                hvid = 104 + csp
            elif (csp < 24):
                hvid = 104 - csp + 15
            elif (csp < 32):
                hvid = 127 - csp + 24
            return hvid


#=============== General functions ====================


def subtract_ped_from_TH1(hist: pyroot.TH1F, ped_width=2.5):

    if (hist.Integral(950,1024) > 100):
        return 0, 0

    nbins = hist.GetNbinsX()
    amps  = np.array([hist.GetBinContent(i+1) for i in range(nbins)])
    max_amp = hist.GetMaximum()

    peaks, peak_stats = find_peaks(amps,
                                #    width=ped_width,
                                   distance=10,
                                   height=max_amp,
                                   prominence=max_amp
                                  )

    ped = peaks[0]
    left_edge, right_edge = peak_stats["left_bases"][0], peak_stats["right_bases"][0]+1

    for i in range(1, nbins+1):
        if (i in range(left_edge, right_edge, 1)):
            hist.SetBinContent(i, 0.)
        elif (hist.Integral(i-1, i+1) == 1):
            hist.SetBinContent(i, 0.)
    
    mean = hist.GetMean() - ped
    rms  = hist.GetStdDev()

    if (mean < 0.):
        mean, rms = 0., 0.

    return mean, rms


def subtract_ped_from_array(bin_contents, bin_edges, ped_width=2.5) -> np.ndarray:

    amps = np.copy(bin_contents)
    nbins = len(bin_edges)
    max_amp = np.max(amps)

    _, peak_pos = find_peaks(amps,
                                width=ped_width,
                                distance=10,
                                height=[0.02*max_amp,max_amp]
                                )

    if (len(peak_pos["left_bases"]) > 1):
        left_edge, right_edge = peak_pos["left_bases"][0]-1, peak_pos["left_bases"][1]-1
    else:
        left_edge, right_edge = 0, nbins-1

    for i in range(left_edge, right_edge, 1):
        amps[i] = 0.

    return amps


def get_num_of_peaks_in_TH1(hist: pyroot.TH1F, peak_width=2.5) -> int:

    nbins = hist.GetNbinsX()
    amps  = np.array([hist.GetBinContent(i+1) for i in range(nbins)])
    max_amp = hist.GetMaximum()

    _, peak_pos = find_peaks(amps,
                                width=peak_width,
                                distance=10,
                                height=[0.05*max_amp,max_amp]
                                )

    return len(peak_pos["left_bases"])


def get_num_of_peaks_in_array(bin_contents, bin_edges, peak_width=2.5) -> int:

    amps = np.copy(bin_contents)
    nbins = len(bin_edges)
    max_amp = np.max(amps)

    _, peak_pos = find_peaks(amps,
                                width=peak_width,
                                distance=10,
                                height=[0.05*max_amp,max_amp]
                                )

    return len(peak_pos["left_bases"])


def get_num_of_peaks_in_array(bin_contents, bin_edges, peak_width=2.5) -> int:

    amps = np.copy(bin_contents)
    nbins = len(bin_edges)
    max_amp = np.max(amps)

    _, peak_pos = find_peaks(amps,
                                 width=peak_width,
                                 distance=10,
                                 height=[0.05*max_amp,max_amp]
                                )

    return len(peak_pos["left_bases"])


def df_to_grid(df, label):

    x_range = range(16)
    z_range = range(30,46,1)
    
    grid = np.empty((len(x_range), len(z_range)))
    grid[:] = np.nan
    for ix in x_range:
        for iz, z in enumerate(z_range):
            chan = f"x{ix}_z{z}"
            grid[iz, ix] = df.at[chan, label]

    return grid


def convert_TH1_to_nparray(hist: pyroot.TH1F) -> np.ndarray:
    bin_contents = np.array([hist.GetBinContent(i) for i in range(1, hist.GetNbinsX()+1)])
    bin_edges = np.array([hist.GetBinLowEdge(i) for i in range(1, hist.GetNbinsX()+2)])

    return {"edges": bin_edges,
            "contents": bin_contents}


def get_fec_csp(channel="x0_z30"):

    # PHOS256 matrix
    x_range = range(0,16)
    z_range = range(30,46)

    (x, z) = (int(part[1:]) for part in channel.split("_"))
    fec = z % z_range[0] // 2
    row = z % 2

    # CSPs are arrenged as in PHOS modules 1, 2, 3
    csp_arr = [ [ 0,  1,  2,  3,  4,  5,  6,  7, 24, 25, 26, 27, 28, 29, 30, 31],   # lower row of CSPs (at z=0)
                [16, 17, 18, 19, 20, 21, 22, 23,  8,  9, 10, 11, 12, 13, 14, 15] ]  # upper row of CSPs (at z=1)

    csp = {(ix, iz): csp_arr[row][ix] for ix, iz in product(x_range, z_range)}.get((x, z))

    return (fec, csp)


# ===== functions for plotting =====

def get_style_map(n=5, gradient="Spectral_r", vmin=0., vmax=1.):

    cmap = plt.cm.get_cmap(gradient)
    colors = cmap(np.linspace(vmin, vmax, n))
    arrange = np.arange(n)

    base_linestyles = [
        '-',                    # solid
        '--',                   # dashed
        '-.',                   # dash-dot
        ':',                    #dotted
        (0, (5, 10)),           # loosely dashed
        (0, (3, 10, 1, 10)),    # loosely dashdotted
        (5, (10, 3)),           # long dash with offset
        (0, (1, 1)),            # densely dotted
        (0, (5, 1)),            # densely dashed
        (0, (3, 1, 1, 1)),      # densely dashdotdotted
        (0, (1, 10)),           # loosely dotted
    ]

    linestyles = [base_linestyles[i % len(base_linestyles)] for i in arrange]

    base_markers = [
        'o',    # circle
        's',    # square
        'X',    # X (filled)        
        '*',    # star
        'P',    # filled plus
        '8',    # octagon
        'p',    # pentagon
        'h',    # hexagon 1
        'H',    # hexagon 2
        'D',    # diamond
        'd',    # thin diamond
        'v',    # triangle down
        '^',    # triangle up
        'x',    # x
        # '<',    # triangle left
        # '>',    # triangle right
        # '|',    # vline
        # '_',    # hline
        # '1',    # tri down
        # '2',    # tri up
        # '3',    # tri left
        # '4',    # tri right
    ]

    marker_styles = []
    for marker in base_markers:
        marker_styles.append({"marker": marker})  # filled markers
        marker_styles.append({"marker": marker, "fillstyle": "none"})  # open markers

    # np.random.shuffle(arrange)
    markers = [marker_styles[i % len(base_markers)] for i in arrange]

    style_map = {"colors": colors, 
                 "linestyles": linestyles,
                 "markers": markers
                }
    
    return style_map
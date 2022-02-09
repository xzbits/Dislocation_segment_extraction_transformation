import numpy as np
import os


class MultipleTemperature:
    def __init__(self, plot_path, start_idx = None, last_idx = None):
        self.plot_path = plot_path
        self.start_idx = start_idx
        self.last_idx = last_idx
        self.file_list = os.listdir(self.plot_path)

    def generate_wv_psd(self):
        data = {}
        for temp in self.file_list:
            if self.start_idx is None or self.last_idx is None:
                if temp.startswith("t_"):
                    wv_psd_path = '\\'.join((self.plot_path, temp, "wavevector_psd.csv"))
                    wv_psd_perfect_path = "\\".join((self.plot_path, temp, "wavevector_psd_perfect.csv"))

                    temp_af_wv_psd = np.genfromtxt(wv_psd_path, delimiter=",")
                    temp_af_wv_psd_perfect = np.genfromtxt(wv_psd_perfect_path, delimiter=",")

                    data["{}_wv_psd".format(temp)] = temp_af_wv_psd
                    data["{}_wv_psd_perfect".format(temp)] = temp_af_wv_psd_perfect

            else:
                if temp.startswith("t_"):
                    wv_psd_path = '\\'.join((self.plot_path, temp, "wavevector_psd.csv"))
                    wv_psd_perfect_path = "\\".join((self.plot_path, temp, "wavevector_psd_perfect.csv"))

                    temp_af_wv_psd = np.genfromtxt(wv_psd_path, delimiter=",")
                    temp_af_wv_psd_perfect = np.genfromtxt(wv_psd_perfect_path, delimiter=",")

                    data["{}_wv_psd".format(temp)] = temp_af_wv_psd[self.start_idx:self.last_idx]
                    data["{}_wv_psd_perfect".format(temp)] = temp_af_wv_psd_perfect[self.start_idx:self.last_idx]

        return data


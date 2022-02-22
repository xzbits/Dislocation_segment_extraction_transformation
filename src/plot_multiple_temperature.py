from src.extract_data_allFrame import ExtractedData
import re
import os


class MultipleTemperature:
    def __init__(self, plot_path, dir_suffix, csv_file_suffix, length_lower_threshold=None, rms_length_list=None,
                 rerun=False, start_idx=None, last_idx=None, temp_list=None):
        self.plot_path = plot_path
        self.dir_suffix = dir_suffix
        self.csv_file_suffix = csv_file_suffix
        self.rerun = rerun
        self.start_idx = start_idx
        self.last_idx = last_idx
        self.temp_list = temp_list
        self.length_lower_threshold = length_lower_threshold
        self.rms_length_list = rms_length_list
        self.file_list = os.listdir(self.plot_path)

        if temp_list is None:
            self.multiple_temperature_wv_psd = self.generate_wv_psd()
        else:
            self.multiple_temperature_wv_psd = self.generate_wv_psd_templist()

    def get_temperature_list(self):
        output_list = list()
        for temp in self.file_list:
            if temp.startswith(self.dir_suffix):
                re_file_name = re.findall(r"t_(\d+)", temp)
                output_list.append(int(re_file_name[0]))
        output_list.sort()
        return output_list

    def __get_wv_psd_rms(self, data, temp):
        # Generate or check the existing of wv_psd and rms csv file
        wv_psd_rms_data = ExtractedData(convert_file_pth=os.path.join(self.plot_path, self.dir_suffix + "%i" % temp),
                                        file_suffix=self.csv_file_suffix,
                                        group_by_axes=2,
                                        sort_by_axes=0,
                                        no_segment_per_group=2,
                                        rerun=self.rerun,
                                        length_lower_threshold=self.length_lower_threshold,
                                        rms_length_list=self.rms_length_list)

        # Add wv, psd, and rms of "temp" temperature to "data"
        if self.start_idx is None or self.last_idx is None:
            temp_af_wv_psd = wv_psd_rms_data.af_wv_psd
            temp_af_wv_psd_perfect = wv_psd_rms_data.af_wv_psd_perfect
            temp_rms = wv_psd_rms_data.rms
            data["t_{}_wv_psd".format(temp)] = temp_af_wv_psd
            data["t_{}_wv_psd_perfect".format(temp)] = temp_af_wv_psd_perfect
            data["t_{}_rms".format(temp)] = temp_rms
        else:
            temp_af_wv_psd = wv_psd_rms_data.af_wv_psd
            temp_af_wv_psd_perfect = wv_psd_rms_data.af_wv_psd_perfect
            temp_rms = wv_psd_rms_data.rms
            data["t_{}_wv_psd".format(temp)] = temp_af_wv_psd[self.start_idx:self.last_idx]
            data["t_{}_wv_psd_perfect".format(temp)] = temp_af_wv_psd_perfect[self.start_idx:self.last_idx]
            data["t_{}_rms".format(temp)] = temp_rms[self.start_idx:self.last_idx]

    def generate_wv_psd_templist(self):
        data = {}
        for temp in self.temp_list:
            self.__get_wv_psd_rms(data, temp)
        return data

    def generate_wv_psd(self):
        data = {}
        for temp in self.file_list:
            self.__get_wv_psd_rms(data, temp)
        return data

# This scrip is using to:
# Extract the csv files that obtained from DXA ovito
# Compute wave vector and PSD
import numpy as np
from src.group_sort_segments import GroupSegments
import os


class ExtractedData:
    def __init__(self, convert_file_pth, file_suffix, rerun=False, group_by_axes=2, sort_by_axes=0,
                 no_segment_per_group=2, length_lower_threshold=None):
        self.convert_file_pth = convert_file_pth
        self.file_suffix = file_suffix
        self.group_by_axes = group_by_axes
        self.sort_by_axes = sort_by_axes
        self.no_segment_per_group = no_segment_per_group
        self.length_lower_threshold = length_lower_threshold
        self.no_component = None
        self.wv_psd_path = os.path.join(self.convert_file_pth, "wavevector_psd.csv")
        self.wv_psd_perfect_path = os.path.join(self.convert_file_pth, "wavevector_psd_perfect.csv")
        if self.__is_wv_psd_file_exit() is True and rerun is False:
            self.af_wv_psd = np.genfromtxt(self.wv_psd_path, delimiter=",")
            self.af_wv_psd_perfect = np.genfromtxt(self.wv_psd_perfect_path, delimiter=",")
        else:
            self.af_wv_psd, self.af_wv_psd_perfect = self.extract_data()

        print("This data contains {} groups,\n"
              "{} segments in total,\n"
              "and {} components per segment".format(
                int(self.af_wv_psd.shape[1] / self.no_segment_per_group / 2),
                int(self.af_wv_psd.shape[1] / 2),
                int(self.af_wv_psd.shape[0])))

    def __is_wv_psd_file_exit(self):
        psd_file = len([name for name in os.listdir(self.convert_file_pth) if name.startswith('wavevector_psd')])
        if psd_file == 2:
            return True
        else:
            return False

    @staticmethod
    def __save_file(data, path):
        # Save "data" in csv format into "path"
        np.savetxt(path, data, delimiter=",")

    def extract_data(self):
        no_frame = len([name for name in os.listdir(self.convert_file_pth) if name.startswith(self.file_suffix)])
        st_wth_name_con = self.file_suffix + "{}.csv"
        af_wv_psd = np.array([])
        af_wv_psd_perfect = np.array([])
        for frame in range(no_frame):
            # Load file:
            convert_file = os.path.join(self.convert_file_pth, st_wth_name_con)
            data_convert = np.genfromtxt(convert_file.format(frame), delimiter=",")

            # Preprocessing data
            group_segment = GroupSegments(data_convert,
                                          group_by_axes=self.group_by_axes,
                                          sort_by_axes=self.sort_by_axes,
                                          no_segment_per_group=self.no_segment_per_group,
                                          length_lower_threshold=self.length_lower_threshold)

            # Calculate Wavevector and PSD
            group_wavevector_psd, group_wavevector_psd_perfect = group_segment.cal_wavevector_psd()
            af_wv_psd = np.append(af_wv_psd, group_wavevector_psd)
            af_wv_psd_perfect = np.append(af_wv_psd_perfect, group_wavevector_psd_perfect)
            if self.no_component is None:
                self.no_component = group_segment.no_component

        af_wv_psd = af_wv_psd.reshape(no_frame, -1)
        af_wv_psd = np.mean(af_wv_psd, axis=0)
        af_wv_psd_perfect = af_wv_psd_perfect.reshape(no_frame, -1)
        af_wv_psd_perfect = np.mean(af_wv_psd_perfect, axis=0)
        af_wv_psd = af_wv_psd.reshape(-1, self.no_component).T
        af_wv_psd_perfect = af_wv_psd_perfect.reshape(-1, self.no_component).T
        self.__save_file(af_wv_psd, self.wv_psd_path)
        self.__save_file(af_wv_psd_perfect, self.wv_psd_perfect_path)

        return af_wv_psd, af_wv_psd_perfect

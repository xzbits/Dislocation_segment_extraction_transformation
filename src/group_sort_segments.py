import numpy as np
from src.hfda import measure


class GroupSegments:
    def __init__(self, data_convert, group_by_axes, sort_by_axes, no_segment_per_group, length_lower_threshold=None):
        self.data_convert = data_convert
        self.length_lower_threshold = length_lower_threshold
        self.group_by_axes = group_by_axes  # 0 is x-axes, 1 is y-axes, 2 is z-axes
        self.sort_by_axes = sort_by_axes    # 0 is x-axes, 1 is y-axes, 2 is z-axes
        self.no_segment_per_group = no_segment_per_group
        self.__get_no_component_segment()
        if self.self_test() is False:
            raise RuntimeError("The number of groups is not integer. \n"
                               "Currently, Total segments: {}, and desired segment per group: {}"
                               .format(self.no_segment, self.no_segment_per_group))
        else:
            pass
        self.group_segment = self.__grouping_sorting()

    def self_test(self):
        if self.no_component % self.no_segment_per_group == 0:
            return True
        else:
            return False

    def __get_no_component_segment(self):
        # Get number of segment in CSV file
        x_no_seg = np.where(self.data_convert[:, 0] == 0.0)
        y_no_seg = np.where(self.data_convert[:, 1] == 0.0)
        self.no_segment = len(np.intersect1d(x_no_seg, y_no_seg)) - 1
        self.no_component = int((len(self.data_convert[:, 0]) - self.no_segment - 1) / self.no_segment)

    def __extract_filter_segment(self, filtering_axes=1, upper_threshold=None):
        # Extract segments from convert segments
        idx_start = 1
        lst_segment = list()
        for i in range(self.no_segment):
            idx_split = idx_start + self.no_component
            segment_len = abs(self.data_convert[idx_start, filtering_axes] -
                              self.data_convert[idx_split-1, filtering_axes])
            if self.length_lower_threshold is not None:
                if self.length_lower_threshold < segment_len:
                    lst_segment.append(self.data_convert[idx_start:idx_split])
                else:
                    pass
            else:
                lst_segment.append(self.data_convert[idx_start:idx_split])
            idx_start = idx_split + 1
        if len(lst_segment) == 0:
            raise RuntimeError("There is no segments after filtering, please check again the lower threshold length")
        print('Number of segments: ', len(lst_segment))
        return lst_segment

    def __grouping_sorting(self):
        lst_segment = self.__extract_filter_segment()

        # Grouping segments by (self.group_by_axes)-axes value
        # with 0 = x-axes, 1 = y-axes, 2 = z-axes
        grouped_segments = list()
        if lst_segment[0].shape[1] == 3:
            lst_segment.sort(key=lambda x: x[0, self.group_by_axes])
            no_group = len(lst_segment) // self.no_segment_per_group
            for i in range(no_group):
                grouped_segments.append(
                    lst_segment[int(i * self.no_segment_per_group):int(i * self.no_segment_per_group
                                                                       + self.no_segment_per_group)])
        else:
            no_group = len(lst_segment) // self.no_segment_per_group
            for i in range(no_group):
                grouped_segments.append(
                    lst_segment[int(i * self.no_segment_per_group):int(i * self.no_segment_per_group
                                                                       + self.no_segment_per_group)])
            Warning("The extract data have only 2 axes data")

        # Sorting segments by mean of (self.sort_by_axes)-axes values
        # with 0 is x-axes, 1 is y-axes, 2 is z-axes
        for i in range(len(grouped_segments)):
            grouped_segments[i].sort(key=lambda x: np.mean(x[:, self.sort_by_axes]))

        return grouped_segments

    def cal_wavevector_psd(self):
        # Calculating Segments
        group_wv_psd = list()
        for one_group in self.group_segment:
            one_group_lst = list()
            for one_segment in one_group:
                # x_array - idx==0, y_array - idx==1
                x_array = one_segment[:, 0]
                y_array = one_segment[:, 1]
                # Calculating
                grid_spacing = np.diff(y_array)[0]
                yft = np.fft.fft(x_array)
                psd = (np.abs(yft) ** 2) / (self.no_component ** 2)
                freq = np.fft.fftfreq(x_array.shape[0], d=grid_spacing)
                wavevectors = 2 * np.pi * freq
                idx = np.argsort(freq)
                one_group_lst.append((wavevectors[idx], psd[idx]))
            group_wv_psd.append(one_group_lst)

        # Calculate Segments perfect
        group_wv_psd_perfect = list()
        for one_group_perfect in self.group_segment:
            x_array_perfect = np.array([])
            y_array_perfect = np.array([])
            for i in range(len(one_group_perfect)):
                x_array_perfect = np.append(x_array_perfect, one_group_perfect[i][:, 0])
                y_array_perfect = np.append(y_array_perfect, one_group_perfect[i][:, 1])
            x_array_perfect = np.mean(x_array_perfect.reshape(-1, self.no_component), axis=0)
            y_array_perfect = np.mean(y_array_perfect.reshape(-1, self.no_component), axis=0)
            grid_spacing_perfect = np.diff(y_array_perfect)[0]
            yft_perfect = np.fft.fft(x_array_perfect)
            psd_perfect = np.abs(yft_perfect) ** 2 / (self.no_component ** 2)
            freq_perfect = np.fft.fftfreq(x_array_perfect.shape[0], d=grid_spacing_perfect)
            wavevectors_perfect = 2 * np.pi * freq_perfect
            idx_perfect = np.argsort(freq_perfect)
            group_wv_psd_perfect.append((wavevectors_perfect[idx_perfect], psd_perfect[idx_perfect]))

        return group_wv_psd, group_wv_psd_perfect

    def cal_root_mean_square(self, a_var=None):
        if a_var is None:
            a_var = [0.9999, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1/64, 1/108]
        length_list = [int(element * self.no_component) for element in a_var]
        group_rms = list()
        for one_group in self.group_segment:
            segments_rms = list()
            for one_segment in one_group:
                one_segment_rms = list()
                one_segment_yy = list()
                x_array = one_segment[:, 0]
                y_array = one_segment[:, 1]
                h_rms = np.sum(x_array ** 2) / self.no_component
                grid_spacing = y_array[1] - y_array[0]
                # for i in range(a, self.no_component):
                for i in length_list:
                    accum_val = 0.0
                    for j in range(0, self.no_component - i):
                        first_term = x_array[j]
                        second_term = x_array[j + i]
                        accum_val += abs(first_term * second_term)
                    print(h_rms - (accum_val/(self.no_component - i)))
                    one_segment_rms.append(np.sqrt(h_rms - (accum_val/(self.no_component - i))))
                    one_segment_yy.append(i * grid_spacing)
                segments_rms.append((one_segment_rms, one_segment_yy))
            group_rms.append(segments_rms)
        return group_rms

    def cal_root_mean_square_old(self, a_var=None):
        if a_var is None:
            a_var = [0.9999, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1/64, 1/108]
        length_list = [int(element * self.no_component) for element in a_var]
        group_rms = list()
        for one_group in self.group_segment:
            segments_rms = list()
            for one_segment in one_group:
                one_segment_rms = list()
                one_segment_yy = list()
                x_array = one_segment[:, 0]
                y_array = one_segment[:, 1]
                grid_spacing = y_array[1] - y_array[0]
                for i in length_list:
                    accum_val = 0.0
                    for j in range(0, self.no_component - i):
                        first_term = x_array[j]
                        second_term = x_array[j + i]
                        accum_val += abs(first_term - second_term)
                    one_segment_rms.append(accum_val/(self.no_component - i))
                    one_segment_yy.append(i * grid_spacing)
                segments_rms.append((one_segment_rms, one_segment_yy))
            group_rms.append(segments_rms)
        return group_rms

    def cal_acf(self):
        group_rms = list()
        a = 1
        for one_group in self.group_segment:
            segments_rms = list()
            for one_segment in one_group:
                one_segment_rms = list()
                one_segment_yy = list()
                x_array = one_segment[:, 0]
                y_array = one_segment[:, 1]
                grid_spacing = y_array[1] - y_array[0]
                for i in range(a, self.no_component):
                    accum_val = 0.0
                    for j in range(0, self.no_component - i):
                        first_term = x_array[j]
                        second_term = x_array[j+i]
                        accum_val += (first_term - second_term)**2
                    one_segment_rms.append(0.5 * accum_val/(self.no_component - i))
                    one_segment_yy.append(i * grid_spacing)
                segments_rms.append((one_segment_rms, one_segment_yy))
            group_rms.append(segments_rms)
        return group_rms

    def cal_hfda(self):
        group_hfda = list()
        no_data = None
        for one_group in self.group_segment:
            segments_hdfa = list()
            for one_segment in one_group:
                no_data = np.floor(one_segment[:, 0].shape / 2).astype(int)
                k, L, _ = measure(one_segment[:, 0], no_data)
                segments_hdfa.append((k, L))
            group_hfda.append(segments_hdfa)
        return group_hfda, no_data

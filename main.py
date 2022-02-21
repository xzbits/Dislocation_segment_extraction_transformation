from src.ovito_extract import DumpDirectory
from src.plot_multiple_temperature import MultipleTemperature
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path_dumpfile = r""
    path_save_file = r""
    temperature_list = []
    dump_dir = DumpDirectory(path_dumpfile, path_save_file, temperature_desire=temperature_list)
    dump_dir.create_dir()
    dump_dir.generate_csv()

    # Average alloy
    temperature_list = []
    plot_path_temp_aa = r""
    dir_suffix_aa = "t_"
    csv_suffix_aa = "segment_points_con_frame_"
    multiple_temperature_aa = MultipleTemperature(plot_path_temp_aa,
                                                  dir_suffix_aa,
                                                  csv_suffix_aa,
                                                  length_lower_threshold=250,
                                                  temp_list=temperature_list)
    plot_data_aa = multiple_temperature_aa.multiple_temperature_wv_psd
    rms = plot_data_aa["t_1200_rms"]
    wv_psd = plot_data_aa["t_1200_wv_psd"]

    rms_seg_1 = rms[:, 0]
    rms_seg_2 = rms[:, 2]
    rms_seg_3 = rms[:, 4]
    rms_seg_4 = rms[:, 6]

    t_seg_1 = rms[:, 1]
    t_seg_2 = rms[:, 3]
    t_seg_3 = rms[:, 5]
    t_seg_4 = rms[:, 7]

    plt.loglog(t_seg_1, rms_seg_1, "b-")
    plt.loglog(t_seg_2, rms_seg_2, "r-")
    plt.loglog(t_seg_3, rms_seg_3, "g-")
    plt.loglog(t_seg_4, rms_seg_4, "y-")

    plt.show()

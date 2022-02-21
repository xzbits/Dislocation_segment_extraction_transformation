# from src.ovito_extract import DumpDirectory
from src.extract_data_allFrame import ExtractedData
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # path_dumpfile = r""
    # path_save_file = r""
    # temperature_list = [1200]
    # dump_dir = DumpDirectory(path_dumpfile, path_save_file, temperature_desire=temperature_list)
    # dump_dir.create_dir()
    # dump_dir.generate_csv()

    csv_path = r"C:\Users\Hoa\Desktop\cur_work\1200_data\extract_file\extracted_data\t_1200"
    data_wp = ExtractedData(convert_file_pth=csv_path,
                            file_suffix="segment_points_con_frame_",
                            group_by_axes=2,
                            sort_by_axes=0,
                            no_segment_per_group=2,
                            length_lower_threshold=250,
                            rms_length_list=[0.9999, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1/64, 1/108],
                            rerun=True)

    rms = data_wp.rms
    wv_psd = data_wp.af_wv_psd
    print(rms.shape)
    rms_seg_1 = rms[:, 0]
    rms_seg_2 = rms[:, 2]
    rms_seg_3 = rms[:, 4]
    rms_seg_4 = rms[:, 6]

    t_seg_1 = rms[:, 1]
    t_seg_2 = rms[:, 3]
    t_seg_3 = rms[:, 5]
    t_seg_4 = rms[:, 7]

    plt.loglog(t_seg_1, rms_seg_1, "b-")
    plt.loglog(t_seg_2, rms_seg_2, "r-", label="dislocation segment 2, RMS = 110.568")
    plt.loglog(t_seg_3, rms_seg_3, "g-", label="dislocation segment 3, RMS = 240.414")
    plt.loglog(t_seg_4, rms_seg_4, "y-", label="dislocation segment 4, RMS = 277.3147")

    plt.show()

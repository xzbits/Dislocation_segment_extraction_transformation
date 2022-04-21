"""This script is using to:
- Extract dislocation lines using DXA which is implemented in Ovito
- Wrap dislocation lines back to simulation cell
- Interpolating the dislocation points to achieve equispaced data point
"""
from ovito.io import import_file, export_file
from ovito.modifiers import DislocationAnalysisModifier
from ovito.modifiers import AffineTransformationModifier
from ovito.modifiers import WrapPeriodicImagesModifier
from ovito.data import DislocationNetwork
from multiprocessing import Process, cpu_count, current_process, Pool
import os
import re
import numpy as np


# Wrapper the dislocation back to simulation cell
class Wrapper:
    def __init__(self, segment_points, ylim):
        self.segment_points = segment_points
        self.ylim = ylim
        self.y_array = segment_points[:, 1]
        self.wrap = self.processing()

    def is_wrap_side(self):
        if np.max(self.y_array) > self.ylim[1]:
            # Right wrap
            return 0
        elif np.min(self.y_array) < self.ylim[0]:
            # Left wrap
            return 1
        else:
            # No points out of simulation box boundaries
            return -1

    def processing(self):
        if self.is_wrap_side() == 0:
            right_idx = np.where(self.y_array > self.ylim[1])[0]
            in_array = np.array(self.segment_points[:right_idx[0], :])
            wrap_array = self.segment_points[right_idx[0]-1:, :]
            wrap_array[:, 1] -= abs(self.y_array[-1] - self.y_array[0])
            fix_array = np.r_[wrap_array, in_array]
            return fix_array
        elif self.is_wrap_side() == 1:
            left_idx = np.where(self.y_array < self.ylim[0])[0]
            in_array = np.array(self.segment_points[left_idx[-1]+1:, :])
            wrap_array = self.segment_points[:left_idx[-1]+2, :]
            wrap_array[:, 1] += abs(self.y_array[-1] - self.y_array[0])
            fix_array = np.r_[in_array, wrap_array]
            return fix_array
        else:
            return self.segment_points


class DumpDirectory:
    def __init__(self, path, save_path, dump_file_suffix, temperature_desire=None):
        """
        IMPORTANT NOTE: An dump folder must have one type of file name.
        :param path: Dump files path
        """
        self.temperature_desire = temperature_desire
        self.path = path
        self.save_path = save_path
        self.dump_file_suffix = dump_file_suffix
        self.files_list = os.listdir(self.path)
        self.file_name_type = self.classify_file_name(self.files_list[0])
        if self.file_name_type == -1:
            raise ValueError(self.files_list[0])
        self.re_term = self.get_re_term(self.files_list[0])
        if self.file_name_type == 0:
            self.temperature_list = self.extract_temperature_list()
        else:
            pass

    @staticmethod
    def classify_file_name(file_name):
        """
        Classify file name based on HEADislocation project
        :param file_name: File name.
        :return: 3 type of files name (file name with Temperature and Time, file name with Time, others)
        """
        re_file_name = re.findall(r"_(\d+)_(\d+).cfg", file_name)
        if re_file_name.__len__() == 1:
            # File name with Temperature and Time
            return 0
        elif re_file_name.__len__() == 0:
            # File name with Time
            return 1
        else:
            # Others
            return -1

    @staticmethod
    def get_re_term(file_name):
        re_file_name = re.findall(r"_(\d+)_(\d+).cfg", file_name)
        if re_file_name.__len__() == 1:
            # File name with Temperature and Time
            return r"_(\d+)_(\d+).cfg"
        elif re_file_name.__len__() == 0:
            # File name with Time
            return r"_(\d+).cfg"
        else:
            raise RuntimeError("Cannot identify regular expresion term for this {} file".format(file_name))

    def extract_temperature_list(self):
        output_list = list()
        for file_name in self.files_list:
            if self.classify_file_name(file_name) != self.file_name_type:
                raise RuntimeError("There are more than 2 type of file names existing in the directory: {} and {}"
                                   .format(self.files_list[0], file_name))
            else:
                re_name = re.findall(self.re_term, file_name)
                if int(re_name[0][0]) in self.temperature_desire:
                    pass
                else:
                    output_list.append(int(re_name[0][0]))
        return output_list

    @staticmethod
    def get_simulation_box(data):
        cell = data.cell[...]  # cell contains the original dislocation coordinate

        origin = cell[:, 3]
        p0 = np.zeros((1, 3), dtype=float) + origin
        p4 = p0 + cell[:, 2]
        p5 = p4 + cell[:, 0]
        p6 = p5 + cell[:, 1]

        xlim = (origin[0], p6[0, 0])
        ylim = (origin[1], p6[0, 1])
        zlim = (origin[2], p6[0, 2])

        return xlim, ylim, zlim

    def extract_pipeline_raw(self):
        for temp in self.temperature_desire:
            pipeline_path = os.path.join(self.path, self.dump_file_suffix + "{}_*.cfg".format(temp))
            save_path = os.path.join(self.save_path, "extracted_wrapped_data_raw", "t_{}".format(temp))

            pipeline = import_file(pipeline_path)
            modifier = DislocationAnalysisModifier(line_point_separation=0, line_smoothing_level=0)
            modifier.input_crystal_structure = DislocationAnalysisModifier.Lattice.FCC
            pipeline.modifiers.append(modifier)
            pipeline.modifiers.append(AffineTransformationModifier(relative_mode=False,
                                                                   target_cell=pipeline.compute().cell[...],
                                                                   only_selected=True))
            pipeline.modifiers.append(WrapPeriodicImagesModifier())
            data = pipeline.compute()
            xlimit, ylimit, zlimit = self.get_simulation_box(data)

            for i in range(pipeline.source.num_frames):
                print("Frame: {}".format(i))
                data = pipeline.compute(i)
                segments_per_frame = np.array([[0, 0, 0]])
                for segment in data.dislocations.segments:
                    aa = Wrapper(segment.points, ylimit)
                    segments_per_frame = np.r_[segments_per_frame, aa.wrap]
                    segments_per_frame = np.r_[segments_per_frame, np.array([[0, 0, 0]])]
                np.savetxt(os.path.join(save_path, "segment_points_wrapped_raw_frame_{}.csv".format(i)),
                           segments_per_frame, delimiter=",")

    def extract_pipeline(self, pipeline_path, save_path, start=None, stop=None):
        pipeline = import_file(pipeline_path)
        modifier = DislocationAnalysisModifier(line_point_separation=0, line_smoothing_level=0)
        modifier.input_crystal_structure = DislocationAnalysisModifier.Lattice.FCC
        pipeline.modifiers.append(modifier)
        pipeline.modifiers.append(AffineTransformationModifier(relative_mode=False,
                                                               target_cell=pipeline.compute().cell[...],
                                                               only_selected=True))
        pipeline.modifiers.append(WrapPeriodicImagesModifier())
        data = pipeline.compute()
        xlimit, ylimit, zlimit = self.get_simulation_box(data)

        start_frame = 0 if start is None else start
        end_frame = pipeline.source.num_frames if start is None else stop

        for i in range(start_frame, end_frame):
            print("Frame: {}".format(i))
            data = pipeline.compute(i)
            segments_per_frame = np.array([[0, 0, 0]])
            segments_con_per_frame = np.array([[0, 0, 0]])
            for segment in data.dislocations.segments:
                aa = Wrapper(segment.points, ylimit)
                segments_per_frame = np.r_[segments_per_frame, aa.wrap]
                segments_per_frame = np.r_[segments_per_frame, np.array([[0, 0, 0]])]

                x_array = aa.wrap[:, 0]
                y_array = aa.wrap[:, 1]
                segments_per_frame_y_min = np.min(y_array)
                segments_per_frame_y_max = np.max(y_array)
                equal_segments = np.linspace(segments_per_frame_y_min, segments_per_frame_y_max, num=3000)

                temp_array = np.array([])
                idx_temp_array = np.array([], dtype=int)
                for j in range(0, y_array.shape[0] - 1):
                    in_segment_idx = np.where((y_array[j] <= equal_segments) & (equal_segments <= y_array[j + 1]))[0]
                    if in_segment_idx.shape[0] != 0:
                        for idd in in_segment_idx:
                            if idd not in idx_temp_array:
                                y_hat = equal_segments[idd]
                                x_i = x_array[j]
                                y_i = y_array[j]
                                x_i1 = x_array[j + 1]
                                y_i1 = y_array[j + 1]
                                x_hat = x_i + (((x_i1 - x_i) * (y_hat - y_i)) / (y_i1 - y_i))
                                temp_array = np.append(temp_array, x_hat)
                            else:
                                continue
                    else:
                        continue
                    idx_temp_array = np.append(idx_temp_array, in_segment_idx)
                temp_array = np.c_[temp_array, equal_segments, np.full(equal_segments.shape, segment.points[0, 2])]
                segments_con_per_frame = np.r_[segments_con_per_frame, temp_array]
                segments_con_per_frame = np.r_[segments_con_per_frame, np.array([[0, 0, 0]])]
            np.savetxt("{}/segment_points_con_frame_{}.csv".format(save_path, i), segments_con_per_frame, delimiter=",")

    def get_dump_file_time_steps(self, dumpfile_list):
        output = []
        for file_name in dumpfile_list:
            time_steps = int(re.findall(r"{}(\d+)_(\d+)".format(self.dump_file_suffix), file_name)[0][1])
            output.append(time_steps)
        output.sort()
        return output

    def generate_csv(self):
        if self.temperature_desire is None:
            if self.file_name_type == 0:
                for temp in self.temperature_list:
                    print("Temperature: {}".format(temp))
                    pipelines_path = os.path.join(self.path, self.dump_file_suffix + "{}_*.cfg".format(temp))
                    save_path = os.path.join(self.save_path, "extracted_data", "t_{}".format(temp))
                    self.extract_pipeline(pipelines_path, save_path)
            elif self.file_name_type == 1:
                pipelines_path = os.path.join(self.path, self.dump_file_suffix + "*.cfg")
                save_path = os.path.join(self.save_path, "extracted_data")
                self.extract_pipeline(pipelines_path, save_path)
            else:
                pass
        else:
            if isinstance(self.temperature_desire, list):
                for temp in self.temperature_desire:
                    print("Temperature: {}".format(temp))
                    pipelines_path = os.path.join(self.path, self.dump_file_suffix + "{}_*.cfg".format(temp))
                    save_path = os.path.join(self.save_path, "extracted_data", "t_{}".format(temp))
                    self.extract_pipeline(pipelines_path, save_path)
            else:
                raise RuntimeError("Please set temperature_desire type is list object")

    def parallel_generate_csv(self, files, start, stop):
        process_name = current_process().name
        print("Starting %s" % process_name)
        if isinstance(files, list):
            if isinstance(start, int) == False or isinstance(stop, int) == False:
                raise ValueError("start and step should be integer")

            for temp in self.temperature_desire:
                pipeline_path_list = [os.path.join(self.path, self.dump_file_suffix + "{}_{}.cfg".format(temp, dump))
                                      for dump in files]
                save_path = os.path.join(self.save_path, "extracted_data", "t_{}".format(temp))
                self.extract_pipeline(pipeline_path_list, save_path, start=start, stop=stop)
        else:
            raise ValueError("the input should be None or a list or not clarified ")

    def parallel_distribute(self, cur_no_cores=None):
        if self.temperature_desire is not None:
            for temp in self.temperature_desire:
                print("Temperature: {}".format(temp))
                cur_no_cores = cpu_count() if cur_no_cores is None else cur_no_cores
                file_list = [name for name in os.listdir(self.path)
                             if name.startswith(self.dump_file_suffix + "{}_".format(temp))]
                no_files = len(file_list)
                no_partition = no_files // cur_no_cores + 1 \
                    if no_files % cur_no_cores > cur_no_cores / 2 or no_files < cur_no_cores \
                    else no_files // cur_no_cores
                timestep_list = self.get_dump_file_time_steps(file_list)
                job_list = []
                start_idx = 0
                no_parallel_run = cur_no_cores if no_files > cur_no_cores else no_files
                for i in range(no_parallel_run):
                    stop_idx = start_idx + no_partition
                    if i != cur_no_cores - 1:
                        job_files = timestep_list[start_idx:stop_idx]
                    else:
                        job_files = timestep_list[start_idx:]

                    if job_files.__len__() == 1:
                        job_list.append(Process(name="Extracting a dump file %i" % (job_files[0]),
                                                target=self.parallel_generate_csv,
                                                args=(job_files, start_idx, stop_idx)))
                    else:
                        job_list.append(Process(name="Extracting %i dump files from %i to %i"
                                                     % (len(job_files), job_files[0], job_files[-1]),
                                                target=self.parallel_generate_csv,
                                                args=(job_files, start_idx, stop_idx)))
                    start_idx = stop_idx

                for process in job_list:
                    process.start()
                for process in job_list:
                    process.join()
        else:
            raise RuntimeError("Please clarify temperature_desire var")

    def parallel_distribute_with_pool(self, cur_no_cores=None):
        if self.temperature_desire is not None:
            for temp in self.temperature_desire:
                print("Temperature: {}".format(temp))
                cur_no_cores = cpu_count() if cur_no_cores is None else cur_no_cores
                file_list = [name for name in os.listdir(self.path)
                             if name.startswith(self.dump_file_suffix + "{}_".format(temp))]
                no_files = len(file_list)
                no_partition = no_files // cur_no_cores + 1 \
                    if no_files % cur_no_cores > cur_no_cores / 2 or no_files < cur_no_cores \
                    else no_files // cur_no_cores
                timestep_list = self.get_dump_file_time_steps(file_list)
                job_list = []
                start_idx = 0
                no_parallel_run = cur_no_cores if no_files > cur_no_cores else no_files
                pool_run = Pool(processes=no_parallel_run)
                for i in range(no_parallel_run):
                    stop_idx = start_idx + no_partition
                    if i != cur_no_cores - 1:
                        job_files = timestep_list[start_idx:stop_idx]
                    else:
                        job_files = timestep_list[start_idx:]

                    job_list.append((job_files, start_idx, stop_idx))

                    start_idx = stop_idx

                pool_run.starmap(self.parallel_generate_csv, job_list)
        else:
            raise RuntimeError("Please clarify temperature_desire var")

    def create_dir(self):
        # Create extracted_data directory
        path_extract = os.path.join(self.save_path, "extracted_data")
        try:
            os.mkdir(path_extract)
        except OSError:
            print("Creation of the temperature with time directory %s failed" % path_extract)
        else:
            print("Successfully created the  temperature with time directory %s" % path_extract)
        if self.temperature_desire is None:
            # Create extracted_data sub-directory
            if self.file_name_type == 0:
                for one_temperature in self.extract_temperature_list():
                    path = os.path.join(self.save_path, "extracted_data", "t_{}".format(one_temperature))
                    try:
                        os.mkdir(path)
                    except OSError:
                        print("Creation of the temperature with time directory %s failed" % path)
                    else:
                        print("Successfully created the  temperature with time directory %s" % path)
            elif self.file_name_type == 1:
                pass
            else:
                raise ValueError("Cannot recognize the type of dump file names!!!!!!!")
        else:
            # Create extracted_data sub-directory
            if self.file_name_type == 0:
                for one_temperature in self.temperature_desire:
                    path = os.path.join(self.save_path, "extracted_data", "t_{}".format(one_temperature))
                    try:
                        os.mkdir(path)
                    except OSError:
                        print("Creation of the temperature with time directory %s failed" % path)
                    else:
                        print("Successfully created the  temperature with time directory %s" % path)
            elif self.file_name_type == 1:
                pass
            else:
                raise ValueError("Cannot recognize the type of dump file names!!!!!!!")

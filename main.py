from src.ovito_extract import DumpDirectory

if __name__ == "__main__":
    path_dumpfile = r""
    path_save_file = r""
    temperature_list = [1200]
    dump_dir = DumpDirectory(path_dumpfile, path_save_file, temperature_desire=temperature_list)
    dump_dir.create_dir()
    dump_dir.generate_csv()

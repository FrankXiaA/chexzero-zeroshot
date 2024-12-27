import argparse
from pathlib import Path
from data_process import get_cxr_paths_list, img_to_hdf5, get_cxr_path_csv, write_report_csv
import matplotlib



if __name__ == "__main__":

    csv_out_path = "C:/Users/26984/PycharmProjects/pythonProject/Github/CheXzero-main/CheXzero-main/data/cxr_paths.csv"
    chest_x_ray_path = "C:/Users/26984/Desktop/Research/PALM/chexlocalize/CheXpert/test"
    chest_x_ray_path_1 = "C:/Users/26984/PycharmProjects/pythonProject/Github/CheXzero-main/CheXzero-main/data/few_shot_imgs"
    cxr_out_path ="C:/Users/26984/PycharmProjects/pythonProject/Github/CheXzero-main/CheXzero-main/data/cxr.h5"
    mimic_impressions_path = "C:/Users/26984/PycharmProjects/pythonProject/Github/CheXzero-main/CheXzero-main/data/mimic_impressions.csv"
    cxr_path = ""


    """cxr_dir = Path(chest_x_ray_path)
    cxr_paths = list(cxr_dir.rglob("*.jpg"))
    cxr_paths = list(filter(lambda x: "view1" in str(x), cxr_paths))  # filter only first frontal views
    cxr_paths = sorted(cxr_paths)  # sort to align with groundtruth
    assert (len(cxr_paths) == 500)

    img_to_hdf5(cxr_paths, cxr_out_path)"""



    get_cxr_path_csv(csv_out_path, chest_x_ray_path_1)
    cxr_paths = get_cxr_paths_list(csv_out_path)
    img_to_hdf5(cxr_paths, cxr_out_path)
    write_report_csv(cxr_paths, mimic_impressions_path)



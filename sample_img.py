import pandas as pd
import os
import shutil

# Define the path to the few-shot samples CSV file
few_shot_csv_path = r"D:\Research\PALM\chexlocalize\CheXpert\data\chexpertchestxrays-u20210408\few_shot_samples.csv"

# Define the base path where all the images are stored
base_image_path = r"D:\Research\PALM\chexlocalize\CheXpert\data\chexpertchestxrays-u20210408\All studies"

# Define the path to the destination folder for few-shot images
few_shot_folder = r"D:\Research\PALM\chexlocalize\CheXpert\data\chexpertchestxrays-u20210408\few_shot_imgs"

os.makedirs(few_shot_folder, exist_ok=True)

# Read the few-shot samples CSV file into a DataFrame
few_shot_df = pd.read_csv(few_shot_csv_path)
print(len(few_shot_df))
count=0
# Iterate over each row in the DataFrame
for index, row in few_shot_df.iterrows():
    count += 1
    # Get the relative path to the image
    relative_image_path = row['Path']
    # Construct the full path to the source image
    source_image_path = os.path.join(base_image_path, relative_image_path)

    # Construct the destination path for the image
    dest_image_path = os.path.join(few_shot_folder, relative_image_path)

    # Ensure the destination directory exists
    os.makedirs(os.path.dirname(dest_image_path), exist_ok=True)

    # Copy the image to the destination folder
    shutil.copy2(source_image_path, dest_image_path)

print(f"Specified images have been copied to {few_shot_folder}")
print(f'count={count}')

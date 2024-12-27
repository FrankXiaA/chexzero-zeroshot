import pandas as pd
import os

# Define the path to your CSV file
csv_path = r"D:\Research\PALM\chexlocalize\CheXpert\data\chexpertchestxrays-u20210408\train_visualCheXbert.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_path)

# Define the columns for the different classes (excluding 'Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA')
class_columns = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
                 "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
                 "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"]

# Create an empty DataFrame to store the samples
few_shot_samples = pd.DataFrame()

# Sample 16 instances for each class
for col in class_columns:
    class_samples = df[df[col] == 1.0].sample(n=16, random_state=42, replace=True)
    few_shot_samples = pd.concat([few_shot_samples, class_samples], ignore_index=True)

# Drop duplicates that might have been added due to the sampling with replacement
few_shot_samples = few_shot_samples.drop_duplicates()

# Save the few-shot samples to a new CSV file
output_path = os.path.join(os.path.dirname(csv_path), "few_shot_samples.csv")
few_shot_samples.to_csv(output_path, index=False)

print(f"Few-shot samples saved to {output_path}")

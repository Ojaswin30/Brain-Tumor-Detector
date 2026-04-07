import kagglehub
import os
import shutil

# Download dataset
path = kagglehub.dataset_download("jakeshbohaju/brain-tumor")

print("Downloaded to:", path)

# Move dataset to your project folder (optional but cleaner)
destination = "Datasets"

if not os.path.exists(destination):
    shutil.copytree(path, destination)

print("Dataset ready in ./Datasets")


# import os
# import pandas as pd
# import shutil

# # Paths
# image_folder = "/workspaces/Brain-Tumor-Detector/Datasets/Brain Tumor/Brain Tumor"  # Adjust if your images are in a subfolder
# csv_file = "/workspaces/Brain-Tumor-Detector/Datasets/Brain Tumor.csv"

# output_dir = "Datasets"
# yes_dir = os.path.join(output_dir, "yes")
# no_dir = os.path.join(output_dir, "no")

# os.makedirs(yes_dir, exist_ok=True)
# os.makedirs(no_dir, exist_ok=True)

# # Load CSV
# df = pd.read_csv(csv_file)

# print("Columns:", df.columns)

# # Loop through CSV
# for _, row in df.iterrows():
#     image_name = str(row["Image"]) + ".jpg"   # 🔥 important fix
#     label = row["Class"]

#     src_path = os.path.join(image_folder, image_name)

#     if not os.path.exists(src_path):
#         print(f"Missing: {image_name}")
#         continue

#     if label == 1:
#         dst_path = os.path.join(yes_dir, image_name)
#     else:
#         dst_path = os.path.join(no_dir, image_name)

#     shutil.copy(src_path, dst_path)

# print("✅ Dataset organized successfully!")
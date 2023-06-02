# import os

# # set the parent directory containing all the folders
# parent_dir = 'D:/Programming/Ayam/ayamYOLO/Image_weighing/duck2'

# # iterate through all the folders and remove all items inside them
# for i in range(1, 41):
#     folder_path = os.path.join(parent_dir, str(i))
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#         os.remove(file_path)

import os
import shutil

image_folder = "D:/Programming/Ayam/ayamYOLO/yolov5/runs/detect/exp14/crops/ayam/"
folder_prefix = ""
output_folder = "D:/Programming/Ayam/ayamYOLO/Image_weighing/duck2"

# Loop through the images and the folders
for i in range(1, 43):
    # Get the image filename
    image_filename = f"tes3_{i}.jpg"
    image_path = os.path.join(image_folder, image_filename)

    # Get the folder name and path
    folder_name = folder_prefix + str(i)
    folder_path = os.path.join(output_folder, folder_name)

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Copy the image to the folder
    shutil.copy(image_path, folder_path)
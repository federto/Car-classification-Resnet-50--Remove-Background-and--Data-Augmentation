"""
This script will be used to remove noisy background from cars images to
improve the quality of our data and get a better model.
The main idea is to use a vehicle detector to extract the car
from the picture, getting rid of all the background, which may cause
confusion to our CNN model.
We must create a new folder to store this new dataset, following exactly the
same directory structure with its subfolders but with new images.
"""
import argparse
from utils.detection import get_vehicle_coordinates
import os
import cv2
from PIL import Image



def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Full path to the directory having all the cars images. Already "
            "splitted in train/test sets. E.g. "
            "`/home/app/src/data/car_ims_v1/`."
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "cropped pictures. E.g. `/home/app/src/data/car_ims_v2/`."
        ),
    )

    args = parser.parse_args()

    return args


def main(data_folder, output_data_folder):
    """
    Parameters
    ----------
    data_folder : str
        Full path to train/test images folder.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        cropped images.
    """
    # For this function, you must:
    #   1. Iterate over each image in `data_folder`, you can
    #      use Python `os.walk()` or `utils.waldir()``
    #   2. Load the image
    #   3. Run the detector and get the vehicle coordinates, use
    #      utils.detection.get_vehicle_coordinates() for this task
    #   4. Extract the car from the image and store it in
    #      `output_data_folder` with the same image name. You may also need
    #      to create additional subfolders following the original
    #      `data_folder` structure.
    # TODO
    
    # Iterate in train data folder
    for (dirpath,dirs,files) in os.walk(data_folder, topdown=True):
        for filename in files:
            
            label = os.path.split(dirpath)

            # Read the photo
            all_file = cv2.imread(os.path.join(dirpath,filename))

            # Mark the cut points and cut
            file_mark = get_vehicle_coordinates(all_file)
            x_min = file_mark[0]
            y_min = file_mark[1]
            x_max = file_mark[2]
            y_max = file_mark[3]
            new_img = Image.open(os.path.join(dirpath,filename))
            box = (x_min,y_min,x_max,y_max)
            # New cut Image
            new_img = new_img.crop(box)

            # Save the new images in a new carpet with the label name
            path_train_test = os.path.join(output_data_folder,label[0])
            subdir = label[0].split('/')
            subdir=subdir[len(subdir)-1]
            path_train_test = os.path.join(output_data_folder,subdir,label[1])
            path_file = os.path.join(output_data_folder,subdir,label[1],filename)
            
            # If we cant detect the car o trucks pass the entire imagen
            if not os.path.exists(path_train_test):
                os.makedirs(path_train_test)
            new_img.save(path_file)
    


if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.output_data_folder)
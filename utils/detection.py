from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
import cv2
# The chosen detector model is "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
# because this particular model has a good balance between accuracy and speed.
# You can check the following Colab notebook with examples on how to run
# Detectron2 models
# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5.
# Assign the loaded detection model to global variable DET_MODEL
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
)
try:
    # It may fail if no GPU was found
    DET_MODEL = DefaultPredictor(cfg)
except:
    # Load the model for CPU only
    print(
        f"Failed to load Detection model on GPU, "
        "trying with CPU. Message: {exp}."
    )
    cfg.MODEL.DEVICE='cpu'
    DET_MODEL = DefaultPredictor(cfg)


def get_vehicle_coordinates(img):
    """
    This function will run an object detector (loaded in DET_MODEL model
    variable) over the the image, get the vehicle position in the picture
    and return it.

    Many things should be taken into account to make it work:
        1. Current model being used can detect up to 80 different objects,
           we're only looking for 'cars' or 'trucks', so you should ignore
           other detected objects.
        2. The object detector may find more than one vehicle in the picture,
           you must then, choose the one with the largest area in the image.
        3. The model can also fail and detect zero objects in the picture,
           in that case, you should return coordinates that cover the full
           image, i.e. [0, 0, width, height].
        4. Coordinates values must be integers, we're making reference to
           a position in a numpy.array, we can't use float values.

    Parameters
    ----------
    img : numpy.ndarray
        Image in RGB format.

    Returns
    -------
    box_coordinates : list
        List having bounding box coordinates as [left, top, right, bottom].
        Also known as [x1, y1, x2, y2].
    """
    # TODO
    
    # Get the img shape
    height, width = img.shape[:2]
    
    # Process the image
    outputs = DET_MODEL(img)
    
    # Get the box values and labels values of the image
    box = outputs["instances"].pred_boxes
    labels = outputs["instances"].pred_classes

    dict_areas = {}
    box_coordinates = []

    # If we detect a car or a truck cut the image in the corrdenates that we get in box
    if  2 in labels or 7 in labels :
        
        for pos, clase in enumerate(labels):
                dict_areas[pos] = box[pos].area().item()

        idx_big_car = max(dict_areas, key=dict_areas.get)

        big_car_box = box[idx_big_car].__getitem__(0)

        for val in big_car_box:
            for coord in val:
                box_coordinates.append(int(coord.item()))
    
    # Pass the entire image
    else :
        box_coordinates=[0, 0, width, height]

    
    return box_coordinates

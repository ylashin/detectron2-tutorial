# version inspection
import detectron2
print(f"Detectron2 version is {detectron2.__version__}")

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import cv2
import requests
import numpy as np

# load an image of Lionel Messi with a ball
image_reponse = requests.get("https://upload.wikimedia.org/wikipedia/commons/4/41/Leo_Messi_v_Almeria_020314_%28cropped%29.jpg")
image_as_np_array = np.frombuffer(image_reponse.content, np.uint8)
image = cv2.imdecode(image_as_np_array, cv2.IMREAD_COLOR)


# create config
cfg = get_cfg()
# below path applies to current installation location of Detectron2
cfgFile = "/usr/local/lib/python3.8/site-packages/detectron2/model_zoo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
cfg.merge_from_file(cfgFile)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
cfg.MODEL.DEVICE = "cpu" # we use a CPU Detectron copy

# create predictor
predictor = DefaultPredictor(cfg)

# make prediction
output = predictor(image)
print(output)


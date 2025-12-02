# FCS-YOLO
A new model for high precise multi-scale and subtle crack detection under complex backgrounds
The corresponding paper title for this project is “Spatial-Temporal Evolution of Landslide Cracks Revealed by UAV Photogrammetry: The FCS-YOLO Model for Precise Crack Extraction Under Complex Backgrounds ”.
In the future, various data and codes in the paper will gradually be opened up.
# Train Your Net
train.py --weights '' --cfg your yaml address --data datasets/data.yaml  --name your name --batch-size 16 --workers 0 --cache --epochs 1000 --save-period 1
# Test the FCS-YOLO Model
FCS-YOLO.py -- The training weights of FCS-YOLO model
YOLOv8n.py -- The training weights of YOLOv8 model
You can use the data in test_data.zip to test the performance of the training weights
# Cracks Datasets
Baidu Disk：cracks.zip -- https://pan.baidu.com/s/1fvL690XuoG51xZYCkwmvkA Extract Code: gsfu 

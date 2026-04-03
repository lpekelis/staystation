import os

from ultralytics import YOLO

model_name = os.environ["YOLO_MODEL_NAME"]
model = YOLO(f"{model_name}.pt")
model.export(format="ncnn")
print(f"NCNN export complete: {model_name}_ncnn_model")

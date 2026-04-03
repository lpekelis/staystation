from ultralytics import YOLO

model = YOLO("yolo26s.pt")
model.export(format="ncnn")
print("NCNN export complete")

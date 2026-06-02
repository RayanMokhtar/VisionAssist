from ultralytics import YOLO

# Load a YOLOv8 model (e.g., yolov8n.pt)
model = YOLO("yolo26l.pt")

# Export the model to ONNX format
# opset=12 is recommended for compatibility
# simplify=True optimizes the model graph
# dynamic=False ensures fixed input size, often better for C++ deployment
# imgsz=640 sets the input image size
model.export(format="onnx", opset=12, simplify=True, dynamic=True)
print("Model exported successfully to yolov8x.onnx")

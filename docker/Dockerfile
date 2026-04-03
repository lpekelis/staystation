FROM ultralytics/ultralytics:latest-arm64

# Downgrade torch for Pi4 compatibility
RUN pip uninstall torch torchvision -y && \
    pip install torch==2.5.1 torchvision

# Install server dependencies
RUN pip install fastapi uvicorn python-multipart

# Export YOLO model to NCNN (cached in image layer)
COPY docker/export_model.py /app/export_model.py
RUN python /app/export_model.py

# Verify export succeeded
RUN ls -la /app/yolo11n_ncnn_model/

# Copy inference server
COPY docker/inference_server.py /app/inference_server.py

EXPOSE 8080
CMD ["uvicorn", "inference_server:app", "--host", "0.0.0.0", "--port", "8080", "--app-dir", "/app"]

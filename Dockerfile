FROM python:3.9-slim

   # Install mlflow (if it's not already in the image)
   RUN pip install mlflow

   # Set working directory
   WORKDIR /app

   # Copy project files
   COPY . .

   # Expose port
   EXPOSE 5000

   # Run mlflow ui
   CMD ["mlflow", "ui", "--backend-store-uri", "file:///app/mlruns", "--host", "0.0.0.0", "--port", "5000"]

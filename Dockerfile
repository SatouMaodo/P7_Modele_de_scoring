FROM buildpack-deps:buster-scm

   # Install Docker client
   RUN curl -fsSL https://get.docker.com -o get-docker.sh && \
       sh get-docker.sh && \
       rm get-docker.sh

   # Install mlflow
   RUN pip install mlflow

   # Set working directory
   WORKDIR /app

   # Copy project files
   COPY . .

   # Expose port
   EXPOSE 5000

   # Run mlflow ui
   CMD ["mlflow", "ui", "--backend-store-uri", "file:///app/mlruns/", "--host", "0.0.0.0", "--port", "5000"]

#web: gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT main:app
web: uvicorn main:app --host 0.0.0.0 --port $PORT

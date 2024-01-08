FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . /app
EXPOSE 3000
CMD ["uvicorn", "app:app", "--reload", "--port" ,"3000" ,"--host" ,"0.0.0.0"]
# CMD ["python", "app.py"]

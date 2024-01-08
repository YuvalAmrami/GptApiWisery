# First stage: build the application
FROM python:3.12-slim AS build
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .

# Second stage: create the final Docker image
FROM python:3.7-slim
WORKDIR /app
COPY --from=build /app .
CMD ["python", "./your-daemon-or-script.py"]
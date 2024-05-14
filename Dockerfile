# syntax=docker/dockerfile:1

FROM python:3.10-slim

WORKDIR /nuimages

COPY flask_requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]

EXPOSE 5000
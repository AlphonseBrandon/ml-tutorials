FROM python:3.11-slim

RUN pip install pipenv

WORKDIR /app
COPY ["requirements.txt", "./"]

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ["./00_app", "./04_model", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "app:app" ]
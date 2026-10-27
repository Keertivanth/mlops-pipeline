# pull the python image from DockerHub
FROM python:3.14-slim
# SET THE WORKING DIRECTORY inside container
WORKDIR /app
# copy dependencies list before copying application code
COPY requirements.txt .
# install the depemdecies
RUN pip install -r requirements.txt

# copy application code and saved model inside app
COPY . .
# DOCUMENT THE PORT USED BY THE FastAPI application
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
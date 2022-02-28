# XTextClassification
This project is a demo of interpreting text classification model using Tensorflow 2 and Streamlit.

## Data
IMDB dataset contains 50k movie reviews and each of them is labeled as either "positive" or "negative". <br>
Data source: [Kaggle](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Docker
Build the docker container of this app by
```
docker build -t x-text-classification .
```
Run the container by
```
docker run -p 8501:8501 x-text-classification
# the app should be up and run on localhost:8501
```

## Deploy on Google Cloud Run
Open the GCP cloud shell or local terminal with gcloud installed, clone this repo and cd into the project root directory, then
```
# build and push to Container Registry
gcloud builds submit --tag gcr.io/<PROJECT_ID>/x-text-classification:v1.0
# deploy it to Cloud Run
gcloud run deploy --image=gcr.io/<PROJECT_ID>/x-text-classification:v1.0 --port 8501 --memory 2Gi
```

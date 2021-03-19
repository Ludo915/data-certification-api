
from fastapi import FastAPI
import joblib
import pandas as pd
app = FastAPI()


# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}

@app.get("/predict/")
def pred_art_pop(acousticness,	
                danceability,	
                duration_ms,	
                energy,		
                explicit,		
                id,		
                instrumentalness,		
                key,	
                liveness,		
                loudness,		
                mode,	
                name,		
                release_date,		
                speechiness,		
                tempo,		
                valence,	
                artist):

    X = pd.DataFrame(dict(
        acousticness = [float(acousticness)],
        danceability = [float(danceability)],
        duration_ms = [int(duration_ms)],
        energy = [float(energy)],
        explicit = [float(explicit)],
        id = [id],
        instrumentalness = [float(instrumentalness)],
        key = [int(key)],
        liveness = [float(liveness)],
        loudness = [float(loudness)],
        mode = [int(mode)],
        name = [name],
        release_date = [release_date],
        speechiness = [float(speechiness)],
        tempo = [float(tempo)],
        valence = [float(valence)],
        artist = [artist]))


    # Implement a /predict endpoint
    # pipeline
    pipeline = joblib.load('model.joblib')

    #make prediction
    results = pipeline.predict(X)

    #convert response from numpy to python type
    pred = float(results[0])

    return dict(artist = artist,
                name = name,
                prediction = pred)
    



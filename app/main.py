from fastapi import FastAPI

app = FastAPI(
    title="JobRadar AI Engine",
    version="1.0.0"
)


@app.get("/")
def root():
    return {"message": "JobRadar AI Engine is running"}
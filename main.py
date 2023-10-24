from fastapi import FastAPI

app = FastAPI()

@app.post("/what_language_is_this")
async def root():
    return [
        {"class":"German","value":0.1},
        {"class":"Esperanto","value":0.1},
        {"class":"French","value":0.1},
        {"class":"Italian","value":0.4},
        {"class":"Spanish","value":0.1},
        {"class":"Turkish","value":0.1},
        {"class":"English","value":0.1}
    ]

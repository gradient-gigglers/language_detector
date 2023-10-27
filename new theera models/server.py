import fastapi
import torch
import tokenizer
import model


app = fastapi.FastAPI()


@app.on_event("startup")
async def startup_event():
  app.state.maybe_model = "Attach my model to some variables"
  app.state.tknz = (tokenizer.Tokenizer()).load_vocab("./vocab.txt")
  app.state.lang = model.Language(torch.rand(len(app.state.tknz.vocab), 50), 7)
  app.state.lang.load_state_dict(torch.load("./lang_epoch_5.pt"))
  app.state.langs = ["German", "Esperanto", "French", "Italian", "Spanish", "Turkish", "English"]
  app.state.lang.eval()


@app.on_event("shutdown")
async def startup_event():
  print("Shutting down")


@app.get("/")
def on_root():
  return { "message": "Hello App" }


@app.post("/what_language_is_this")
async def on_language_challenge(request: fastapi.Request):

  # The POST request body has a text filed,
  # take it and tokenize it. Then feed it to
  # the language model and return the result.
  text = (await request.json())["text"]
  tknz = app.state.tknz.encode(text)
  tknz = torch.tensor(tknz, dtype=torch.long).unsqueeze(0)
  if tknz.shape[1] == 0: return [
    {"class": class_name, "value": 1/len(app.state.langs)}
    for class_name in app.state.langs
  ]

  lang = app.state.lang(tknz)
  lang = torch.nn.functional.softmax(lang, dim=1)
  lang = lang.squeeze(0).tolist()
  result = [{"class": class_name, "value": value} for class_name, value in zip(app.state.langs, lang)]
  return result

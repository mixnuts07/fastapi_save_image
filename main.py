# from cmath import nan
# from fastapi import FastAPI
# from fastapi import UploadFile, File
# import uvicorn
# import prediction as pd

# app = FastAPI()

# @app.get('/index')
# def hello_world(name: str):
#     return f"Hello {name}!"

# @app.post('/api/predict')
# async def predict_image(file: UploadFile = File(...)):
#     contents = await file.read()
#     if contents == False:
#         contents = 88
#     return contents
#     #  extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
#     # if not extension:
#     #     return "Image must be jpg or png format!"
#     # image = pd.read_image(await file.read())
#     # return image
#     # image = pd.preprocess(image)

#     # pred = pd.predict(image)
#     # print(pred)
#     # return pred


# if __name__ == "__main__":
#     uvicorn.run(app, port=8080, host='0.0.0.0')

# ------------
from fastapi import FastAPI
from fastapi import UploadFile, File
import uvicorn
import prediction as pd

app = FastAPI()

@app.get('/index')
def hello_world(name: str):
    return f"Hello {name}!"

@app.post('/api/predict')
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = pd.read_image(contents)
    image = pd.preprocess(image)
    pred = pd.predict(image)
    print(pred)
    return pred


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='0.0.0.0')

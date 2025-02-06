from fastapi import FastAPI
import asyncio
from classifierPost import train_model_with_base_and_new_data

application = FastAPI()

@application.get("/greeting")
async def greeting():
    await asyncio.sleep(20)
    return {"message": "Hello User welcome to FastAPI clss 6Feb25 2307"}


#greeting2 with 2 seconds sleep
@application.get("/greeting2")
async def greeting2():
    await asyncio.sleep(2)
    return {"message": "Hello User welcome to FastAPI Greet2 clss 6Feb25 2307"}

#submit through postman
@application.post("/postgreeting4")
def postgreeting4(data:dict):
    return {"message": "data submitted successfully 2307", "data":data["name"]}

# call the train_model_with_base_and_new_data function from classifierPost.py using fastapi post method
@application.post("/trainme")
def trainme(data:dict):
    print(data)
    train_model_with_base_and_new_data(data["base_data"], data["new_data"])
    return {"message": "Model trained successfully via FastAPI 2307"}


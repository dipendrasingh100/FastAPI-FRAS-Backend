from pydantic import BaseModel

class Faces(BaseModel): 
    name: str
    email: str
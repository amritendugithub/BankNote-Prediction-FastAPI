from pydantic import BaseModel

# Class which describes Bank Notes mesurements
class BankNote(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy:  float
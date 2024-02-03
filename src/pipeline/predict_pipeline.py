import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            # data_scaled=preprocessor.transform(features)
            data_scaled = features.values
            print(data_scaled)
            print('after scaling')
            preds=model.predict(data_scaled)
            print(preds)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(  self,
        Age: int,
        SystolicBP: int,
        DiastolicBP: int,
        BS: int,
        BodyTemp: int,
        HeartRate: int):

        self.Age = Age

        self.SystolicBP = SystolicBP

        self.DiastolicBP = DiastolicBP

        self.BS = BS

        self.BodyTemp = BodyTemp

        self.HeartRate = HeartRate


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.Age],
                "SystolicBP": [self.SystolicBP],
                "DiastolicBP": [self.DiastolicBP],
                "BS": [self.BS],
                "BodyTemp": [self.BodyTemp],
                "HeartRate": [self.HeartRate],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

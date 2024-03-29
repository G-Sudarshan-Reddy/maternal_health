import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.metrics import r2_score, accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models
import numpy as np

@dataclass
class modeltrainerconfig:
    model_path = os.path.join('artifacts', 'model.pkl')

class modeltrainer:
    def __init__(self):
        self.trainerconfig = modeltrainerconfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Model training begun")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            lg_model = LogisticRegression()
            lg_model.fit(X_train, y_train)
            
            predicted=lg_model.predict(X_test)
            print(predicted)

            # X_new = np.array([[54,172,133,45,89,39]])
            # pred = lg_model.predict(X_new)
            # print(pred)

            save_object(
                file_path=self.trainerconfig.model_path,
                obj=lg_model
            )

            r2_square = accuracy_score(y_test, predicted)
            return r2_square
        

        except Exception as e:
            raise CustomException(e, sys)

    # def initiate_model_trainer(self, train_array, test_array):
    #     try:
    #         logging.info("Model training process begun")
    #         X_train,y_train,X_test,y_test=(
    #             train_array[:,:-1],
    #             train_array[:,-1],
    #             test_array[:,:-1],
    #             test_array[:,-1]
    #         )
    #         models = {
    #             # "Random Forest": RandomForestRegressor(),
    #             "Decision Tree": DecisionTreeRegressor(),
    #             # "Gradient Boosting": GradientBoostingRegressor(),
    #             # # "XGBRegressor": XGBRegressor(),
    #             # "AdaBoost Regressor": AdaBoostRegressor(),
    #         }
    #         params={
    #             "Decision Tree": {
    #                 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    #                 # 'splitter':['best','random'],
    #                 # 'max_features':['sqrt','log2'],
    #             }
    #             # "Random Forest":{
    #             #     # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
    #             #     # 'max_features':['sqrt','log2',None],
    #             #     'n_estimators': [8,16,32,64,128,256]
    #             # }
    #             # "Gradient Boosting":{
    #             #     # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
    #             #     'learning_rate':[.1,.01,.05,.001],
    #             #     'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
    #             #     # 'criterion':['squared_error', 'friedman_mse'],
    #             #     # 'max_features':['auto','sqrt','log2'],
    #             #     'n_estimators': [8,16,32,64,128,256]
    #             # },

    #             # "AdaBoost Regressor":{
    #             #     'learning_rate':[.1,.01,0.5,.001],
    #             #     # 'loss':['linear','square','exponential'],
    #             #     'n_estimators': [8,16,32,64,128,256]
    #             # }
                
    #         }
    #         model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
    #                                          models=models,param=params)
            
    #         ## To get best model score from dict
    #         best_model_score = max(sorted(model_report.values()))

    #         ## To get best model name from dict

    #         best_model_name = list(model_report.keys())[
    #             list(model_report.values()).index(best_model_score)
    #         ]
    #         best_model = models[best_model_name]
    #         print(best_model)

    #         if best_model_score<0.6:
    #             raise CustomException("No best model found")
    #         logging.info(f"Best found model on both training and testing dataset")

    #         save_object(
    #             file_path=self.trainerconfig.model_path,
    #             obj=best_model
    #         )

    #         predicted=best_model.predict(X_test)
    #         print(X_test)

    #         r2_square = r2_score(y_test, predicted)
    #         return r2_square
        
    #     except Exception as e:
    #         raise CustomException(e, sys)
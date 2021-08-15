from math import nan
from os import path, remove
import os
import numpy as np

from pandas.core.frame import DataFrame
from sklearn.base import RegressorMixin
from csvdata.services import DIR_NAME, BASE_DIR
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import pickle
from rest_framework.exceptions import ValidationError, APIException

DIR_DEFAULT = path.join(str(BASE_DIR), 'static', 'default')

POLYNOMIAL_REG = 1
RANDOM_FOREST = 2
POLYNOMIAL_REG_NAME = 'poly_reg'
RANDOM_FOREST_NAME = 'rand_forest'

class LearningService:

    def __read_csv_file__(self,_id):
        try:
            csv_file = path.join(DIR_NAME, f'{_id}.csv')
            df = pd.read_csv(csv_file)
            return DataFrame(df)
        except:
            raise ValidationError({"message": "CSV file not found"})
    
    def __save_model__(self, model, _id, x_plot, y_plot, algorithm, labels):
        pickle.dump(model, open(path.join(DIR_NAME, f'{_id}.model.sav'), 'wb'))
        pickle.dump(
            {
                'x_plot': x_plot, 
                'y_plot': y_plot,
                'algorithm': algorithm, 
                'labels': labels
            }, open(path.join(DIR_NAME, f'{_id}.meta.sav'), 'wb'))
    
    def __load_model__(slef, _id, algorithm):
        if algorithm and algorithm != '' and algorithm != '0':
            algorithm = int(algorithm)
            directory = DIR_DEFAULT
            if algorithm == POLYNOMIAL_REG:
                filename = POLYNOMIAL_REG_NAME
            elif algorithm == RANDOM_FOREST:
                filename = RANDOM_FOREST_NAME
            else:
                raise APIException("Algorithm option not allowed.")
        elif _id and _id != '':
            directory = DIR_NAME
            filename = _id
        else:
            raise APIException("Please provide an id or algorithm.")
        # Load model from server
        try:
            model = pickle.load(
                open(path.join(directory, f'{filename}.model.sav'), 'rb'))
            meta = pickle.load(open(path.join(directory, f'{filename}.meta.sav'), 'rb'))
        except:
            raise ValidationError({"message": "Model not found"})
        return model, meta
    
    def __format_json__(self, xlabel, ylabel, score, fit, test, title):
        return {
                "title": title,
                "xlabel": xlabel,
                "ylabel": ylabel,
                "score": score,
                "datasets": [
                    {
                        "label": "Predicted",
                        "backgroundColor": "#ff0000",
                        "data": [{"x": x, "y": y} for x, y in zip(fit[0], fit[1])]
                    },
                    {
                        "label": "Actual",
                        "backgroundColor": "#ffff00",
                        "data": [{"x": x, "y": y} for x, y in zip(test[0], test[1])]
                    }
                ]
            }

    def __format_json_predicted__(self, xlabel, ylabel, score, fit, title):
        return {
                "title": title,
                "xlabel": xlabel,
                "ylabel": ylabel,
                "score": score,
                "datasets": [
                    {
                        "label": "Predicted",
                        "backgroundColor": "#ff0000",
                        "data": [{"x": x, "y": y} for x, y in zip(fit[0], fit[1])]
                    }
                ]
            }


    def fit(self, _id, x_plot: str,  y_plot: str, group_by: str, labels: list, test_size=0.2, algorithm=1):
        df = self.__read_csv_file__(_id)
        
        try:
            columns = df.columns.tolist()
            for c in columns:
                if c not in [x_plot, y_plot, group_by] + labels:
                    df = df.drop([c], axis=1)
            # Scaling dataset to remove difference in distributions within columns
            scaler = MinMaxScaler()
            df[labels] = scaler.fit_transform(df[labels])
            # Splitting dataset into training and test sets
            x = df.drop([y_plot], axis=1)
            y = df[[y_plot, group_by]]
            X_train, X_test, y_train, y_test = train_test_split(
                x, y, test_size=test_size, random_state=42)
            x_train_final = X_train.drop([x_plot, group_by], axis=1)
            x_test_final = X_test.drop([x_plot, group_by], axis=1)
            y_test_final = y_test[y_plot]
            y_train_final = y_train[y_plot]
            if algorithm == POLYNOMIAL_REG:
                poly_reg = PolynomialFeatures(degree=4)
                X_poly = poly_reg.fit_transform(x_train_final)
                lin_reg = LinearRegression()
                model = lin_reg.fit(X_poly, y_train_final)
                # save model
                self.__save_model__(model, _id, x_plot, y_plot, algorithm, labels)
                x_pol_test = poly_reg.fit_transform(x_test_final)
                score = lin_reg.score(x_pol_test, y_test_final) * 100
                predictions = self.predict_poly_reg(
                    lin_reg, poly_reg, X_test, y_test, group_by, y_plot, x_plot)
            elif algorithm == RANDOM_FOREST:
                # Definition of specific parameters for Random forest 
                # Number of trees in random forest
                n_estimators = [int(x) for x in np.linspace(start = 2, stop = 2000, num = 20)]
                # Number of features to consider at every split
                max_features = ['auto', 'sqrt']
                # Maximum number of levels in tree
                max_depth = [int(x) for x in np.linspace(4, 30, num = 2)]
                max_depth.append(None)
                # Minimum number of samples required to split a node
                min_samples_split = [2, 3, 4, 5, 10]
                # Minimum number of samples required at each leaf node
                min_samples_leaf = [1, 2, 4]
                # Method of selecting samples for training each tree
                bootstrap = [True, False]
                
                # Create the random grid
                random_grid = {
                        'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap
                    }

                rf = RandomForestRegressor(random_state=42)
                model = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter = 15, scoring='neg_mean_absolute_error',  cv = 3, verbose=2, random_state=42, n_jobs=-1, return_train_score=True)
                model.fit(x_train_final, y_train_final)
                model.best_params_
                model = model.best_estimator_.fit(x_train_final, y_train_final) 
                # save model
                self.__save_model__(model, _id, x_plot, y_plot, algorithm, labels)
                score = model.score(x_test_final, y_test_final) * 100
                predictions = self.predict_rand_forest(model, X_test, y_test, group_by, y_plot, x_plot)
            else:
                raise APIException("Algorithm option not allowed.")
            return {"score": score, "predictions": predictions}
        except Exception as e:
            raise APIException({'message': str(e)})

    def predict_poly_reg(self, lin_reg: LinearRegression, poly_reg, X_test, y_test, group_by, y_plot, x_plot):
        group_df = X_test.groupby([group_by]).agg({x_plot: "count"})
        items = group_df.to_dict()[x_plot].keys()
        predicts = []
        for i in items:
            x_test_i = X_test[X_test[group_by] == i]
            y_test_i = y_test[y_test[group_by] == i]

            x_test_i_f = x_test_i.drop([x_plot, group_by], axis=1)
            x_pol_test_i = poly_reg.fit_transform(x_test_i_f)
            y_poly = lin_reg.predict(x_pol_test_i)
            score = lin_reg.score(x_pol_test_i, y_poly) * 100
            fit = (x_test_i[x_plot].tolist(), y_poly.tolist())
            test = (x_test_i[x_plot].tolist(), y_test_i[y_plot].tolist())
            data = self.__format_json__(x_plot, y_plot, score, fit, test, f"The R2 value for Polynomial regression(Degree - 4): {i}")
            predicts.append(data)

        return predicts
    
    def predict_rand_forest(self, model: RandomizedSearchCV, X_test, y_test, group_by, y_plot, x_plot):
        group_df = X_test.groupby([group_by]).agg({x_plot: "count"})
        items = group_df.to_dict()[x_plot].keys()
        predicts = []
        for i in items:
            x_test_i = X_test[X_test[group_by] == i]
            y_test_i = y_test[y_test[group_by] == i]
            x_test_i_f = x_test_i.drop([x_plot, group_by], axis=1)
            y_predict = model.predict(x_test_i_f)
            score = model.score(x_test_i_f, y_predict) * 100
            fit = (x_test_i[x_plot].tolist(), y_predict.tolist())
            test = (x_test_i[x_plot].tolist(), y_test_i[y_plot].tolist())
            data = self.__format_json__(x_plot, y_plot, score, fit, test, f"Random Forest: {i}")
            predicts.append(data)

        return predicts          

    def get_prediction_prod(self, _id, csvfile, algorithm):
        if not _id and not algorithm:
            raise APIException("Please provide an id or algorithm.")
            
        model, meta = self.__load_model__(_id, algorithm)
        
        x_plot = meta['x_plot']
        y_plot = meta['y_plot']
        labels = meta['labels']
        algorithm = meta['algorithm']
        prod_data = pd.read_csv(csvfile)
        data = pd.DataFrame(prod_data)
        if len(data) < 2:
            raise ValidationError({"message": "Please provide at least two rows in csv file."})
        columns = data.columns.tolist()
        features = [x_plot] + labels

        scaler = MinMaxScaler()
        data[labels] = scaler.fit_transform(data[labels])

        # Remove columns that are not in features
        for c in columns:
            if c not in features:
                data = data.drop([c], axis=1)
        if len(data.columns) != len(features):
            raise ValidationError({"message": "Not all features found in the file"})
        try:
            x_prod_f = data.drop([x_plot], axis=1)
            if algorithm == POLYNOMIAL_REG:
                poly_reg = PolynomialFeatures(degree=4)
                x_pol_prod = poly_reg.fit_transform(x_prod_f)
                y_poly = model.predict(x_pol_prod)
                score = model.score(x_pol_prod, y_poly) * 100
                fit = (data[x_plot].tolist(), y_poly.tolist())
                title = f"The R2 value for Polynomial regression(Degree - 4)"
            elif algorithm == RANDOM_FOREST:
                y_predict = model.predict(x_prod_f)
                score = model.score(x_prod_f, y_predict) * 100
                fit = (data[x_plot].tolist(), y_predict.tolist())
                title = f"Random Forest"
            out = self.__format_json_predicted__(x_plot, y_plot, score, fit, title)
            out["algorithm"] = algorithm
            return out
        except Exception as e:
            raise APIException({"message": str(e)})

    def get_model_meta(self, _id):
        try:
            meta = pickle.load(open(path.join(DIR_NAME, f'{_id}.meta.sav'), 'rb'))
        except:
            raise APIException({"message": "Model not found"})
        y_plot = meta['y_plot']
        x_plot = meta['x_plot']
        labels = meta['labels']
        algorithm = meta['algorithm']
        return {"y_plot": y_plot, "x_plot": x_plot, "labels": labels, "algorithm": algorithm}

    def delete_model(self, _id):
        try:
            remove(path.join(DIR_NAME, f'{_id}.meta.sav'))
            remove(path.join(DIR_NAME, f'{_id}.model.sav'))
            remove(path.join(DIR_NAME, f'{_id}.csv'))
            return {"message": "Model deleted"}
        except:
            return {"message": "Model not found"}
import pandas as pd
import pickle
from rest_framework.exceptions import ValidationError, APIException
from os import path

DIR_NAME = ""

@api_view(['POST'])
@permission_classes([AllowAny])
@authentication_classes([])
def predict(request):
    _id = request.data['_id']
    data = request.data['data']
    code = request.data['code']
    return Response(data=learning_service.get_prediction_prod(_id, data, code))


@api_view(['POST'])
@permission_classes([AllowAny])
@authentication_classes([])
def predict(request):
    _id = request.data['_id']
    data = request.FILES['data']
    code = request.data['code']
    x_plot = request.data['x_plot']
    y_plot = request.data['y_plot']
    labels = request.data['labels']
    labels = json.loads(labels)
    return Response(data=learning_service.get_prediction_prod(_id, data, code, x_plot, y_plot, labels))

def get_prediction_prod(self, _id, csvfile, code, x_plot, y_plot, labels):
    try:
        model = pickle.load(
            open(path.join(DIR_NAME, f'{_id}.model.sav'), 'rb'))
        meta = pickle.load(open(path.join(DIR_NAME, f'{_id}.meta.sav'), 'rb'))
    except:
        raise ValidationError({"message": "Model not found"})
    try:
        algorithm = meta['algorithm']
        prod_data = pd.read_csv(csvfile)
        data = pd.DataFrame(prod_data)
        # x_prod = data.drop([y_plot], axis=1)
        y_prod = data[[y_plot]]
        # labels.append(y_plot)
        x_prod_f = data[labels]
        x_prod_x = data[[x_plot]]
        # x_prod = x_prod_l + x_prod_x
        
        # x_prod_f = x_prod.drop([x_plot], axis=1)
        if algorithm == POLYNOMIAL_REG:
            poly_reg = PolynomialFeatures(degree=4)
            x_pol_prod = poly_reg.fit_transform(x_prod_f)
            y_poly = model.predict(x_pol_prod)
            score = model.score(x_pol_prod, y_poly) * 100
            fit = (x_prod_x[x_plot].tolist(), y_poly.tolist())
            test = (x_prod_x[x_plot].tolist(), y_prod[y_plot].tolist())
            title = f"The R2 value for Polynomial regression(Degree - 4): {code}"
        elif algorithm == RANDOM_FOREST:
            y_predict = model.predict(x_prod_f)
            score = model.score(x_prod_f, y_predict) * 100
            fit = (x_prod_x[x_plot].tolist(), y_predict.tolist())
            test = (x_prod_x[x_plot].tolist(), y_prod[y_plot].tolist())
            title = f"Random Forest: {code}"
        else:
            raise 
        data = self.__format_json__(x_plot, y_plot, score, fit, test, title)
        data["algorithm"] = algorithm
        return data
        
    except Exception as e:
        raise APIException({"message": str(e)})
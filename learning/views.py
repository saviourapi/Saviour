from .services import LearningService
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.permissions import AllowAny

learning_service = LearningService()


@api_view(['POST'])
@permission_classes([AllowAny])
@authentication_classes([])
def learn(request):
    _id = request.data['_id']
    x_plot = request.data['x_plot']
    y_plot = request.data['y_plot']
    group_by = request.data['group_by']
    labels = request.data['labels']
    algorithm = request.data.get('algorithm', 1) # 1: Polinomial, 2: Random forest
    test = request.data.get('test', 0.2)
    return Response(data=learning_service.fit(_id, x_plot, y_plot, group_by, labels, test, algorithm))

@api_view(['POST'])
@permission_classes([AllowAny])
@authentication_classes([])
def predict(request):
    _id = request.data.get('_id', None)
    data = request.data.get('data', None)
    algorithm = request.data.get('algorithm', None)
    return Response(data=learning_service.get_prediction_prod(_id, data, algorithm))

@api_view(['GET'])
@authentication_classes([])
@permission_classes([AllowAny])
def get_model_meta(request, id):
    data = learning_service.get_model_meta(id)
    return Response(data=data)

@api_view(['DELETE'])
@authentication_classes([])
@permission_classes([AllowAny])
def delete_model(request):
    _id = request.data.get('_id', None)
    return Response(data=learning_service.delete_model(_id))

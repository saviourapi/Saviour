from django.views.decorators.csrf import csrf_exempt
from csvdata.services import CsvdDtaService
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.permissions import AllowAny

csv_service = CsvdDtaService()


@api_view(['POST'])
@authentication_classes([])
@permission_classes([AllowAny])
def load_data(request):
    csvfile = request.FILES['data']
    data = csv_service.read_csv_header(csvfile)
    return Response(data=data)

@api_view(['POST'])
@authentication_classes([])
@permission_classes([AllowAny])
def get_csv_header(request):
    csvfile = request.FILES['data']
    data = csv_service.get_csv_headers(csvfile)
    return Response(data=data)

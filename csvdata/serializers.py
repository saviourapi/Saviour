from rest_framework import serializers

class CsvSumarizeSerializer(serializers.Serializer):
    columns = serializers.ListField()
    covariance = serializers.DictField()
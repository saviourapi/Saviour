from django.db import models

class CsvSumarize:
    columns: list
    covariance: dict

    def __init__(self, columns, covariance):
        self.columns = columns
        self.covariance = covariance

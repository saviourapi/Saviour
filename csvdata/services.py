from PetrolPredict.settings import BASE_DIR
from datetime import datetime
import pandas as pd
from pandas.core.frame import DataFrame
from os import path
from rest_framework.exceptions import APIException

DIR_NAME = path.join(str(BASE_DIR), 'static', 'files')

class CsvdDtaService:


    def read_csv_header(self, csvfile):
        data = pd.read_csv(csvfile)
        df = DataFrame(data)
        cov = df.corr()
        cov = cov.where(pd.notnull(cov), "nan")
        id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        with open(path.join(DIR_NAME, f'{id}.csv'), 'wb+') as destination:
            for chunk in csvfile.chunks():
                destination.write(chunk)
        return {"_id": id, "columns": df.columns, "corr": cov.to_dict()}
    
    def get_csv_headers(self, csvfile):
        data = pd.read_csv(csvfile)
        df = DataFrame(data)
        return {'columns': df.columns}


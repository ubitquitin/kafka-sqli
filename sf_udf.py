import numpy as np
from snowflake_util import snowflake_connector
from snowflake.snowpark.functions import udf
import requests
import os
from dotenv import load_dotenv

import pandas as pd
from snowflake.snowpark.functions import pandas_udf
from snowflake.snowpark.types import IntegerType, PandasSeriesType, VariantType

load_dotenv()
hf_token = os.getenv('HF_API_TOKEN')

API_URL = "https://dk4m9w1xfx473hj4.us-east-1.aws.endpoints.huggingface.cloud"
headers = {
	"Accept" : "application/json",
	"Authorization": hf_token,
	"Content-Type": "application/json" 
}

session = snowflake_connector()

session.clear_imports()
session.clear_packages()

#Register above uploded model as import of UDF
session.add_import("@SQLI_CLASSIFICATION/joblib_RL_Model.pkl")

#map packege dependancies
session.add_packages("joblib", "scikit-learn", "numpy", "pandas", "requests")

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def read_file(filename):
    import joblib
    import sys
    import os
    
    #where all imports located at
    import_dir = sys._xoptions.get("snowflake_import_directory")

    if import_dir:
        with open(os.path.join(import_dir, filename), 'rb') as file:
            m = joblib.load(file)
            return m

#Put everything on sagemaker endpoint/s3, set up api integration, 
#call api and warp in this udf
#register UDF
@udf(name = 'classify_injection', is_permanent = True, replace = True, stage_location = '@SQLI_CLASSIFICATION', external_access_integrations=['HF_ACCESS_INTEGRATION_RS'])
def classify_injection(x: str) -> float:
    if x:
        embeds = query({
            "inputs": x,
            "truncate": True
        })['embeddings']
        
        inp = np.asarray(embeds).reshape(1, -1)
        lr = read_file('joblib_RL_Model.pkl')
        return lr.predict(inp)
    else:
        return 0


@pandas_udf(name='batch_classify_injection', input_types=[PandasSeriesType(VariantType())], return_type=PandasSeriesType(IntegerType()), is_permanent = True, replace = True, stage_location = '@SQLI_CLASSIFICATION', external_access_integrations=['HF_ACCESS_INTEGRATION_RS'])
def batch_classify_injection(queries: pd.Series) -> pd.Series:
    embeds = query({
        "inputs": queries,
        "truncate": True
    })['embeddings']
        
    inp = np.asarray(embeds).reshape(-1, 768) #n samples, 768 emb dimension
        
    lr = read_file('joblib_RL_Model.pkl')
    return lr.predict(inp)


    

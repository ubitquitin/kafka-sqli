import joblib
from sentence_transformers import SentenceTransformer
from snowflake_util import snowflake_connector

model = SentenceTransformer('C:/Users/ROHAN/time-series-kafka-demo/model/')
joblib.dump(model, 'ft_llm.joblib')

#define a session
session = snowflake_connector()

#create the stage for storing the ML models
session.sql('CREATE OR REPLACE STAGE SQLI_CLASSIFICATION').show()

#upload into the ML_MODELS SNowfla
session.file.put(
    "joblib_RL_Model.pkl", "@SQLI_CLASSIFICATION", auto_compress=False, overwrite=True
)

session.file.put(
    "ft_llm.joblib", "@SQLI_CLASSIFICATION", auto_compress=False, overwrite=True
)
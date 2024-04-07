from snowflake.snowpark.session import Session
import os
from dotenv import load_dotenv

load_dotenv()

accountname = os.getenv('SF_ACCOUNT') # your accountname
username = os.getenv('SF_USER') #your snowflake username
password = os.getenv('SF_PASS') #snowflake password
wh = os.getenv('WAREHOUSE') #snowflake wh
db = os.getenv('DATABASE')
schema = os.getenv('SCHEMA')

connection_parameters = {
    "account": accountname,
    "user": username,
    "password": password,
    "role": "ACCOUNTADMIN",
    "warehouse": wh,
    "database": db,
    "schema": schema
}

def snowflake_connector():
    try:
        session = Session.builder.configs(connection_parameters).create()
        print("connection successful!")
    except:
        raise ValueError("error while connecting with db")
    return session
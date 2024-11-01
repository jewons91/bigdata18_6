import requests
import json
import numpy as np
import pandas as pd
import os
import logging

file_path=os.getenv('ini_file_path', '/root/airflow/dags/hangukInvestAPI')
logging.info(f'### Current working directory: {os.getcwd()} ###')
logging.info(f'### Config file path: {file_path} ###')

with open(file_path + '/api.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
APP_KEY = lines[0].strip()
APP_SECRET = lines[1].strip()

URL_BASE = "https://openapi.koreainvestment.com:9443" #실전투자

# Auth
def auth():
    headers = {"content-type":"application/json"}
    body = {
        "grant_type":"client_credentials",
        "appkey":APP_KEY, 
        "appsecret":APP_SECRET
        }
    PATH = "oauth2/tokenP"
    URL = f"{URL_BASE}/{PATH}"
    res = requests.post(URL, headers=headers, data=json.dumps(body))
    
    if res.status_code == 200 and "access_token" in res.json():
        with open(file_path + '/token.txt', 'w', encoding='utf-8') as f:
            f.write(res.json()['access_token'])
        return res.json()['access_token']
    else:
        logging.info("Failed to get access token.")
        logging.info(res.json())
        raise Exception("Authentication failed")

# if __name__ == '__main__':
#     print(auth(ACCESS_TOKEN))
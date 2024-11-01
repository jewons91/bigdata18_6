from model import prediction_to_json, insert_json_to_db, get_code_list

codes = get_code_list()
for code in codes:
    json_result = prediction_to_json(code, model_path='C:\\big18\\final\DB\model.pth', scaler_path='C:\\big18\\final\DB\scaler.pkl')
    insert_json_to_db(json_result)

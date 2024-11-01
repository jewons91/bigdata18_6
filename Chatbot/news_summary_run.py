from news_summary import news_summary_json, insert_json_summary_to_db
from datetime import datetime

# start_date = '2024-08-31'
# end_date = '2024-09-07'

# 현재 날짜와 시간 (당일)
end_date = datetime.now()
# 어제 날짜와 시간
start_date = end_date - timedelta(days=1)

json_result = news_summary_json(start_date, end_date)
insert_json_summary_to_db(json_result)

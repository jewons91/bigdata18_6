from naver_news_function import naver_news_scraping

# 삼성전자 SK하이닉스 한미반도체 SK텔레콤 KT 삼양식품 농심 CJ제일제당

search_word = 'SK하이닉스'
start_date = '20240801'
end_date = '20240831'

naver_news_scraping(search_word, start_date, end_date)

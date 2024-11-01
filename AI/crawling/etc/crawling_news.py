import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_news(keyword, page=2):
    start_index = (page - 1) * 10 + 1
    url = f"https://search.naver.com/search.naver?where=news&query={keyword}&start={start_index}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    news_list = soup.find_all("div", class_="news_area")

    results = []
    for news in news_list:
        title = news.find("a", class_="news_tit").get_text(strip=True)
        source = news.find("a", class_="info").get_text(strip=True)
        date = news.find("span", class_="info").get_text(strip=True)
        #link = news.find("a", class_="news_tit")["href"]
        #content = get_news_content(link)
        content = news.find("a", class_="api_txt_lines dsc_txt_wrap").get_text(strip=True)
        print(content)
        results.append({"Title": title, "Source": source, "Date": date, "Content": content})

    return results

def get_news_content(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    article_body = soup.find("div", class_="article_body")
    if article_body:
        content = article_body.get_text(separator="\n", strip=True)
    else:
        content = "No content found"

    return content

def save_to_csv(news, filepath):
    df = pd.DataFrame(news)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    keyword = "코로나19"  # 원하는 검색어로 변경 가능
    total_pages = 3  # 가져올 전체 페이지 수
    all_news = []
    for page in range(1, total_pages + 1):
        news = get_news(keyword, page=page)
        all_news.extend(news)
    
    save_to_csv(all_news, r"C:\Users\TJ\Desktop\naver_news_with_content_paged.csv")
    print(f"{total_pages}페이지의 뉴스 데이터 저장이 완료되었습니다.")

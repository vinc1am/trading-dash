import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import yfinance as yf
from datetime import datetime
import json
from tqdm import tqdm
import time


def get_snp500_stock_list():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = requests.get(url)
    soup = BeautifulSoup(req.content, 'html.parser')
    table = soup.find('table', {'class': 'wikitable sortable'})
    snp_stock_list = pd.read_html(StringIO(str(table)))[0]
    snp_stock_list['Symbol'] = snp_stock_list['Symbol'].replace({'BRK.B': 'BRK-B', 'BF.B': 'BF-B'})
    snp_stock_list = snp_stock_list[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]
    return snp_stock_list


def get_stock_info(ticker, start_date, end_date):
    start_date = datetime.strptime(start_date, '%Y%m%d').date()
    end_date = datetime.strptime(end_date, '%Y%m%d').date()
    
    stock = yf.Ticker(ticker)
    
    try:
        history_df = stock.history(start=start_date, end=end_date)
        history_df.index = history_df.index.strftime('%Y%m%d')
        history_dict = history_df.to_dict(orient='index')
        info_dict = {
            # 'news': stock.news,
            'website': stock.info.get('website'),
            'industry': stock.info.get('industry'),
            'sector': stock.info.get('sector'),
            'description': stock.info.get('longBusinessSummary'),
            'history': history_dict
        }

    except Exception as e:
        print(f"Couldn't fetch data for {ticker}. Error: {str(e)}")
        info_dict = {}

    return info_dict
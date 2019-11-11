import requests
import pandas as pd
import numpy as np
import time
import math

def get_url_page(begin_date, end_date, page, COMPANY_NAME, SUBWORD_1):
    if COMPANY_NAME == "Facebook":
        url = 'https://api.nytimes.com/svc/search/v2/articlesearch.json?&begin_date={}&end_date={}&page={}&api-key=G2p8dvETu35Fk9KwmNceMqk3gYkvYcLS&fq=organizations.contains:({}, {})'.format(begin_date, end_date, page, COMPANY_NAME, SUBWORD_1)
    else:
        url = 'https://api.nytimes.com/svc/search/v2/articlesearch.json?&begin_date={}&end_date={}&page={}&api-key=G2p8dvETu35Fk9KwmNceMqk3gYkvYcLS&fq=organizations.contains:({})'.format(begin_date, end_date, page, COMPANY_NAME)
    print(url)
    return url
    # return 'https://api.nytimes.com/svc/search/v2/articlesearch.json?&begin_date={}&end_date={}&page={}&api-key=G2p8dvETu35Fk9KwmNceMqk3gYkvYcLS&fq=headline:({})'.format(begin_date, end_date, page, COMPANY_NAME)

def get_num_pages(URL):
    response = requests.get(URL)
    json = response.json()
    num_items = json['response']['meta']['hits']
    num_pages = math.ceil(num_items / 10)
    return num_pages

def main():
    COMPANY_NAME = "Facebook"
    SUBWORD_1 = "Instagram"
    SUBWORD_2 = "Mark Zuckerberg"

    df = pd.read_csv('./FB_data.csv', index_col="date", parse_dates=True)
    begin_date = str(df.index[0].date()).replace("-", "")
    end_date = str(df.index[-1].date()).replace("-", "")
    # URL = 'https://api.nytimes.com/svc/search/v2/articlesearch.json?q={}&begin_date={}&end_date={}&page={}&api-key=G2p8dvETu35Fk9KwmNceMqk3gYkvYcLS&fq=headline:({})'.format(COMPANY_NAME, begin_date, end_date, 0, COMPANY_NAME)
    URL = get_url_page(begin_date, end_date, 0, COMPANY_NAME, SUBWORD_1)
    # sample: https://api.nytimes.com/svc/search/v2/articlesearch.json?q=google&begin_date=20130208&end_date=20180207&page=0&api-key=G2p8dvETu35Fk9KwmNceMqk3gYkvYcLS&fq=headline:(google)
    
    num_pages = get_num_pages(URL)

    df['main_news'] = ""
    df['tagged_news'] = ""

    # for i in range(4):
    try:
        for i in range(0, num_pages):
            time.sleep(7)
            response = requests.get(get_url_page(begin_date, end_date, i, COMPANY_NAME, SUBWORD_1))
            json = response.json()
            # json = {
            #     'response': {
            #         'docs': [
            #             { 
            #                 'headline': "Google is awesome",
            #                 'pub_date': '2017-11-28T12:33:12+0000'
            #             }
            #         ]
            #     }
            # }

            if 'response' in json.keys():
                for item in json['response']['docs']:
                    try:
                        pub_date = pd.to_datetime(item['pub_date']).date()
                        if pub_date in df.index:
                            if COMPANY_NAME in item['headline']['main'] or COMPANY_NAME in item['snippet'] or SUBWORD_1 in item['headline']['main'].lower() or SUBWORD_1 in item['snippet'].lower() or SUBWORD_2 in item['headline']['main'].lower() or SUBWORD_2 in item['snippet'].lower():
                                df.loc[pub_date, ['main_news']] = df.loc[pub_date, ['main_news']] + "|||" + item['headline']['main'] + ": " + item['snippet']
                                print("The main_news is")
                                print(df.loc[pub_date, ['main_news']])
                            else:
                                df.loc[pub_date, ['tagged_news']] = item['headline']['main'] + ": " + item['snippet']
                                print("The tagged_news is")
                                print(df.loc[pub_date, ['tagged_news']])
                    except Exception as e:
                        print("ERROR: Unable to convert datetime")
                        print(e)

    except Exception as e:
        print("Error")
        print(e)
        df.to_csv ('./output/{}_with_news.csv'.format(COMPANY_NAME), header=True) 
    df.to_csv ('./output/{}_with_news.csv'.format(COMPANY_NAME), header=True)

if __name__ == '__main__':
    main()

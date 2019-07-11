import pandas as pd
import os
import urllib
import fileinput
from bs4 import BeautifulSoup


home_url = 'http://www.football-data.co.uk/'  # data source
data_path = 'Data/'  # where downloaded csv files are stored
leagues = [
    'england',
    'scotland',
    'germany',
    'italy',
    'spain',
    'france',
    'netherlands',
    'belgium',
    'portugal',
    'turkey',
    'greece'
]  # crawled from http://www.football-data.co.uk/data.php


def get_csv_urls(league_url):  # crawling csv urls
    csv_urls = []
    html = urllib.request.urlopen(league_url).read()
    soup = BeautifulSoup(html)
    tags = soup('a')
    for tag in tags:
        href = (tag.get('href', None))
        if href.endswith(".csv"):
            csv_urls.append(href)
    return csv_urls


def download_csvs():  # download original csvs for flexible manual cleaning
    for league in leagues:
        csv_urls = get_csv_urls(home_url + league + 'm.php')
        for csv_url in csv_urls:
            path = data_path + ''.join(csv_url.split('/')[-2:])
            with open(path, 'wb') as f:
                f.write(urllib.request.urlopen(home_url + csv_url).read())


def clean_csvs():  # preprocess csv files to make them readable for pd.read_csv
    for csv_path in os.listdir(data_path):
        print(csv_path)
        # remove non-utf-8 chars
        with open(data_path + csv_path, encoding='utf-8', errors='ignore') as f:
            content = f.read()
        with open(data_path + csv_path, 'w') as f:
            f.write(content)
        # remove extra commas at the end of csv lines
        first_line = True
        for line in fileinput.input(data_path + csv_path, inplace=True):
            line = line[:-1].split(',')
            ncol = 0
            if first_line:
                ncol = len(line)
                while line[ncol - 1] == '':
                    ncol -= 1
                first_line = False
            print(','.join(line[:ncol]))


def concat_csvs():  # concatenate all csvs together (different csv has a different subset of the final columns)
    df = pd.DataFrame()
    for f in os.listdir(data_path):
        print(f)
        df2 = pd.read_csv(data_path + f, engine='python')
        year = int(f[2:4])
        df2['Year'] = 1900 + year if year > 80 else 2000 + year
        df = pd.concat([df, df2], ignore_index=True)
    return df


if __name__ == '__main__':
    download_csvs()
    clean_csvs()
    concat_csvs().to_csv('pre_data.csv', columns=[
        'Div', 'Date', 'Year', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS',
        'HST', 'AST', 'HHW', 'AHW', 'HC', 'AC', 'HF', 'AF', 'HFKC', 'AFKC', 'HO', 'AO', 'HY', 'AY', 'HR', 'AR',
        'HBP', 'ABP', 'B365H', 'B365D', 'B365A', 'BSH', 'BSD', 'BSA', 'BWH', 'BWD', 'BWA', 'GBH', 'GBD', 'GBA',
        'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'PSH', 'PSD', 'PSA', 'SOH', 'SOD', 'SOA', 'SBH', 'SBD', 'SBA',
        'SJH', 'SJD', 'SJA', 'SYH', 'SYD', 'SYA', 'VCH', 'VCD', 'VCA', 'WHH', 'WHD', 'WHA', 'Bb1X2', 'BbMxH', 'BbAvH',
        'BbMxD', 'BbAvD', 'BbMxA', 'BbAvA', 'BbOU', 'BbMx>2.5', 'BbAv>2.5', 'BbMx<2.5', 'BbAv<2.5', 'GB>2.5', 'GB<2.5',
        'B365>2.5', 'B365<2.5', 'BbAH', 'BbAHh', 'BbMxAHH', 'BbAvAHH', 'BbMxAHA', 'BbAvAHA', 'GBAHH', 'GBAHA', 'GBAH',
        'LBAHH', 'LBAHA', 'LBAH', 'B365AHH', 'B365AHA', 'B365AH', 'PSCH', 'PSCD', 'PSCA'
    ], index=False)
    # followed by manual cleaning via R

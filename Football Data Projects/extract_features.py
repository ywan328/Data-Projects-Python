import pandas as pd
import numpy as np
import csv


decay_rate, n_years = 0.2, 4
idx_stat, idx_odds, idx_res, n_str, n_stat, n_odds, n_num, n_all = 5, 29, 102, 5, 24, 73, 103, 108


def summarize(df, teamA, teamB, date, year):
    year = int(year)
    norm_AvsB, norm_BvsA, norm_ABvsC, norm_CvsAB = 0, 0, 0, 0
    sum_AvsB, sum_BvsA, sum_ABvsC, sum_CvsAB = np.zeros(n_num), np.zeros(n_num), np.zeros(n_num * 2), np.zeros(n_num * 2)
    for t in range(n_years + 1):
        dfoi = df[(df['Date'] < date) & (df['Year'] == year - t)]
        AvsB = dfoi[(dfoi['HomeTeam'] == teamA) & (dfoi['AwayTeam'] == teamB)].iloc[:, idx_stat:]
        BvsA = dfoi[(dfoi['HomeTeam'] == teamB) & (dfoi['AwayTeam'] == teamA)].iloc[:, idx_stat:]
        AvsX = dfoi[dfoi['HomeTeam'] == teamA].drop(columns=['Div', 'Date', 'Year', 'HomeTeam'])
        BvsX = dfoi[dfoi['HomeTeam'] == teamB].drop(columns=['Div', 'Date', 'Year', 'HomeTeam'])
        XvsA = dfoi[dfoi['AwayTeam'] == teamA].drop(columns=['Div', 'Date', 'Year', 'AwayTeam'])
        XvsB = dfoi[dfoi['AwayTeam'] == teamB].drop(columns=['Div', 'Date', 'Year', 'AwayTeam'])
        ABvsC = pd.merge(AvsX, BvsX, on='AwayTeam', suffixes=('_A', '_B')).drop(columns=['AwayTeam'])
        CvsAB = pd.merge(XvsA, XvsB, on='HomeTeam', suffixes=('_A', '_B')).drop(columns=['HomeTeam'])
        wt = decay_rate ** t
        norm_AvsB += wt * AvsB.shape[0]
        sum_AvsB += wt * AvsB.shape[0] * np.nan_to_num(AvsB.mean())
        norm_BvsA += wt * BvsA.shape[0]
        sum_BvsA += wt * BvsA.shape[0] * np.nan_to_num(BvsA.mean())
        norm_ABvsC += wt * ABvsC.shape[0]
        sum_ABvsC += wt * ABvsC.shape[0] * np.nan_to_num(ABvsC.mean())
        norm_CvsAB += wt * CvsAB.shape[0]
        sum_CvsAB += wt * CvsAB.shape[0] * np.nan_to_num(CvsAB.mean())
    if norm_AvsB != 0:
        sum_AvsB /= norm_AvsB
    if norm_BvsA != 0:
        sum_BvsA /= norm_BvsA
    if norm_ABvsC != 0:
        sum_ABvsC /= norm_ABvsC
    if norm_CvsAB != 0:
        sum_CvsAB /= norm_CvsAB
    return np.concatenate([sum_AvsB, sum_BvsA, sum_ABvsC, sum_CvsAB])


if __name__ == '__main__':
    df = pd.read_csv('cleaned_data.csv', dtype={
        'Div': 'str',
        'Date': 'str',
        'Year': 'int',
        'HomeTeam': 'str',
        'AwayTeam': 'str',
        'FTHG': 'float',
        'FTAG': 'float',
        'FTR': 'str',
        'HTHG': 'float',
        'HTAG': 'float',
        'HTR': 'str',
        'HS': 'float',
        'AS': 'float',
        'HST': 'float',
        'AST': 'float',
        'HHW': 'float',
        'AHW': 'float',
        'HC': 'float',
        'AC': 'float',
        'HF': 'float',
        'AF': 'float',
        'HFKC': 'float',
        'AFKC': 'float',
        'HO': 'float',
        'AO': 'float',
        'HY': 'float',
        'AY': 'float',
        'HR': 'float',
        'AR': 'float',
        'HBP': 'float',
        'ABP': 'float',
        'B365H': 'float',
        'B365D': 'float',
        'B365A': 'float',
        'BSH': 'float',
        'BSD': 'float',
        'BSA': 'float',
        'BWH': 'float',
        'BWD': 'float',
        'BWA': 'float',
        'GBH': 'float',
        'GBD': 'float',
        'GBA': 'float',
        'IWH': 'float',
        'IWD': 'float',
        'IWA': 'float',
        'LBH': 'float',
        'LBD': 'float',
        'LBA': 'float',
        'PSH': 'float',
        'PSD': 'float',
        'PSA': 'float',
        'SOH': 'float',
        'SOD': 'float',
        'SOA': 'float',
        'SBH': 'float',
        'SBD': 'float',
        'SBA': 'float',
        'SJH': 'float',
        'SJD': 'float',
        'SJA': 'float',
        'SYH': 'float',
        'SYD': 'float',
        'SYA': 'float',
        'VCH': 'float',
        'VCD': 'float',
        'VCA': 'float',
        'WHH': 'float',
        'WHD': 'float',
        'WHA': 'float',
        'Bb1X2': 'float',
        'BbMxH': 'float',
        'BbAvH': 'float',
        'BbMxD': 'float',
        'BbAvD': 'float',
        'BbMxA': 'float',
        'BbAvA': 'float',
        'BbOU': 'float',
        'BbMx>2.5': 'float',
        'BbAv>2.5': 'float',
        'BbMx<2.5': 'float',
        'BbAv<2.5': 'float',
        'GB>2.5': 'float',
        'GB<2.5': 'float',
        'B365>2.5': 'float',
        'B365<2.5': 'float',
        'BbAH': 'float',
        'BbAHh': 'float',
        'BbMxAHH': 'float',
        'BbAvAHH': 'float',
        'BbMxAHA': 'float',
        'BbAvAHA': 'float',
        'GBAHH': 'float',
        'GBAHA': 'float',
        'GBAH': 'float',
        'LBAHH': 'float',
        'LBAHA': 'float',
        'LBAH': 'float',
        'B365AHH': 'float',
        'B365AHA': 'float',
        'B365AH': 'float',
        'PSCH': 'float',
        'PSCD': 'float',
        'PSCA': 'float'
    }, parse_dates=[1])
    df = pd.get_dummies(df, columns=['FTR', 'HTR'], dtype=np.int)
    header = list(df)

    with open('features.csv', 'w') as ff, open('labels.csv', 'w') as fl:
        wf, wl = csv.writer(ff, delimiter=','), csv.writer(fl, delimiter=',')
        for idx, row in df.iterrows():
            row = list(row)
            labels = row[-6:-3]
            if np.any(np.isnan(labels)):
                continue
            features = np.zeros(n_num * 6 + n_odds)
            features[-n_odds:] = np.nan_to_num(row[idx_odds:idx_res])
            date, year, teamA, teamB = row[1:5]
            print(idx)
            features[:-n_odds] = summarize(df, teamA, teamB, date, year)
            wf.writerow(map(str, list(features)))
            wl.writerow(map(str, list(labels)))






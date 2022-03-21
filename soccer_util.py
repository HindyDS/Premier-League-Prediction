#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np

def who_won(row):
    splitted = row.split(":")
    if int(splitted[0]) > int(splitted[1]):
        return 0
    
    elif int(splitted[0]) < int(splitted[1]):
        return 1
    
    elif int(splitted[0]) == int(splitted[1]):
        return 2
    

def get_streak(df, team_encode, win_lose_draw):
    
    def get_single_streak(df, win_lose_draw):
        streak_count = 0
        streak = [0]
        
        if win_lose_draw == 'win':
            win_lose_draw_encode = 0

        elif win_lose_draw == 'lose':
            win_lose_draw_encode = 1

        elif win_lose_draw == 'draw':
            win_lose_draw_encode = 2
        
        winner = df.winner.to_list()
        for i in range(len(winner)):
            if i + 1 < len(winner):
                if winner[i] == win_lose_draw_encode and winner[i + 1] == win_lose_draw_encode:
                    streak_count += 1

                else:
                    streak_count = 0

                streak.append(streak_count)       

        return streak
    
    
    if team_encode == 'home':
        team_encode = 'home_team'

    elif team_encode == 'away':
        team_encode = 'away_team'

    group_by_team = list(df.groupby(team_encode))
    result = []
    for g in range(len(group_by_team)):
        grp = group_by_team[g][1].sort_values('datetime').copy()
        grp[f'{team_encode[:1]}_{win_lose_draw}_streak'] = get_single_streak(grp, win_lose_draw)
        grp[f'{team_encode[:1]}_{win_lose_draw}_streak'] = grp[f'{team_encode[:1]}_{win_lose_draw}_streak'].shift(1).fillna(0)
        result.append(grp)
    
    result = pd.concat(result)
    result.sort_index(inplace=True)
    
    return result


def get_count(df, team_encode, win_lose_draw):
    if win_lose_draw == 'win':
        win_lose_draw_encode = 0

    elif win_lose_draw == 'lose':
        win_lose_draw_encode = 1

    elif win_lose_draw == 'draw':
        win_lose_draw_encode = 2
        
    if team_encode == 'home':
        team_encode = 'home_team'

    elif team_encode == 'away':
        team_encode = 'away_team'
    
    group_by_team = list(df.groupby(team_encode))
    
    result = []
    for g in range(len(group_by_team)):
        grp = group_by_team[g][1].sort_values('datetime').copy()
        grp[f'{team_encode[:1]}_{win_lose_draw}_counts'] = np.where(grp.winner == win_lose_draw_encode, 1, 0)
        grp[f'{team_encode[:1]}_{win_lose_draw}_counts'] = grp[f'{team_encode[:1]}_{win_lose_draw}_counts'].shift(1)
        grp[f'{team_encode[:1]}_{win_lose_draw}_counts'] = grp[f'{team_encode[:1]}_{win_lose_draw}_counts'].fillna(0)
        grp[f'{team_encode[:1]}_{win_lose_draw}_counts'] = grp[f'{team_encode[:1]}_{win_lose_draw}_counts'].cumsum()
        grp[f'{team_encode[:1]}_{win_lose_draw}_counts'] = grp[f'{team_encode[:1]}_{win_lose_draw}_counts'].astype(int)
        result.append(grp)
        
    return pd.concat(result)


def get_match_count(df, team_encode):
    if team_encode == 'home':
        team_encode = 'home_team'

    elif team_encode == 'away':
        team_encode = 'away_team'
    
    result = []
    group_by_team = list(df.groupby(team_encode))
    for g in range(len(group_by_team)):
        grp = group_by_team[g][1].sort_values('datetime').copy()
        grp[f'{team_encode[:1]}_total_match_count'] = range(1, len(grp) + 1)
        result.append(grp)
        
    return pd.concat(result)


def get_match_specific_streak(df, team_encode, win_lose_draw):
    
    def get_single_streak(df, win_lose_draw):
        streak_count = 0
        streak = [0]
        
        if win_lose_draw == 'win':
            win_lose_draw_encode = 0

        elif win_lose_draw == 'lose':
            win_lose_draw_encode = 1

        elif win_lose_draw == 'draw':
            win_lose_draw_encode = 2
        
        winner = df.winner.to_list()
        for i in range(len(winner)):
            if i + 1 < len(winner):
                if winner[i] == win_lose_draw_encode and winner[i + 1] == win_lose_draw_encode:
                    streak_count += 1

                else:
                    streak_count = 0

                streak.append(streak_count)       

        return streak
    
    
    if team_encode == 'home':
        team_encode = 'home_team'

    elif team_encode == 'away':
        team_encode = 'away_team'
    
    group_by_team = list(df.groupby(['home_team', 'away_team', team_encode]))
    result = []
    for g in range(len(group_by_team)):
        grp = group_by_team[g][1].sort_values('datetime').copy()
        grp[f'{team_encode[:1]}_match_specific_{win_lose_draw}_streak'] = get_single_streak(grp, win_lose_draw)
        grp[f'{team_encode[:1]}_match_specific_{win_lose_draw}_streak'] = grp[f'{team_encode[:1]}_match_specific_{win_lose_draw}_streak'].shift(1).fillna(0)
        result.append(grp)
    
    result = pd.concat(result)
    result.sort_index(inplace=True)
    
    return result


def get_match_specific_count(df, team_encode, win_lose_draw):
    if win_lose_draw == 'win':
        win_lose_draw_encode = 0

    elif win_lose_draw == 'lose':
        win_lose_draw_encode = 1

    elif win_lose_draw == 'draw':
        win_lose_draw_encode = 2
        
    if team_encode == 'home':
        team_encode = 'home_team'

    elif team_encode == 'away':
        team_encode = 'away_team'
    
    group_by_team = list(df.groupby(['home_team', 'away_team', team_encode]))
    
    result = []
    for g in range(len(group_by_team)):
        grp = group_by_team[g][1].sort_values('datetime').copy()
        grp[f'{team_encode[:1]}_match_specific_{win_lose_draw}_counts'] = np.where(grp.winner == win_lose_draw_encode, 1, 0)
        grp[f'{team_encode[:1]}_match_specific_{win_lose_draw}_counts'] = grp[f'{team_encode[:1]}_match_specific_{win_lose_draw}_counts'].shift(1)
        grp[f'{team_encode[:1]}_match_specific_{win_lose_draw}_counts'] = grp[f'{team_encode[:1]}_match_specific_{win_lose_draw}_counts'].fillna(0)
        grp[f'{team_encode[:1]}_match_specific_{win_lose_draw}_counts'] = grp[f'{team_encode[:1]}_match_specific_{win_lose_draw}_counts'].cumsum()
        grp[f'{team_encode[:1]}_match_specific_{win_lose_draw}_counts'] = grp[f'{team_encode[:1]}_match_specific_{win_lose_draw}_counts'].astype(int)
        result.append(grp)
        
    result = pd.concat(result)
    result.sort_index(inplace=True)
        
    return result


def get_match_specific_match_count(df):
    group_by_teams = list(df.groupby(['home_team', 'away_team']))
    result = []
    for g in range(len(group_by_teams)):
        grp = group_by_teams[g][1].sort_values('datetime').copy()
        grp['match_specific_total_match_count'] = range(1, len(grp) + 1)
        grp['match_specific_total_match_count'] = grp['match_specific_total_match_count'].shift(1).fillna(0).astype(int)
        result.append(grp)
    
    result = pd.concat(result)
    result.sort_index(inplace=True)
        
    return result

def get_match_specific_pct(df, team_encode, win_lose_draw):
    if win_lose_draw == 'win':
        win_lose_draw_encode = 0

    elif win_lose_draw == 'lose':
        win_lose_draw_encode = 1

    elif win_lose_draw == 'draw':
        win_lose_draw_encode = 2
        
    if team_encode == 'home':
        team_encode = 'home_team'

    elif team_encode == 'away':
        team_encode = 'away_team'
        
    df[f'{team_encode[:1]}_match_specific_%{win_lose_draw}'] = df[f'{team_encode[:1]}_match_specific_{win_lose_draw}_counts'] / df[f'match_specific_total_match_count']
    df[f'{team_encode[:1]}_match_specific_%{win_lose_draw}'] = df[f'{team_encode[:1]}_match_specific_%{win_lose_draw}'].fillna(0)
    return df


def get_odd_ratio(df, h_team_encode, a_team_encode): 
    df[f'{h_team_encode[:1]}_to_{a_team_encode[:1]}_odd_ratio'] = df[f'{h_team_encode[:1]}_odd'].astype(float) / df[f'{a_team_encode[:1]}_odd'].astype(float)
    
    return df   

    
def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data


def get_ewm(df):
    for t in ['h', 'a']:
        group_by_team = list(df.groupby('home_team'))
        result = []
        for g in range(len(group_by_team)):
            grp = group_by_team[g][1].sort_values('datetime').copy()
            grp[f'{t}_total_match_count_ewm'] = grp[f'{t}_total_match_count'].ewm(1).mean()
            grp[f'{t}_win_streak_ewm'] = grp[f'{t}_win_streak'].ewm(1).mean()
            grp[f'{t}_win_counts_ewm'] = grp[f'{t}_win_counts'].ewm(1).mean()
            grp[f'{t}_lose_streak_ewm'] = grp[f'{t}_lose_streak'].ewm(1).mean()
            grp[f'{t}_draw_streak_ewm'] = grp[f'{t}_draw_streak'].ewm(1).mean()
            grp[f'{t}_draw_counts_ewm'] = grp[f'{t}_draw_counts'].ewm(1).mean()
            result.append(grp)
            
        result = pd.concat(result)
        result.sort_index(inplace=True)
        df = result.copy()
        
    return result


def get_match_specific_ewm(df):
    for t in ['home_team', 'away_team']:
        group_by_team = list(df.groupby(['home_team', 'away_team', t]))
        result = []
        for g in range(len(group_by_team)):
            grp = group_by_team[g][1].sort_values('datetime').copy()
            grp[f'{t[:1]}_match_specific_win_streak_ewm'] = grp[f'{t[:1]}_match_specific_win_streak'].ewm(1).mean()
            grp[f'{t[:1]}_match_specific_win_counts_ewm'] = grp[f'{t[:1]}_match_specific_win_counts'].ewm(1).mean()
            grp[f'{t[:1]}_match_specific_lose_streak_ewm'] = grp[f'{t[:1]}_match_specific_lose_streak'].ewm(1).mean()
            grp[f'{t[:1]}_match_specific_draw_streak_ewm'] = grp[f'{t[:1]}_match_specific_draw_streak'].ewm(1).mean()
            grp[f'{t[:1]}_match_specific_draw_counts_ewm'] = grp[f'{t[:1]}_match_specific_draw_counts'].ewm(1).mean()
            result.append(grp)
            
        result = pd.concat(result)
        result.sort_index(inplace=True)
        df = result.copy()
        
    return result


def elo_score(ra, rb):
    '''
    ra - previous rating of player a
    rb - previous rating of player b
    ea - expected score of player a
    eb - expected score of player b
    '''
    
    ea = 1 / (1 + 10 ** ((rb - ra) / 400))
    eb = 1 / (1 + 10 ** ((ra - rb) / 400))
    
    return ea, eb


def elo_update(ra, sa, ea, rb, sb, eb, k):
    '''
    ra - previous rating of player a
    rb - previous rating of player b
    ea - expected score of player a
    eb - expected score of player b
    sa - actual score of player a
    sb - actual score of player b
    k - weight constant
    '''

    ra_prime = ra + k * (sa - ea)
    rb_prime = rb + k * (sb - eb)
    
    return ra_prime, rb_prime

def shift_elo(df, initial_elo=0):
    group_by_home = list(df.groupby('home_team'))
    result = []
    for g in range(len(group_by_home)):
        grp = group_by_home[g][1].sort_values('datetime').copy()
        grp['home_elo'] = grp['home_elo'].shift(1).fillna(initial_elo)
        grp['away_elo'] = grp['away_elo'].shift(1).fillna(initial_elo)
        result.append(grp)

    return pd.concat(result).sort_index()


import itertools
import pandas as pd
from sklearn.metrics import accuracy_score

class Loop_Stacker:
    def __init__(self, predictions):
        self.predictions = predictions
    
    def fit(self, verbosity=1):
        best_combinations = {}
        best_combinations_frames = {}
        count = 1
        while True:
            print(f"Round: {count}")
            com_keys = [itertools.combinations(self.predictions.keys(), i) for i in range(2, len(self.predictions) + 1)]
            com_vals = [itertools.combinations(self.predictions.values(), i) for i in range(2, len(self.predictions) + 1)]
            
            if len(com_keys) == 0:
                break
            
            curr_best_score = 0
            curr_best_combinations = ""
            for key, val in zip(com_keys, com_vals):
                for k, v in zip(key, val):
                    y_pred = pd.DataFrame(v).mode().iloc[0]
                    score = accuracy_score(y_pred, y_test)

                    if curr_best_score < score:
                        curr_best_score = score
                        curr_best_combinations = k
                        best_combinations[count] = curr_best_combinations
                        best_combinations_frames[count] = y_pred
                        
                        if verbosity > 0:
                            print(k, score)

            for com in curr_best_combinations:
                self.predictions.pop(com)   

            count += 1
            
            if verbosity > 0:
                print("\n")

        y_pred = pd.DataFrame([v for v in best_combinations_frames.values()]).mode().iloc[0]
        all_stacked_score = accuracy_score(y_pred, y_test)
        
        if verbosity > 0:
            print(f"All stacked: {all_stacked_score}")
            
        best_combinations_frames['all_stacked'] = y_pred
        
        self.best_combinations = best_combinations
        self.best_combinations_frames = best_combinations_frames

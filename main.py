import requests
import pandas as pd
import numpy as np
import pickle
import os
import itertools
from tqdm import tqdm
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import yeojohnson

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from nba_api.stats.endpoints import playercareerstats, commonplayerinfo
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

CURRENT_SEASON = '2023-24'

weights_myleague = {
    "FGM": 35,
    "FGA": -10,
    "FG3M": 2.5,
    "FTM": 8,
    "FTA": -3,
    "OREB": 5,
    "REB": 8,
    "AST": 17,
    "STL": 30,
    "BLK": 30,
    "TOV": -15,
    "PTS": 5,
}

# weights_myleague = {
#     "FGM": 2,
#     "FGA": -1,
#     "FG3M": 1,
#     "FTM": 1,
#     "FTA": -1,
#     "OREB": 0,
#     "REB": 1,
#     "AST": 2,
#     "STL": 4,
#     "BLK": 4,
#     "TOV": -2,
#     "PTS": 1,
# }
weights_ESPN = {
    "FGM": 2,
    "FGA": -1,
    "FG3M": 1,
    "FTM": 1,
    "FTA": -1,
    "OREB": 0,
    "REB": 1,
    "AST": 2,
    "STL": 4,
    "BLK": 4,
    "TOV": -2,
    "PTS": 1,
}

# Create a custom requests session
session = requests.Session()
retries = Retry(total=15, backoff_factor=1, status_forcelist=[ 429, 500, 502, 503, 504 ])
session.mount('https://', HTTPAdapter(max_retries=retries))

# Directory for caching
cache_dir = 'player_stats_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Cache functions
def cache_exists(player_id, season, data_type):
    return os.path.exists(os.path.join(cache_dir, f'{player_id}_{season}_{data_type}.pkl'))

def load_from_cache(player_id, season, player_name, data_type):
    file_path = os.path.join(cache_dir, f'{player_id}_{season}_{data_type}.pkl')
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except (EOFError, FileNotFoundError):
        print(f"Cache file for player {player_id} {player_name}, season {season}, type {data_type} is corrupted or missing.")
        if os.path.exists(file_path):
            os.remove(file_path)
        return None

def save_to_cache(player_id, season, data_type, data):
    file_path = os.path.join(cache_dir, f'{player_id}_{season}_{data_type}.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def fetch_player_game_data(player_id, season, player_name):
    cached_games = load_from_cache(player_id, season, player_name, "games")
    today = datetime.today()

    # Convert 'GAME_DATE' in cached data to datetime
    if cached_games is not None and not cached_games.empty:
        cached_games['GAME_DATE'] = pd.to_datetime(cached_games['GAME_DATE'], format='%b %d, %Y')

    # Check if cached data is outdated or missing
    if cached_games is None or (season == CURRENT_SEASON and (cached_games.empty or cached_games['GAME_DATE'].max() < today)):
        # Fetch new game data from API
        game_log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        new_game_data = game_log.get_data_frames()[0]
        new_game_data['GAME_DATE'] = pd.to_datetime(new_game_data['GAME_DATE'], format='%b %d, %Y')

        # Append new data to cached data if necessary
        if cached_games is not None and not cached_games.empty and season == CURRENT_SEASON:
            combined_data = pd.concat([cached_games, new_game_data]).drop_duplicates().sort_values(by=['GAME_DATE'], ascending=False).reset_index(drop=True)
            save_to_cache(player_id, season, "games", combined_data)
            return combined_data
        else:
            save_to_cache(player_id, season, "games", new_game_data)
            return new_game_data
    else:
        return cached_games

#main code
data_type = 'season'

# Fetch current NBA players
current_players = players.get_active_players()

# List to store player stats DataFrames
player_stats_list = []
all_players_game_data = []

for player in tqdm(current_players, desc='Loading player data'):
    player_id = player['id']
    player_name = player['full_name']

    try:
        # Fetch player career stats
        career_stats = playercareerstats.PlayerCareerStats(player_id=player_id, timeout=60)
        career_stats_df = career_stats.get_data_frames()[0]

        for season in career_stats_df['SEASON_ID'].unique():
            season_data = career_stats_df[career_stats_df['SEASON_ID'] == season].copy()
            cached_data = load_from_cache(player_id, season, player_name, data_type) if season != CURRENT_SEASON else None

            if cached_data is None or not isinstance(cached_data, pd.DataFrame) or 'TEAM_ABBREVIATION' not in cached_data.columns:
                # Fetch detailed player info for position
                player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id, timeout=60).get_data_frames()[0]
                position = player_info['POSITION'][0]

                # Add player info to the season data using .loc
                season_data.loc[:, 'PLAYER_NAME'] = player_name
                season_data.loc[:, 'POSITION'] = position

                # Save the season data to cache
                save_to_cache(player_id, season, data_type, season_data)
                player_stats_list.append(season_data)
            else:
                # Check if cached data is a DataFrame and append
                if isinstance(cached_data, pd.DataFrame):
                    player_stats_list.append(cached_data)
                else:
                    print(f"Cached data format error for player {player_id} {player_name}, season {season}")

            # Fetch individual game data for the player and season
            game_data = fetch_player_game_data(player_id, season, player_name)

            # Add player name and season to the game data for reference
            game_data.loc[:, 'PLAYER_NAME'] = player_name
            game_data.loc[:, 'SEASON_ID'] = season

            # Append the game data to the list
            all_players_game_data.append(game_data)

    except requests.exceptions.RequestException as e:
        print(f"Request failed for player {player_id} {player_name}: {e}")
        continue

# Concatenate all DataFrames in the list
player_stats_df = pd.concat(player_stats_list, ignore_index=True)
player_stats_df.to_pickle('player_stats.pkl')

# Concatenate all individual game data into a single DataFrame
all_games_df = pd.concat(all_players_game_data, ignore_index=True)
all_games_df.to_pickle('all_games.pkl')

player_stats_df = pd.read_pickle('player_stats.pkl')
# all_games_df = pd.read_pickle('all_games.pkl')

def custom_round(value, decimals):
    return float(Decimal(value).quantize(Decimal('0.' + '0' * decimals), rounding=ROUND_HALF_UP))

# List of stat columns to convert to per game stats
stat_columns = ['MIN', 'FGM', 'FGA', 'FTM', 'FTA', 'FG3M', 'PTS', 'OREB', 'REB', 'AST', 'STL', 'BLK', 'TOV']

# Create a new DataFrame for per game stats
per_game_stats_df = player_stats_df.copy()

# Rename original stat columns temporarily
temp_column_names = {col: f'{col}_total' for col in stat_columns}
per_game_stats_df.rename(columns=temp_column_names, inplace=True)

for column in stat_columns:
    per_game_stats_df[column] = per_game_stats_df[f'{column}_total'] / per_game_stats_df['GP']
    per_game_stats_df[column] = per_game_stats_df[column].apply(lambda x: custom_round(x, 1))

# Drop the original (now renamed) cumulative stat columns
per_game_stats_df.drop(list(temp_column_names.values()), axis=1, inplace=True)


def get_game_info_df(weights):
    all_games_df=pd.read_pickle('all_games.pkl')
    # Add weighted stat columns
    for stat, weight in weights.items():
        per_game_stats_df[f'{stat}_PTS'] = (per_game_stats_df[stat] * weight).round(2)

    # List of all weighted stat columns
    weighted_stat_columns = [f'{stat}_PTS' for stat in weights.keys()]

    # Calculate total points
    per_game_stats_df['TOTAL'] = per_game_stats_df[weighted_stat_columns].sum(axis=1).round(2)

    #2. number 1 + the latest team the player was in on the traded season
    # DataFrame to store the final filtered data
    final_df = pd.DataFrame()

    # Process data for each player
    for player_id in per_game_stats_df['PLAYER_ID'].unique():
        # Filter to get all rows for this player
        player_df = per_game_stats_df[per_game_stats_df['PLAYER_ID'] == player_id]

        # Iterate through each unique season for the player
        for season in player_df['SEASON_ID'].unique():
            player_season_df = player_df[player_df['SEASON_ID'] == season]

            # Check if 'TOT' is present for this season
            if 'TOT' in player_season_df['TEAM_ABBREVIATION'].values:
                # Get the 'TOT' row
                tot_row = player_season_df[player_season_df['TEAM_ABBREVIATION'] == 'TOT']

                # Get the row for the latest team in that season (last row before 'TOT')
                latest_team_row = player_season_df[(player_season_df['TEAM_ABBREVIATION'] != 'TOT') & 
                                                (player_season_df['TEAM_ID'] != 0)].iloc[-1:]

                # Append both 'TOT' row and latest team row to final_df
                final_df = pd.concat([final_df, tot_row, latest_team_row])
            else:
                # If 'TOT' is not present, append the whole season data
                final_df = pd.concat([final_df, player_season_df])

    # Removing potential duplicates (if any)
    final_df = final_df.drop_duplicates(subset=['PLAYER_ID', 'SEASON_ID', 'TEAM_ID'])

    # Apply weights to each game stat
    for stat, weight in weights.items():
        all_games_df[f'{stat}_weighted'] = (all_games_df[stat] * weight).round(2)

    # Calculate total score for each game
    all_games_df['total_score'] = all_games_df[[f'{stat}_weighted' for stat in weights.keys()]].sum(axis=1).round(2)

    # Assuming 'all_games_df' is your DataFrame
    # Define the columns to consider when identifying duplicates (all columns except 'PLUS_MINUS')
    columns_to_consider = all_games_df.columns.difference(['PLUS_MINUS'])

    # Drop duplicates based on these columns
    all_games_df = all_games_df.drop_duplicates(subset=columns_to_consider)

    # Aggregate total scores per player per season
    player_season_totals = all_games_df.groupby(['Player_ID', 'SEASON_ID'])['total_score'].sum().reset_index()

    # Rank players based on aggregated total scores per season
    player_season_totals['rank_in_season'] = player_season_totals.groupby('SEASON_ID')['total_score'].rank(method='max', ascending=False)
    # Filter to get the top 250 players per season
    top_players = player_season_totals[player_season_totals['rank_in_season'] <= 250]

    # Now merge this back with all_games_df to filter only relevant player-season combinations
    top_players_df = all_games_df.merge(top_players[['Player_ID', 'SEASON_ID']], on=['Player_ID', 'SEASON_ID'])

    # Group by player and season
    grouped = top_players_df.groupby(['Player_ID', 'SEASON_ID'])

    # Calculate mean and standard deviation for each player-season group
    summary_df = grouped['total_score'].agg(['mean', 'std']).reset_index().round(2)
    summary_df.rename(columns={'mean': 'average_total', 'std': 'std_dev_total'}, inplace=True)

    # Calculate global mean and std for each season
    global_stats = top_players_df.groupby('SEASON_ID')['total_score'].agg(['mean', 'std']).reset_index().round(2)
    global_stats.rename(columns={'mean': 'mean_global', 'std': 'std_global'}, inplace=True)

    # Merge global stats with summary_df
    summary_df = summary_df.merge(global_stats, on='SEASON_ID')

    # Calculate z-score
    summary_df['z_score'] = summary_df.apply(lambda row: (row['average_total'] - row['mean_global']) / row['std_global'] if row['std_global'] != 0 else 0, axis=1).round(2)

    # Merge with player names for readability
    summary_df = summary_df.merge(top_players_df[['Player_ID', 'PLAYER_NAME']].drop_duplicates(), on='Player_ID')
    # Calculate global mean and std for each weighted stat per season
    global_stats_per_stat = top_players_df.groupby('SEASON_ID')[[f'{stat}_weighted' for stat in weights.keys()]].agg(['mean', 'std']).reset_index()

    # Flatten the column multi-index
    global_stats_per_stat.columns = ['_'.join(col).strip() for col in global_stats_per_stat.columns.values]
    global_stats_per_stat.rename(columns={'SEASON_ID_': 'SEASON_ID'}, inplace=True)

    # Initialize a DataFrame to store z-scores
    z_scores_df = top_players_df[['Player_ID', 'SEASON_ID', 'PLAYER_NAME']].drop_duplicates().reset_index(drop=True)
    # Merge global stats with individual game data
    all_games_merged = top_players_df.merge(global_stats_per_stat, on='SEASON_ID')

    # Calculate z-scores for each weighted stat
    for stat in weights.keys():
        mean_col = f'{stat}_weighted_mean'
        std_col = f'{stat}_weighted_std'
        z_scores_df[f'{stat}_z_score'] = all_games_merged.apply(
            lambda row: (row[f'{stat}_weighted'] - row[mean_col]) / row[std_col] if row[std_col] != 0 else 0, axis=1).round(2)

    # Group by player and season to get the average z-scores
    z_scores_df = z_scores_df.groupby(['Player_ID', 'SEASON_ID']).mean(numeric_only=True).reset_index()

    # Merge with player names for readability
    z_scores_df = z_scores_df.merge(top_players_df[['Player_ID', 'PLAYER_NAME']].drop_duplicates(), on='Player_ID')
    return all_games_df, summary_df, z_scores_df, top_players_df

def calculate_recent_days_avg(player_id, all_games_df, day_step, max_days=210):
    player_games = all_games_df[(all_games_df['Player_ID'] == player_id) & (all_games_df['SEASON_ID'] == CURRENT_SEASON)].sort_values('GAME_DATE', ascending=False)
    num_games_played = len(player_games)
    if num_games_played == 0:
        return 0

    recent_values = []
    total_weight = 0
    current_date = datetime.today()

    # Calculate unique day intervals based on actual games played
    last_game_date = player_games['GAME_DATE'].min()
    total_days = (current_date - last_game_date).days
    days_intervals = [i for i in range(day_step, min(total_days, max_days) + 1, day_step)]

    for index, days in enumerate(days_intervals):
        end_date = current_date - timedelta(days=days)
        interval_games = player_games[player_games['GAME_DATE'] >= end_date]
        # print(interval_games)

        if not interval_games.empty:
            interval_avg = interval_games['total_score'].mean().round(2)
            # print(interval_avg)
            weight = 1 / (index + 1)  # Higher weight for more recent intervals
            # print(weight)
            recent_values.append(interval_avg * weight)
            total_weight += weight

    combined_recent_value = sum(recent_values) / total_weight if total_weight > 0 else 0
    return combined_recent_value

def calculate_interval_values(player_id, all_games_df, interval_step, max_games=82):
    player_games = all_games_df[(all_games_df['Player_ID'] == player_id) & (all_games_df['SEASON_ID'] == CURRENT_SEASON)].sort_values('GAME_DATE', ascending=False)
    num_games_played = len(player_games)
    if num_games_played <= 6:
        return 0, 0
    
    recent_values = []
    current_values = []
    total_recent_weight = 0  # Sum of weights for recent values
    total_current_weight = 0  # Sum of weights for current values

    # Season-long average (highest weight for current value)
    season_avg = player_games['total_score'].mean() if not player_games.empty else 0
    season_weight = 20  # High weight for the full season average
    current_values.append(season_avg * season_weight)
    total_current_weight += season_weight

    # Generate intervals and adjust them based on the number of games played
    intervals = [i for i in range(interval_step, max_games + 1, interval_step) if i <= num_games_played]
    if num_games_played not in intervals:
        intervals.append(num_games_played)  # Add the remainder interval

    for index, interval in enumerate(intervals):
        interval_games = player_games.head(interval)
        interval_avg = interval_games['total_score'].mean() if not interval_games.empty else 0

        # Apply weights for recent value (more recent intervals have higher weights)
        recent_weight = 1 / (index + 1)
        recent_values.append(interval_avg * recent_weight)
        total_recent_weight += recent_weight

        # For current value, weights increase as intervals get longer
        current_weight = index + 1
        current_values.append(interval_avg * current_weight)
        total_current_weight += current_weight

    combined_recent_value = sum(recent_values) / total_recent_weight
    combined_current_value = sum(current_values) / total_current_weight

    return combined_recent_value, combined_current_value

def get_player_trend_ratio(player_id, all_games_df, interval_steps, days_list):
    interval_ratios = []
    day_ratios = []
    current_values = []

    for interval_step in interval_steps:
        recent_value, current_value = calculate_interval_values(player_id, all_games_df, interval_step)
        if (current_value != 0 and recent_value != 0):  # To avoid division by zero
            ratio = recent_value / current_value
            interval_ratios.append(ratio)
        current_values.append(current_value)

    # Calculate day ratios for each day step
    for days in days_list:
        days_avg = calculate_recent_days_avg(player_id, all_games_df, days)
        # Use the last current value for comparison, which corresponds to the full season average
        if current_values[-1] != 0:  # To avoid division by zero
            day_ratios.append(days_avg / current_values[-1])
    
    # Calculate the average ratio across all interval steps and days
    average_interval_ratio = np.mean(interval_ratios) if interval_ratios else 0
    average_day_ratio = np.mean(day_ratios) if day_ratios else 0

    # Weights for combining the ratios
    weight_interval = 0.2  # Weight for the interval-based trend ratio
    weight_days = 0.8  # Weight for the day-based trend ratio

    # Calculate the combined trend ratio using the weighted average of interval and day ratios
    combined_trend_ratio = (weight_interval * average_interval_ratio) + (weight_days * average_day_ratio)
    return combined_trend_ratio

def get_final_df(weights):
    all_games_df, summary_df, z_scores_df, top_players_df = get_game_info_df(weights)

    interval_steps = [3, 4, 5, 6]
    days_list = [5, 6, 7, 8, 9, 10]
    player_trend_ratios = {}

    player_ids = top_players_df['Player_ID'].unique()
    for player_id in player_ids:
        trend_ratio = get_player_trend_ratio(player_id, all_games_df, interval_steps, days_list)
        player_trend_ratios[player_id] = trend_ratio
        # print(f"Player ID {player_id} - Trend Ratio: {trend_ratio}")

    # Create a DataFrame from player_trend_ratios
    trend_ratio_df = pd.DataFrame(player_trend_ratios.items(), columns=['Player_ID', 'Trend_Ratio'])
    trend_ratio_df.sort_values('Trend_Ratio', ascending=False, inplace=True)
    trend_ratio_df = trend_ratio_df[trend_ratio_df['Trend_Ratio'] != 0]

    # Define past seasons and season_weights
    past_seasons = ['2022-23', '2021-22', '2020-21', '2019-20', '2018-19']  # Replace with actual season IDs
    season_weights = [1.0, 0.5, 0.2, 0.1, 0.1]

    # Assuming 'Player_ID' is the column in the DataFrame that contains the player IDs
    player_ids = top_players_df['Player_ID'].unique()

    # Define a threshold for significant deviation (example: 0.5 standard deviations)
    deviation_threshold = 0.7

    # Iterate over players to identify buy low/sell high candidates
    buy_low_candidates = []
    sell_high_candidates = []

    # DataFrame to store all players with their stats
    candidates_df = pd.DataFrame(columns=['Player_ID', 'Player_Name', 'Current_Season_Avg', 'Current_Season_Std', 'Past_Season_Weighted_Avg', 'Past_Season_Weighted_Std', 'Current_Season_Z_Score', 'Category'])

    # First, calculate the mean and standard deviation of past season averages for all players
    past_season_averages = []
    for player_id in player_ids:
        player_data = summary_df[summary_df['Player_ID'] == player_id]
        player_past_seasons = player_data[player_data['SEASON_ID'].isin(past_seasons)]
        player_past_avg = player_past_seasons['average_total'].mean() if not player_past_seasons.empty else np.nan
        past_season_averages.append(player_past_avg)

    mean_past_avg = np.nanmean(past_season_averages)
    std_past_avg = np.nanstd(past_season_averages)

    # DataFrame to store all players with their stats
    candidates_df = pd.DataFrame()
    normalized_weighted_sums = []

    # Lists to store individual components
    components = {'Z_Score_Diff': [], 'Deviation_Percentage': [], 'Current_Method_Score': []}

    for player_id in player_ids:
        player_data = summary_df[summary_df['Player_ID'] == player_id]
        current_season_games_played = all_games_df[(all_games_df['Player_ID'] == player_id) & (all_games_df['SEASON_ID'] == CURRENT_SEASON)].shape[0]

        weighted_avg = 0
        total_weight = 0
        weighted_variance = 0

        # Initialize current season variables
        current_season_avg = np.nan
        current_season_std_dev = np.nan
        current_season_z_score = np.nan

        for season, weight in zip(past_seasons, season_weights):
            season_data = player_data[player_data['SEASON_ID'] == season]
            if not season_data.empty:
                season_avg = season_data['average_total'].iloc[0]
                season_std = season_data['std_dev_total'].iloc[0]
                season_variance = season_std ** 2

                weighted_avg += season_avg * weight
                weighted_variance += season_variance * weight
                total_weight += weight

        if total_weight > 0:
            weighted_avg /= total_weight
            weighted_std = np.sqrt(weighted_variance / total_weight)
            past_season_z_score = ((weighted_avg - mean_past_avg) / std_past_avg).round(2)

            # Retrieve current season data
            current_season_data = player_data[player_data['SEASON_ID'] == CURRENT_SEASON]
            if not current_season_data.empty:
                current_season_avg = current_season_data['average_total'].iloc[0]
                current_season_std_dev = current_season_data['std_dev_total'].iloc[0]
                current_season_z_score = current_season_data['z_score'].iloc[0]

                # Calculate components
                z_score_diff = current_season_z_score - past_season_z_score
                deviation_percentage = ((current_season_avg - weighted_avg) / weighted_avg) * 100
                current_method_score = current_season_avg - (weighted_avg - deviation_threshold * weighted_std)

                components['Z_Score_Diff'].append(z_score_diff)
                components['Deviation_Percentage'].append(deviation_percentage)
                components['Current_Method_Score'].append(current_method_score)

                # Prepare row for DataFrame
                new_row = {
                    'Player_ID': player_id,
                    'Player_Name': player_data['PLAYER_NAME'].iloc[0],
                    'Current_Season_Avg': current_season_avg,
                    'Current_Season_Std': current_season_std_dev,
                    'Current_Season_Games_Played': current_season_games_played,
                    'Past_Season_Weighted_Avg': weighted_avg.round(1),
                    'Past_Season_Weighted_Std': weighted_std.round(1),
                    'Past_Season_Z_Score': past_season_z_score,
                    'Current_Season_Z_Score': current_season_z_score,
                }
                candidates_df = pd.concat([candidates_df, pd.DataFrame([new_row])], ignore_index=True)

    # Calculate the median of the Trend_Ratio for players with sufficient games
    median_trend_ratio = trend_ratio_df['Trend_Ratio'].median()

    # Merge with candidates_df
    candidates_df = candidates_df.merge(trend_ratio_df, on='Player_ID', how='left') #current season trend ratio

    # Fill missing trend ratios with the median value
    candidates_df['Trend_Ratio'].fillna(median_trend_ratio, inplace=True)

    candidates_df['Pct_Diff_from_Previous_Seasons'] = components['Deviation_Percentage']

    # Convert components dictionary to DataFrame
    components_df = pd.DataFrame(components)

    # Apply the Yeo-Johnson transformation to the Deviation_Percentage
    normalized_deviation_percentage, lmbda = yeojohnson(components_df['Deviation_Percentage'])
    components_df['Deviation_Percentage'] = normalized_deviation_percentage.round(2)

    # Apply Min-Max Normalization
    scaler = MinMaxScaler()
    components_df[['Normalized_Z_Score_Diff', 'Normalized_Deviation_Percentage', 'Normalized_Current_Method_Score']] = scaler.fit_transform(components_df[['Z_Score_Diff', 'Deviation_Percentage', 'Current_Method_Score']])

    # Reshape the 'Trend_Ratio' column into a 2D array
    trend_ratio_reshaped = candidates_df['Trend_Ratio'].values.reshape(-1, 1)
    # Apply MinMaxScaler to the reshaped data
    normalized_trend_ratio = scaler.fit_transform(trend_ratio_reshaped)
    # Add the normalized trend ratio back to your DataFrame
    components_df['Normalized_Trend_Ratio'] = normalized_trend_ratio.flatten()

    # Define weights for each component
    final_weights = {
        'Normalized_Z_Score_Diff': 0.9,  # Weight for Z-Score Difference
        'Normalized_Deviation_Percentage': 0.5,  # Weight for Deviation Percentage
        'Normalized_Current_Method_Score': 0.7,  # Weight for Current Method Score
        'Normalized_Trend_Ratio': 2  # Higher weight for Trend Ratio
    }

    # Calculate the weighted sum of the components
    weighted_components = (components_df['Normalized_Z_Score_Diff'] * final_weights['Normalized_Z_Score_Diff'] +
                        components_df['Normalized_Deviation_Percentage'] * final_weights['Normalized_Deviation_Percentage'] +
                        components_df['Normalized_Current_Method_Score'] * final_weights['Normalized_Current_Method_Score'] +
                        components_df['Normalized_Trend_Ratio'] * final_weights['Normalized_Trend_Ratio'])

    prev_season_weighted_components = (components_df['Normalized_Z_Score_Diff'] * final_weights['Normalized_Z_Score_Diff'] +
                                    components_df['Normalized_Deviation_Percentage'] * final_weights['Normalized_Deviation_Percentage'] +
                                    components_df['Normalized_Current_Method_Score'] * final_weights['Normalized_Current_Method_Score'])

    # Assign the weighted sum to Combined_Score
    candidates_df['Combined_Score'] = weighted_components.round(2)
    candidates_df['Prev_Season_Combined_Score'] = prev_season_weighted_components.round(2)

    # Normalize the weighted sum (Z-score normalization)
    mean_weighted = candidates_df['Combined_Score'].mean()
    std_weighted = candidates_df['Combined_Score'].std()
    candidates_df['Normalized_Score'] = ((candidates_df['Combined_Score'] - mean_weighted) / std_weighted).round(2)

    # Assign categories based on normalized score
    def assign_category(normalized_score):
        if normalized_score > 1.5:
            return 'Sell High'
        elif normalized_score < -1.5:
            return 'Buy Low'
        else:
            return 'Neutral'

    candidates_df['Category'] = candidates_df['Normalized_Score'].apply(assign_category)

    candidates_df['Pct_Diff_from_Current_Season'] = ((candidates_df['Trend_Ratio'] - 1) * 100).apply(lambda x: f"{x:.0f}%")
    candidates_df['Pct_Diff_from_Previous_Seasons'] = candidates_df['Pct_Diff_from_Previous_Seasons'].apply(lambda x: f"{x:.0f}%")

    # filter players with games less than 4 
    candidates_df = candidates_df[(candidates_df['Current_Season_Games_Played'] > 4)]# & (candidates_df['Current_Season_Avg'] >= 230)]
    candidates_df.sort_values('Current_Season_Avg', ascending=False, inplace=True)
    candidates_df = candidates_df.head(185)
    candidates_df.sort_values('Normalized_Score', ascending=False, inplace=True)

    # Filter to find buy low and sell high candidates
    buy_low_candidates_df = candidates_df[candidates_df['Category'] == 'Buy Low']
    sell_high_candidates_df = candidates_df[candidates_df['Category'] == 'Sell High']

    # Output the DataFrames
    print(f"All Players Data for {weights}:")
    # candidates_df.to_csv('candidates.csv', index=False)
    print("\nBuy Low Candidates:")
    print(buy_low_candidates_df[["Player_Name", "Normalized_Score"]])
    print("\nSell High Candidates:")
    print(sell_high_candidates_df[["Player_Name", "Normalized_Score"]])

    # # Histograms for Total Scores for Each Season
    # plt.figure(figsize=(10, 3))
    # plt.subplot(1, 2, 1)
    # candidates_df['Normalized_Score'].hist(bins=30)
    # plt.title('Normalized_Score')
    # plt.xlabel('Total Score')
    # plt.ylabel('Frequency')

    # # Histograms for Total Scores for Each Season
    # plt.subplot(1, 2, 2)
    # candidates_df['Combined_Score'].hist(bins=30)
    # plt.title('Combined_Score')
    # plt.xlabel('Total Score')
    # plt.ylabel('Frequency')
    # plt.show()

    # # Histograms for Total Scores for Each Season
    # plt.figure(figsize=(10, 8))
    # plt.subplot(2, 2, 1)
    # components_df['Normalized_Z_Score_Diff'].hist(bins=30)
    # plt.title('Normalized_Z_Score_Diff')
    # plt.xlabel('Total Score')
    # plt.ylabel('Frequency')

    # # Histograms for Total Scores for Each Season
    # plt.subplot(2, 2, 2)
    # components_df['Normalized_Deviation_Percentage'].hist(bins=30)
    # plt.title('Normalized_Deviation_Percentage')
    # plt.xlabel('Total Score')
    # plt.ylabel('Frequency')

    # # Histograms for Total Scores for Each Season
    # plt.subplot(2, 2, 3)
    # components_df['Normalized_Current_Method_Score'].hist(bins=30)
    # plt.title('Normalized_Weighted_Sum')
    # plt.xlabel('Total Score')
    # plt.ylabel('Frequency')

    # plt.subplot(2, 2, 4)
    # components_df['Normalized_Trend_Ratio'].hist(bins=30)
    # plt.title('Normalized_Trend_Ratio')
    # plt.xlabel('Total Score')
    # plt.ylabel('Frequency')
    # plt.show()
    
    if weights == weights_myleague:
        return candidates_df
    elif weights == weights_ESPN:
        candidates_df_posting = candidates_df[['Player_Name', 'Current_Season_Avg', 'Pct_Diff_from_Previous_Seasons', 'Pct_Diff_from_Current_Season', 'Normalized_Score']]
        candidates_df_posting.to_csv('candidates_posting.csv', index=False)

weights_list = [weights_myleague, weights_ESPN]
for weights in weights_list:
    if weights == weights_myleague:
        candidates_df = get_final_df(weights)
    elif weights == weights_ESPN:
        get_final_df(weights)

all_games_df, summary_df, z_scores_df, top_players_df = get_game_info_df(weights_myleague)

class Team:
    def __init__(self, team_name, player_names, all_games_df):
        self.team_name = team_name
        self.all_games_df = all_games_df
        self.roster = player_names
        self.injured_list = []
        self.tot_roster = self.roster + self.injured_list
        self.max_roster_size = 13
        self.max_injured_size = 2
        self.no_trade_list = []

    def _get_recent_avg(self, player_id, days):
        if days == 'season':
            recent_games = self.all_games_df[(self.all_games_df['Player_ID'] == player_id) & 
                                             (self.all_games_df['SEASON_ID'] == CURRENT_SEASON)]
        else:
            recent_games = self.all_games_df[(self.all_games_df['Player_ID'] == player_id) & 
                                             (self.all_games_df['GAME_DATE'] >= datetime.today() - timedelta(days=days))]
        return recent_games['total_score'].mean().round(2) if not recent_games.empty else 0

    def _get_player_data(self, player_name):
        return self.all_games_df[(self.all_games_df['PLAYER_NAME'] == player_name) & (self.all_games_df['SEASON_ID'] == CURRENT_SEASON)]
    
    def update_no_trade_list(self, player_names):
        """Update the list of players that should not be considered in trade recommendations."""
        self.no_trade_list = player_names

    def get_player_avg_score(self, player_name, days=None):
        player_data = self._get_player_data(player_name)
        if player_data.empty:
            print(f"No data found for player {player_name}")
            return None
        # Implement time frame filtering logic here if needed
        return player_data['total_score'].mean().round(2)

    def add_player(self, player_name):
        if len(self.roster) < self.max_roster_size:
            self.roster.append(player_name)
        else:
            print(f"Roster is full. Remove a player before adding a new one.")

    def remove_player(self, player_name):
        if player_name in self.roster:
            self.roster.remove(player_name)
        else:
            print(f"Player {player_name} not found in the roster.")

    def add_injured_player(self, player_name):
        if len(self.injured_list) < self.max_injured_size:
            self.injured_list.append(player_name)
            self.tot_roster = self.roster + self.injured_list
        else:
            print(f"Injured list is full. Remove a player before adding a new one.")

    def remove_injured_player(self, player_name):
        if player_name in self.injured_list:
            self.injured_list.remove(player_name)
            self.tot_roster.remove(player_name)
        else:
            print(f"Player {player_name} not found in the injured list.")

    def get_team_avg_score(self, days=None):
        scores = [self.get_player_avg_score(player, days) for player in self.roster]
        return (sum(filter(None, scores)) / len(scores)).round(2) if scores else 0
    
    def get_team_avg_score_inc_injury(self, days=None):
        scores = [self.get_player_avg_score(player, days) for player in self.tot_roster]
        return (sum(filter(None, scores)) / len(scores)).round(2) if scores else 0
    
    def view_roster(self):
        print(f"Roster for {self.team_name}:")
        for player in self.roster:
            print(player)
        print()  # Print a newline for better readability

    def view_injury_list(self):
        print(f"Injury list for {self.team_name}:")
        for player in self.injured_list:
            print(player)
        print()  # Print a newline for better readability

    def get_trade_candidates(self, candidates_df):
        # Filter the candidates DataFrame for players in this team's roster
        team_candidates = candidates_df[candidates_df['Player_Name'].isin(self.tot_roster)]
        return team_candidates

    def find_trade_matches(self, other_team, candidates_df, trade_range, trade_threshold, min_avg_score, trade_size=1):
        # my_candidates = self.get_trade_candidates(candidates_df)
        # my_sell_highs = my_candidates[my_candidates['Normalized_Score'] > 0]
        my_sell_highs = self.get_trade_candidates(candidates_df)
        my_sell_highs = my_sell_highs[~my_sell_highs['Player_Name'].isin(self.no_trade_list)]
        
        # Filter out players from the no_trade_list
        other_candidates = other_team.get_trade_candidates(candidates_df)
        other_candidates = other_candidates[~other_candidates['Player_Name'].isin(other_team.no_trade_list)]
        
        # Calculate the total number of combinations for the progress bar
        total_combinations = sum(1 for _ in itertools.combinations(my_sell_highs.iterrows(), trade_size)) * sum(1 for _ in itertools.combinations(other_candidates.iterrows(), trade_size-1))

        # Create a tqdm progress bar
        progress_bar = tqdm(total=total_combinations, desc=f'Trading with {other_team.team_name}')

        possible_trades = []
        for my_combination in itertools.combinations(my_sell_highs.iterrows(), trade_size):
            for their_combination in itertools.combinations(other_candidates.iterrows(), trade_size): #change trade_size-1 to trade_size if regarding 3v3 or 2v2 trades
                buy_low_count = sum(player[1]['Normalized_Score'] < 0 for player in their_combination)
                low_avg_score_count = sum(player[1]['Current_Season_Avg'] < min_avg_score for player in their_combination)
                # print(f'{other_team}, buy_low_count {buy_low_count}, low_avg_score_count {low_avg_score_count}') # if low avg score count is 0 then skip to next team needed

                avg_scores = {}
                favorable_criteria_count = 0  # Count of favorable criteria
                for days in [7, 14, 30, 'season']:
                    my_avg = round(sum(self._get_recent_avg(player[1]['Player_ID'], days) for player in my_combination) / trade_size, 2)
                    their_avg = round((sum(other_team._get_recent_avg(player[1]['Player_ID'], days) for player in their_combination) + min_avg_score) / (trade_size), 2) #change trade_size-1 to trade_size if regarding 3v3 or 2v2 trades
                    avg_scores[days] = (my_avg, their_avg)

                    # Check if the criteria is met for each time frame
                    if (my_avg/their_avg <= (1 + (trade_range+trade_threshold)/100) and my_avg/their_avg >= (1 - (trade_range-trade_threshold)/100)) :# or (my_avg - their_avg) <= trade_threshold:
                        favorable_criteria_count += 1
                # print(avg_scores)

                if (favorable_criteria_count >= 2 and
                    ((trade_size == 2 and buy_low_count >= 1) or (trade_size > 2 and buy_low_count >= 2)) and
                    low_avg_score_count >= 0): #change low_avg_score_count to 1 if regarding 2v2 or 3v3 trades
                    my_players = [player[1]['Player_Name'] for player in my_combination]
                    their_players = [player[1]['Player_Name'] for player in their_combination]
                    possible_trades.append((my_players, their_players, avg_scores))
                progress_bar.update(1)

        progress_bar.close()

        return possible_trades

# Helper function to find trades between two teams
def find_fair_trades(my_team, other_team, candidates_df, trade_range, trade_threshold, min_avg_score, trade_size=1):
    return my_team.find_trade_matches(other_team, candidates_df, trade_range, trade_threshold, min_avg_score, trade_size)

# Example usage
team_name1 = '원준'
team_name2 = '루이'
team_name3 = '장혁'
team_name4 = 'yhp'
team_name5 = '로나'
team_name6 = '원기'
team_name7 = '젶'
team_name8 = '종건'
team_name9 = '데릭'
team_name10 = '귤파'

player_names1 = ['Jamal Murray', 'Jalen Green', 'Luka Doncic', 'Mikal Bridges', 'LeBron James', 'Jabari Smith Jr.', 'Mark Williams', 'Jarrett Allen', 'Kristaps Porzingis', 'Cade Cunningham', 'Dereck Lively II', 'Kelly Oubre Jr.']
player_names2 = ["De'Aaron Fox", 'Tyrese Maxey', 'Jaylen Brown', 'Michael Porter Jr.', 'Paolo Banchero', 'Giannis Antetokounmpo', 'Brook Lopez', 'Jonas Valanciunas', "D'Angelo Russell", 'Devin Vassell', 'Donovan Mitchell', 'Jordan Poole', 'Marcus Smart']
player_names3 = ['Dennis Schroder', 'James Harden', 'Kevin Durant', 'Pascal Siakam', 'Zion Williamson', 'Domantas Sabonis', 'Clint Capela', 'Jerami Grant', 'Jusuf Nurkic', 'Julius Randle', 'Tyler Herro']
player_names4 = ['Trae Young', 'Kyrie Irving', 'Zach LaVine', 'Jalen Johnson', 'Kyle Kuzma', 'Karl-Anthony Towns', 'Bam Adebayo', 'Evan Mobley', 'Darius Garland', 'Khris Middleton', 'Ausar Thompson', 'Russell Westbrook', 'Cameron Johnson']
player_names5 = ['Fred VanVleet', 'LaMelo Ball', 'Shai Gilgeous-Alexander', 'Franz Wagner',  'Keegan Murray', 'Chet Holmgren', 'Miles Bridges', 'Jaren Jackson Jr.', 'Josh Giddey',  'Derrick White', 'Shaedon Sharpe', 'Klay Thompson', 'Jalen Duren']
player_names6 = ['Jordan Clarkson', 'Brandon Ingram', 'Devin Booker', 'Gordon Hayward', 'Bobby Portis', 'Jakob Poeltl', 'Ivica Zubac', 'Jayson Tatum', 'Aaron Gordon', 'Jalen Brunson', 'Deandre Ayton', 'Lauri Markkanen', 'Anfernee Simons']
player_names7 = ['Dejounte Murray', 'Malcolm Brogdon', 'Bogdan Bogdanovic', 'Nikola Jokic', 'Joel Embiid', 'Tyus Jones', 'Nic Claxton', 'Spencer Dinwiddie', 'OG Anunoby', 'Cole Anthony', 'CJ McCollum']
player_names8 = ['Ja Morant', 'Walker Kessler', 'Tyrese Haliburton', 'Tobias Harris', 'DeMar DeRozan', 'Scottie Barnes', 'Herbert Jones',  'Moritz Wagner', 'Alperen Sengun', 'Victor Wembanyama', 'Zach Collins', 'Bojan Bogdanovic']
player_names9 = ['Stephen Curry', 'Chris Paul', 'Keldon Johnson', 'Anthony Edwards', 'Nikola Vucevic', 'Collin Sexton', 'Jimmy Butler', 'Paul George', 'RJ Barrett', 'Rudy Gobert', 'Coby White', 'Cam Thomas']
player_names10 = ['Damian Lillard', 'Jrue Holiday', 'Desmond Bane', 'Anthony Davis', 'Austin Reaves', 'Myles Turner', 'Daniel Gafford', 'Kawhi Leonard', 'Jalen Williams', 'Terry Rozier', 'John Collins', 'Jamie Jaquez Jr.']

# # player_names1 = ['Kevin Durant']
# player_names4 = ['Terry Rozier', 'Coby White']

team1 = Team(team_name1, player_names1, all_games_df)
# team1.update_no_trade_list(['Kristaps Porzingis'])
team1.update_no_trade_list(['Mark Williams'])
print(f'{team_name1} average score: {team1.get_team_avg_score()}, including injury: {team1.get_team_avg_score_inc_injury()}')
# team1.view_roster()
# team1.view_injury_list()

team2 = Team(team_name2, player_names2, all_games_df)
# team2.add_injured_player('Wendell Carter Jr.')
team2.update_no_trade_list(['Donovan Mitchell'])
print(f'{team_name2} average score: {team2.get_team_avg_score()}, including injury: {team2.get_team_avg_score_inc_injury()}')

team3 = Team(team_name3, player_names3, all_games_df)
# team3.add_injured_player('Tyler Herro')
team3.add_injured_player('Ben Simmons')
team3.update_no_trade_list(['Zion Williamson', 'Ben Simmons'])
print(f'{team_name3} average score: {team3.get_team_avg_score()}, including injury: {team3.get_team_avg_score_inc_injury()}')

team4 = Team(team_name4, player_names4, all_games_df)
team4.add_injured_player('Zach LaVine')
# team4.add_injured_player('Jalen Johnson')
# team4.update_no_trade_list(['Jalen Johnson', 'Zach LaVine'])
print(f'{team_name4} average score: {team4.get_team_avg_score()}, including injury: {team4.get_team_avg_score_inc_injury()}')

team5 = Team(team_name5, player_names5, all_games_df)
# team5.add_injured_player('Marcus Smart')
# team5.add_injured_player('Jalen Duren')
team5.update_no_trade_list(['LaMelo Ball', 'Shaedon Sharpe'])
print(f'{team_name5} average score: {team5.get_team_avg_score()}, including injury: {team5.get_team_avg_score_inc_injury()}')

team6 = Team(team_name6, player_names6, all_games_df)
# team6.add_injured_player('Anfernee Simons')
team6.update_no_trade_list(['Aaron Gordon'])
print(f'{team_name6} average score: {team6.get_team_avg_score()}, including injury: {team6.get_team_avg_score_inc_injury()}')

team7 = Team(team_name7, player_names7, all_games_df)
team7.update_no_trade_list(['Tre Jones'])
print(f'{team_name7} average score: {team7.get_team_avg_score()}, including injury: {team7.get_team_avg_score_inc_injury()}')

team8 = Team(team_name8, player_names8, all_games_df)
team8.add_injured_player('Markelle Fultz')
team8.update_no_trade_list(['Markelle Fultz'])
print(f'{team_name8} average score: {team8.get_team_avg_score()}, including injury: {team8.get_team_avg_score_inc_injury()}')

team9 = Team(team_name9, player_names9, all_games_df)
team9.add_injured_player('Bradley Beal')
team9.update_no_trade_list(['Bradley Beal', 'Jimmy Butler'])
print(f'{team_name9} average score: {team9.get_team_avg_score()}, including injury: {team9.get_team_avg_score_inc_injury()}')

team10 = Team(team_name10, player_names10, all_games_df)
team10.update_no_trade_list(['Kawhi Leonard'])
print(f'{team_name10} average score: {team10.get_team_avg_score()}, including injury: {team10.get_team_avg_score_inc_injury()}')

teams = [team10, team9, team8, team7, team6, team5, team4, team3, team2, team1]
# teams = [team4, team1]
trade_range = 3 #means 3% difference(-3% to 3%)
trade_threshold = 3 #means 1% from trade range (-4% to 2%)
my_team = team4
# min_avg_score = 250
min_avg_score = 0
# min_avg_score = 21.9

# print(f'finding trade range of {-1*trade_range - trade_threshold}% to {trade_range - trade_threshold}%')
# for other_team in tqdm(teams, desc='Finding trades'):
#     if other_team != my_team:
#         for trade_size in range(2, 4):
#             fair_trades = find_fair_trades(my_team, other_team, candidates_df, trade_range=trade_range, trade_threshold=trade_threshold, min_avg_score=min_avg_score, trade_size=trade_size)
#             for trade in fair_trades:
#                 my_players, their_players, avg_scores = trade
#                 print(f"Trade suggestion:\n{my_players} ({my_team.team_name}) for {their_players} ({other_team.team_name})")
#                 for days, scores in avg_scores.items():
#                     my_avg, their_avg = scores
#                     diff = round(my_avg - their_avg, 2) * -1
#                     pct_diff = round((1 - my_avg / their_avg) * 100, 2)
#                     if days == 'season':
#                         print(f"Season Avg Score: {my_avg} vs {their_avg}, Diff: {diff}, % Diff: {pct_diff}%")
#                     else:
#                         print(f"{days}-day Avg Score: {my_avg} vs {their_avg}, Diff: {diff}, % Diff: {pct_diff}%")
#                 print("\n")  # Print a newline for better readability

# Open a file to log the trade suggestions
with open('trade_suggestions.txt', 'w', encoding='utf-8') as file:
    file.write(f'Finding trade range of {-1*trade_range - trade_threshold}% to {trade_range - trade_threshold}%\n')
    for other_team in tqdm(teams, desc='Finding trades'):
        if other_team != my_team:
            for trade_size in range(2, 4):
                fair_trades = find_fair_trades(my_team, other_team, candidates_df, trade_range=trade_range, trade_threshold=trade_threshold, min_avg_score=min_avg_score, trade_size=trade_size)
                for trade in fair_trades:
                    my_players, their_players, avg_scores = trade
                    trade_suggestion_text = f"Trade suggestion:\n{my_players} ({my_team.team_name}) for {their_players} ({other_team.team_name})\n"
                    for days, scores in avg_scores.items():
                        my_avg, their_avg = scores
                        diff = round(my_avg - their_avg, 2) * -1
                        pct_diff = round((1 - my_avg / their_avg) * 100, 2)
                        if days == 'season':
                            trade_suggestion_text += f"Season Avg Score: {my_avg} vs {their_avg}, Diff: {diff}, % Diff: {pct_diff}%\n"
                        else:
                            trade_suggestion_text += f"{days}-day Avg Score: {my_avg} vs {their_avg}, Diff: {diff}, % Diff: {pct_diff}%\n"
                    file.write(trade_suggestion_text + "\n")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("games.csv")
#print(df.head())

#print(df.shape)          # Rows and columns
#print(df.columns)        # Column names
#print(df.info())         # Data types + nulls
#print(df.describe())     # Summary stats for numeric columns

df['created_at'] = pd.to_datetime(df['created_at'], unit='ms')
df['last_move_at'] = pd.to_datetime(df['last_move_at'], unit='ms')
df['game_duration_sec'] = (df['last_move_at'] - df['created_at']).dt.total_seconds()
#print(df['game_duration_sec' ])

# print((df['game_duration_sec'] == 0).sum()) Holy crap 8548 out of our 20058 games lasted 0 seconds, I'm guessing these are because of aborting games/disconnects (they're online chess games)

df_clean = df[df['game_duration_sec'] > 0].copy() # Filtering out those instant losers

#Below, check victory reasoning / winners
#print(df_clean['victory_status'].value_counts()) Resign: 6450, Mate: 3462, OutOfTime: 1035, Draw: 564
#print(df_clean['winner'].value_counts()) White: 5753, Black: 5169, Draw: 588

#Add some new columns for future analysis ease 
df_clean['rating_diff'] = df_clean['white_rating'] - df_clean['black_rating']
df_clean['is_white_win'] = df_clean['winner'] == 'white'
df_clean['is_black_win'] = df_clean['winner'] == 'black'
df_clean['is_draw'] = df_clean['winner'] == 'draw'

# print("White win %:", df_clean['is_white_win'].mean() * 100)
# print("Black win %:", df_clean['is_black_win'].mean() * 100)
# print("Draw %:", df_clean['is_draw'].mean() * 100)

# print("Average rating difference:", df_clean['rating_diff'].mean())
# print("Average game duration (minutes):", df_clean['game_duration_sec'].mean() / 60)

# Below is a histogram to view the rating variance, we can see for both black/white players they cluster around the rating 1400-1600, both are almost perfectly normal, slight bias towards white
# being stronger, which is completely random as there is no actual logic to either color being a higher rating
# plt.figure(figsize=(10, 5))
# sns.histplot(df_clean['white_rating'], color='blue', label='White Rating', kde=True, bins=30)
# sns.histplot(df_clean['black_rating'], color='red', label='Black Rating', kde=True, bins=30)
# plt.title("Player Rating Distributions")
# plt.xlabel("Rating")
# plt.ylabel("Count")
# plt.legend()
# plt.show()

# Below is another histogram, telling us about the difference in rating within games, it being near perfectly normal tells most are fairly balanced, some games being >1000 elo difference
# (ex: me vs Grant), could lead to outliers (fast resigns/predictable wins)
# plt.figure(figsize=(8, 4))
# sns.histplot(df_clean['rating_diff'], bins=40, kde=True)
# plt.axvline(0, color='black', linestyle='--', label='Equal Ratings')
# plt.title("Rating Difference (White - Black)")
# plt.xlabel("Rating Difference")
# plt.legend()
# plt.show()

# Below boxplot is kind of interesting, we can see rating is not absolute, somehow even when higher rated, frequently higher rated players can lose to lower rated (even when difference is large)
# overall this reveals upsets are common.
# sns.boxplot(x='winner', y='rating_diff', data=df_clean)
# plt.title("Rating Difference by Winner")
# plt.axhline(0, linestyle='--', color='gray')
# plt.ylabel("White Rating - Black Rating")
# plt.show()

# Check chess openings by count, we want to see what openings lead to wins more often
# print(df_clean['opening_name'].value_counts().head(10))

opening_outcomes = df_clean.groupby(['opening_name', 'winner']).size().unstack(fill_value=0)
opening_outcomes['total_games'] = opening_outcomes.sum(axis=1)

# Add win rate columns
opening_outcomes['white_win_rate'] = (opening_outcomes['white'] / opening_outcomes['total_games']) * 100
opening_outcomes['black_win_rate'] = (opening_outcomes['black'] / opening_outcomes['total_games']) * 100
opening_outcomes['draw_rate'] = (opening_outcomes['draw'] / opening_outcomes['total_games']) * 100

top_openings = opening_outcomes.sort_values(by='total_games', ascending=False).head(10)
print(top_openings[['white_win_rate', 'black_win_rate', 'draw_rate']])

top_openings[['white_win_rate', 'black_win_rate', 'draw_rate']].plot(
    kind='bar',
    stacked=True,
    figsize=(12, 6),
    colormap='coolwarm'
)

plt.title("Win Rate by Opening (Top 10 Most Played)")
plt.ylabel("Percentage")
plt.xlabel("Opening Name")
plt.xticks(rotation=45, ha='right')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()


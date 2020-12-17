import pandas as pd
import numpy as np

# Get the last friendly action time of each friendship
def get_last_friendly_action(source):
    # Load all the seasons and remember their friendship id
    seasons = pd.json_normalize(source, record_path=['seasons'], meta=['idx'])
    # Select only the seasons that correspond to a friendly action (support)
    seasons_with_support = seasons[(seasons['interaction.victim'] == 'support') | (seasons['interaction.betrayer'] == 'support')]
    # Get the last friendly action for each friendship
    last_friendly_action = seasons_with_support.groupby('idx', as_index=False).last()
    last_friendly_action = last_friendly_action[['idx', 'season']]
    last_friendly_action = last_friendly_action.rename(columns={'season': 'last_season'})

    return last_friendly_action

# Get the season feature of a player type from the source file information
def load_season_features(source, last_friendly_action, player_type):
    # Load messages
    messages = pd.json_normalize(source,
                                 # Get the informations at the message level for the targeted type of player
                                 record_path = ['seasons', 'messages', player_type],
                                 # Take the information of the betrayal, a friendship id and the season to allow merge
                                 # also take the interaction information which correspond to support
                                 meta = ['idx', 'betrayal', ['seasons', 'season'], ['seasons', 'interaction']])
    # Add the last seasons for each messages based on their friendship id
    messages = pd.merge(messages, last_friendly_action, on='idx')
    # Remove all messages that occurs after the last friendly action
    messages = messages[messages['seasons.season'] <= messages['last_season']]
    # Rename the message features
    messages = messages.rename(columns={'seasons.season' : 'season', 'lexicon_words.disc_temporal_rest' : 'temporal', 'lexicon_words.allsubj' : 'subjectivity', 'lexicon_words.disc_expansion' : 'expansion', 'lexicon_words.disc_contingency' : 'contingency', 'lexicon_words.premise' : 'premise', 'lexicon_words.disc_temporal_future' : 'planning', 'lexicon_words.disc_comparison' : 'comparison', 'lexicon_words.claim' : 'claim'})
    
    # Get the support information from the interaction of the player type
    messages['support'] = messages['seasons.interaction'].apply(lambda x: 1 if x[player_type] == 'support' else 0)
    # Remove useless columns (interactions converted in support and frequent_words not used)
    messages = messages.drop(columns=['frequent_words', 'seasons.interaction'])
    
    # Convert all arrays into numerical feature (the number of words)
    messages['temporal'] = array_to_count(messages['temporal'])
    messages['subjectivity'] = array_to_count(messages['subjectivity'])
    messages['expansion'] = array_to_count(messages['expansion'])
    messages['contingency'] = array_to_count(messages['contingency'])
    messages['premise'] = array_to_count(messages['premise'])
    messages['planning'] = array_to_count(messages['planning'])
    messages['comparison'] = array_to_count(messages['comparison'])
    messages['claim'] = array_to_count(messages['claim'])
    
    # Compute the features of the season which corresponds to the mean of message features
    seasons = messages.groupby(['idx', 'season', 'betrayal', 'support'], as_index=False).mean()
    
    # Add the number of message in each season as a feature
    messages_count = messages.groupby(['idx', 'season'], as_index=False)['support'].count()
    messages_count = messages_count.rename(columns={'support' : 'n_messages'})
    seasons = seasons.merge(messages_count, on=['idx', 'season'])

    # Add the player type column to allow a concat with the other player type later
    seasons['player_type'] = player_type

    return seasons

# Convert an array into an int by taking the number of element in it
# If the array is not defined, set it to 0
def array_to_count(col):
    col = col.apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    return col

# Get the friendship features from the season features (mean and variance)
def get_friendship_features(season_features):
    # The following cols will not be aggregated
    static_cols = ['idx', 'player_type', 'betrayal']

    # Compute the mean features and add a prefix to differentiate them from the variance
    mean_features = season_features.groupby(static_cols, as_index=False).mean().drop(columns=['season', 'last_season'])
    mean_features.columns = ['mean_' + str(col) if col not in static_cols else str(col) for col in mean_features.columns]
    
    # Compute the variance features and add a prefix to differentiate them from the mean
    var_features = season_features.groupby(static_cols, as_index=False).var(ddof=0).drop(columns=['season', 'last_season'])
    var_features.columns = ['var_' + str(col) if col not in static_cols else str(col) for col in var_features.columns]
        
    # Merge the mean and variance features to have a single list of features
    result = pd.merge(mean_features, var_features, on = static_cols)

    # Remove the victims or betrayers that do not have a friend in the list (absence of message in all seasons)
    two_occu = result['idx'].value_counts() == 2 # This can be represented by an occurence of 2 of the friendship id
    valid_indices = result['idx'].value_counts()[two_occu].index # Build the list of the valid friendship ids
    result = result[result['idx'].isin(valid_indices)]
    
    return result
    
# Put the features of the two player types on the same row
def merge_player_features(data, on):
    victims = data[data['player_type'] == 'victim'].drop(columns=['player_type'])
    betrayers = data[data['player_type'] == 'betrayer'].drop(columns=['player_type'])
    # Add a suffix to differentiate between victim and betrayer
    result = pd.merge(victims, betrayers, on=on, suffixes=['_victim', '_betrayer'])

    return result
    

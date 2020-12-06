import pandas as pd
import numpy as np

# Get the mean of politeness in each season for a certain type of player (victim or betrayer)
def load_season_mean_politeness(source, last_friendly_action, player_type):
    messages = pd.json_normalize(source,
                                 # Get the informations at the message level for the targeted type of player
                                 record_path = ['seasons', 'messages', player_type],
                                 # Take the information of the betrayal and a friendship id to allow merge
                                 meta = ['idx', 'betrayal', ['seasons', 'season']])
    # Add the last seasons for each messages based on their friendship id
    messages = pd.merge(messages, last_friendly_action, on='idx')
    # Remove all messages that occurs after the last friendly action
    messages = messages[messages['seasons.season'] <= messages['last_season']]
    # Take only the relevant columns
    messages = messages[['idx', 'seasons.season', 'betrayal', 'politeness']]
    messages = messages.rename(columns={'seasons.season': 'season'})
    # Compute the mean politeness for each season
    messages = messages.groupby(['idx', 'season', 'betrayal'], as_index=False).mean()
    # Add the player type column to allow a concat with the other player type later
    messages['player_type'] = player_type
    
    return messages

def load_season_features(source, last_friendly_action, player_type):
    messages = pd.json_normalize(source,
                                 # Get the informations at the message level for the targeted type of player
                                 record_path = ['seasons', 'messages', player_type],
                                 # Take the information of the betrayal and a friendship id to allow merge
                                 meta = ['idx', 'betrayal', ['seasons', 'season'], ['seasons', 'interaction']])
    # Add the last seasons for each messages based on their friendship id
    messages = pd.merge(messages, last_friendly_action, on='idx')
    # Remove all messages that occurs after the last friendly action
    messages = messages[messages['seasons.season'] <= messages['last_season']]
    messages = messages.rename(columns={'seasons.season' : 'season', 'lexicon_words.disc_temporal_rest' : 'temporal', 'lexicon_words.allsubj' : 'subjectivity', 'lexicon_words.disc_expansion' : 'expansion', 'lexicon_words.disc_contingency' : 'contingency', 'lexicon_words.premise' : 'premise', 'lexicon_words.disc_temporal_future' : 'planning', 'lexicon_words.disc_comparison' : 'comparison', 'lexicon_words.claim' : 'claim'})
    messages['support'] = messages['seasons.interaction'].apply(lambda x: 1 if x[player_type] == 'support' else 0)
    messages = messages.drop(columns=['frequent_words', 'seasons.interaction'])
    
    messages['temporal'] = array_to_count(messages['temporal'])
    messages['subjectivity'] = array_to_count(messages['subjectivity'])
    messages['expansion'] = array_to_count(messages['expansion'])
    messages['contingency'] = array_to_count(messages['contingency'])
    messages['premise'] = array_to_count(messages['premise'])
    messages['planning'] = array_to_count(messages['planning'])
    messages['comparison'] = array_to_count(messages['comparison'])
    messages['claim'] = array_to_count(messages['claim'])
    
    messages_count = messages.groupby(['idx', 'season'], as_index=False)['support'].count()
    messages_count = messages_count.rename(columns={'support' : 'n_messages'})
    messages = messages.groupby(['idx', 'season', 'betrayal', 'support'], as_index=False).mean()
    
    messages = messages.merge(messages_count, on=['idx', 'season'])

    # Add the player type column to allow a concat with the other player type later
    messages['player_type'] = player_type

    return messages

def array_to_count(col):
    #col.loc[col.isnull()] = col.loc[col.isnull()].apply(lambda x: [])
    col = col.apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    return col
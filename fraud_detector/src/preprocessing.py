# Import standard libraries
import json
import os
import pandas as pd
import numpy as np
import logging

# Import extra modules
from geopy.distance import great_circle
from sklearn.impute import SimpleImputer 

logger = logging.getLogger(__name__)
RANDOM_STATE = 42

# Lookup tables derived from the competition train.csv by tools/build_encoders.py
ENCODERS_PATH = os.getenv('ENCODERS_PATH', './models/encoders.json')

def add_time_features(df):
    logger.debug('Adding time features...')
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    dt = df['transaction_time'].dt
    df['hour'] = dt.hour
    df['year'] = dt.year
    df['month'] = dt.month
    df['day_of_month'] = dt.day
    df['day_of_week'] = dt.dayofweek
    df.drop(columns='transaction_time', inplace=True)
    return df


def cat_encode(encoders, input_df, col):
    
    logger.debug('Encoding category: %s', col)
    new_col = col + '_cat'
    mapping = encoders['category_maps'][col]
    
    # Apply the category mapping built from the training data
    input_df[new_col] = input_df[col].map(mapping).fillna('cat_NAN')
    input_df = input_df.drop(columns=col)
    
    return input_df


def add_distance_features(df):
    
    logger.debug('Calculating distances...')
    df['distance'] = df.apply(
        lambda x: great_circle(
            (x['lat'], x['lon']), 
            (x['merchant_lat'], x['merchant_lon'])
        ).km,
        axis=1
    )
    return df.drop(columns=['lat', 'lon', 'merchant_lat', 'merchant_lon'])


# Load the precomputed encoding tables at docker container start
def load_encoders():

    logger.info('Loading encoders...')

    # Import the lookup tables built from the training data
    with open(ENCODERS_PATH, encoding='utf-8') as fh:
        encoders = json.load(fh)

    logger.info('Encoders imported. Source rows: %s', encoders['meta']['source_rows'])

    return encoders


# Main preprocessing function
def run_preproc(encoders, input_df):

    # Define column types
    categorical_cols = ['gender', 'merch', 'cat_id', 'one_city', 'us_state', 'jobs']
    continuous_cols = ['amount', 'population_city']
    drop_col = ['name_1', 'name_2', 'street', 'post_code']
    input_df = input_df.drop(columns=drop_col)
    
    # Run category encoding
    for col in categorical_cols:
        input_df = cat_encode(encoders, input_df, col)

    logger.info('Categorical merging completed. Output shape: %s', input_df.shape)
    
    # Add some simple time features
    input_df = add_time_features(input_df)

    logger.info('Added time features. Output shape: %s', input_df.shape)

    categorical_cols = [x + '_cat' for x in categorical_cols]
    categorical_cols.extend(['hour', 'year', 'month', 'day_of_month', 'day_of_week'])
    
    # Run mean ecoding for categorical variables
    for col in categorical_cols:
        # Fill empty values of categorical columns with some default category
        input_df[col] = input_df[col].fillna('cat_NAN')
    
        # Look up the target mean for every category
        means_tb = encoders['mean_encodings'][col]
        input_df[f'{col}_mean_enc'] = input_df[col].astype(str).map(means_tb)

    logger.info('Categorical mean encoding completed. Output shape: %s', input_df.shape)

    # Calculate distance between a client and a merchant
    input_df = add_distance_features(input_df)
    continuous_cols.extend(['distance'])

    # Impute empty values with mean value
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(pd.DataFrame(
        [[encoders['imputer_stats'][c] for c in continuous_cols]], columns=continuous_cols
    ))

    output_df = pd.concat([
        input_df.drop(columns=continuous_cols),
        pd.DataFrame(imputer.transform(input_df.copy()[continuous_cols]), columns=continuous_cols)
    ], axis=1)

    # Add log transformation
    for col in continuous_cols:
        output_df[col + '_log'] = np.log(output_df[col] + 1)
        output_df.drop(columns=col, inplace=True)
        
    logger.info('Continuous features preprocessing completed. Output shape: %s', output_df.shape)
    
    # Return resulting dataset
    return output_df
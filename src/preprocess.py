import pandas as pd
import scipy.stats


# The main function to be used
def preprocess(target_df):

    # =====================================================    
    # Preprocess the original 7million bidding data first
    # =====================================================

    # read the bids.csv
    bids_df = pd.read_csv("./data/bids.csv")

    # The bids are NOT sorted by time. So, we will sort it by time
    bids_df = bids_df.sort_values('time').reset_index(drop=True)

    # Set the initial time to be zero
    bids_df['time'] = bids_df['time'] - bids_df['time'].min()

    # =====================================================    
    # Fetch bidding data of bidders inside the target_df
    # =====================================================

    # Create a new bids dataframe which only have bidders from the target_bids_df
    target_bids_df = bids_df[bids_df['bidder_id'].isin(target_df['bidder_id'])]

    # =====================================================    
    # Create features
    # =====================================================

    # Merging the features here
    features = extract_activity_features(target_bids_df) \
        .merge(extract_time_features(target_bids_df), on='bidder_id') \
        .merge(extract_diversity_features(target_bids_df), on='bidder_id') \
        .merge(extract_price_features(target_bids_df), on='bidder_id') \
        .merge(extract_response_features(target_bids_df), on='bidder_id') \
        .merge(extract_deviceCount_feature(target_bids_df), on='bidder_id')
    
    # =====================================================    
    # Handling the missing bidders
    # They are the bidders who never bets
    # =====================================================

    # Bidders in train_df but NOT in train_bids_df
    missing_bidders = target_df[~target_df['bidder_id'].isin(target_bids_df['bidder_id'])]
    print(f"Number of missing bidders: {len(missing_bidders)}")

    # Create a row of zeros for each missing bidder
    missing_features = pd.DataFrame(0, index=range(len(missing_bidders)), columns=features.columns)
    missing_features['bidder_id'] = missing_bidders['bidder_id'].values

    # Add to features
    features = pd.concat([features, missing_features], ignore_index=True)

    # Printing the shape of DataFrame
    print(f"Features DF shape: {features.shape}")
    print(f"Train DF shape: {target_df.shape}")

    return features



# This function extracts the activity features from bids data
def extract_activity_features(target_bids_df):

    # =====================================================    
    # This function is used to extract activity features
    # from target_bids_df
    # =====================================================

    activity_features = target_bids_df.groupby('bidder_id').agg(
        bids_count=('bid_id', 'count'),
        auction_count=('auction', 'nunique')
    ).reset_index()

    activity_features['mean_bids_per_auction'] = activity_features['bids_count'] / activity_features['auction_count']

    return activity_features



# This function extracts the time features from bids data
def extract_time_features(target_bids_df):

    # =====================================================    
    # This function is used to extract time features
    # from target_bids_df
    # =====================================================

    # Sort by bidder and time
    bids_sorted = target_bids_df.sort_values(['bidder_id', 'time'])

    # Mark first bid of each bidder
    bids_sorted['is_first_bid'] = bids_sorted.groupby('bidder_id').cumcount() == 0

    # Time difference between consecutive bids by same bidder
    bids_sorted['tdiff'] = bids_sorted.groupby('bidder_id')['time'].diff()

    # Sort back to bidder-level
    bids_sorted = bids_sorted.sort_values(['bidder_id', 'time'])

    # IP change detection and time between changes
    bids_sorted['ip_changed'] = bids_sorted.groupby('bidder_id')['ip'].transform(lambda x: x != x.shift())
    bids_sorted['tdiff_ip'] = bids_sorted['time'].diff()
    bids_sorted.loc[~bids_sorted['ip_changed'], 'tdiff_ip'] = float('nan')
    bids_sorted.loc[bids_sorted['is_first_bid'], 'tdiff_ip'] = float('nan')

    # Aggregate
    speed_features = bids_sorted.groupby('bidder_id').agg(
        tdiff_max=('tdiff', 'max'),
        tdiff_min=('tdiff', 'min'),
        tdiff_mean=('tdiff', 'mean'),
        tdiff_median=('tdiff', 'median'),
        tdiff_std=('tdiff', 'std'),
        tdiff_zeros=('tdiff', lambda x: (x == 0).sum()),
        tdiff_ip=('tdiff_ip', 'mean')
    ).reset_index()

    # Fill NaNs with 0
    for col in ['tdiff_std']:
        speed_features[col] = speed_features[col].fillna(0)

    # Fill NaNs with column medians
    for col in ['tdiff_median', 'tdiff_ip', 'tdiff_max', 'tdiff_min', 'tdiff_mean']:
        speed_features[col] = speed_features[col].fillna(speed_features[col].median())

    return speed_features



# Just a helper function to calculate the entropy
def entropy(series):

    # =====================================================    
    # This helper function is to calculate the entropy
    # of a series
    # =====================================================

    counts = series.value_counts(normalize=True)
    return scipy.stats.entropy(counts)



# This function extracts the diversity features from bids data
def extract_diversity_features(target_bids_df):

    # =====================================================    
    # This function is used to extract diversity of the
    # bids using entropy
    # =====================================================

    diversity_features = target_bids_df.groupby('bidder_id').agg(
        country_cnt=('country', 'nunique'),
        ip_entropy=('ip', entropy),
        url_entropy=('url', entropy)
    ).reset_index()

    return diversity_features



# This function extracts the price features from bids data
def extract_price_features(target_bids_df):

    # =====================================================    
    # This function is used to extract price features
    # =====================================================

    # Sort by auction and time
    bids_sorted_auction = target_bids_df.sort_values(['auction', 'time'])

    # The price level is the cumulative count of bids within each auction
    bids_sorted_auction['price_level'] = bids_sorted_auction.groupby('auction').cumcount() + 1

    # Now aggregate price stats per bidder
    price_features = bids_sorted_auction.groupby('bidder_id').agg(
        price_max=('price_level', 'max'),
        price_min=('price_level', 'min'),
        price_mean=('price_level', 'mean'),
        price_median=('price_level', 'median'),
        price_std=('price_level', 'std')
    ).reset_index()

    # Fill NaNs with 0
    for col in ['price_std']:
        price_features[col] = price_features[col].fillna(0)

    # Fill NaNs with median (if there is any NaN in these columns)
    for col in ['price_max', 'price_min', 'price_mean', 'price_median']:
        price_features[col] = price_features[col].fillna(price_features[col].median())
    
    return price_features



# This function extracts the device count feature from bids data
def extract_deviceCount_feature(target_bids_df):
    device_cnt = target_bids_df.groupby('bidder_id').agg(
    device_cnt=('device', 'nunique')
).reset_index()
    
    return device_cnt



# This function extracts the response features from bids data
def extract_response_features(target_bids_df):

    # Response time: time difference from previous bid in same AUCTION
    bids_sorted = target_bids_df.sort_values(['auction', 'time'])
    bids_sorted['response'] = bids_sorted.groupby('auction')['time'].diff()

    response_features = bids_sorted.groupby('bidder_id').agg(
        response_max=('response', 'max'),
        response_min=('response', 'min'),
        response_mean=('response', 'mean'),
        response_median=('response', 'median'),
        response_std=('response', 'std')
    ).reset_index()

    # Fill NaNs with 0
    for col in ['response_std']:
        response_features[col] = response_features[col].fillna(0)

    for col in ['response_max', 'response_min', 'response_mean', 'response_median']:
        response_features[col] = response_features[col].fillna(response_features[col].median())

    return response_features

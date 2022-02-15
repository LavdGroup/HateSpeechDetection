"""
train_dev_test_split.py

Usage:
    train_dev_test_split.py <data_path> <output_dir> [options]

Options:
    -h --help                               show this screen.
    --dev=<int>                             evaluation set size proportion [default: 0.1]
    --test=<int>                            test set size proportion [default: 0.1]
    --random_state=<int>                    random state for pandas dataframe shuffle [default: 42]

"""

from docopt import docopt
import pandas as pd

args = docopt(__doc__)
print(f"running train_dev_test.py with the following argumnets:\n{args}\n")

# comand line arguments
df_path = str(args["<data_path>"])
output_path = str(args["<output_dir>"])
random_state = int(args["--random_state"])
dev_proportion = float(args["--dev"])
test_proportion = float(args["--test"])


df_tweets = pd.read_csv(df_path)  # load the dataframe
df_tweets = df_tweets.sample(frac=1, random_state=random_state)  # shuffle the data

n_rows = len(df_tweets)
n_val = int(dev_proportion * n_rows)
n_test = int(test_proportion * n_rows)
n_train = n_rows - n_val - n_test

# split dataframe
df_train = df_tweets.iloc[ : n_train]
df_val = df_tweets.iloc[n_train : n_train + n_val]
df_test = df_tweets.iloc[n_train + n_val : ]

# store dataframes as csv
df_train.to_csv(str(output_path) + "/train.csv", index=False)
df_val.to_csv(str(output_path) + "/val.csv", index=False)
df_test.to_csv(str(output_path) + "/test.csv", index=False)
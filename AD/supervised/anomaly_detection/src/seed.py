import pandas as pd
import MLDB as ml

"""
Run this script if you need to seed the database with initial usernames
"""


# read in text file with usernames
df = pd.read_csv("users.txt", delimiter=" ", header=None)
# print(df)
# write usernames to user table
for user in df[0]:
    ml.new_user(user, "", "")

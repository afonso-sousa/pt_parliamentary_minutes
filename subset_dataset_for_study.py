# %%
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

tqdm.pandas()
nltk.download("stopwords")


df = pd.read_csv("data/out_with_text_processed.csv")
df = df[~df.isnull().any(axis=1)]
df = df.reset_index(drop=True)


# %%
df = df[["dep_id", "dep_name", "dep_parl_group", "text", "vote"]]
df.to_csv("data/parliamentary_minutes.csv", index=False)

# %%

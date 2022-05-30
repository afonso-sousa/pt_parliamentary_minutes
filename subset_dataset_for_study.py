# %%
import pandas as pd

df = pd.read_csv("data/out_with_text_processed.csv")
dataset = df[~df.isnull().any(axis=1)]

df = df[["dep_id", "dep_name", "dep_parl_group", "text", "vote"]]

# %%
df.to_csv("data/parliamentary_minutes.csv", index=False)

# %%

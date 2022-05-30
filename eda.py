# %%
import ast
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk import RegexpParser, Tree, ne_chunk, pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from wordcloud import WordCloud

# %%
df = pd.read_csv("data/out_with_text_processed.csv")
df = df[~df.isnull().any(axis=1)]

# %%
print(f"# unique initiatives: {df['ini_num'].nunique()}")  # 383
print(f"# unique deputies: {df['dep_id'].nunique()}")  # 389

# %%
####### Who votes with whom? #######
def build_who_votes_with_whom_plot(ax, field, topX, title):
    a, cnt = np.unique(df[field].values, return_counts=True)
    topX_indices = cnt.argsort()[-topX:][::-1]
    value_tuples = a[topX_indices]
    value_tuples = [tuple(ast.literal_eval(s)) for s in list(value_tuples)]
    values = [",".join(i) for i in value_tuples]
    freqs = cnt[topX_indices]
    # top_results = [r for r in values if len(r) > 1]
    if "" in values:
        values[values.index("")] = "No votes"

    ax = sns.barplot(x=values, y=freqs, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.set(title=title, xlabel="Party Groups", ylabel="Frequency")


fig, axs = plt.subplots(1, 3, figsize=(20, 5))
topX = 8
build_who_votes_with_whom_plot(axs[0], "vot_in_favour", topX, title="In Favour")
build_who_votes_with_whom_plot(axs[1], "vot_against", topX, title="Against")
build_who_votes_with_whom_plot(axs[2], "vot_abstention", topX, title="Abstention")
plt.suptitle("Who votes with whom?")
plt.tight_layout()
plt.show()

# %%
ini_df = df[["ini_num", "ini_leg", "ini_type", "leg_begin_date", "leg_end_date"]]
ini_df = ini_df.drop_duplicates()

# %%
ini_df.groupby("ini_num").count()

# %%
# There are initiatives with same number for different legislatures
unique_num_leg = (
    ini_df.groupby(["ini_num", "ini_leg"])
    .size()
    .reset_index()
    .rename(columns={0: "count"})
)
unique_num_leg[unique_num_leg["count"] > 1]

# %%
# There are initiatives with same number for different types
unique_num_type = (
    ini_df.groupby(["ini_num", "ini_type"])
    .size()
    .reset_index()
    .rename(columns={0: "count"})
)
unique_num_type[unique_num_type["count"] > 1]

# %%
# Initiative numbers are ONLY unique per type and legislature
unique_num_leg_type = (
    ini_df.groupby(["ini_num", "ini_leg", "ini_type"])
    .size()
    .reset_index()
    .rename(columns={0: "count"})
)
unique_num_leg_type[unique_num_leg_type["count"] > 1]

# %%
# array([   1, 6103, 5649, 5742, 4131, ...]
print(df.loc[df["ini_leg"] == "X"]["doc_first_page"].unique())
# array([1])
print(df.loc[df["ini_leg"] == "XI"]["doc_first_page"].unique())
# array([1])
print(df.loc[df["ini_leg"] == "XII"]["doc_first_page"].unique())

# %%
# Entries with middle name
df.loc[df["dep_name"].str.split().str.len() == 3]

# %%
# array(['PS', 'CDS-PP', 'PCP', 'BE', 'PSD', 'PEV', 'Ninsc'])
df["dep_parl_group"].unique()

# %%
"""
array(['PS', 'PSD', 'BE', 'CDS-PP', 'PEV', 'PCP', '1-PSD',
       'Luísa Mesquita (Ninsc)', 'José Paulo Areia De Carvalho (Ninsc)',
       '2-PS', '1-PS', 'Isabel Alves Moreira (PS)',
       'Pedro Nuno Santos (PS)', 'Maria Antónia De Almeida Santos (PS)',
       'Pedro Delgado Alves (PS)', 'João Galamba (PS)',
       'Duarte Cordeiro (PS)', 'Elza Pais (PS)', 'Manuel Mota (PS)',
       'Jacinto Serrão (PS)']
"""

df["vot_in_favour"].apply(ast.literal_eval).explode().unique()

# %%
# 577 - rows with different publication and initiative sessions
df[df["pub_session"] != df["ini_session"]][["pub_session", "ini_session"]]

# %%
# Proposals vs Voting Behaviour
vot_in_favour_flattened = df["vot_in_favour"].apply(ast.literal_eval).explode()
top_votes_in_favour = vot_in_favour_flattened.value_counts().head(6)
vot_against_flattened = df["vot_against"].apply(ast.literal_eval).explode()
top_votes_against = vot_against_flattened.value_counts().head(6)
vot_abstention_flattened = df["vot_abstention"].apply(ast.literal_eval).explode()
top_votes_abstention = vot_abstention_flattened.value_counts().head(6)

aux_df = pd.concat(
    [top_votes_in_favour, top_votes_against, top_votes_abstention], axis=1
)
aux_df = aux_df.reset_index().rename(columns={"index": "party"})
aux_df_melted = aux_df.melt("party", var_name="labels", value_name="# votes")

aux_df_melted = aux_df_melted.replace(
    {
        "vot_in_favour": "in favour",
        "vot_against": "against",
        "vot_abstention": "abstention",
    }
)

ax = sns.catplot(x="party", y="# votes", hue="labels", data=aux_df_melted, kind="point")
ax.set(title="Proposals vs Voting Behaviour")


# %%
ax = sns.countplot(x="dep_parl_group", data=df)
ax.set(title="Proposals per Party", xlabel=None, ylabel="frequency")


# %%
ax = sns.countplot(
    x="vote",
    data=df.replace(
        {
            "vot_in_favour": "in favour",
            "vot_against": "against",
            "vot_abstention": "abstention",
        }
    ),
)
ax.set(title="Instances per Label", xlabel=None, ylabel="frequency")

# # %%
# vot_favor_flattened = df["vot_in_favour"].apply(ast.literal_eval).explode()
# top_votos_favor = vot_favor_flattened.value_counts().head(6)
# sns.barplot(top_votos_favor.index, top_votos_favor.values)

# # %%
# vot_abs_flattened = df["vot_against"].apply(ast.literal_eval).explode()
# top_votos_abs = vot_abs_flattened.value_counts().head(6)
# sns.barplot(top_votos_abs.index, top_votos_abs.values)

# # %%
# vot_contra_flattened = df["vot_abstention"].apply(ast.literal_eval).explode()
# top_votos_contra = vot_contra_flattened.value_counts().head(6)
# sns.barplot(top_votos_contra.index, top_votos_contra.values)

# %%
# mean word count
mean_words_per_intervention = df["text"].apply(lambda x: len(str(x).split())).mean()
print(f"Mean # words per intervention: {int(mean_words_per_intervention)}")  # 621

# %%
# Check 100 most common words in dataset

most_common = Counter(
    word_tokenize(" ".join(df["text"].str.lower()), language="portuguese")
).most_common(100)

# %%
# vobab size
allwords = []
for _, speech in df["text"].items():
    tokens_list = word_tokenize(speech, language="portuguese")
    allwords += [token.lower() for token in tokens_list]

print(f"# tokens: {len(allwords)}")  # 3288672

# %%
vocabulary = sorted(set(allwords))
print(f"Vocabulary size: {len(vocabulary)}")  # 47878

# %%
# num sentences per intervention
mean_sentences_per_intervention = df["text"].apply(lambda x: len(sent_tokenize(x, language="portuguese"))).mean()
print(f"Mean # sentences per intervention: {int(mean_sentences_per_intervention)}")  # 24

# %%
# num sentences
allsentences = []
for _, speech in df["text"].items():
    sentence_list = sent_tokenize(speech, language="portuguese")
    allsentences += [token.lower() for token in sentence_list]

print(f"# sentences: {len(allsentences)}")  # 111614

# %%
##### Word Cloud #####
pt_stopwords = stopwords.words("portuguese")
wordcloud = WordCloud(
    width=800,
    height=800,
    background_color="white",
    stopwords=pt_stopwords,
    min_font_size=10,
).generate(" ".join(allwords))

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()


# %%
##### Get noun phrases #####
def get_continuous_chunks(text, chunk_func=ne_chunk):
    chunked = chunk_func(pos_tag(word_tokenize(text, language="portuguese")))
    continuous_chunk = []
    current_chunk = []

    for subtree in chunked:
        if type(subtree) == Tree:
            current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    return continuous_chunk

# Defining a grammar & Parser
NP = "NP: {(<V\w+>|<NN\w?>)+.*<NN\w?>}"
chunker = RegexpParser(NP)
df["noun_phrases"] = df["text"].apply(lambda sent: get_continuous_chunks(sent, chunker.parse))


# %%
# num sentences
num_noun_phrases = 0
for _, noun_phrases in df["noun_phrases"].items():
    num_noun_phrases += len(noun_phrases)

print(f"# noun phrases: {num_noun_phrases}")  # 389243

# %%
# Normalized dataset (built in `npl_vote_prediction.py`)
dataset = pd.read_csv("data/normalized_dataset.csv")

# %%
# mean word count
mean_words_per_intervention = dataset["text"].apply(lambda x: len(word_tokenize(x, language="portuguese"))).mean()
print(f"Mean # words per intervention: {int(mean_words_per_intervention)}")  # 621

# %%

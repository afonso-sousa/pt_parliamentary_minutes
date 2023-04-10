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
# df = df[~df.isnull().any(axis=1)]

# %%
df_with_text = df[df.text_process_label == 1]

# %%
relevant_cols = df.drop(columns=['pages', 'pdf_file_path', "doc_first_page", "text_process_label"])

# %%
ini_df = df[["ini_num", "ini_leg", "ini_type", "leg_begin_date", "leg_end_date"]]
ini_df = ini_df.drop_duplicates()
unique_num_leg_type = (
    ini_df.groupby(["ini_num", "ini_leg", "ini_type"])
    .size()
    .reset_index()
    .rename(columns={0: "count"})
)
assert len(unique_num_leg_type[unique_num_leg_type["count"] > 1]) == 0

# %%
parties = df["dep_parl_group"].unique()

# %%
# =============== GENERAL STATISTICS ===============
print(f"# entries: {len(df)}")
print(f"# entries with text: {len(df_with_text)}")
print(f"# relevant attributes: {len(relevant_cols.columns)}")
print(f"# initiatives: {len(unique_num_leg_type)}")
print(f"Parties: {parties}")


# %%
# =============== TEXT STATISTICS ===============
allwords = []
for _, speech in df_with_text["text"].items():
    tokens_list = word_tokenize(speech, language="portuguese")
    allwords += [token.lower() for token in tokens_list]

vocabulary = sorted(set(allwords))

mean_sentences_per_intervention = (
    df_with_text["text"].apply(lambda x: len(sent_tokenize(x, language="portuguese"))).mean()
)

print(f"# tokens: {len(allwords)}")  # 3790086
print(f"Vocabulary size: {len(vocabulary)}")  # 47878
print(f"Mean # words per intervention: {len(allwords) // len(df_with_text)}")  # 621
print(
    f"Mean # sentences per intervention: {int(mean_sentences_per_intervention)}"
)

# %%
# Check 100 most common words in dataset

most_common = Counter(
    word_tokenize(" ".join(df_with_text["text"].str.lower()), language="portuguese")
).most_common(100)


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
df["noun_phrases"] = df["text"].apply(
    lambda sent: get_continuous_chunks(sent, chunker.parse)
)


# %%
# num sentences
num_noun_phrases = 0
for _, noun_phrases in df["noun_phrases"].items():
    num_noun_phrases += len(noun_phrases)

print(f"# noun phrases: {num_noun_phrases}")  # 389243


# %%
# =================== PLOTS ==================
# %%
####### Who votes with whom? #######
def build_who_votes_with_whom_plot(ax, field, topX, title):
    a, cnt = np.unique(df[field].values, return_counts=True)
    topX_indices = cnt.argsort()[-topX:][::-1]
    value_tuples = a[topX_indices]
    value_tuples = [tuple(ast.literal_eval(s)) for s in list(value_tuples)]
    values = [",".join(i) for i in value_tuples]
    freqs = cnt[topX_indices]
    if "" in values:
        values[values.index("")] = "No votes"

    # colors = {"PSD,CDS-PP": "#FFA602", "PS": "#ff9cdd", "PSD,PS,CDS": "blue", "PS,PCP,BE,PEV": "#fe0100", "PSD,PS,CDS-PP,PCP,BE,PEV": "green", "PCP,BE,PEV": "#330000", "PS,PSD,CDS-PP": "pink", "PSD,PS,CDS-PP,PCP,BE,PEV": "#3500FE"}
    colors = ["#ffa602", "#ff9cdd", "#5d8cae", "#fe0100", "#00cc00", "#330000", "#ff66cc", "#3500fe"]
    ax = sns.barplot(x=values, y=freqs, ax=ax, palette=colors)
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
df_no_ninsc = df[df.dep_parl_group != "Ninsc"]
colors = {"PS": "#ff9cdd", "CDS-PP": "#3299fe", "PCP": "#fe0100", "BE": "#330000", "PSD": "#FFA602", "PEV": "#26AE09", "PAN": "#40DE12", "CH": "#6600CD", "IL": "#3500FE", "L": "#AA2B00"}
ax = sns.countplot(x="dep_parl_group", data=df_no_ninsc, palette=colors, order=df_no_ninsc.dep_parl_group.value_counts().index)
ax.set(title="Interventions per Party", xlabel=None, ylabel="frequency")


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



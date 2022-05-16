# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import ast
from nltk.tokenize import word_tokenize



# %%
df = pd.read_csv("data/out_with_text_X-XII_processed.csv")
df = df[~df.isnull().any(axis=1)]
df.drop(["pub_serie", "pub_legislatura", "pdf_file_path"], axis=1, inplace=True)

df.describe()


# %%
# array([   1, 6103, 5649, 5742, 4131, ...]
df.loc[df["ini_leg"] == 'X']["doc_first_page"].unique()
# array([1])
df.loc[df["ini_leg"] == 'XI']["doc_first_page"].unique()
# array([1])
df.loc[df["ini_leg"] == 'XII']["doc_first_page"].unique()

# %%
df["name_split"] = df["dep_nome"].str.split()
middle_name_speakers = df.loc[df["name_split"].str.len() == 3]
df.drop("name_split", axis=1, inplace=True)
middle_name_speakers

# %%
# array(['PS', 'CDS-PP', 'PCP', 'BE', 'PSD', 'PEV', 'Ninsc'])
df["dep_gp"].unique()



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

df["vot_favor"].apply(ast.literal_eval).explode().unique()

# %%
df["text"]

# %%
# 577 - rows with different publication and initiative sessions
df[df["pub_sessao"] != df["ini_sessao"]][["pub_sessao", "ini_sessao"]]

# %%
sns.set(color_codes=True)


def show_values_on_bars_v(axs):
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, y1, y2 + 1))

    def _show_on_single_plot(ax):
        for p in ax.patches:
            _x = p.get_x() - 0.01 + p.get_width() / 2.0
            _y = p.get_y() + p.get_height() * 1.02
            value = "{:.0f}".format(p.get_height())
            ax.text(_x, _y + 0.3, value, ha="center")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


# %%

# fig, ax = plt.subplots()
# ax = sns.countplot(x="vot_favor", data=vot_favor_flattened)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
# ax.set(xlabel="Party", ylabel="Number of speeches")

# show_values_on_bars_v(ax)
# plt.savefig('speeches_per_party.png')

# %%
vot_favor_flattened = df["vot_favor"].apply(ast.literal_eval).explode()
top_votos_favor = vot_favor_flattened.value_counts().head(6)
sns.barplot(top_votos_favor.index, top_votos_favor.values)

# %%
vot_abs_flattened = df["vot_abstencao"].apply(ast.literal_eval).explode()
top_votos_abs = vot_abs_flattened.value_counts().head(6)
sns.barplot(top_votos_abs.index, top_votos_abs.values)

# %%
vot_contra_flattened = df["vot_contra"].apply(ast.literal_eval).explode()
top_votos_contra = vot_contra_flattened.value_counts().head(6)
sns.barplot(top_votos_contra.index, top_votos_contra.values)

# %%
top_gp = df["dep_gp"].value_counts().head(6)
sns.barplot(top_gp.index, top_gp.values)

# %%
# 383 - number of initiatives
df.groupby(['ini_num']).count()

# %%
# mean word count
df["text"].apply(lambda x: len(str(x).split())).mean()

# %%
# vobab size
text_series = df['text']
allwords = []
for _, speech in text_series.items():
    tokens_list = word_tokenize(speech, language='portuguese')
    tokens = [token.lower() for token in tokens_list]
    allwords += tokens

# %%
from wordcloud import WordCloud
from nltk.corpus import stopwords

pt_stopwords = stopwords.words("portuguese")
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = pt_stopwords,
                min_font_size = 10).generate(" ".join(allwords))

# plot the WordCloud image                      
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()

# %%
vocabulary = sorted(set(allwords))
# %%
import collections
counter = collections.Counter(allwords)
counter.most_common(10)
# %%

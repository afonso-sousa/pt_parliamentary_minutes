# %%
from pathlib import Path
import pandas as pd
import pdftotext
import re


# df = pd.read_csv("data/out.csv")
df = pd.read_pickle("data/initial_corpus_meta.pkl")

# leg_num_list = ["X", "XI", "XII"]
# df = df.loc[df['ini_leg'].isin(leg_num_list)]
# df = df.reset_index(drop=True)
df = df.explode("pages")
df = df.reset_index(drop=True)

def actual_page(row, last=True):
        return int(row["pages"].split("-")[last]) - int(row["doc_first_page"]) + 1

# %%
# for row_idx, _ in df.iterrows():
    # print(f"Entry #{row_idx} - Done successfully")
    # if 5000 != row["index"]:
    #     continue
    
    # if row_idx > 10:
    #     break
    
    # first_pages = []
    # last_pages = []
    # doc_first_page = df.iloc[row_idx]["doc_first_page"]
    # pairs = df.iloc[row_idx]["pages"]        
    
    # df["first_page"] = df.apply(lambda r: actual_page(r, last=False), axis=1)
    # df["last_page"] = df.apply(lambda r: actual_page(r, last=True), axis=1)
    # for pair in pairs:
    #     first, last = pair.split("-")
    #     first_pages.append(first)
    #     last_pages.append(last)

    # if pairs.startswith('['):
    #     pages_list = re.findall(r'\d+', pairs)
    #     first_pages = pages_list[0::2]
    #     last_pages = pages_list[1::2]
    # else:
    #     first, last = pairs.split("-")
    #     first_pages = [first]
    #     last_pages = [last]
    
    # actual_first_list = []
    # actual_last_list = []
    # for first_page, last_page in zip(first_pages, last_pages):
    #     actual_first_page = int(first_page) - int(doc_first_page) + 1
    #     actual_last_page = int(last_page) - int(doc_first_page) + 1

    #     actual_first_list.append(actual_first_page)
    #     actual_last_list.append(actual_last_page)

# %%  
for row_idx, row in df.iterrows():
    first = actual_page(row, last=False)
    last = actual_page(row, last=True)
    
    file_path = Path(
        "data/parliament/atas_parlamentares_pdfs",
        row["ini_leg"],
        row["pdf_file_path"],
    )
    try:
        with open(file_path, "rb") as f:
            pdf = pdftotext.PDF(f)
            pages_list = []
            for j, page in enumerate(pdf):
                if first - 1 <= j <= last: # add one to `last` to minimize truncated speeches
                    if "text" not in df:
                        df.at[row_idx, 'text'] = page
                    elif pd.isnull(df.at[row_idx, 'text']):
                        df.at[row_idx, 'text'] = page
                    else:
                        df.at[row_idx, 'text'] = df.at[row_idx, 'text'] + "|||" + page

        print(f"Entry #{row_idx + 1} - Done successfully")
    except IOError:
        print(f"Entry #{row_idx + 1} - File not accessible")

# replace cells with empty spaces (namely in text
# when it cannot be extracted)
df = df.replace(r'^s*$', float('NaN'), regex = True)
# removes before-mentioned records
df = df[~df.isnull().any(axis=1)]

# %%
# process votes
# explode all three vote columns
exp_df = df.explode('vot_in_favour').explode('vot_against').explode('vot_abstention')
# compare labels with votes to find matches
exp_df = exp_df.eq(exp_df.pop('dep_parl_group'), axis=0)
# remove all False rows and get the matches in each row
exp_df = exp_df[exp_df.any(1)].idxmax(1)
# remove duplicated indices
df['vote'] = exp_df[~exp_df.index.duplicated()]

# %%
# df.drop("pdf_file_path", axis=1, inplace=True)
df.to_csv("data/out_with_text_X-XII.csv", index=False)

# %%

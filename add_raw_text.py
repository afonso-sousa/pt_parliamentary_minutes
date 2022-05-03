from pathlib import Path
import pandas as pd
import pdftotext
import re


df = pd.read_csv("data/out.csv")

leg_num_list = ["X", "XI", "XII"]
df = df.loc[df['ini_leg'].isin(leg_num_list)]
df = df.reset_index(drop=True)


for row_idx, row in df.iterrows():
    
    # if 5000 != row["index"]:
    #     continue
    
    # if row_idx > 10:
    #     break
    
    first_pages = []
    last_pages = []
    doc_first_page = df.iloc[row_idx]["doc_first_page"]
    pairs = df.iloc[row_idx]["pages"]

    if pairs.startswith('['):
        pages_list = re.findall(r'\d+', pairs)
        first_pages = pages_list[0::2]
        last_pages = pages_list[1::2]
    else:
        first, last = pairs.split("-")
        first_pages = [first]
        last_pages = [last]
    
    actual_first_list = []
    actual_last_list = []
    for first_page, last_page in zip(first_pages, last_pages):
        actual_first_page = int(first_page) - int(doc_first_page) + 1
        actual_last_page = int(last_page) - int(doc_first_page) + 1

        actual_first_list.append(actual_first_page)
        actual_last_list.append(actual_last_page)
    
    
    file_path = Path(
        "data/parliament/atas_parlamentares_pdfs",
        df.iloc[row_idx]["ini_leg"],
        df.iloc[row_idx]["pdf_file_path"],
    )
    try:
        with open(file_path, "rb") as f:
            pdf = pdftotext.PDF(f)
            pages_list = []
            for j, page in enumerate(pdf):
                for (first, last) in zip(actual_first_list, actual_last_list):
                    # print(first, last)
                    # print(k)
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

df = df.replace(r'^s*$', float('NaN'), regex = True)
df = df[~df.isnull().any(axis=1)]
df.to_csv("data/out_with_text_X-XII.csv", index=False)

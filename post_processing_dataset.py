# %%
import pandas as pd

df = pd.read_csv("data/out_with_text_processed.csv")
df = df[~df.isnull().any(axis=1)]
# df = df.drop(["pdf_file_path", "doc_first_page", "first_page", "last_page"], axis=1)

legislature_df = df[["ini_leg", "leg_begin_date", "leg_end_date"]]
legislature_df = legislature_df.drop_duplicates()

initiatives_df = df[
    [
        "ini_leg",
        "ini_num",
        "ini_type",
        "ini_title",
        "ini_session",
        "authors",
        "vot_results",
        "vot_in_favour",
        "vot_against",
        "vot_abstention",
    ]
]
initiatives_df = initiatives_df.drop_duplicates()

interventions_df = df[["ini_num", "dep_id", "dep_name", "dep_parl_group", "text", "vote"]]

legislature_df.to_csv("data/legislature.csv", index=False)
initiatives_df.to_csv("data/initiatives.csv", index=False)
interventions_df.to_csv("data/interventions.csv", index=False)
# %%

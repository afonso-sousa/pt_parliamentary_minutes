# Portuguese Parliamentary Minutes Dataset

## Quick Start
You can download every resource in [this link](https://uporto-my.sharepoint.com/:x:/g/personal/up201709001_up_pt/EZvUHZqF5xVGvwX2AD2RZ9gBvH7VexlRdb63GSvEsdJRzg?e=iJ8njz).

## Requirements and Installation

**a. Clone the repository.**
```shell
git clone https://github.com/afonso-sousa/pt_parliamentary_minutes.git
```
**b. Install dependencies.**
This project uses Anaconda as its package manager. Make sure you have it installed. For more info check [Anaconda's official documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) on environment management.
We have compiled an `enviroment.yml` file with all the required dependencies. To install them, simply run:
```shell
conda env create -f environment.yml
```

## Download Official Resources
### Download required content to compile Corpus
**a. Download PDFs.**
The script for downloading the PDFs is built for the Chrome browser, but can easily be modified to suit your prefered browser.
The following example downloads every available plenary meeting for the 13<sup>th</sup> legislature, 1<sup>st</sup> session (omitting the session value will download all available sessions).
```shell
python download_pdfs.py --leg XIII --session 1
```
**b. Download Initiatives' Metadata**
You can download the official open-access initiatives' metadata [here](https://www.parlamento.pt/Cidadania/Paginas/DAIniciativas.aspx). The current code requires the JSON versions and for these to be placed in a subfolder of `data` named `initiatives`.

### Build Corpus
After downloading the PDFs and Initiatives metadata, we can build the dataset. The dataset compilation is split into three different scripts: `init_corpus_meta.py`, `add_raw_text.py`, `process_corpus_text.py`. You can set the legislatures you want to process and the input and output file paths for each script. Type `python ${SCRIPTNAME} --help` to check the available optional arguments.

**1. Execute `init_corpus_meta.py`.**
This file will process the entries in the initiatives files and compile a dataset of intervention metadata. You can run it with:
```shell
python init_corpus_meta.py [optional arguments]
```
**2. Execute `add_raw_text.py`.**
This file will extract the text for each entry from the corresponding PDF pages in the metadata. You can run it with:
```shell
python add_raw_text.py [optional arguments]
```
This script may take several hours to run. We use multiprocessing, but this script still relies on extracting text from PDFs which is time-consuming.
**3. Execute `process_corpus_text.py`.**
This file will process each intervention's text. You can run it with:
```shell
python process_corpus_text.py [optional arguments]
```

### Model Training and Evaluation
Simply run:
```shell
python nlp_vote_prediction.py --mode ${MODE} --model ${MODELNAME}
```
MODELNAME can take values `nb` (NaiveBayes), `lr` (Logistic Regression), or any model hosted under the Hugging Face Hub (e.g., `PORTULAN/albertina-100m-portuguese-ptpt-encoder`).
MODE can take values `train`, `test`, or `attention_vis` (for transformer-based models).

Alternatively, you can run a script under the [folder with the same name](https://github.com/afonso-sousa/pt_parliamentary_minutes/blob/main/scripts).

### Exploratory Data Analysis
In `eda.py` you can find a Jupyter-like Python file. It contains a variety of different statistical information (frequencies, plots, etc.). If ran in Visual Studio Code, you can execute the code as notebook cells.

# Portuguese Parliamentary Minutes Dataset

## Requirements and Installation

**a. Clone the repository.**
```shell
git clone https://github.com/AfonsoSalgadoSousa/pt_parliamentary_minutes.git
```
**b. Install dependencies.**
This project uses Anaconda as its package manager. Make sure you have it installed. For more info check [Anaconda's official documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) on environment management.
We have compiled an `enviroment.yml` file with all the required dependencies. To install them, simply run:
```shell
conda env create -f environment.yml
```

## Quick Start
### Download required content to compile Corpus
**a. Download PDFs.**
The script for downloading the PDFs is built for the Chrome browser, but can easily be modified to suit your prefered browser.
The following example downloads every available plenary meeting for the 13<sup>th</sup> legislature, 1<sup>st</sup> session (omitting the session value will download all available sessions).
```shell
python download_pdfs.py --leg XIII --session 1
```
**b. Download Initiatives' Metadata**
You can download the official open-access initiatives' metadata [here](https://www.parlamento.pt/Cidadania/Paginas/DAIniciativas.aspx).

### Build Corpus
After downloading the PDFs and Initiatives metadata, we can build the dataset. The dataset compilation is split into three different scripts: `init_corpus_meta.py`, `add_raw_text.py`, `process_corpus_text.py`.

**1. Execute `init_corpus_meta.py`.**
This file will process the entries in the initiatives files and compile a dataset of intervention metadata. You can run it with:
```shell
python init_corpus_meta.py
```
**2. Execute `add_raw_text.py`.**
This file will extract the text for each entry from the corresponding PDF pages in the metadata. You can run it with:
```shell
python add_raw_text.py
```
**3. Execute `process_corpus_text.py`.**
This file will process each intervention's text. You can run it with:
```shell
python process_corpus_text.py
```

### Model Training and Evaluation
Simply run:
```shell
python nlp_vote_prediction.py --model <'nb'|'lr'|'bert'>
```

### Exploratory Data Analysis
In `eda.py` you can find a Jupyter-like Python file. It contains a variety of different statistical information (frequencies, plots, etc.). If ran in Visual Studio Code, you can execute the code as notebook cells.

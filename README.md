# Fake News Detection

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="pandas" />
</p>

Classifying news articles as fake or real with TF-IDF features and logistic regression.
Deliberately simple: a linear model over sparse text features is a strong, fast baseline
for this task and trains in seconds on a laptop.

## Dataset

The Kaggle "Fake and Real News" dataset: two CSVs, `fake.csv` and `true.csv`, with the
article text in a `text` column. Download from Kaggle and place both files in the project
root (they are not committed).

## Pipeline

1. Label the two corpora (`fake = 0`, `true = 1`) and concatenate.
2. 80/20 train/test split.
3. TF-IDF vectorization with a 50,000-term vocabulary cap.
4. Logistic regression classifier.
5. Classification report and confusion matrix on the held-out split, plus a prompt to
   classify any article you paste in.

## Run it

```bash
pip install -r requirements.txt
# place fake.csv and true.csv next to main.py
python main.py
```

## A caveat worth knowing

This dataset is easy: the fake and real articles come from different publishers, so the
model partly learns publisher style, not truthfulness. Scores here do not transfer to
arbitrary news in the wild. That gap, and closing it with source-mixed data, is the
interesting follow-up problem.

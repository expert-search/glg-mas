# GLG - NLP Capstone 
**Named Entity Recognition (NER) + Hierarchical Topic Modeling**
---

https://user-images.githubusercontent.com/18731036/129255631-8c37c4ae-794d-4968-a667-00f9ff9c81b3.mov

### Repo for FourthBrain GLG Project
Extracting Topic and Entity data useful in the Technology and Healthcare domains from unstructured text.

GLG is a management consultancy and self-styled World’s Insight Network. One of GLG’s main business roles is to connect clients with experts from its network of over 900,000 subject matter experts.  It identifies Technology and Healthcare as particular areas of focus for its business delivery.

The problem at hand is that GLG lacks a suitable automated, AI-driven, scalable system to route requests to the appropriate experts. To do this, the verbiage of these requests can be leveraged to extract signifying information that will highlight the domain area and ultimately yield an expert recommendation.

The task, therefore, is to create a self-service application, into which the user will input a specific request, which may take the form:

“What is the 3-5 year outlook for e-commerce in Latin America?”
Or:
“What regulatory threats exist to the development of the genetic testing industry?”
The requirement is that the finished tool will assign the incoming request to a the right expert or experts by means of machine learning techniques. Topic modeling and Named Entity Recognition are logical ways to ensure that requests are correctly routed and that noteworthy names, acronyms and organizations are highlighted.

# Methodology

Build LDA model. Run on provided dataset of 2.5 million articles from All The News dataset.

Below is the dictionary output created from the hand-labeled LDA output (185 LDA topics specified).

![2021-08-01 16_52_09-U-AC Pred - Jupyter Notebook](https://user-images.githubusercontent.com/37546038/128775322-c7b10242-0c57-4bbc-928d-fd5db879fe8c.png)

Use saved LDA model to predict topics for incoming queries:

### Request: 'To what extent can the government / Central Bank influence the macro-economy?'
##### primary topic: global macroeconomics

Provide dendrogram plot of topics, with focus on topics relevant to the problem statement, specifically Healthcare and Technology:

![newplot (2)](https://user-images.githubusercontent.com/37546038/128777400-79a76175-042d-47a3-a67c-dff6d1d34865.png)

## Overall System Architecture:

![image](https://user-images.githubusercontent.com/37546038/128779412-3f64dc9f-51b0-4fe2-bfbf-e5d0eeedd6b9.png)




Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

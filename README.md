# README
This folder contains all of th python scripts used for our projects each of these scripts have a use in the pipeline of creating the model.

### Scripts
We have in this folder contains scripts for fouyr different stages of the pipeline.

1. Files used for scraping games.
    > scrape\_multi.py<br>
    This file can scrape data from multiple regions simultaniously, which is nice given the rate limit on the API.

2. Files used for preprocessing data.
    > preprocess\_all.py<br>
    This file preprocesses all the data. this includes feature engineering and applying labels to observations.

3. Files used for training baseline models.
    > xgboost\_model.py<br>
    This file is used for training an xgboost model in order for usb to compare performance.

    > spatial\_only\_model.py<br>
    This file is used for training a model entirely dependent on the spatial data created, this for a baseline of contribution.

    > transformer\_only\_model.py<br>
    This file is analogous to the spatial\_only\_model.py but fot sequential data.

4. File used for traing the final model.
    > hybrid\_final.py<br>
    This file is used for training the hybrid model architecture proposed in the article.


### Dependencies
To install necessary depencies to run the scripts it is advicable to run the following command.

```bash
pip install -r requirements.txt
```


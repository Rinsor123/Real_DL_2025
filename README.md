# README
# Taming the Dragon: Predicting Epic Monster Kills in LoL

This repository contains the implementation for the paper **"Taming the Dragon: Predicting Epic Monster Deaths in League of Legends"**. 

It features a **Hybrid Multi-Modal Transformer-CNN** model that fuses temporal game state data (gold, XP curves) with spatial map data (champion positions, vision) to predict Dragon and Baron kills 1, 2, and 3 minutes into the future.
This folder contains all of the python scripts used for our projects each of these scripts have a use in the pipeline of creating the model.

### Scripts
This folder contains scripts for four different stages of the pipeline.

1. Files used for scraping games.
    > scrape\_multi.py<br>
    This file can scrape data from multiple regions simultaniously, which is helpful given the rate limit on the API (per region).

2. Files used for preprocessing data.
    > preprocess\_all.py<br>
    This file preprocesses all the data. This includes feature engineering and applying labels to observations.

    > pack\_spatial.py<br>
    Packs raw spatial dictionaries into dense tensors to reduce RAM and speed downstream training.

3. Files used for training baseline models.
    > xgboost-godview_final.ipynb<br>
    This file is used for training an xgboost model in order for us to compare performance.

    > spatial\_only\_model.py<br>
    This file is used for training a model entirely dependent on the spatial data created.

    > transformer\_only\_model.py<br>
    This file is analogous to the spatial\_only\_model.py but for sequential data.

4. File used for traing the final model.
    > hybrid\_final.py<br>
    This file is used for training the hybrid model architecture proposed in the article.


### Dependencies
To install necessary depencies to run the scripts it is advicable to run the following command.

```bash
pip install -r requirements.txt
```


# ShopSmart

## Description

ShopSmart is a recommendation system that uses collaborative filtering to suggest products to users based on their past preferences. The project includes data cleaning and preparation, exploratory analysis, training of collaborative filtering models, and evaluation of their performance.

## Project Structure

- `notebook/`: Contains Jupyter notebooks for preparation, analysis, and model comparison.
  - `01-preprocessing.ipynb`
  - `02-EDA.ipynb`
  - `03-train.ipynb`
  - `04-test.ipynb`
  - `05-compare.ipynb`
- `source/`: Contains source code for models and utilities.
  - `models/`
    - `collabFilter_ItemItem.py`
    - `collabFilter_UserUser.py`
    - `collabFilter.py`
  - `utils/`
    - `dataHandler.py`
    - `utils.py`
- `README.md`: Project description.
- `LICENSE`: Project license.
- `requirements.txt`: Project dependencies.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/DanieleT25/ShopSmart.git
    cd ShopSmart
    ```

2. Create and activate a virtual environment:
    ```bash
    python3 -m venv venvShopSmart
    source venvShopSmart/bin/activate  # On Windows, use `venvShopSmart\Scripts\activate`
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Data Preprocessing:
   - Run `01-preprocessing.ipynb` to clean and prepare the data.

2. Exploratory Data Analysis:
   - Run `02-EDA.ipynb` to visualize and analyze the data.

3. Model Training:
   - Run `03-train.ipynb` to train the collaborative filtering models (item-item and user-user).

4. Model Testing:
   - Run `04-test.ipynb` to test the trained models and calculate the MAE (Mean Absolute Error).

5. Model Comparison:
   - Run `05-compare.ipynb` to compare the MAE of the two models and recommend 5 items using the model with the lowest MAE.
   
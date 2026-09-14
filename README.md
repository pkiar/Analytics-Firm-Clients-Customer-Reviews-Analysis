# Customer Reviews Analysis

## Overview

This project analyzes Amazon customer reviews using three data science approaches:

1. **NLP** – cleans review text and identifies common words and phrases.
2. **Time Series Analysis** – analyzes monthly ratings and forecasts future ratings.
3. **Neural Networks** – uses TF-IDF features to predict customer ratings from 1–5 stars.

**Notebook:** `C09_M08(1).ipynb`
**Dataset:** `amazon_reviews_lab.csv`

---

## Dataset

The dataset contains **997 reviews and 9 columns**, including:

* Product ID
* Rating
* Review text
* Review summary
* Review date
* Sentiment label
* Review length

The ratings are highly imbalanced, with **580 out of 997 reviews rated 5 stars**.

---

## NLP Analysis

The review text was processed using:

* Lowercasing
* URL and punctuation removal
* Tokenization
* Stopword removal
* Lemmatization

### Key Results

* **Total tokens:** 72,140
* **Unique tokens:** 8,587
* Common words include `nook`, `book`, `kindle`, `screen`, `read`, and `device`.
* Common bigrams include **“battery life,” “nook color,” “kindle fire,”** and **“customer service.”**

These terms highlight common customer discussions about products, usability, performance, and service.

---

## Time Series Analysis

Monthly average ratings were analyzed using stationarity tests, decomposition, ARIMA models, and residual diagnostics.

### Key Results

* **Monthly observations:** 68
* **Mean monthly rating:** 3.97
* Initial ADF test showed the series was non-stationary.
* First differencing made the series stationary.
* **Best model:** ARIMA(2,1,1)
* **MAE:** 0.1882
* **RMSE:** 0.2232

The ARIMA model performed better than the tested SARIMA model and produced a 12-month forecast of relatively stable ratings.

---

## Neural Network

Review text was converted into TF-IDF features and used to train a Keras neural network.

### Model

```text
TF-IDF
   ↓
Dense(128, ReLU)
   ↓
Dropout(50%)
   ↓
Dense(64, ReLU)
   ↓
Dropout(30%)
   ↓
Dense(5, Softmax)
```

### Results

* **Training reviews:** 797
* **Testing reviews:** 200
* **TF-IDF features:** 7,357
* **Test accuracy:** 57.0%
* **Macro F1:** 0.176

The model strongly favors 5-star reviews because of the large class imbalance.

---

## Key Findings

| Analysis                | Main Result                                                |
| ----------------------- | ---------------------------------------------------------- |
| Dataset                 | 997 reviews                                                |
| NLP                     | Nook, Kindle, books, screens and devices are common topics |
| Time Series             | ARIMA(2,1,1) performed best                                |
| ARIMA RMSE              | 0.2232                                                     |
| Neural Network Accuracy | 57.0%                                                      |
| Main ML Limitation      | Strong class imbalance                                     |

---

## Limitations

* Small number of monthly observations.
* Strong imbalance toward 5-star reviews.
* Neural network performs poorly on lower rating classes.
* TF-IDF does not capture deeper language context.

### Future Improvements

* Apply class weighting or oversampling.
* Compare the neural network with Logistic Regression and SVM.
* Try more advanced NLP models.
* Test additional forecasting models.
* Use macro F1 as an important evaluation metric.

---

## Technologies

Python, Pandas, NumPy, Matplotlib, Seaborn, NLTK, Scikit-learn, Statsmodels, and TensorFlow/Keras.

## Running the Project

Place the notebook and dataset in the same folder:

```text
project/
├── C09_M08(1).ipynb
├── amazon_reviews_lab.csv
└── README.md
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn statsmodels tensorflow
```

Then open the notebook in Jupyter Notebook, JupyterLab, or VS Code and run the cells sequentially.

## Conclusion

The project demonstrates how **NLP, time-series forecasting, and neural networks** can be combined to analyze customer feedback. The ARIMA model provides useful rating forecasts, while the neural network shows potential for rating prediction but requires better handling of class imbalance.

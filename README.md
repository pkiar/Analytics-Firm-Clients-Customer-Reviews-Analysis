# Customer Reviews Analysis

## Summative Lab – Analytics Firm Clients

This project analyzes customer reviews using three major data analytics and machine learning approaches:

1. Natural Language Processing (NLP)
2. Time Series Analysis
3. Neural Networks

The project uses an Amazon customer reviews dataset containing product information, ratings, review text, review summaries, timestamps, sentiment labels, and review-length information.

---

## Project Overview

The objective of this project is to extract useful insights from customer reviews and demonstrate how different analytical techniques can be applied to real-world customer data.

The analysis focuses on:

- Cleaning and preprocessing customer review text
- Identifying frequent words and bigrams
- Analyzing customer ratings over time
- Testing and transforming time-series data
- Decomposing time-series patterns
- Selecting and evaluating forecasting models
- Applying neural network techniques to customer review data
- Evaluating model performance and drawing business insights

---

## Dataset

The dataset used in this project is:

`amazon_reviews_lab.csv`

The dataset contains the following main variables:

| Column | Description |
|---|---|
| `product_id` | Identifier of the reviewed product |
| `rating` | Customer rating |
| `review_text` | Full customer review |
| `review_summary` | Short review summary |
| `review_time_raw` | Original review date |
| `unix_review_time` | Review time in Unix format |
| `timestamp` | Converted review timestamp |
| `sentiment_label` | Sentiment classification |
| `review_length_words` | Number of words in the review |

---

# Part 1: Natural Language Processing

The NLP section focuses on extracting meaningful patterns from customer review text.

### Main tasks

- Text cleaning
- Tokenization
- Stop-word removal
- Stemming and lemmatization
- Word-frequency analysis
- Bigram analysis
- Sentiment-related analysis
- Visualization of language patterns

The analysis helps identify common terms and phrases used by customers and provides insight into the language associated with customer experiences.

---

# Part 2: Time Series Analysis

The time-series section analyzes **monthly average customer ratings** rather than review volume.

The review timestamps were converted into a time-series index and monthly average ratings were calculated.

### Time Series Workflow

1. Convert timestamps to datetime format
2. Set the timestamp as the time-series index
3. Calculate monthly average ratings
4. Test for stationarity
5. Apply first-order differencing
6. Decompose the time series
7. Analyze ACF and PACF
8. Compare ARIMA models
9. Test a seasonal SARIMA model
10. Perform residual diagnostics
11. Generate forecasts

### Stationarity

The original monthly average rating series was found to be non-stationary.

The Augmented Dickey-Fuller (ADF) test produced:

- ADF p-value: `0.5553`

The KPSS test produced:

- KPSS p-value: `0.01`

Both tests indicated that the original series was non-stationary.

After first-order differencing, the ADF test produced:

- ADF p-value: `2.48 × 10⁻⁶`

This indicates that the differenced series is stationary.

Therefore:

```text
d = 1

# Tesla Tweet Sentiment Analysis
## LSA Topic Modeling & VADER Sentiment Analysis

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Academic-green.svg)](LICENSE)
[![NLP](https://img.shields.io/badge/NLP-NLTK%20%7C%20VADER-orange.svg)](README.md)
[![ML](https://img.shields.io/badge/ML-LSA%20%7C%20TF--IDF-red.svg)](README.md)

## Table of Contents
- [Project Overview](#-project-overview)
- [Analysis Framework](#-analysis-framework)
- [Key Findings](#-key-findings)
- [Demo & Visualizations](#-demo--visualizations)
- [Data Source](#-data-source)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Technical Details](#-technical-details)
- [Results Summary](#-results-summary)
- [Business Insights](#-business-insights)
- [Limitations & Future Work](#-limitations--future-work)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)
- [Citation](#-citation)

## Project Overview

**Social Media Text Mining & Sentiment Analysis for Tesla Brand Perception**

This project implements a complete NLP pipeline for analyzing Tesla-related tweets, combining:
- **Latent Semantic Analysis (LSA)** for topic discovery
- **VADER Sentiment Analysis** for social media sentiment scoring
- **TF-IDF Vectorization** for text feature extraction

The analysis provides actionable insights for brand management, marketing strategy, and investor relations.

## Analysis Framework

### Text Preprocessing Pipeline
1. URL and @mention removal
2. Lowercase normalization
3. Punctuation removal
4. NLTK tokenization
5. Stopword filtering (English, 179 words)
6. WordNet lemmatization
7. Length filtering (≥3 characters)

### Topic Modeling (LSA)
- **Algorithm**: TruncatedSVD on TF-IDF matrix
- **Components**: 6 latent topics
- **Vocabulary**: 5,000 features (max_df=0.7, min_df=10)
- **Cumulative Variance**: 13.8% (typical for sparse short-text)

### Sentiment Analysis (VADER)
- **Tool**: VADER (Valence Aware Dictionary for Sentiment Reasoning)
- **Optimized for**: Social media, short text, emoticons, slang
- **Output**: Compound score [-1, +1]
- **Thresholds**: Positive ≥0.05, Negative ≤-0.05

## Key Findings

### Sentiment Distribution
| Category | Percentage | Description |
|----------|------------|-------------|
| **Positive** | 45.7% | Favorable sentiment |
| **Neutral** | 30.0% | Objective/factual |
| **Negative** | 24.4% | Critical sentiment |

### Discovered Topics

| Topic | Theme | Top Keywords | Sentiment |
|-------|-------|--------------|-----------|
| T0 | Twitter Acquisition | twitter, elon, musk, stock, deal, buy | Neutral (观望) |
| T1 | Product Experience | car, like, model, electric, think | Positive (62%) |
| T2 | Tech Figure Image | elon, musk, spacex, trump, electric | Warm (50%) |
| T3 | Trading & Investment | stock, twitter, sell, buy, tsla, price | Negative (唯一负向) |
| T4 | Purchase Intent | like, look, buy, want, feel | Highest Positive (65%) |
| T5 | Product Innovation | model, plaid, battery, new, year | Positive (55%) |

## Demo & Visualizations

### Sample Output Gallery

<details>
<summary><b>Word Cloud</b></summary>

Core vocabulary coverage: brand (Tesla), person (Elon Musk), event (Twitter), technology (electric, battery, plaid).

**Output:** `figures/wordcloud.png`
</details>

<details>
<summary><b>Sentiment Distribution</b></summary>

Bimodal distribution with:
- 30% neutral peak at compound=0
- Positive tail (≥0.5): 14%
- Negative tail (≤-0.5): 6%

**Output:** `figures/sentiment_dist.png`
</details>

<details>
<summary><b>Topic-Word Heatmap</b></summary>

Visual representation of top-15 words per topic with TF-IDF weights.

**Output:** `figures/topic_word_heatmap.png`
</details>

<details>
<summary><b>Topic Weight Distribution</b></summary>

Document-topic weight boxplots showing:
- Topic 0 (Twitter) has highest median weight
- Topic 4 (Purchase) has narrowest distribution

**Output:** `figures/topic_weight_box.png`
</details>

### Quick Statistics
```
Dataset: Tesla Twitter Corpus (2022-07-11)
Original Tweets: 10,016
English Tweets: 7,358
After Cleaning: 7,341
Mean Sentiment: 0.113 (slightly positive)
Std Sentiment: 0.426
```

## Data Source

**Tesla Twitter Dataset**
- **Source**: Twitter API collection (2022-07-11)
- **Raw records**: 10,016 tweets
- **Language filter**: English only (7,358 tweets)
- **Post-cleaning**: 7,341 valid documents
- **Features**: tweet text, language, timestamp, user info

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/tao-hpu/tesla-sentiment-analysis.git
cd tesla-sentiment-analysis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Analysis

```bash
# Run complete pipeline
python src/text_mining.py

# Expected output:
# - Topic modeling results (6 topics)
# - Sentiment statistics
# - 4 visualization figures in figures/
```

### Expected Output

```
============================================================
Tesla Tweet Sentiment & Topic Analysis
============================================================

原始 tweet 条数：7358
清洗后 tweet 条数：7341

Topic 0: twitter | elon | musk | stock | deal | buy | billion | car | ceo | like
Topic 1: car | like | one | model | get | electric | think | make | would | people
Topic 2: car | elon | musk | ceo | electric | model | spacex | trump | first | new
Topic 3: car | stock | twitter | sell | buy | billion | electric | tsla | price | cover
Topic 4: like | look | car | buy | elon | would | musk | feel | twitter | want
Topic 5: model | one | new | get | year | elon | time | plaid | battery | first

========== 情感统计 ==========
count    7341.000000
mean        0.113000
std         0.426000
...
Positive: 45.7%  Neutral: 30.0%  Negative: 24.4%
```

## Project Structure

```
tesla-sentiment-analysis/
│
├── src/                          # Source code
│   └── text_mining.py            # Main analysis pipeline
│
├── data/                         # Data files
│   └── Tesla.csv                 # Raw tweet dataset
│
├── figures/                      # Generated visualizations
│   ├── wordcloud.png
│   ├── sentiment_dist.png
│   ├── topic_word_heatmap.png
│   └── topic_weight_box.png
│
├── reports/                      # Analysis reports
│   └── kimi_report.txt           # Detailed analysis report (Chinese)
│
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

## Technical Details

### Text Preprocessing

```python
# Pipeline steps:
1. Remove URLs: re.compile(r'http\S+|www\S+')
2. Remove @mentions: re.compile(r'@\w+')
3. Lowercase + remove punctuation
4. NLTK word_tokenize
5. Filter stopwords (179 English words)
6. WordNet lemmatization
7. Keep tokens with len >= 3 and isalpha()
```

### TF-IDF Vectorization

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| max_features | 5,000 | Balance vocabulary coverage |
| max_df | 0.7 | Exclude overly common terms |
| min_df | 10 | Exclude rare typos/noise |

### LSA (Latent Semantic Analysis)

| Parameter | Value |
|-----------|-------|
| Algorithm | TruncatedSVD |
| n_components | 6 |
| random_state | 42 |
| Explained variance | 13.8% |

### VADER Sentiment

| Score | Classification |
|-------|----------------|
| compound ≥ 0.05 | Positive |
| -0.05 < compound < 0.05 | Neutral |
| compound ≤ -0.05 | Negative |

## Results Summary

### Topic-Sentiment Cross Analysis

| Topic | Theme | Mean Sentiment | Positive % | Interpretation |
|-------|-------|----------------|------------|----------------|
| T0 | Twitter Acquisition | 0.02 | 29% | Wait-and-see attitude |
| T1 | User Experience | 0.28 | 62% | Strong positive |
| T2 | Tech Figure | 0.15 | 50% | Moderate optimism |
| T3 | Trading | -0.04 | 25% | Only negative topic |
| T4 | Purchase Intent | 0.31 | 65% | Highest positive |
| T5 | Innovation | 0.19 | 55% | Tech enthusiasm |

### Key Insights

1. **Product-Positive**: Topics about Tesla products (T1, T5) show strong positive sentiment
2. **Trading-Negative**: Investment discussion (T3) is the only negative-leaning topic
3. **Acquisition-Neutral**: Twitter deal (T0) dominates volume but sentiment is neutral
4. **Purchase-Highest**: Purchase intent (T4) has highest positive sentiment (65%)

## Business Insights

### Brand & Marketing

- Leverage T1/T4 positive sentiment for user-generated content campaigns
- T5 tech enthusiasm suggests opportunity for Plaid/4680 battery educational content
- Time promotional content to avoid T0 negative sentiment spikes

### Investor Relations

- T3 shows terms like "overvalued", "bearish" — proactive communication needed
- Consider releasing production/delivery data during negative sentiment periods
- Set up real-time keyword alerts for "bearish", "crash" in T3

### Sales & Customer Service

- Identify high-intent users (T4 + compound ≥ 0.6 + "want/buy") for test drive invitations
- Monitor T0 sentiment before launching promotions to avoid brand confusion

## Limitations & Future Work

### Current Limitations

- **Single-day snapshot**: May have calendar effects
- **LSA linearity**: Short text sparsity limits explained variance
- **No emoji analysis**: Emoticons not included in VADER enhancement
- **Binary language filter**: Only English analyzed

### Future Enhancements

1. **Advanced Topic Models**: LDA, BERTopic, c-TF-IDF
2. **Time Series**: Hourly sentiment trends correlated with news events
3. **Emoji Sentiment**: Extend VADER with emoji lexicon
4. **Entity Recognition**: NER to separate "Tesla company" vs "Tesla vehicle" vs "Musk personal"
5. **Multilingual**: Extend to Chinese, German, Japanese markets

## Technologies Used

### Core Libraries

| Library | Purpose |
|---------|---------|
| pandas | Data manipulation |
| numpy | Numerical computing |
| nltk | Text preprocessing, tokenization |
| scikit-learn | TF-IDF, LSA (TruncatedSVD) |
| vaderSentiment | Social media sentiment |
| wordcloud | Word cloud visualization |
| matplotlib | Plotting |
| seaborn | Statistical visualization |

### Environment

- Python 3.9+
- No GPU required
- ~500MB RAM

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-analysis`)
3. Commit changes (`git commit -m 'Add multilingual support'`)
4. Push to branch (`git push origin feature/new-analysis`)
5. Open Pull Request

## Citation

```bibtex
@software{tesla_sentiment_2025,
  title = {Tesla Tweet Sentiment Analysis: LSA Topic Modeling & VADER Sentiment},
  author = {Chen, Baocheng (Alan)},
  year = {2025},
  url = {https://github.com/tao-hpu/tesla-sentiment-analysis},
  version = {1.0}
}
```

## License

Academic use only - MSAI Program 2025

## Related Projects

- [WHO Mortality Statistical Analysis](https://github.com/tao-hpu/who-mortality-statistical-analysis) - Classical, Bayesian & ML statistical framework

---

**Last Updated: 2025-12-07**

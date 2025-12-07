# -*- coding: utf-8 -*-
"""
Tesla tweets – 潜在语义分析与情感分析
运行：python tweets_lsa_sentiment.py
"""

import re, string, os, ssl
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

try: _create_unverified_https_context = ssl._create_unverified_context
except AttributeError: pass
else: ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# 1. 读数据 -----------------------------------------------------------------
FILE = "Tesla.csv"
df = pd.read_csv(FILE, usecols=["language", "tweet"])
df = df.dropna(subset=["language", "tweet"])
texts = df[df["language"]=="en"]["tweet"].astype(str).tolist()
print(f"原始 tweet 条数：{len(texts)}")

# 2. 预处理 -----------------------------------------------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
url_pattern = re.compile(r'http\S+|www\S+')
mention_pattern = re.compile(r'@\w+')

def clean(tweet: str) -> str:
    tweet = url_pattern.sub('', tweet)          # 去 URL
    tweet = mention_pattern.sub('', tweet)      # 去 @xxx
    tweet = tweet.lower()
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(tweet)
    tokens = [lemmatizer.lemmatize(w) for w in tokens
              if w not in stop_words and w.isalpha() and len(w) > 2]
    return " ".join(tokens)

cleaned = [clean(t) for t in texts]
cleaned = [t for t in cleaned if t]           # 去空
print(f"清洗后 tweet 条数：{len(cleaned)}")

# 3. TF-IDF 矩阵 -------------------------------------------------------------
vectorizer = TfidfVectorizer(max_df=0.7, min_df=10, max_features=5000)
X = vectorizer.fit_transform(cleaned)
terms = vectorizer.get_feature_names_out()

# 4. LSA 主题建模 ------------------------------------------------------------
N_TOPICS = 6
svd = TruncatedSVD(n_components=N_TOPICS, random_state=42)
doc_topic = svd.fit_transform(X)              # 文档-主题矩阵
topic_word = svd.components_                  # 主题-词矩阵

# 打印每个主题 top-10 关键词
n_words = 10
for i, topic in enumerate(topic_word):
    top_idx = topic.argsort()[-n_words:][::-1]
    top_words = [terms[j] for j in top_idx]
    print(f"\nTopic {i}: {' | '.join(top_words)}")

# 5. 情感分析 ---------------------------------------------------------------
analyzer = SentimentIntensityAnalyzer()
sentiments = [analyzer.polarity_scores(t)['compound'] for t in cleaned]
df_clean = pd.DataFrame({'tweet': cleaned, 'sentiment': sentiments})

# 6. 可视化 ------------------------------------------------------------------
os.makedirs("figs", exist_ok=True)

# 6.1 词云
wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(cleaned))
plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Tesla Tweets – Word Cloud")
plt.tight_layout()
plt.savefig("figs/wordcloud.png", dpi=300)
plt.show()

# 6.2 情感分布
plt.figure(figsize=(6, 4))
sns.histplot(df_clean['sentiment'], bins=50, kde=True, color='teal')
plt.title("Sentiment Distribution (VADER compound)")
plt.xlabel("compound score")
plt.ylabel("count")
plt.tight_layout()
plt.savefig("figs/sentiment_dist.png", dpi=300)
plt.show()

# 6.3 主题-词热力图
topic_word_df = pd.DataFrame(topic_word.T, index=terms,
                             columns=[f"T{i}" for i in range(N_TOPICS)])
# 每主题取 top-15 词绘图
topn = 15
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i in range(N_TOPICS):
    topw = topic_word_df.iloc[:, i].sort_values(ascending=False).head(topn)
    sns.heatmap(topw.values.reshape(-1, 1), annot=np.array(topw.index).reshape(-1, 1),
                fmt='', cmap='Blues', cbar=False, ax=axes[i])
    axes[i].set_title(f"Topic {i}")
plt.tight_layout()
plt.savefig("figs/topic_word_heatmap.png", dpi=300)
plt.show()

# 6.4 文档主题权重分布
doc_topic_df = pd.DataFrame(doc_topic, columns=[f"T{i}" for i in range(N_TOPICS)])
doc_topic_df.boxplot(figsize=(8, 4))
plt.title("Document–Topic Weight Distribution")
plt.ylabel("weight")
plt.tight_layout()
plt.savefig("figs/topic_weight_box.png", dpi=300)
plt.show()

# 7. 统计输出 ---------------------------------------------------------------
print("\n========== 情感统计 ==========")
print(df_clean['sentiment'].describe())
print("正/负/中比例：")
pos = (df_clean['sentiment'] >= 0.05).sum()
neu = (df_clean['sentiment'].between(-0.05, 0.05)).sum()
neg = (df_clean['sentiment'] <= -0.05).sum()
total = len(df_clean)
print(f"Positive: {pos/total:.1%}  Neutral: {neu/total:.1%}  Negative: {neg/total:.1%}")

# 8. 保存中间结果（可选） -----------------------------------------------------
# df_clean.to_csv("tesla_tweets_clean_sentiment.csv", index=False)
# pd.DataFrame(doc_topic, columns=[f"Topic{i}" for i in range(N_TOPICS)]).to_csv("doc_topic_weights.csv", index=False)
# print("\n已保存：tesla_tweets_clean_sentiment.csv | doc_topic_weights.csv | figs/*.png")
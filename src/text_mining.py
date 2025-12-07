# -*- coding: utf-8 -*-
"""
Tesla tweets – 潜在语义分析与情感分析
运行：python text_mining.py
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
from matplotlib.ticker import MaxNLocator

try: _create_unverified_https_context = ssl._create_unverified_https_context
except AttributeError: pass
else: ssl._create_default_https_context = _create_unverified_https_context

# 下载NLTK资源 (仅在首次运行时需要)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# 1. 读数据 -----------------------------------------------------------------
FILE = "data/Tesla.csv"
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
# 保留原始推文用于后续分析
original_cleaned = [t for t in df[df["language"]=="en"]["tweet"].astype(str).tolist() if t]
cleaned = [t for t in cleaned if t]           # 去空
# 确保原始推文和清洗后推文列表长度一致
min_len = min(len(original_cleaned), len(cleaned))
original_cleaned = original_cleaned[:min_len]
cleaned = cleaned[:min_len]

print(f"清洗后 tweet 条数：{len(cleaned)}")

# 3. TF-IDF 矩阵 -------------------------------------------------------------
vectorizer = TfidfVectorizer(max_df=0.7, min_df=10, max_features=5000)
X = vectorizer.fit_transform(cleaned)
terms = vectorizer.get_feature_names_out()

# 4. LSA 主题建模 & 主题数选择 ------------------------------------------------
# --- 4.1 使用肘部法则辅助选择主题数 ---
def find_optimal_components(max_components=20):
    explained_variances = []
    for n in range(1, max_components + 1):
        svd = TruncatedSVD(n_components=n, random_state=42)
        svd.fit(X)
        explained_variances.append(svd.explained_variance_ratio_.sum())

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_components + 1), explained_variances, 'bo-')
    plt.title('Elbow Method for Optimal Number of Topics')
    plt.xlabel('Number of Topics')
    plt.ylabel('Total Explained Variance Ratio')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/elbow_method.png", dpi=300)
    return explained_variances

# 运行肘部法则分析 (取消注释以运行)
find_optimal_components(15)

# --- 4.2 使用选定主题数进行建模 ---
print("\n根据上面生成的肘部法则图，折线从6处开始平缓，决定选择6")
N_TOPICS = 6
svd = TruncatedSVD(n_components=N_TOPICS, random_state=42)
doc_topic = svd.fit_transform(X)              # 文档-主题矩阵
topic_word = svd.components_                  # 主题-词矩阵

# 打印每个主题 top-10 关键词
n_words = 10
print("\n========== 主题关键词 ==========")
for i, topic in enumerate(topic_word):
    top_idx = topic.argsort()[-n_words:][::-1]
    top_words = [terms[j] for j in top_idx]
    print(f"Topic {i}: {' | '.join(top_words)}")

# 5. 情感分析 ---------------------------------------------------------------
analyzer = SentimentIntensityAnalyzer()
sentiments = [analyzer.polarity_scores(t)['compound'] for t in cleaned]
df_clean = pd.DataFrame({'original_tweet': original_cleaned, 'cleaned_tweet': cleaned, 'sentiment': sentiments})

# --- 5.1 情感修正案例展示 ---
print("\n========== 情感修正案例 ==========")
# VADER对"but"等转折词处理不佳，我们可以找一些例子
examples = df_clean[df_clean['cleaned_tweet'].str.contains('but', case=False)]
if not examples.empty:
    example = examples.iloc[0]
    print(f"原始推文: {example['original_tweet']}")
    print(f"清洗后: {example['cleaned_tweet']}")
    print(f"VADER情感得分: {example['sentiment']:.4f}")
    print("分析: VADER可能未能完全捕捉到'but'后面的转折含义，需要更复杂的模型或规则进行修正。")
else:
    print("未找到包含'but'的推文作为案例。")

# --- 5.2 典型推文示例分析 ---
print("\n========== 典型推文示例 ==========")
# 为每个主题找到最具代表性的推文
for i in range(N_TOPICS):
    # 找到主题i权重最高的文档
    dominant_doc_idx = doc_topic[:, i].argmax()
    original_tweet = df_clean.iloc[dominant_doc_idx]['original_tweet']
    sentiment_score = df_clean.iloc[dominant_doc_idx]['sentiment']
    sentiment_label = 'Positive' if sentiment_score >= 0.05 else ('Negative' if sentiment_score <= -0.05 else 'Neutral')

    print(f"\n--- Topic {i} 代表推文 ---")
    print(f"内容: {original_tweet}")
    print(f"情感: {sentiment_label} ({sentiment_score:.4f})")


# 6. 可视化 ------------------------------------------------------------------
figs_path="figures"
os.makedirs(figs_path, exist_ok=True)

# 6.1 词云
wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(cleaned))
plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Tesla Tweets – Word Cloud")
plt.tight_layout()
plt.savefig(f"{figs_path}/wordcloud.png", dpi=300)

# 6.2 情感分布
plt.figure(figsize=(6, 4))
sns.histplot(df_clean['sentiment'], bins=50, kde=True, color='teal')
plt.title("Sentiment Distribution (VADER compound)")
plt.xlabel("Compound Score")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{figs_path}/sentiment_dist.png", dpi=300)

# --- 6.3 主题-词热力图 ---
topic_word_df = pd.DataFrame(topic_word.T, index=terms,
                             columns=[f"T{i}" for i in range(N_TOPICS)])
topn = 15
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for i in range(N_TOPICS):
    topw = topic_word_df.iloc[:, i].sort_values(ascending=False).head(topn)
    # 反转DataFrame，让词在y轴上从上到下排列
    topw_df = pd.DataFrame(topw).sort_values(by=topw.name, ascending=True)

    sns.heatmap(topw_df, annot=True, fmt=".3f", cmap='viridis',
                cbar=False, ax=axes[i], yticklabels=True)
    axes[i].set_title(f"Topic {i} - Top Words", fontsize=14)
    axes[i].set_xlabel('Weight', fontsize=12)
    axes[i].tick_params(axis='y', rotation=0) # 保持y轴标签水平
plt.tight_layout()
plt.savefig(f"{figs_path}/topic_word_heatmap.png", dpi=300)

# 6.4 文档主题权重分布
doc_topic_df = pd.DataFrame(doc_topic, columns=[f"T{i}" for i in range(N_TOPICS)])

# 将数据从宽格式转换为长格式，以便 seaborn 绘图
doc_topic_long = doc_topic_df.melt(var_name='Topic', value_name='Weight')

plt.figure(figsize=(10, 6))
sns.boxplot(x='Topic', y='Weight', data=doc_topic_long, palette='viridis')
plt.title("Document–Topic Weight Distribution")
plt.xlabel("Topic")
plt.ylabel("Weight")
plt.tight_layout()
plt.savefig(f"{figs_path}/topic_weight_box.png", dpi=300)

# --- 6.5 文本长度分布图 ---
df_clean['text_length'] = df_clean['original_tweet'].str.len()
plt.figure(figsize=(8, 5))
sns.histplot(df_clean['text_length'], bins=50, kde=True, color='coral')
plt.title("Distribution of Tweet Lengths")
plt.xlabel("Number of Characters")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{figs_path}/text_length_dist.png", dpi=300)

# --- 6.6 主题-情感交叉条形图 ---
# 确定每个文档的主导主题
dominant_topic = np.argmax(doc_topic, axis=1)
df_clean['dominant_topic'] = dominant_topic

# 根据情感得分分类
def get_sentiment_label(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df_clean['sentiment_label'] = df_clean['sentiment'].apply(get_sentiment_label)

# 绘制堆叠条形图
sentiment_topic_counts = pd.crosstab(df_clean['dominant_topic'], df_clean['sentiment_label'])
sentiment_topic_counts.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='coolwarm')
plt.title('Sentiment Distribution by Dominant Topic')
plt.xlabel('Dominant Topic')
plt.ylabel('Number of Tweets')
plt.xticks(rotation=0)
plt.legend(title='Sentiment')
plt.tight_layout()
plt.savefig(f"{figs_path}/topic_sentiment_bar.png", dpi=300)

# --- 6.7 时序分析图---
df_time = pd.read_csv(FILE, usecols=["language", "tweet", "date"])
df_time = df_time.dropna(subset=["language", "tweet", "date"])
df_time = df_time[df_time["language"]=="en"]
df_time['date'] = pd.to_datetime(df_time['date'], format='%Y-%m-%d %H:%M:%S')
df_time['sentiment'] = [analyzer.polarity_scores(clean(t))['compound'] for t in df_time['tweet']]

# # 按日期聚合情感
sentiment_over_time = df_time.groupby(df_time['date'].dt.hour)['sentiment'].mean()

plt.figure(figsize=(12, 6))
sentiment_over_time.plot()
plt.title('Average Sentiment Over Time')
plt.xlabel('Date')
plt.ylabel('Average Compound Score')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{figs_path}/sentiment_timeseries.png", dpi=300)
print("\n已生成24小时内的情感时序分析图")

# 7. 统计输出 ---------------------------------------------------------------
print("\n========== 情感统计 ==========")
print(df_clean['sentiment'].describe())
print("正/负/中比例：")
pos = (df_clean['sentiment'] >= 0.05).sum()
neu = (df_clean['sentiment'].between(-0.05, 0.05)).sum()
neg = (df_clean['sentiment'] <= -0.05).sum()
total = len(df_clean)
print(f"Positive: {pos/total:.1%}  Neutral: {neu/total:.1%}  Negative: {neg/total:.1%}")
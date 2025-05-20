import matplotlib.pyplot as plt
import pandas as pd
import os
import nltk
import jieba
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
import re
import glob

# 確保下載過資源
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.word_tokenize("test")
except LookupError:
    nltk.download('punkt')

# === Data file configuration ===
# By default, load all CSVs in the 'data/' folder. Place your data files there.
DATA_FOLDER = 'data'  # Change this if you want a different folder
csv_files = glob.glob(os.path.join(DATA_FOLDER, '*.csv'))

if not csv_files:
    print(f"No CSV files found in '{DATA_FOLDER}/'. Please add your data files (see README for format).")
    exit()

dfs = []
for csv_file in csv_files:
    if os.path.exists(csv_file):
        df_temp = pd.read_csv(csv_file, sep=',', encoding='utf-8')
        dfs.append(df_temp)
        print(f"Read: {os.path.basename(csv_file)}")
    else:
        print(f"File not found: {csv_file}")

if not dfs:
    print("No CSV loaded.")
    exit()

df = pd.concat(dfs, ignore_index=True)

required_columns = ['Platform', 'Source', 'Customer ID', 'Message Content']
if not all(col in df.columns for col in required_columns):
    print("Missing required columns.")
    exit()

customer_data = df[df['Source'] == 'customer'].copy()
print(f"Total customer messages: {len(customer_data)}")
print(f"Unique customer IDs: {customer_data['Customer ID'].nunique()}")

stop_words = set(stopwords.words('english'))

#Token 是從 Message Content 裡來的，經過「斷詞」後的結果
def preprocess_text(text):  
    if not isinstance(text, str):
        return []
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text.lower())
    jieba_tokens = jieba.lcut(text)
    en_tokens = nltk.word_tokenize(text)
    tokens = list(set(jieba_tokens + en_tokens))
    return [token for token in tokens if token not in stop_words and len(token) > 1]

customer_data['Tokens'] = customer_data['Message Content'].apply(preprocess_text)
docs = [doc for doc in customer_data['Tokens'].tolist() if len(doc) > 0]
if not docs:
    print("No documents after preprocessing.")
    exit()

dictionary = corpora.Dictionary(docs)
corpus = [dictionary.doc2bow(doc) for doc in docs]
num_topics = 9
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15, random_state=42)

print("\nLDA Topics:")
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}")

# LDA topic keywords for each category.
# Each sublist contains keywords for one topic/category.
# Index 8 is reserved for "Other" and is intentionally left empty.
topic_keywords = [
    # 0: Booking
    ['room', 'hotel', 'book', 'booking', 'reserve', 'reservation', 'checkin', 'checkout', 'stay', 'night',
     'cancel', 'reschedule', 'confirm', 'deposit', 'type', 'bed', 'price', 'rate', 'available', 'availability',
     'early checkin', 'late checkout', 'single', 'double'],
    # 1: Promotion
    ['discount', 'promotion', 'coupon', 'voucher', 'offer', 'deal', 'special', 'price drop', 'limited time', 'code',
     'promo', 'campaign', 'member price', 'bonus'],
    # 2: Food Ordering
    ['food', 'restaurant', 'menu', 'order', 'eat', 'meal', 'dinner', 'lunch', 'breakfast', 'table', 'buffet',
     'dish', 'snack', 'drink', 'coffee', 'beverage', 'takeout', 'delivery', 'vegan', 'halal', 'chef'],
    # 3: Facility
    ['wifi', 'internet', 'pool', 'gym', 'spa', 'jacuzzi', 'sauna', 'shower', 'bathroom', 'toilet', 'towel',
     'aircon', 'television', 'tv', 'clean', 'laundry', 'machine', 'elevator', 'key card', 'locker', 'amenities'],
    # 4: Itinerary
    ['trip', 'tour', 'travel', 'sightseeing', 'guide', 'schedule', 'plan', 'shuttle', 'map', 'direction',
     'route', 'pickup', 'dropoff', 'airport', 'taxi', 'bus', 'location', 'nearby', 'place', 'car rental'],
    # 5: Service
    ['service', 'help', 'support', 'staff', 'reception', 'rude', 'friendly', 'attentive', 'complain', 'problem',
     'issue', 'broken', 'fix', 'no', 'not work', 'unavailable', 'slow', 'delay', 'refund', 'unsatisfied', 'late',
     'disappointed', 'bad', 'dirty', 'smell'],
    # 6: Nearby
    ['nearby', 'around', 'location', 'convenient', 'distance', 'area', 'neighborhood', 'walk', 'store',
     'restaurant', 'attraction', 'mall', 'transport', 'station', 'bus stop', 'access'],
    # 7: Complaint
    ['complaint', 'complain', 'dissatisfied', 'dissatisfaction', 'unhappy', 'bad experience', 'poor service', 'terrible', 'awful', 'frustrating', 'disappointed', 'issue', 'problem', 'unacceptable', 'not satisfied', 'negative feedback'],
    # 8: Other (empty on purpose)
    []
]

# Mapping from topic index to human-readable category name.
topic_to_category = {
    0: '訂房 (Booking)',
    1: '優惠 (Promotion)',
    2: '訂餐 (Food Ordering)',
    3: '設施 (Facility)',
    4: '行程 (Itinerary)',
    5: '服務 (Service)',
    6: '附近 (Nearby)',
    7: '抱怨 (Complaint)',
    8: '其他 (Other)'
}

#把每個訊息的單字（token）和 topic_keywords 裡的每個分類詞彙做完全比對。
#如果有哪一組關鍵字出現次數最多，那它就會被指定為該分類。

def keyword_based_topic(doc):
    scores = [0] * len(topic_keywords)
    for i, keywords in enumerate(topic_keywords):
        if keywords:  # 避免空列表影響
            scores[i] = sum(token in keywords for token in doc)
    if max(scores) >= 2:
        return scores.index(max(scores))
    return None

def get_dominant_topic(bow):
    if not bow:
        return 5  # 預設分類為服務 (Service)
    topic_dist = lda_model[bow]
    return max(topic_dist, key=lambda x: x[1])[0]

def final_topic(doc):
    bow = dictionary.doc2bow(doc)
    keyword_topic = keyword_based_topic(doc)
    if keyword_topic is not None:
        return keyword_topic
    lda_topic = get_dominant_topic(bow)
    # 如果LDA是Other (7)，再選一次最接近的非Other分類
    if lda_topic == 7:
        topic_dist = lda_model[bow]
        filtered = [(topic_id, prob) for topic_id, prob in topic_dist if topic_id != 7]
        if filtered:
            return max(filtered, key=lambda x: x[1])[0]
    return lda_topic



valid_indices = customer_data.index[customer_data['Tokens'].apply(len) > 0]
customer_data.loc[valid_indices, 'Topic'] = [final_topic(doc) for doc in docs]
customer_data['Topic'] = customer_data['Topic'].fillna(7).astype(int)
customer_data['Category'] = customer_data['Topic'].map(topic_to_category)

category_counts = customer_data['Category'].value_counts()
print("\nNumber of messages per category:")
print(category_counts)

plt.figure(figsize=(8, 6))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90,
        colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#b3e6ff'])
plt.title('Distribution of Customer Issue Categories',fontsize=16, fontweight='bold', color='navy')
plt.axis('equal')
plt.show()

platform_counts = customer_data.groupby('Platform')['Customer ID'].nunique()
plt.figure(figsize=(8, 6))
plt.pie(platform_counts, labels=platform_counts.index, autopct='%1.1f%%', startangle=90,
        colors=['#ff9999', '#66b3ff', '#99ff99'])
plt.title('Platform Comparison to Source: Customer (Unique Users)',fontsize=16, fontweight='bold', color='navy')
plt.axis('equal')
plt.savefig('platform_customer_pie_chart.png')
plt.show()

if 'Send TIme' in customer_data.columns:
    customer_data['Send TIme'] = pd.to_datetime(customer_data['Send TIme'], errors='coerce')
    customer_data = customer_data.dropna(subset=['Send TIme'])
    customer_data['Date'] = customer_data['Send TIme'].dt.date
    daily_unique_customers = customer_data.groupby('Date')['Customer ID'].nunique().reset_index()
    daily_unique_customers.columns = ['Date', 'Unique Customer Count']
    print("\nDaily Unique Customer Count:")
    print(daily_unique_customers.head())

    plt.figure(figsize=(12, 6))
    plt.plot(daily_unique_customers['Date'], daily_unique_customers['Unique Customer Count'], marker='o', color='skyblue')
    plt.title('Daily Unique Customer Count',fontsize=16, fontweight='bold', color='navy')
    plt.xlabel('Date')
    plt.ylabel('Unique Customer Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('daily_unique_customer_count.png')
    plt.show()
else:
    print("⚠️ Column 'Send Time' not found.")

export_columns = ['Send TIme', 'Platform', 'Customer ID', 'Message Content', 'Tokens', 'Topic', 'Category']
categorized_output = customer_data[export_columns].copy()
output_path = 'categorized_customer_messages.csv'
categorized_output.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n✅ Category successfully saved to: {output_path}")
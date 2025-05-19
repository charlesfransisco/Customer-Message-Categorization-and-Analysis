Customer Message Categorization and Analysis
This project processes and analyzes customer messages from CSV files, categorizing them into predefined topics using LDA (Latent Dirichlet Allocation) and keyword-based matching. It also generates visualizations to summarize the results.

Features
Data Preprocessing:

Combines multiple CSV files containing customer messages.
Filters messages from customers and tokenizes text using both English and Chinese tokenizers (nltk and jieba).
Removes stopwords and non-essential characters.
Topic Modeling:

Uses LDA to identify latent topics in the messages.
Matches messages to predefined categories using keyword-based scoring.
Categorization:

Maps messages to one of the following categories:
訂房 (Booking)
優惠 (Promotion)
訂餐 (Food Ordering)
設施 (Facility)
行程 (Itinerary)
服務 (Service)
附近 (Nearby)
抱怨 (Complaint)
其他 (Other)
Visualization:

Generates pie charts for:
Distribution of customer issue categories.
Platform comparison based on unique customer counts.
Plots daily unique customer counts over time.
Export:

Saves categorized messages to a CSV file (categorized_customer_messages.csv).

# Customer Message Categorization and Analysis

A portfolio project that demonstrates how customer-service messages can be organized into useful service categories using **Python**, **Natural Language Processing (NLP)**, **topic modeling**, and **data visualization**.

> **Privacy Notice:** This repository is intended for demonstration using **synthetic/example data only**. It does not include real customer conversations, confidential company information, API keys, or production data.

---

## Project Overview

Customer-service teams may receive many messages from different platforms. This project explores a simple analysis workflow to organize those messages into categories such as booking requests, facility questions, complaints, and nearby recommendations.

The project demonstrates my interest in applying **data analysis and AI-related techniques** to service operations and hospitality-related business scenarios.

---

## Key Features

* Combines customer-message CSV files from a local folder
* Filters messages sent by customers
* Cleans and tokenizes English and Chinese text
* Uses **LDA topic modeling** and **keyword-based matching**
* Assigns messages to service-related categories
* Visualizes message distribution and platform usage
* Exports categorized results for further analysis

---

## Message Categories

| Category      | Example Use Case                                        |
| ------------- | ------------------------------------------------------- |
| Booking       | Room reservations, check-in, check-out, availability    |
| Promotion     | Discounts, coupons, special offers                      |
| Food Ordering | Breakfast, restaurant, meal, delivery requests          |
| Facility      | Wi-Fi, gym, laundry, air conditioning, amenities        |
| Itinerary     | Transportation, travel planning, pickup arrangements    |
| Service       | General assistance and service-related requests         |
| Nearby        | Restaurants, attractions, stores, transportation nearby |
| Complaint     | Negative feedback or service dissatisfaction            |
| Other         | Messages that do not match the categories above         |

---

## Workflow

```text
Synthetic CSV Data
        ↓
Filter Customer Messages
        ↓
Text Cleaning and Tokenization
        ↓
LDA Topic Modeling + Keyword Matching
        ↓
Message Categorization
        ↓
Charts and CSV Output
```

---

## Tech Stack

| Area                 | Tools       |
| -------------------- | ----------- |
| Programming Language | Python      |
| Data Processing      | pandas      |
| Text Processing      | NLTK, jieba |
| Topic Modeling       | gensim LDA  |
| Visualization        | matplotlib  |

---

## Output

The current script produces:

* A category-distribution pie chart displayed during execution
* `platform_customer_pie_chart.png` — comparison of unique customers by platform
* `daily_unique_customer_count.png` — daily customer trend, when timestamp data is available
* `categorized_customer_messages.csv` — categorized analysis results

> The exported CSV may include message content and customer identifiers from the input file. For public portfolio use, only synthetic/demo data should be processed and shared.

---

## Required Data Format

The script reads CSV files from a local `data/` folder.

Required columns:

| Column            | Description                                   |
| ----------------- | --------------------------------------------- |
| `Platform`        | Message platform, such as LINE or Web         |
| `Source`          | Sender type; customer messages use `customer` |
| `Customer ID`     | Synthetic identifier for analysis             |
| `Message Content` | Message text                                  |
| `Send TIme`       | Timestamp for daily trend analysis            |

### Example Synthetic Data

Create a local file such as `data/example_data.csv` using made-up data only:

```csv
Platform,Source,Customer ID,Message Content,Send TIme
LINE,customer,CUST001,"I would like to book a room for two nights.",2025-01-01 12:00:00
Web,customer,CUST002,"Do you have any weekend discounts?",2025-01-02 13:45:00
LINE,customer,CUST003,"The air conditioner is not working.",2025-01-03 09:20:00
Web,customer,CUST004,"Is there a restaurant near the hotel?",2025-01-04 18:30:00
```

---

## How to Run

### 1. Install Dependencies

```bash
pip install pandas matplotlib nltk jieba gensim
```

### 2. Prepare Demo Data

Create a local folder named:

```text
data/
```

Then add a CSV file containing **synthetic/example data only**.

### 3. Run the Script

```bash
python Categorization.py
```

### 4. Review Results

After execution, review the charts and categorized CSV output.

---

## Privacy and Responsible Use

This repository is a **portfolio proof of concept**.

For safety and confidentiality:

* Do not upload real customer messages.
* Do not upload names, phone numbers, reservation IDs, room numbers, or internal company records.
* Do not commit CSV output generated from real-world service data.
* Do not commit API keys, credentials, or environment files if integrations are added in the future.
* Use only synthetic or publicly shareable example data in this public repository.

---

## Future Improvements

* Add a fully synthetic sample dataset for reproducible demonstrations
* Add model evaluation using labeled test data
* Improve multilingual classification accuracy
* Create an interactive dashboard for service-message insights

---

## License

Licensed under the Apache License 2.0.

# Customer Message Categorization and Analysis

This project processes and analyzes customer messages from CSV files, categorizing them into predefined topics using LDA (Latent Dirichlet Allocation) and keyword-based matching. It also generates visualizations to summarize the results.

## Features

- **Data Preprocessing:**
  - Combines multiple CSV files containing customer messages (from the `data/` folder).
  - Filters messages from customers and tokenizes text using both English and Chinese tokenizers (`nltk` and `jieba`).
  - Removes stopwords and non-essential characters.

- **Topic Modeling:**
  - Uses LDA to identify latent topics in the messages.
  - Matches messages to predefined categories using keyword-based scoring.

- **Categorization:**
  - Maps messages to one of the following categories:
    - 訂房 (Booking)
    - 優惠 (Promotion)
    - 訂餐 (Food Ordering)
    - 設施 (Facility)
    - 行程 (Itinerary)
    - 服務 (Service)
    - 附近 (Nearby)
    - 抱怨 (Complaint)
    - 其他 (Other)

- **Visualization:**
  - Generates pie charts for:
    - Distribution of customer issue categories.
    - Platform comparison based on unique customer counts.
  - Plots daily unique customer counts over time.

- **Export:**
  - Saves categorized messages to a CSV file (`categorized_customer_messages.csv`).

## Usage

1. **Prepare Data:**
   - Place your CSV files in the `data/` folder. **Do not include real or sensitive company/customer data in the repository.**
   - Each CSV should have the following columns:
     - `Platform`, `Source`, `Customer ID`, `Message Content`, `Send TIme`
   - You may use the provided `example_data.csv` as a template (with dummy data).

2. **Install Requirements:**
   - Install Python 3.7+ and the required packages:
     ```bash
     pip install pandas matplotlib nltk jieba gensim
     ```

3. **Run the Script:**
   - Execute the script:
     ```bash
     python Categorization.py
     ```

4. **View Results:**
   - Visualizations will be displayed and saved as PNG files.
   - The categorized results will be saved as `categorized_customer_messages.csv`.

## Data Privacy

- **No real company/customer data is included in this repository.**
- The `data/` folder is excluded via `.gitignore` to prevent accidental upload of sensitive files.
- Only use example or dummy data for demonstration and sharing.

## Example Data

You can create a file like `data/example_data.csv`:

```csv
Platform,Source,Customer ID,Message Content,Send TIme
LINE,customer,12345,"I want to book a room",2025-01-01 12:00:00
WhatsApp,customer,67890,"Do you have any discounts?",2025-01-02 13:45:00
```

## License

This project is provided for educational and demonstration purposes. Please review and update the license as appropriate for your use case.

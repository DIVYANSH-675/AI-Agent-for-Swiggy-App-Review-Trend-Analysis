# AI Agent for Swiggy App Review Trend Analysis

This project builds an AI system that analyzes Google Play Store reviews for **Swiggy** — a popular food delivery app in India — and generates daily trend reports. The system automatically detects user issues, requests, and feedback patterns from thousands of reviews and monitors how they change over time.

## ⚠️ Important Disclaimer

**API KEY NOTICE:**
The API keys used in this project are **revoked**.
You must use your own valid API keys to run this project.

---

## 📌 Project Overview

The **AI Agent for Swiggy App Review Trend Analysis** is an automated pipeline that fetches app reviews from the Google Play Store, cleans and processes the data, and uses advanced AI techniques to extract, cluster, and summarize user sentiments and topics.
It then generates daily and monthly reports to visualize key issues and customer feedback trends over time.

---

## 🧠 Key Features

* **Automated Review Scraping:** Gathers the latest Swiggy reviews from Google Play Store.
* **Sentiment and Topic Detection:** Uses NLP models to identify user pain points, feature requests, and feedback.
* **Dynamic Topic Grouping:** Clusters similar comments to create coherent topics automatically.
* **Trend Tracking:** Tracks how issues evolve daily, providing comparative insights.
* **Data Storage:** Stores all raw, processed, and analyzed data for further analytics.
* **Daily Report Generation:** Creates concise summaries and structured CSV reports every 24 hours.

---

## 🏗️ Project Pipeline

1. **Review Extraction**

   * Fetches recent Swiggy app reviews using the Google Play Scraper.
   * Supports pagination and date-wise filtering for continuous data collection.

2. **Data Cleaning & Preprocessing**

   * Removes duplicates, irrelevant text, emojis, and special characters.
   * Filters out non-English or incomplete reviews.

3. **AI-based Review Categorization**

   * Uses LLM (OpenAI or compatible model) to classify reviews into key categories:

     * Bugs / Technical Issues
     * Delivery & Service Problems
     * Feature Requests
     * Positive Feedback
     * Others

4. **Topic Clustering**

   * Groups semantically similar reviews using embeddings.
   * Creates summarized labels for each cluster for readability.

5. **Trend Analysis**

   * Calculates daily frequency of recurring topics.
   * Tracks issue spikes or resolution trends over time.

6. **Report Generation**

   * Produces both CSV summaries and markdown-style daily reports.
   * Ready for visualization or dashboard integration.

---

## 🧹 Folder Structure

```
AI-Agent-Swiggy-Review-Trend/
│
├── data/
│   ├── raw_reviews.csv
│   ├── processed_reviews.csv
│   └── trend_reports/
│
├── scripts/
│   ├── fetch_reviews.py
│   ├── analyze_reviews.py
│   ├── trend_analysis.py
│   └── generate_report.py
│
├── requirements.txt
├── README.md
└── config.json
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Agent-Swiggy-Review-Trend.git
cd AI-Agent-Swiggy-Review-Trend
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

Update your API keys in `config.json`:

```json
{
  "openai_api_key": "YOUR_API_KEY_HERE",
  "model": "gpt-4-turbo"
}
```

### 4. Run the Pipeline

```bash
python scripts/fetch_reviews.py
python scripts/analyze_reviews.py
python scripts/trend_analysis.py
python scripts/generate_report.py
```

---

## 📊 Example Output

**Daily Trend Report (sample)**

| Date       | Topic                        | Count | Sentiment | Trend       |
| ---------- | ---------------------------- | ----- | --------- | ----------- |
| 2025-11-11 | Late Delivery Complaints     | 132   | Negative  | 🔺 Rising   |
| 2025-11-11 | App Login Issues             | 64    | Negative  | 🔻 Dropping |
| 2025-11-11 | Cashback Feature Requests    | 28    | Neutral   | ➖ Stable    |
| 2025-11-11 | Positive Delivery Experience | 115   | Positive  | 🔺 Rising   |

---

## 🧾 Requirements

* Python 3.10 or higher
* pip 24+
* Compatible with both CPU and GPU environments
* Tested on Ubuntu 22.04 (Linux)

---

## 📈 Future Extensions

* Integrate visualization dashboards (Streamlit / Plotly)
* Add real-time sentiment monitoring
* Expand to multi-app comparison (Swiggy, Zomato, Blinkit)
* Automate email alerts for major trend changes

---

## 🤝 Contribution Guidelines

Contributions are welcome!
Feel free to:

* Submit issues for bugs or improvements
* Open pull requests for enhancements
* Share suggestions for better review analytics

---

## 🧑‍💻 Author

**Divyansh Gupta**
Rajiv Gandhi Institute of Petroleum Technology (RGIPT)

---

## 🪪 License

This project is licensed under the **MIT License**.

---

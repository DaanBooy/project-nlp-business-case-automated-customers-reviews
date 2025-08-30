# Project NLP | Business Case: Automated Customer Reviews

The goal of this project is to develop a product review system. That can classify sentiment, cluster by product categories and write product summaries into recommendation articles.

For this project a dataset with **Amazon reviews** is used, more specifically reviews from the **Video Games** category. This data set can be found here 👉 https://amazon-reviews-2023.github.io/ 👈

## 📊 Best Results

Best-performing model for review sentiment classification was **RoBERTa-base**:

- **Accuracy:** 0.875  
- **F1 Score (Macro Avg):** 0.734  
- **Precision (Macro Avg):** 0.719  
- **Recall (Macro Avg):** 0.766  
- **Weighted F1 Score:** 0.885  

## 🔍 Clusters found

 - **Keyboards & Mice**
 - **Headsets & Audio**
 - **Games**
 - **Controllers**

## 🤗 Models and deployment on HuggingFace

👉 https://huggingface.co/spaces/DaanBooy/games_and_accessories_reviews 👈

**What you can do:**
- **Classification tab:** Upload a review and see its sentiment classification (RoBERTa-base).
- **Clustering view:** Explore product category clusters.
- **Summaries:** Read AI-generated articles highlighting top products, main complaints, and the worst product per category.

The model files generated in this project can be found here: 👇

 - https://huggingface.co/DaanBooy/review-sentiment-distilbert-base-uncased
 - https://huggingface.co/DaanBooy/review-sentiment-bert-base-uncased
 - https://huggingface.co/DaanBooy/review-sentiment-roberta-base

## ⚙️ How to Run & Reproduce Results

**Clone the repository**

```bash
git clone <your-repo-url>
cd <repo-folder>
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Data Preparation**  
Run `data_prep.ipynb` to preprocess the Amazon Video Games review dataset.

**Train Models**
- `review_classification_dbu.ipynb` → DistilBERT  
- `review_classification_bbu.ipynb` → BERT-base  
- `review_classification_rb.ipynb` → RoBERTa-base

**Evaluate Models**  
Use `model_comparison_viz.ipynb` to visualize results (precision, recall, F1, confusion matrix).

**Clustering**
- Run `review_clustering_sample.ipynb` for exploratory clustering.  
- Run `review_clustering_full.ipynb` to cluster the full dataset into 4 categories.

**Summarization**  
Execute `review_summarization.ipynb` to generate product recommendation articles.

**Deployment**  
Use `model_deployment_HF.ipynb` to deploy the system to Hugging Face Spaces.

**Notes:**  
To ensure you have enough RAM to run notebooks, use Google Colab on GPU T4 (High-RAM).  
To run `review_summarization.ipynb`, make sure you create a `.env` file in your project folder/Google Drive containing your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

---

## 📂 Repository Contents

| File Name | Description |
:-----------:|:-------------:|
| `requirements.txt` | Lists all Python dependencies needed to run and reproduce the entire project |
| `data_prep.ipynb` | Used to prepare review data for the rest of the project |
| `review_classification_dbu.ipynb` | Notebook with distilbert-base-uncased model training |
| `review_classification_bbu.ipynb` | Notebook with bert-base-uncased model training |
| `review_classification_rb.ipynb` | Notebook with roberta-base model training |
| `model_comparison_viz.ipynb` | Notebook used to visualize model results|
| `review_clustering_sample.ipynb` | Notebook used to cluster sample of data to inspect dataset |
| `review_clustering_full.ipynb` | Notebook used to full cluster the entire dataset |
| `review_summarization.ipynb` | Notebook used to make the generative review summarization articles |
| `model_deployment_HF` | File used to deploy to HuggingFace space |
| `Project NLP _ Business Case_ Automated Customer Reviews - Final report.pdf` | PDF containing the final report on this project|
| `To be added later` | Presentation file, to be added later |

---

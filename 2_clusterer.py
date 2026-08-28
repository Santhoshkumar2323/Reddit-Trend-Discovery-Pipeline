import os
import json
import glob
import re
import pandas as pd
import umap  
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

with open("config.json", "r") as f:
    config = json.load(f)

os.makedirs("output_reports", exist_ok=True)
timestamp = pd.Timestamp.now().strftime("%Y%m%d")

print("\n Loading Vector Model...")
model = SentenceTransformer(config["ai_settings"]["embedding_model"])

domain_name = config["domain_name"]
raw_files = glob.glob(f"raw_data/{domain_name}_*.csv")

if not raw_files:
    print(" Error: No collected CSV files found in raw_data/. Run scraper first.")
    exit()

CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS.union(config["ai_settings"]["custom_stopwords"]))

def clean_text_for_nlp(text):
    text = re.sub(r'http\S+|www\.\S+', '', str(text))
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text.strip()

print("\n STARTING SEMANTIC CLUSTERING CALCULATIONS...")
print("-" * 89)

for file_path in raw_files:
    filename = os.path.basename(file_path)
    year = filename.replace(f"{domain_name}_", "").replace(".csv", "")
    
    df = pd.read_csv(file_path)
    df = df[df['post_id'] != 'post_id'] 
    df["title"] = df["title"].fillna("")
    df["body_text"] = df["body_text"].fillna("")
    df["body_fingerprint"] = df["body_text"].apply(lambda x: re.sub(r'\W+', '', str(x)[:200].lower()))
    df = df.drop_duplicates(subset=["body_fingerprint"]).copy()
    
    df["normalized_title"] = df["title"].apply(lambda x: re.sub(r'\d+|january|february|march|april|may|june|july|august|september|october|november|december', '', str(x).lower()).strip())
    df = df.drop_duplicates(subset=["normalized_title", "subreddit"]).copy()
    
    total_rows = len(df)
    
    if total_rows < config["ai_settings"]["min_cluster_size"]:
        print(f"Year {year}: Skipped (Only {total_rows} unique posts available. Need {config['ai_settings']['min_cluster_size']})")
        continue
        
    df["clean_content"] = (df["title"] + " " + df["body_text"]).apply(clean_text_for_nlp)
    
    embeddings = model.encode(df["clean_content"].tolist(), batch_size=32, show_progress_bar=False)
    
    print(f" Reducing dimensions with UMAP for year {year}...")
    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        metric='cosine',
        random_state=42,
        n_jobs=1  
    )
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    clusterer = HDBSCAN(
        min_cluster_size=config["ai_settings"]["min_cluster_size"],
        min_samples=2,
        metric='euclidean',
        copy=True  
    )
    
    df["cluster_id"] = clusterer.fit_predict(reduced_embeddings)
    
    df["cluster_keywords"] = "Noise / Miscellaneous"
    unique_clusters = [c for c in df["cluster_id"].unique() if c != -1]
    
    for cluster in unique_clusters:
        cluster_docs = df[df["cluster_id"] == cluster]["clean_content"]
        try:
            tfidf = TfidfVectorizer(stop_words=CUSTOM_STOP_WORDS, max_features=100)
            tfidf_matrix = tfidf.fit_transform(cluster_docs)
            word_scores = tfidf_matrix.sum(axis=0).A1
            feature_names = tfidf.get_feature_names_out()
            top_indices = word_scores.argsort()[::-1][:4]
            top_words = [feature_names[i] for i in top_indices]
            
            df.loc[df["cluster_id"] == cluster, "cluster_keywords"] = ", ".join(top_words)
        except Exception:
            df.loc[df["cluster_id"] == cluster, "cluster_keywords"] = "General Fitness"
        
    html_path = f"output_reports/dashboard_{domain_name}_{year}_{timestamp}.html"
    csv_path = f"output_reports/report_{domain_name}_{year}_run_{timestamp}.csv"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fitness Trends Dashboard - {year}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; display: flex; height: 100vh; background: #f5f7fb; }}
            .sidebar {{ width: 340px; background: #ffffff; border-right: 1px solid #e1e6eb; overflow-y: auto; padding: 20px; box-sizing: border-box; }}
            .main-content {{ flex: 1; padding: 30px; overflow-y: auto; box-sizing: border-box; }}
            h2 {{ color: #1e293b; margin-top: 0; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }}
            .topic-btn {{ width: 100%; text-align: left; padding: 12px; margin-bottom: 8px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; cursor: pointer; transition: all 0.2s; }}
            .topic-btn:hover {{ background: #edf2f7; border-color: #cbd5e1; }}
            .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 15px; border-left: 4px solid #3b82f6; }}
            .card h4 {{ margin: 0 0 10px 0; color: #1e293b; font-size: 16px; }}
            .card p {{ margin: 0; color: #475569; font-size: 14px; line-height: 1.5; white-space: pre-line; }}
            .meta {{ margin-top: 10px; font-size: 12px; color: #94a3b8; font-weight: bold; }}
            .hidden {{ display: none; }}
        </style>
        <script>
            function showTopic(clusterId) {{
                document.querySelectorAll('.topic-group').forEach(el => el.classList.add('hidden'));
                document.getElementById('topic-' + clusterId).classList.remove('hidden');
            }}
        </script>
    </head>
    <body>
        <div class="sidebar">
            <h3>📌 Discovered Topics ({year})</h3>
            <button class="topic-btn" onclick="showTopic('noise')">⚠️ Unclustered Items</button>
    """

    for cluster in sorted(unique_clusters):
        keywords = df[df["cluster_id"] == cluster]["cluster_keywords"].values[0]
        html_content += f'<button class="topic-btn" onclick="showTopic(\'{cluster}\')">📁 Topic {cluster}: {keywords}</button>\n'
        
    html_content += '</div><div class="main-content">'
    
    for cluster in sorted(unique_clusters):
        keywords = df[df["cluster_id"] == cluster]["cluster_keywords"].values[0]
        html_content += f'<div id="topic-{cluster}" class="topic-group hidden"><h2>Topic Group {cluster}: {keywords}</h2>'
        cluster_df = df[df["cluster_id"] == cluster]
        for _, row in cluster_df.iterrows():
            html_content += f"""
            <div class="card">
                <h4>{row['title']}</h4>
                <p>{row['body_text']}</p>
                <div class="meta">📍 r/{row['subreddit']} | 💬 Comments: {row['comments']} | 👍 Upvotes: {int(row['upvote_ratio']*100)}%</div>
            </div>
            """
        html_content += '</div>'
        
    html_content += '<div id="topic-noise" class="topic-group"><h2>⚠️ Unclustered Posts</h2><p>These posts were filtered out because they did not form any recurring semantic patterns.</p>'
    noise_df = df[df["cluster_id"] == -1]
    for _, row in noise_df.head(50).iterrows():
        html_content += f"""
        <div class="card" style="border-left-color: #94a3b8;">
            <h4>{row['title']}</h4>
            <p>{row['body_text']}</p>
            <div class="meta">📍 r/{row['subreddit']}</div>
        </div>
        """
    html_content += '</div></div></body></html>'
    
    with open(html_path, "w", encoding="utf-8") as html_file:
        html_file.write(html_content)
        

    output_df = df.drop(columns=["body_fingerprint", "normalized_title", "clean_content"], errors="ignore")
    output_df.sort_values(by="cluster_id").to_csv(csv_path, index=False)
    
    noise_count = len(df[df["cluster_id"] == -1])
    print(f"Year {year} Cleaned! Unique Posts: {total_rows:<4} | Isolated: {len(unique_clusters)} Topics | Noise: {noise_count} posts.")

print("\n ALL CLUSTERS GENERATED!\n")
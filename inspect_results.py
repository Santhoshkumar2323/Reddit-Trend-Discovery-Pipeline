import os
import glob
import pandas as pd

print("\n INITIATING DATA AUDIT...")
print("=" * 80)

report_files = glob.glob("output_reports/report_*.csv")

if not report_files:
    print("❌ Error: No processed report files found in output_reports/.")
    print("👉 Run your clustering script (2_clusterer.py) first.")
    exit()

report_files.sort()

for target_file in report_files:
    filename = os.path.basename(target_file)
    print("\n" + "━" * 80)
    print(f"📂 REPORT FILE: {filename}")
    print("━" * 80)

    try:
        df = pd.read_csv(target_file)
        total_rows = len(df)

        print(" 1. DATASET STRUCTURAL AUDIT")
        print(f"   • Total High-Quality Posts : {total_rows} rows")
        print(f"   • Data Attributes Tracked  : {len(df.columns)} columns")
        
        if "created_year" in df.columns:
            print("\n CHRONOLOGICAL DISTRIBUTION:")
            year_counts = df["created_year"].value_counts()
            for yr, count in year_counts.items():
                print(f"   • Year {int(yr)} Data Volume     : {count} posts")
                
        if "subreddit" in df.columns:
            print("\n SUBREDDIT ENGAGEMENT BREAKDOWN (TOP 5):")
            sub_counts = df["subreddit"].value_counts().head(5)
            for sub, count in sub_counts.items():
                print(f"   • r/{sub:<24} : {count} posts saved")
                
        print("\n 2. TOPIC POPULARITY (Sorted by Volume)")
        print("-" * 80)
        print(f" {'CLUSTER ID':<12} | {'RANKED AI TOPIC KEYWORDS':<45} | {'VOL (POSTS)':<10}")
        print("-" * 80)

        cluster_summary = df.groupby("cluster_id").agg({
            "cluster_keywords": "first",
            "post_id": "count"
        }).rename(columns={"post_id": "count"}).sort_values(by="count", ascending=False)
        
        for cid, row in cluster_summary.iterrows():
            kw = row["cluster_keywords"]
            kw_clean = (kw[:42] + '...') if len(str(kw)) > 42 else str(kw)
            
            if cid == -1:
                print(f" {'-1 (Noise)':<12} | {kw_clean:<45} | {row['count']:<10}")
            else:
                print(f" Topic {cid:<7} | {kw_clean:<45} | {row['count']:<10}")
                

        print("-" * 80)
        print("\n 3.TOP VALIDATED TOPIC ENTRIES")
        
        valid_clusters = cluster_summary.index[cluster_summary.index != -1]
        
        if len(valid_clusters) > 0:
            top_topic_id = valid_clusters[0]
            top_keywords = cluster_summary.loc[top_topic_id, "cluster_keywords"]
            
            print(f"   [Displaying Sample Titles for Topic {top_topic_id}: {top_keywords}]")
            
            sample_df = df[df["cluster_id"] == top_topic_id].head(3)
            for idx, (_, row) in enumerate(sample_df.iterrows(), 1):
                title_preview = row["title"] if len(row["title"]) < 65 else row["title"][:62] + "..."
                print(f"   {idx}. \"{title_preview}\" (r/{row['subreddit']})")
        else:
            print("   No distinct topic clusters formed in this file yet.")
            
    except Exception as e:
        print(f" Analysis Failed for file {filename}: {str(e)}")

print("\n" + "=" * 80)
print("AUDIT LOG COMPLETE!\n")

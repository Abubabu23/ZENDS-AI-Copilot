# ============================================
# ZENDS Communications
# Exploratory Data Analysis (EDA)
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# ============================================
# Setup
# ============================================

# Plot style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

# Output path for saving plots
PLOT_PATH = "data/processed/eda_plots"
os.makedirs(PLOT_PATH, exist_ok=True)

# Load dataset
df = pd.read_csv("data/raw/ZenDS_Communications_queries.csv")

# Add message length column
df["message_length"] = df["text"].apply(len)

print("=" * 50)
print("ZENDS Communications — EDA Analysis")
print("=" * 50)
print(f"\n✅ Dataset loaded successfully!")
print(f"✅ Shape: {df.shape}")
print(f"✅ Columns: {list(df.columns)}")
print(f"\n📊 First 5 rows:")
print(df.head())


# ============================================
# Univariate Analysis
# ============================================

def univariate_analysis():
    print("\n" + "=" * 50)
    print("1. UNIVARIATE ANALYSIS")
    print("=" * 50)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Univariate Analysis — ZENDS Dataset", 
                 fontsize=16, fontweight="bold")

    # Plot 1: Intent Distribution
    intent_counts = df["intent"].value_counts()
    axes[0].bar(intent_counts.index, intent_counts.values, 
                color=sns.color_palette("husl", 5))
    axes[0].set_title("Intent Distribution")
    axes[0].set_xlabel("Intent")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=45)
    for i, v in enumerate(intent_counts.values):
        axes[0].text(i, v + 10, str(v), ha="center")

    # Plot 2: Sentiment Distribution
    sentiment_counts = df["sentiment"].value_counts()
    axes[1].bar(sentiment_counts.index, sentiment_counts.values,
                color=sns.color_palette("husl", 3))
    axes[1].set_title("Sentiment Distribution")
    axes[1].set_xlabel("Sentiment")
    axes[1].set_ylabel("Count")
    for i, v in enumerate(sentiment_counts.values):
        axes[1].text(i, v + 10, str(v), ha="center")

    # Plot 3: Message Length Distribution
    axes[2].hist(df["message_length"], bins=30, 
                 color="steelblue", edgecolor="white")
    axes[2].set_title("Message Length Distribution")
    axes[2].set_xlabel("Message Length (characters)")
    axes[2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(f"{PLOT_PATH}/univariate_analysis.png", 
                dpi=150, bbox_inches="tight")
    plt.show()

    # Print stats
    print(f"\n📊 Intent Distribution:\n{intent_counts}")
    print(f"\n📊 Sentiment Distribution:\n{sentiment_counts}")
    print(f"\n📊 Message Length Stats:")
    print(df["message_length"].describe())

    # ============================================
# Bivariate Analysis
# ============================================

def bivariate_analysis():
    print("\n" + "=" * 50)
    print("2. BIVARIATE ANALYSIS")
    print("=" * 50)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Bivariate Analysis — ZENDS Dataset",
                 fontsize=16, fontweight="bold")

    # Plot 1: Sentiment vs Intent (Grouped Bar)
    sentiment_intent = df.groupby(
        ["intent", "sentiment"]).size().unstack()
    sentiment_intent.plot(kind="bar", ax=axes[0],
                         colormap="tab10")
    axes[0].set_title("Sentiment vs Intent")
    axes[0].set_xlabel("Intent")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(title="Sentiment")

    # Plot 2: Message Length by Intent
    df.boxplot(column="message_length", by="intent",
               ax=axes[1])
    axes[1].set_title("Message Length by Intent")
    axes[1].set_xlabel("Intent")
    axes[1].set_ylabel("Message Length")
    axes[1].tick_params(axis="x", rotation=45)
    plt.sca(axes[1])
    plt.title("Message Length by Intent")

    # Plot 3: Refund Queries — Sentiment
    refund_df = df[df["intent"] == "Refund"]
    refund_sentiment = refund_df["sentiment"].value_counts()
    axes[2].pie(refund_sentiment.values,
                labels=refund_sentiment.index,
                autopct="%1.1f%%",
                colors=["#FF6B6B", "#4ECDC4", "#45B7D1"])
    axes[2].set_title("Refund Queries — Sentiment Split")

    plt.tight_layout()
    plt.savefig(f"{PLOT_PATH}/bivariate_analysis.png",
                dpi=150, bbox_inches="tight")
    plt.show()

    # Print stats
    print(f"\n📊 Sentiment vs Intent:\n{sentiment_intent}")
    print(f"\n📊 Refund Sentiment Split:\n{refund_sentiment}")
    # ============================================
# Multivariate Analysis
# ============================================

def multivariate_analysis():
    print("\n" + "=" * 50)
    print("3. MULTIVARIATE ANALYSIS")
    print("=" * 50)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Multivariate Analysis — ZENDS Dataset",
                 fontsize=16, fontweight="bold")

    # Plot 1: Message Length vs Intent vs Sentiment (Violin)
    sns.violinplot(data=df, x="intent", y="message_length",
                   hue="sentiment", ax=axes[0], 
                   palette="husl", split=False)
    axes[0].set_title("Message Length vs Intent vs Sentiment")
    axes[0].set_xlabel("Intent")
    axes[0].set_ylabel("Message Length")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(title="Sentiment", loc="upper right")

    # Plot 2: Heatmap — Avg Message Length per Intent/Sentiment
    heatmap_data = df.groupby(
        ["intent", "sentiment"])["message_length"].mean().unstack()
    sns.heatmap(heatmap_data, annot=True, fmt=".0f",
                cmap="YlOrRd", ax=axes[1])
    axes[1].set_title("Avg Message Length\nIntent vs Sentiment")
    axes[1].set_xlabel("Sentiment")
    axes[1].set_ylabel("Intent")

    plt.tight_layout()
    plt.savefig(f"{PLOT_PATH}/multivariate_analysis.png",
                dpi=150, bbox_inches="tight")
    plt.show()

    # Print stats
    print(f"\n📊 Avg Message Length per Intent/Sentiment:")
    print(heatmap_data)

    # ============================================
# Outlier Detection
# ============================================

def outlier_detection():
    print("\n" + "=" * 50)
    print("4. OUTLIER DETECTION")
    print("=" * 50)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Outlier Detection — ZENDS Dataset",
                 fontsize=16, fontweight="bold")

    # Calculate IQR
    Q1 = df["message_length"].quantile(0.25)
    Q3 = df["message_length"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = df[(df["message_length"] < lower_bound) |
                  (df["message_length"] > upper_bound)]

    # Plot 1: Boxplot
    axes[0].boxplot(df["message_length"], vert=True,
                    patch_artist=True,
                    boxprops=dict(facecolor="steelblue"))
    axes[0].set_title("Message Length Boxplot")
    axes[0].set_ylabel("Message Length")
    axes[0].axhline(y=upper_bound, color="red",
                    linestyle="--", label=f"Upper: {upper_bound:.0f}")
    axes[0].axhline(y=lower_bound, color="orange",
                    linestyle="--", label=f"Lower: {lower_bound:.0f}")
    axes[0].legend()

    # Plot 2: Outliers by Intent
    if len(outliers) > 0:
        outlier_intent = outliers["intent"].value_counts()
        axes[1].bar(outlier_intent.index, outlier_intent.values,
                    color=sns.color_palette("husl", 5))
        axes[1].set_title("Outliers by Intent")
        axes[1].set_xlabel("Intent")
        axes[1].set_ylabel("Outlier Count")
        axes[1].tick_params(axis="x", rotation=45)
        for i, v in enumerate(outlier_intent.values):
            axes[1].text(i, v + 0.5, str(v), ha="center")
    else:
        axes[1].text(0.5, 0.5, "No Outliers Found!",
                    ha="center", va="center",
                    fontsize=14, color="green")
        axes[1].set_title("Outliers by Intent")

    plt.tight_layout()
    plt.savefig(f"{PLOT_PATH}/outlier_detection.png",
                dpi=150, bbox_inches="tight")
    plt.show()

    # Print stats
    print(f"\n📊 IQR Analysis:")
    print(f"   Q1: {Q1:.0f}")
    print(f"   Q3: {Q3:.0f}")
    print(f"   IQR: {IQR:.0f}")
    print(f"   Lower Bound: {lower_bound:.0f}")
    print(f"   Upper Bound: {upper_bound:.0f}")
    print(f"\n⚠️ Total Outliers Found: {len(outliers)}")
    if len(outliers) > 0:
        print(f"\n📊 Outliers by Intent:\n{outliers['intent'].value_counts()}")
        print(f"\n📊 Sample Outliers:")
        print(outliers[["text", "intent", 
                        "sentiment", "message_length"]].head())
        
# ============================================
# Correlation Analysis
# ============================================

def correlation_analysis():
    print("\n" + "=" * 50)
    print("5. CORRELATION ANALYSIS")
    print("=" * 50)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Correlation Analysis — ZENDS Dataset",
                 fontsize=16, fontweight="bold")

    # Plot 1: Sentiment Frequency per Intent
    sentiment_intent = df.groupby(
        ["intent", "sentiment"]).size().unstack()
    sentiment_intent_pct = sentiment_intent.div(
        sentiment_intent.sum(axis=1), axis=0) * 100
    sentiment_intent_pct.plot(kind="bar", ax=axes[0],
                          colormap="tab10", stacked=True)
    axes[0].set_title("Sentiment % per Intent")
    axes[0].set_xlabel("Intent")
    axes[0].set_ylabel("Percentage %")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(title="Sentiment", loc="upper right")

    # Plot 2: High Severity Complaints
    complaint_df = df[df["intent"] == "Complaint"]
    angry_complaints = complaint_df[
        complaint_df["sentiment"] == "Angry"]
    
    # Message length distribution for angry complaints
    axes[1].hist(angry_complaints["message_length"], 
                 bins=20, color="red", 
                 alpha=0.7, label="Angry Complaints")
    axes[1].hist(complaint_df[
        complaint_df["sentiment"] != "Angry"]["message_length"],
        bins=20, color="steelblue",
        alpha=0.7, label="Other Complaints")
    axes[1].set_title("High Severity Complaints\nMessage Length")
    axes[1].set_xlabel("Message Length")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{PLOT_PATH}/correlation_analysis.png",
                dpi=150, bbox_inches="tight")
    plt.show()

    # Print stats
    print(f"\n📊 Sentiment % per Intent:")
    print(sentiment_intent_pct.round(2))
    print(f"\n📊 High Severity Complaints:")
    print(f"   Total Complaints: {len(complaint_df)}")
    print(f"   Angry Complaints: {len(angry_complaints)}")
    print(f"   Angry %: {len(angry_complaints)/len(complaint_df)*100:.1f}%")


    # ============================================
# Main Block
# ============================================

if __name__ == "__main__":
    
    print("=" * 50)
    print("ZENDS Communications")
    print("Exploratory Data Analysis")
    print("=" * 50)

    # Run all analysis
    univariate_analysis()
    bivariate_analysis()
    multivariate_analysis()
    outlier_detection()
    correlation_analysis()

    print("\n" + "=" * 50)
    print("✅ EDA Complete!")
    print(f"✅ Plots saved to: {PLOT_PATH}")
    print("=" * 50)
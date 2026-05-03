# ============================================
# ZENDS Communications
# Synthetic Dataset Generation
# ============================================

import pandas as pd
import numpy as np
import random
import json
import os
from tqdm import tqdm

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# Output path
OUTPUT_PATH = "data/raw/ZenDS_Communications_queries.csv"
TOTAL_RECORDS = 20000


# ============================================
# ZENDS Entity Dictionary
# ============================================

entities = {
    "products": [
        "Prepaid Basic", "Prepaid Plus", "Prepaid Unlimited",
        "Postpaid Silver", "Postpaid Gold", "Postpaid Platinum",
        "ZENDFiber Home 100Mbps", "ZENDFiber Home 300Mbps", "ZENDFiber Home 1Gbps",
        "ZENDOffice Net 200", "ZENDOffice Net 500", "ZENDOffice Net 1G",
        "ZENDBiz Connect 100", "ZENDBiz Connect 500", "ZENDBiz Connect 1G",
        "ZENDEnterprise Ultra", "ZENDEnterprise Dedicated",
        "ZENDCloud VM Basic", "ZENDCloud VM Pro", "ZENDCloud VM Enterprise",
        "ZENDStorage 1TB", "ZENDStorage 10TB", "ZENDArchive Storage",
        "ZENDSmart Traffic", "ZENDSmart Lighting", "ZENDSmart Parking",
        "ZENDIndustrial Sensor", "ZENDFleet IoT"
    ],
    "countries": ["India", "USA", "Singapore", "Thailand"],
    "amounts": ["$30", "$48", "$60", "$72", "$80", "$100", "$120", "$150", "$200"],
    "durations": ["3 days", "5 days", "1 week", "2 weeks", "1 month"],
    "ticket_ids": [f"TKT{random.randint(10000,99999)}" for _ in range(100)],
    "account_ids": [f"ACC{random.randint(10000,99999)}" for _ in range(100)],
}

# ============================================
# Intent Templates
# ============================================

intent_templates = {
    "Billing": [
        "Why is my bill so high this month for {product}?",
        "I was charged {amount} but my plan is only {amount}.",
        "Can you explain the charges on my {account_id} account?",
        "I did not receive my invoice for {product} this month.",
        "There is an incorrect charge of {amount} on my account.",
        "My auto-payment failed for {product} plan.",
        "I was double charged for {product} this month.",
        "When is my next billing date for {product}?",
        "I need a copy of my bill for {product}.",
        "Why was {amount} deducted without any notice?",
        "My bill shows {amount} but I should be charged less.",
        "Can I get a detailed breakdown of my {product} charges?",
        "I have not received my bill for the past {duration}.",
        "Please clarify the tax charges on my {product} plan.",
        "My enterprise account {account_id} has wrong billing details."
    ],

    "Refund": [
        "I want a refund for {product} as it is not working.",
        "I cancelled {product} within 7 days, please process my refund.",
        "I was charged {amount} wrongly, I need a refund.",
        "My refund for {product} has not been processed yet.",
        "It has been {duration} and I still have not received my refund.",
        "I need a refund for the unused portion of {product}.",
        "Please refund {amount} to my account immediately.",
        "I cancelled my {product} plan but no refund was given.",
        "My ticket {ticket_id} regarding refund has no update.",
        "Why is my refund taking more than {duration}?",
        "I am eligible for a refund as per ZENDS policy.",
        "Refund request for account {account_id} is still pending.",
        "I did not use {product} at all, please refund {amount}.",
        "My refund was rejected but I meet all the criteria.",
        "Please escalate my refund request for {product}."
    ],

    "Technical": [
        "My {product} internet is not working since {duration}.",
        "I am facing slow speed on {product} plan.",
        "My {product} connection keeps dropping every few minutes.",
        "I cannot connect to {product} service at all.",
        "My router for {product} is not responding.",
        "The {product} app is crashing on my device.",
        "I am unable to make calls on {product} plan.",
        "My {product} data speed is very slow.",
        "There is network outage in my area for {duration}.",
        "I cannot login to my ZENDS account since {duration}.",
        "My {product} eSIM is not activating.",
        "I am getting error while setting up {product}.",
        "My IoT device {product} is not sending data.",
        "Cloud service {product} is showing downtime.",
        "My {product} connection is unstable since {duration}."
    ],

    "Complaint": [
        "I am very unhappy with the service quality of {product}.",
        "Your support team was very rude to me.",
        "I have been waiting for {duration} with no resolution.",
        "My complaint ticket {ticket_id} has been ignored.",
        "This is the worst experience I have had with ZENDS.",
        "I was promised a resolution in {duration} but nothing happened.",
        "I want to escalate my complaint regarding {product}.",
        "ZENDS service has been very disappointing lately.",
        "I am planning to switch providers due to poor service.",
        "My issue with {product} has not been resolved since {duration}.",
        "I keep getting transferred between agents with no help.",
        "Nobody from ZENDS has followed up on my issue.",
        "I filed a complaint {duration} ago and got no response.",
        "Your SLA was breached for my account {account_id}.",
        "I am extremely frustrated with ZENDS customer support."
    ],

    "Product Inquiry": [
        "What are the features of {product} plan?",
        "How much does {product} cost in {country}?",
        "Can I upgrade from {product} to a higher plan?",
        "What is the difference between Postpaid Gold and Platinum?",
        "Does {product} support international roaming?",
        "What is the data limit for {product} plan?",
        "Can I get a bundle offer for mobile and broadband?",
        "What IoT solutions does ZENDS offer for smart cities?",
        "Tell me about ZENDCloud VM plans and pricing.",
        "What is included in the Enterprise Dedicated support?",
        "Does ZENDS offer any discount for annual payment?",
        "What is the SLA for {product} enterprise plan?",
        "Can I add family members to my {product} plan?",
        "What are the available plans in {country}?",
        "How do I migrate to {product} from my current plan?"
    ]
}

# ============================================
# Sentiment Modifiers
# ============================================

sentiment_modifiers = {
    "Angry": {
        "prefix": [
            "This is absolutely unacceptable! ",
            "I am extremely frustrated! ",
            "This is ridiculous! ",
            "I am very angry about this! ",
            "This is the worst service ever! ",
            "I cannot believe this! ",
            "I am fed up with ZENDS! ",
            "This is outrageous! ",
        ],
        "suffix": [
            " This needs to be fixed immediately!",
            " I demand an immediate resolution!",
            " I will escalate this issue!",
            " This is completely unacceptable!",
            " I am very disappointed with ZENDS!",
            " Fix this now or I will switch providers!",
            " I expect an urgent response!",
            " This has gone too far!",
        ]
    },

    "Neutral": {
        "prefix": [
            "I would like to know ",
            "Can you please help me with ",
            "I need assistance with ",
            "Kindly help me with ",
            "I have a query regarding ",
            "Please provide information about ",
            "I want to understand ",
            "Could you clarify ",
        ],
        "suffix": [
            " Please respond at your earliest convenience.",
            " Kindly look into this matter.",
            " Please provide an update.",
            " I would appreciate your help.",
            " Thank you for your assistance.",
            " Please let me know the details.",
            " Awaiting your response.",
            " Please guide me on this.",
        ]
    },

    "Happy": {
        "prefix": [
            "Hi, I love ZENDS service! ",
            "Great service so far! ",
            "I am happy with ZENDS! ",
            "Your service is amazing! ",
            "I really enjoy using ZENDS! ",
            "ZENDS has been wonderful! ",
            "Thank you ZENDS! ",
            "I am very satisfied! ",
        ],
        "suffix": [
            " Keep up the great work!",
            " ZENDS is the best provider!",
            " I will definitely recommend ZENDS!",
            " Very happy with the service!",
            " Thank you for the excellent support!",
            " Looking forward to more great service!",
            " ZENDS never disappoints!",
            " Truly satisfied customer here!",
        ]
    }
}
# ============================================
# Entity Injection Function
# ============================================

def inject_entities(template):
    """Replace placeholders with ZENDS entities"""
    text = template
    
    if "{product}" in text:
        text = text.replace("{product}", 
            random.choice(entities["products"]))
    
    if "{country}" in text:
        text = text.replace("{country}", 
            random.choice(entities["countries"]))
    
    if "{amount}" in text:
        text = text.replace("{amount}", 
            random.choice(entities["amounts"]))
    
    if "{duration}" in text:
        text = text.replace("{duration}", 
            random.choice(entities["durations"]))
    
    if "{ticket_id}" in text:
        text = text.replace("{ticket_id}", 
            random.choice(entities["ticket_ids"]))
    
    if "{account_id}" in text:
        text = text.replace("{account_id}", 
            random.choice(entities["account_ids"]))
    
    return text


# ============================================
# Sentiment Application Function
# ============================================

def apply_sentiment(text, sentiment):
    """Add sentiment prefix and suffix to text"""
    prefix = random.choice(
        sentiment_modifiers[sentiment]["prefix"])
    suffix = random.choice(
        sentiment_modifiers[sentiment]["suffix"])
    
    # Neutral - only sometimes add prefix/suffix
    if sentiment == "Neutral":
        choice = random.randint(1, 3)
        if choice == 1:
            return prefix + text
        elif choice == 2:
            return text + suffix
        else:
            return text
    
    # Angry and Happy - always add
    choice = random.randint(1, 2)
    if choice == 1:
        return prefix + text
    else:
        return text + suffix
    
    # ============================================
# Main Dataset Generation Function
# ============================================

def generate_dataset(total_records=20000):
    """Generate synthetic dataset for ZENDS Communications"""
    
    intents = ["Billing", "Refund", "Technical", 
               "Complaint", "Product Inquiry"]
    sentiments = ["Angry", "Neutral", "Happy"]
    
    # Records per intent (balanced)
    records_per_intent = total_records // len(intents)
    # 20000 / 5 = 4000 records per intent
    
    # Records per sentiment per intent (balanced)
    records_per_sentiment = records_per_intent // len(sentiments)
    # 4000 / 3 = 1333 records per sentiment
    
    data = []
    
    for intent in intents:
        print(f"\n Generating {intent} records...")
        
        for sentiment in sentiments:
            for _ in tqdm(range(records_per_sentiment), 
                         desc=f"{intent} - {sentiment}"):
                
                # Step 1: Pick random template
                template = random.choice(
                    intent_templates[intent])
                
                # Step 2: Inject entities
                text = inject_entities(template)
                
                # Step 3: Apply sentiment
                text = apply_sentiment(text, sentiment)
                
                # Step 4: Add to data
                data.append({
                    "text": text,
                    "intent": intent,
                    "sentiment": sentiment
                })
    
    # Shuffle dataset
    random.shuffle(data)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    print(f"\n✅ Total records generated: {len(df)}")
    print(f"✅ Intent distribution:\n{df['intent'].value_counts()}")
    print(f"✅ Sentiment distribution:\n{df['sentiment'].value_counts()}")
    
    return df


# ============================================
# Save Dataset Function
# ============================================

def save_dataset(df):
    """Save dataset to CSV"""
    
    # Create directory if not exists
    os.makedirs("data/raw", exist_ok=True)
    
    # Save to CSV
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Dataset saved to: {OUTPUT_PATH}")
    print(f"✅ Shape: {df.shape}")

    # ============================================
# Main Block
# ============================================

if __name__ == "__main__":
    
    print("=" * 50)
    print("ZENDS Communications")
    print("Synthetic Dataset Generation")
    print("=" * 50)
    
    # Step 1: Generate dataset
    print("\n🚀 Starting dataset generation...")
    df = generate_dataset(total_records=20000)
    
    # Step 2: Save dataset
    print("\n💾 Saving dataset...")
    save_dataset(df)
    
    # Step 3: Preview
    print("\n📊 Dataset Preview:")
    print(df.head(10))
    
    print("\n✅ Dataset Generation Complete!")
    print("=" * 50)
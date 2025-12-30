import streamlit as st
import requests
import numpy as np
from numpy.linalg import norm

# ===============================
# 1. CONFIGURATION & STYLING
# ===============================
LMSTUDIO_EMBEDDING_URL = "http://192.168.1.38:1234/v1/embeddings"
EMBEDDING_MODEL = "text-embedding-all-minilm-l6-v2"
SERPAPI_KEY = "651aa2bf909fc8b5ca95abcb6234237a4f7da352098cd8daf2f74bb77d5d436d" # Replace with your key

st.set_page_config(page_title="AI Shopping Partner", layout="wide")

# Custom CSS for impressive UI
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .product-card {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        height: 520px;
        margin-bottom: 20px;
        display: flex;
        flex-direction: column;
    }
    .img-container {
        text-align: center;
        height: 180px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .img-container img { max-height: 100%; max-width: 100%; object-fit: contain; }
    .product-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #2c3e50;
        margin-top: 15px;
        height: 3em;
        overflow: hidden;
    }
    .price-tag {
        font-size: 1.5rem;
        color: #27ae60;
        font-weight: 800;
        margin: 10px 0;
    }
    .match-badge {
        background: #e1f5fe;
        color: #01579b;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .buy-btn {
        display: block;
        text-align: center;
        background-color: #ff4b4b;
        color: white !important;
        text-decoration: none;
        padding: 10px;
        border-radius: 8px;
        font-weight: bold;
        margin-top: 10px;
    }
    .buy-btn:hover { background-color: #d43f3f; }
</style>
""", unsafe_allow_html=True)

# ===============================
# 2. CORE LOGIC FUNCTIONS
# ===============================

@st.cache_data
def get_embedding(text):
    try:
        payload = {"model": EMBEDDING_MODEL, "input": text}
        response = requests.post(LMSTUDIO_EMBEDDING_URL, json=payload, timeout=5)
        return np.array(response.json()["data"][0]["embedding"])
    except:
        return np.zeros(384)

def cosine_similarity(a, b):
    if np.all(a == 0) or np.all(b == 0): return 0
    return np.dot(a, b) / (norm(a) * norm(b))

@st.cache_data
def search_google_products(query):
    params = {
        "engine": "google",
        "q": query + " buy online",
        "tbm": "shop",
        "api_key": SERPAPI_KEY
    }
    try:
        response = requests.get("https://serpapi.com/search.json", params=params)
        results = response.json().get("shopping_results", [])
        products = []
        for r in results:
            price_clean = r.get("price", "0").replace("$", "").replace(",", "")
            products.append({
                "name": r.get("title", "No Name"),
                "price": price_clean,
                "image": r.get("thumbnail", ""),
                "link": r.get("link", "#"),
                "description": r.get("snippet", "Detailed specs available at store.")
            })
        return products
    except:
        return []

# ===============================
# 3. APP UI & INTERACTION
# ===============================

# Session State for Comparison
if "compare_list" not in st.session_state:
    st.session_state.compare_list = []

# Sidebar Comparison
with st.sidebar:
    st.header("⚖️ Product Compare")
    if st.session_state.compare_list:
        if st.button("Clear All"):
            st.session_state.compare_list = []
            st.rerun()
        for item in st.session_state.compare_list:
            with st.expander(f"📦 {item['name'][:30]}..."):
                st.write(f"**Price:** ${item['price']}")
                st.write(f"**Relevance:** {item['score']}")
    else:
        st.info("Check boxes on products to compare them.")

# Main Screen
st.title("🛒 Your Personal AI Shopper")
st.write("Find the best products....")

user_input = st.text_input("Describe what you are looking for:", placeholder="e.g. A durable smartphone for photography under $800")

if user_input:
    with st.spinner("Analyzing products..."):
        raw_products = search_google_products(user_input)
        
        if raw_products:
            query_emb = get_embedding(user_input)
            scored_products = []

            for p in raw_products:
                content = f"{p['name']} {p['description']}"
                item_emb = get_embedding(content)
                score = cosine_similarity(query_emb, item_emb)
                scored_products.append((score, p))

            # Sort by score
            scored_products.sort(reverse=True, key=lambda x: x[0])

            # Grid Display
            cols = st.columns(3)
            for idx, (score, p) in enumerate(scored_products[:9]):
                with cols[idx % 3]:
                    # Format prices and scores
                    display_price = p['price']
                    match_percent = f"{int(score * 100)}%"
                    
                    # RENDER PRODUCT CARD
                    st.markdown(f"""
                    <div class="product-card">
                        <div class="img-container">
                            <img src="{p['image']}">
                        </div>
                        <div style="margin-top:10px;">
                            <span class="match-badge">Match: {match_percent}</span>
                            <div class="product-title">{p['name']}</div>
                            <div class="price-tag">${display_price}</div>
                            <p style="font-size:0.85rem; color:#555;">{p['description'][:100]}...</p>
                        </div>
                        <div style="margin-top:auto;">
                            <a class="buy-btn" href="{p['link']}" target="_blank">View on Store</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # SELECTION CHECKBOX (Underneath the card)
                    if st.checkbox("Select to compare", key=f"check_{idx}"):
                        item = {"name": p['name'], "price": display_price, "score": match_percent}
                        if item not in st.session_state.compare_list:
                            if len(st.session_state.compare_list) < 3:
                                st.session_state.compare_list.append(item)
                                st.rerun()
                            else:
                                st.warning("Limit: 3 products.")
        else:
            st.error("No results found. Please try a different search.")
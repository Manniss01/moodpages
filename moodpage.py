import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

# Load books data
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"].fillna("cover_not_found.png") + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["thumbnail"].isna(),
    "cover_not_found.png",
    books["large_thumbnail"],
)

# Load and split text data
raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(raw_documents)

# Create or load persistent vector DB
persist_dir = "chroma_db"
if os.path.exists(persist_dir):
    db_books = Chroma(persist_directory=persist_dir, embedding_function=OpenAIEmbeddings())
else:
    db_books = Chroma.from_documents(documents, OpenAIEmbeddings(), persist_directory=persist_dir)

# Recommendation engine
def retrieve_semantic_recommendations(query: str, category: str = None, tone: str = None,
                                      initial_top_k: int = 50, final_top_k: int = 16) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs if rec.page_content.strip('"').split()[0].isdigit()]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category and category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)
    # else: All tones, no sorting

    return book_recs

# Output formatting as modern cards grid with clickable cards if URL exists
def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    if recommendations.empty:
        return """
        <p style='text-align:center; font-style:italic; color:#c0c8f7;'>
            Sorry, no books found matching your criteria. Please try a different query or adjust your filters.
        </p>
        """

    cards_html = []
    for _, row in recommendations.iterrows():
        description = row.get("description", "No description available.")
        truncated_description = " ".join(description.split()[:25]) + ("..." if len(description.split()) > 25 else "")

        authors_field = row.get("authors", "Unknown Author")
        authors_split = [a.strip() for a in authors_field.split(";") if a.strip()]
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = authors_split[0] if authors_split else "Unknown Author"

        title_text = row.get("title", "Untitled")
        alt_text = f"Cover of {title_text} by {authors_str}"

        # Use a link if URL exists, else just div
        has_url = "url" in row and pd.notna(row["url"]) and row["url"].strip() != ""
        if has_url:
            link_start = f'<a href="{row["url"]}" target="_blank" rel="noopener" class="book-card-link" aria-label="Open details for {title_text}">'
            link_end = "</a>"
        else:
            link_start = f'<div class="book-card" tabindex="0" role="article" aria-label="Book titled {title_text} by {authors_str}">'
            link_end = "</div>"

        card_html = f"""
        {link_start}
            <div class="book-card-inner">
                <img src="{row['large_thumbnail']}" alt="{alt_text}" class="cover-img" loading="lazy"/>
                <div class="card-content">
                    <h3 class="book-title">{title_text}</h3>
                    <p class="book-authors">{authors_str}</p>
                    <p class="book-desc">{truncated_description}</p>
                </div>
            </div>
        {link_end}
        """
        cards_html.append(card_html)

    return f'<div class="cards-container">{"".join(cards_html)}</div>'

# Dropdown values
categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

custom_css = """
body, .gradio-container {
    background: linear-gradient(135deg, #0b194a 0%, #152a73 100%);
    color: #f0f4ff;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 0 30px 40px 30px;
}

#main-title {
    text-align: center;
    margin-bottom: 30px;
    padding: 0 15px;
    max-width: 700px;
    margin-left: auto;
    margin-right: auto;
}

#main-title h1 {
    font-weight: 700;
    font-size: 2rem;
    margin-bottom: 8px;
    color: #ffffff;
    letter-spacing: 0.8px;
    user-select: none;
}

#main-title p {
    font-size: 1rem;
    color: #d0d4db;
    margin-top: 0;
    margin-bottom: 0;
    font-weight: 400;
    font-style: normal;
    user-select: none;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
}

/* Align input controls nicely */
.gr-row {
    justify-content: center !important;
    gap: 20px;
    margin-bottom: 25px;
}

/* Make inputs consistent width */
.gr-textbox, .gr-dropdown {
    max-width: 450px;
    min-width: 350px;
    font-size: 1rem;
}

/* Center and style button */
.gr-button {
    background-color: #164a9f !important;
    color: white !important;
    font-weight: 700;
    font-size: 1.1rem;
    border-radius: 14px !important;
    border: none !important;
    padding: 12px 36px !important;
    cursor: pointer;
    margin: 0 auto;
    display: block;
    transition: background-color 0.3s ease;
}

.gr-button:hover:not(:disabled) {
    background-color: #2a5fc1 !important;
}

.gr-button:disabled {
    background-color: #2a3f86 !important;
    cursor: not-allowed;
}

/* Cards container with fixed height and scroll */
.cards-container {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 18px;
    max-height: 620px;
    overflow-y: auto;
    padding: 8px;
    user-select: none;
}

/* Individual card styling */
.book-card, .book-card-link {
    display: block;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(15, 23, 55, 0.2);
    background: #18327e;
    color: #e5e9f7;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    text-decoration: none;
}

.book-card:focus, .book-card-link:focus {
    outline: 2px solid #9fbbff;
    outline-offset: 3px;
    box-shadow: 0 4px 15px rgba(26, 115, 255, 0.7);
}

.book-card:hover, .book-card-link:hover {
    transform: translateY(-6px);
    box-shadow: 0 6px 20px rgba(26, 115, 255, 0.9);
}

.book-card-inner {
    padding: 14px;
    display: flex;
    flex-direction: column;
    height: 100%;
}

.cover-img {
    width: 100%;
    height: 280px;
    border-radius: 10px;
    object-fit: cover;
    margin-bottom: 12px;
}

.card-content {
    flex-grow: 1;
}

.book-title {
    font-weight: 700;
    font-size: 1.15rem;
    margin: 0 0 5px 0;
}

.book-authors {
    font-weight: 600;
    font-size: 0.9rem;
    color: #a0b4ff;
    margin: 0 0 8px 0;
}

.book-desc {
    font-size: 0.85rem;
    color: #d0d4db;
    margin: 0;
}
"""

# Build Gradio UI
with gr.Blocks(css=custom_css, title="MoodPages: Semantic Book Recommendations") as moodpage:
    gr.HTML(
        """
        <section id="main-title">
            <h1>MoodPages 📚</h1>
            <p>Discover books through semantic search powered by AI — filter results by genre and emotional tone to match your interests and mood.</p>
        </section>
        """
    )
    with gr.Row():
        query_input = gr.Textbox(
            label="Search for a book by description...",
            placeholder="Enter description, topic, or mood...",
            lines=1,
            max_lines=1,
        )
        category_dropdown = gr.Dropdown(choices=categories, value="All", label="Category Filter")
        tone_dropdown = gr.Dropdown(choices=tones, value="All", label="Emotion/Tone Filter")
        submit_btn = gr.Button("Get Recommendations")

    output_html = gr.HTML()

    submit_btn.click(
        fn=recommend_books,
        inputs=[query_input, category_dropdown, tone_dropdown],
        outputs=output_html,
        api_name="recommend_books"
    )

    # Footer row
    with gr.Row():
        gr.Markdown(
            """
            <footer style="text-align:center; margin-top:40px; font-size:0.9rem; color:#7a8dbd; user-select:none;">
                Developed by <strong>Unicodax</strong>
            </footer>
            """
        )

if __name__ == "__main__":
    moodpage.launch()

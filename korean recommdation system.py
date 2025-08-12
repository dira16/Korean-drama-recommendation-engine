import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
import sys
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def setup_logging():
    """Sets up a logger for the application."""
    log = logging.getLogger('K-DramaRecommender')
    log.setLevel(logging.INFO)
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return log

logger = setup_logging()

class KnowledgeBasedRecommender:
    """A recommender system based on user-defined rules and filters."""
    def __init__(self, df):
        logger.info("Initializing Knowledge-Based Recommender.")
        self.df = df

    def recommend(self, genres=[], min_rating=5.0, top_n=10):
        """Recommends K-Dramas based on selected genres and minimum rating."""
        if not genres:
            return pd.DataFrame()
        
        filtered_df = self.df[self.df['rating'] >= min_rating].copy()
        genre_filter_regex = '|'.join(genres)
        filtered_df = filtered_df[filtered_df['genres'].str.contains(genre_filter_regex, case=False, na=False)]
        sorted_df = filtered_df.sort_values(by='rating', ascending=False)
        return sorted_df.head(top_n)

class ContentBasedRecommender:
    """A recommender system that explains its choices based on content features."""
    def __init__(self, df):
        logger.info("Initializing Content-Based Recommender.")
        self.df = df.reset_index(drop=True)
        self.indices = pd.Series(self.df.index, index=self.df['name']).drop_duplicates()
        self.tfidf_matrix, self.vectorizer = self._compute_similarity_matrix()
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def _compute_similarity_matrix(self):
        """Computes the TF-IDF matrix and vectorizer."""
        features = self.df["genres"].fillna('') + " " + self.df["tags"].fillna('') + " " + self.df["country"].fillna('') + " " + self.df["main_role"].fillna('') + " " + self.df["support_role"].fillna('')
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(features)
        return tfidf_matrix, vectorizer

    def recommend(self, title, top_n=5):
        """Generates recommendations with explanations."""
        if title not in self.indices:
            return []
        
        idx = self.indices[title]
        sim_scores = sorted(list(enumerate(self.cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:top_n+1]
        
        recommendations = []
        source_vector = self.tfidf_matrix[idx]
        feature_names = self.vectorizer.get_feature_names_out()

        for i, score in sim_scores:
            rec_df_row = self.df.iloc[i]
            target_vector = self.tfidf_matrix[i]
            
            common_features_indices = source_vector.multiply(target_vector).nonzero()[1]
            top_common_features = sorted(
                [(feature_names[j], target_vector[0, j]) for j in common_features_indices],
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            explanation = "it shares features like " + ", ".join([f"'{f[0]}'" for f in top_common_features]) + "." if top_common_features else "it has similar overall themes."
            recommendations.append({"name": rec_df_row["name"], "rating": rec_df_row["rating"], "genres": rec_df_row["genres"], "explanation": explanation})
        return recommendations

def load_and_prepare_data(filepath="kdrama.csv"):
    """Loads, preprocesses, and adds simulated data."""
    logger.info(f"Loading and preprocessing data from {filepath}")
    df = pd.read_csv(filepath)
    df.dropna(subset=["name"], inplace=True)
    df.drop_duplicates(subset=["name"], inplace=True)
    
    for col in ["genres", "tags", "country", "main_role", "support_role"]:
        df[col] = df[col].fillna('').astype(str)
    
    np.random.seed(42)
    df['rating'] = np.random.uniform(6.0, 10.0, df.shape[0]).round(1)
    
    all_genres = set()
    df['genres'].str.split(',').apply(lambda genres: [all_genres.add(g.strip()) for g in genres if g.strip()])
    unique_genres = sorted(list(all_genres))
    
    return df, unique_genres

def create_dashboard_plots(df):
    """Creates a set of plots for the dashboard based on a given dataframe."""
    logger.info(f"Generating dashboard plots for a dataframe with {len(df)} entries.")
    
    if df.empty:
        # Return None for all plots if there's no data, to avoid errors
        return None, None, None, None

    # Plot 1: Rating Distribution
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.histplot(df['rating'], bins=15, kde=True, ax=ax1, color='skyblue')
    ax1.set_title('Distribution of Ratings in Selection')
    ax1.set_xlabel('Rating')
    ax1.set_ylabel('Count')
    plt.tight_layout()

    # Plot 2: Top Genres in Selection
    genre_counts = df['genres'].str.split(',').explode().str.strip().value_counts()
    top_genres = genre_counts.nlargest(10)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_genres.values, y=top_genres.index, ax=ax2, palette='magma')
    ax2.set_title('Top 10 Most Common Genres in Selection')
    ax2.set_xlabel('Count')
    plt.tight_layout()

    # Plot 3: Word Cloud of Tags
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    tags_text = ' '.join(df['tags'].dropna())
    if tags_text:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(tags_text)
        ax3.imshow(wordcloud, interpolation='bilinear')
        ax3.set_title('Most Common Themes & Tags')
        ax3.axis('off')
    else:
        ax3.text(0.5, 0.5, 'No tags available for this selection', ha='center', va='center')
        ax3.axis('off')
    plt.tight_layout()
    
    # Plot 4: Top Countries by Production
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    country_counts = df['country'].value_counts().nlargest(10)
    sns.barplot(x=country_counts.values, y=country_counts.index, ax=ax4, palette='viridis')
    ax4.set_title('Top 10 Countries by Production in Selection')
    ax4.set_xlabel('Number of Dramas')
    plt.tight_layout()

    return fig1, fig2, fig3, fig4

def launch_ui(content_recommender, knowledge_recommender, full_df, all_drama_titles, unique_genres):
    """Launches a fully connected, multi-tab Gradio UI."""

    def get_content_recommendations(drama_title):
        if not drama_title: return gr.update(visible=False)
        recs = content_recommender.recommend(drama_title)
        if not recs: return gr.update(visible=False)
            
        output_html = f"### Because you liked **{drama_title}**, you might also enjoy...\n"
        for rec in recs:
            output_html += f"<div style='border:1px solid #eee;padding:10px;border-radius:8px;margin-bottom:10px;'><h4>{rec['name']}</h4><p><strong>Rating:</strong> {rec['rating']} | <strong>Genres:</strong> {rec['genres']}</p><p style='color:#555;'><strong>💡 Why:</strong> {rec['explanation']}</p></div>"
        return gr.update(value=output_html, visible=True)

    def get_knowledge_recommendations(genres, min_rating):
        if not genres: return gr.update(visible=False), "Please select at least one genre."
        recs_df = knowledge_recommender.recommend(genres=genres, min_rating=min_rating)
        
        if recs_df.empty: return gr.update(visible=False), "No dramas found matching your criteria!"
            
        output_html = f"### Showing dramas with a rating of **{min_rating}+** in genres: **{', '.join(genres)}**\n"
        for _, row in recs_df.iterrows():
            output_html += f"<div style='border:1px solid #eee;padding:10px;border-radius:8px;margin-bottom:10px;'><h4>{row['name']}</h4><p><strong>Rating:</strong> {row['rating']} | <strong>Genres:</strong> {row['genres']}</p></div>"
        return gr.update(value=output_html, visible=True), ""

    def update_dashboard(genres, min_rating):
        """This function is triggered when the dashboard tab is selected."""
        logger.info("Updating dashboard based on new filter selection.")
        # If no genres are selected, use the full dataset for the dashboard
        if not genres:
            filtered_df = full_df[full_df['rating'] >= min_rating]
        else:
            filtered_df = knowledge_recommender.recommend(genres, min_rating, top_n=len(full_df))
        
        plot1, plot2, plot3, plot4 = create_dashboard_plots(filtered_df)
        
        message = f"Showing plots for **{len(filtered_df)} dramas** matching your filters."
        if filtered_df.empty:
            message = "No data matches your current filters. Dashboard is empty."

        return plot1, plot2, plot3, plot4, message

    # --- UI Layout ---
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="pink", secondary_hue="blue")) as interface:
        gr.Markdown("<div style='text-align:center;font-family:\"Garamond\",serif;'><h1>🌸 K-Drama Recommendation & Insights Hub 🌸</h1></div>")

        with gr.Tabs():
            # --- Tab 1: Content-Based ---
            with gr.TabItem("Recommend by Similar Drama"):
                gr.Markdown("### Find shows similar to one you already love!")
                content_input = gr.Dropdown(choices=all_drama_titles, label="Select a K-Drama", info="Start typing to search for a drama...")
                content_button = gr.Button("Find Similar Dramas", variant="primary")
                content_output = gr.Markdown(visible=False)

            # --- Tab 2: Knowledge-Based ---
            with gr.TabItem("Filter by Preference"):
                gr.Markdown("### Discover new dramas based on your taste.")
                with gr.Row():
                    knowledge_genres_input = gr.CheckboxGroup(choices=unique_genres, label="Select Genre(s)")
                    knowledge_rating_input = gr.Slider(minimum=6.0, maximum=10.0, step=0.1, value=7.0, label="Minimum Rating")
                knowledge_button = gr.Button("Find My K-Drama", variant="primary")
                knowledge_message = gr.Markdown()
                knowledge_output = gr.Markdown(visible=False)

            # --- Tab 3: Interactive Dashboard ---
            with gr.TabItem("Interactive Dashboard") as dashboard_tab:
                dashboard_message = gr.Markdown("### Dashboard reflecting filters from the 'Filter by Preference' tab. Select that tab to change filters.")
                with gr.Row():
                    plot_output_1 = gr.Plot()
                    plot_output_2 = gr.Plot()
                with gr.Row():
                    plot_output_3 = gr.Plot()
                    plot_output_4 = gr.Plot()

            # --- Tab 4: Documentation ---
            with gr.TabItem("Documentation"):
                gr.Markdown("""
                ##  How to Use This App

                Welcome to the K-Drama Recommendation and Insights Hub! This guide will help you navigate the app's features.

                ### **Recommend by Similar Drama** Tab
                - **What it does:** This feature uses a *Content-Based Recommender*. It analyzes the genres, tags, actors, and country of the drama you select and finds other dramas with the most similar profiles.
                - **How to use it:**
                    1. Select a K-Drama you like from the dropdown menu.
                    2. Click the "Find Similar Dramas" button.
                    3. The app will show you 5 recommendations, along with an explanation of why each one was chosen.

                ### **Filter by Preference** Tab
                - **What it does:** This feature uses a *Knowledge-Based Recommender*. It filters the entire dataset based on the criteria you provide.
                - **How to use it:**
                    1. Check one or more genres you are interested in.
                    2. Use the slider to set the minimum rating you're willing to accept.
                    3. Click the "Find My K-Drama" button to see a list of dramas that match your preferences, sorted by the highest rating.

                ### **Interactive Dashboard** Tab
                - **What it does:** This tab provides a visual analysis of the K-Dramas that match the filters set in the **"Filter by Preference"** tab.
                - **How to use it:**
                    1. First, go to the "Filter by Preference" tab and select your desired genres and minimum rating.
                    2. Click on the "Interactive Dashboard" tab.
                    3. The plots will automatically update to reflect your current selection. If you haven't selected any genres, the dashboard will show data for all dramas above the selected minimum rating.
                - **The plots include:**
                    - **Distribution of Ratings:** A histogram showing the spread of ratings for the selected dramas.
                    - **Top 10 Genres:** A bar chart of the most common genres in your selection.
                    - **Most Common Themes & Tags:** A word cloud visualizing prominent tags and themes.
                    - **Top 10 Countries by Production:** A bar chart showing where the selected dramas were produced.

                ### **About the Data**
                The recommendations are based on a dataset of K-Dramas (`kdrama.csv`). The ratings in this app have been simulated for demonstration purposes.
                """)

        # --- Event Wiring ---
        content_button.click(fn=get_content_recommendations, inputs=[content_input], outputs=[content_output])
        knowledge_button.click(fn=get_knowledge_recommendations, inputs=[knowledge_genres_input, knowledge_rating_input], outputs=[knowledge_output, knowledge_message])
        
        # This is the key connection: when the dashboard tab is selected, update the plots
        dashboard_tab.select(
            fn=update_dashboard,
            inputs=[knowledge_genres_input, knowledge_rating_input],
            outputs=[plot_output_1, plot_output_2, plot_output_3, plot_output_4, dashboard_message]
        )

    logger.info("Launching Gradio UI.")
    interface.launch(share=True) # Share=True enables the share link

if __name__ == "__main__":
    # Ensure you have the 'kdrama.csv' file in the same directory as this script.
    try:
        kdrama_df, unique_genres = load_and_prepare_data()
        all_drama_titles = sorted(kdrama_df['name'].unique().tolist())
        
        content_model = ContentBasedRecommender(kdrama_df)
        knowledge_model = KnowledgeBasedRecommender(kdrama_df)
        
        print("\n--- Testing Content-Based Model ---")
        print(pd.DataFrame(content_model.recommend("Crash Landing on You")))
        print("\n")

        launch_ui(content_model, knowledge_model, kdrama_df, all_drama_titles, unique_genres)
    except FileNotFoundError:
        print("Error: 'kdrama.csv' not found. Please make sure the dataset file is in the correct directory.")

from flask import Flask, render_template, jsonify, send_from_directory
import json
import os

app = Flask(__name__)
here = os.path.dirname(os.path.abspath(__file__))

mapper_path = os.path.join(here, 'app_data/article_id_folder_mapper.json')
article_id_image_folder_mapper = json.load(open(mapper_path))
doc_source_data = json.load(open(os.path.join(here, 'app_data/doc_source_data.json')))
# load all article data onces 
all_article_data = {}
with open(os.path.join(here, 'app_data/science_articles_full.jsonl'), 'r') as f:
    for line in f:
        line = json.loads(line.strip())
        id = line.get('id')
        if str(id) in doc_source_data:
            all_article_data[id] = line
# all_article_ids_and_headlines = {k: f"{v['headline']}, {str(v['date'])}" for k, v in all_article_data.items()}
all_article_ids_and_headlines = {k: f"{v['headline']}" for k, v in all_article_data.items()}

def load_article_data(article_id=None):
    """Load article data for a specific ID."""
    with open(os.path.join(here, 'app_data/science_article_sample.jsonl'), 'r') as f:
        to_return = None
        all_ids_and_headlines = {}
        for line in f:
            article = json.loads(line)
            all_ids_and_headlines[article.get('id')] = article.get('headline')
            if article.get('id') == article_id:
                to_return = article
    return to_return, all_ids_and_headlines

def load_image_captions(article_id):
    """Load image captions for a specific article ID."""
    image_captions = []
    with open(os.path.join(here, 'app_data/image_descriptions.jsonl'), 'r') as f:
        for line in f:
            item = json.loads(line)
            if item.get('id') == article_id:
                image_captions.append(item)
    return image_captions

def get_image_type(image_id, image_dir):
    candidate = image_id.replace('.pdf', '.png')
    if os.path.exists(os.path.join(image_dir, candidate)):
        return image_id, candidate
    candidate = image_id.replace('.pdf', '.jpg')
    if os.path.exists(os.path.join(image_dir, candidate)):
        return image_id, candidate
    return image_id, image_id

def get_image_paths(article_id, image_captions):
    """Generate image paths based on article ID and captions."""
    image_paths = {}
    # Extract article headline to find folder name
    article, _ = load_article_data(article_id)
    if article is None:
        return {}
    
    # Convert headline to folder-friendly format (simplified version)
    folder_path = article_id_image_folder_mapper[str(article_id)]
    image_dir = os.path.join(here, 'app_data', 'science_article_figures', folder_path)
    for caption in image_captions:
        image_id, image_filename = get_image_type(caption.get('image_id', ''), image_dir)
        # image_paths[image_id] = f"/images/science_article_figures/{folder_path}/{image_filename}"
        image_paths[image_id] = f"/images/{folder_path}/{image_filename}"
    
    return image_paths


def get_diff_data(article_id):
    diff_summary = diff_details= None 
    with open(os.path.join(here, 'app_data/diffs_summaries.jsonl'), 'r') as f:
        for line in f:
            summary = json.loads(line)
            if summary.get('id') == article_id:
                diff_summary = summary['diff_summary']
    with open(os.path.join(here, 'app_data/diffs_specific_categories.jsonl'), 'r') as f:
        for line in f:
            details = json.loads(line)
            if details.get('id') == article_id:
                diff_details = details['output']
    return diff_summary, diff_details


def get_doc_source_data(article_id):
    # sort by 'Centrality': 'High' >  'Medium' > 'Low'
    centrality_order = {'High': 0, 'Medium': 1, 'Low': 2}
    doc_source_data_for_article = doc_source_data[str(article_id)]
    doc_source_data_for_article = sorted(doc_source_data_for_article, key=lambda x: centrality_order[x['Centrality']])
    return doc_source_data_for_article


def calculate_source_statistics(doc_source_data_for_article):
    """Calculate statistics about sources for the fact box."""
    # Initialize counters
    high_centrality = 0
    medium_centrality = 0
    low_centrality = 0
    
    perspectives = {}
    spoken_yes = 0
    spoken_no = 0
    
    # Count sources by centrality
    for source in doc_source_data_for_article:
        if source.get('Centrality'):
            if source['Centrality'].startswith('High'):
                high_centrality += 1
            elif source['Centrality'].startswith('Medium'):
                medium_centrality += 1
            elif source['Centrality'].startswith('Low'):
                low_centrality += 1
        
        # Count sources by perspective
        if source.get('Perspective'):
            if isinstance(source['Perspective'], str):
                perspective = source['Perspective']
                perspectives[perspective] = perspectives.get(perspective, 0) + 1
            else:
                for perspective in source['Perspective']:
                    perspectives[perspective] = perspectives.get(perspective, 0) + 1
        
        # Count sources by spoken_to
        if source.get('Spoken_to') == "Yes":
            spoken_yes += 1
        elif source.get('Spoken_to') == "No":
            spoken_no += 1
    
    return {
        'centrality': {
            'high': high_centrality,
            'medium': medium_centrality,
            'low': low_centrality
        },
        'perspectives': perspectives,
        'spoken_to': {
            'yes': spoken_yes,
            'no': spoken_no
        }
    }


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/news_summary_tree')
def news_summary_tree():
    # Load the data from the JSON file
    with open('app_data/tree_data.json', 'r') as f:
        tree_data = json.load(f)
    
    return render_template(
        'tree_template_horizontal.html',
        config={
            'margin': {'top': 20, 'right': 290, 'bottom': 30, 'left': 190},
            'width': 1200,
            'height': 500
        },
        data=tree_data
    )

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(os.path.join(here, 'app_data', 'science_article_figures'), filename)

@app.route('/dig_deeper/')
@app.route('/dig_deeper/<int:article_id>')
def dig_deeper(article_id=None):
    # If article_id is provided, show the specific article
    article_data, ids_and_headlines = load_article_data(article_id)
    if article_id is None:
        # If no article_id is provided, show the list of available articles
        with open(os.path.join(here, 'app_data/science_article_sample.jsonl'), 'r') as f:
            articles = []
            for line in f:
                article = json.loads(line)
                articles.append({
                    'id': article.get('id'),
                    'title': article.get('title', 'No Title'),
                    'summary': article.get('summary', 'No Summary')
                })
            return render_template(
                'dig_deeper_article_viewer.html', 
                ids_and_headlines=ids_and_headlines,
                article_data=article_data,
            )
    
    if article_data is None:
        return jsonify({"error": "Article not found"}), 404

    diff_summary, diff_details = get_diff_data(article_id)

    image_captions = load_image_captions(article_id)
    image_paths = get_image_paths(article_id, image_captions)
    
    print(image_paths)
    # Prepare data for the template
    images_with_captions = []
    for i, caption_data in enumerate(image_captions):
        caption = caption_data.get('output', {}).get('Caption', '').strip()
        significance = caption_data.get('output', {}).get('Significance', '').strip()
        fun_fact = caption_data.get('output', {}).get('Fun fact', '').strip()
        image_path = image_paths[caption_data.get('image_id')]
        images_with_captions.append({
            'image_path': image_path,
            'caption': caption,
            'significance': significance,
            'fun_fact': fun_fact
        })
    
    return render_template(
        'dig_deeper_article_viewer.html', 
        article_data=article_data,
        ids_and_headlines=ids_and_headlines,
        images=images_with_captions,
        diff_summary=diff_summary,
        diff_details=diff_details
    )


@app.route('/explore_reporting/doc_source_data/<int:article_id>')
def explore_reporting_doc_source_data(article_id):
    article_data, all_ids_and_headlines = all_article_data[article_id], all_article_ids_and_headlines
    print(all_ids_and_headlines)
    doc_source_data_for_article = get_doc_source_data(article_id)
    
    # Calculate source statistics
    source_stats = calculate_source_statistics(doc_source_data_for_article)
    
    return render_template(
        'explore_reporting_doc_source_data.html',
        article_data=article_data,
        all_ids_and_headlines=all_ids_and_headlines,
        doc_source_data_for_article=doc_source_data_for_article,
        source_stats=source_stats
    )



if __name__ == '__main__':
    # Get port from environment variable or default to 8080
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

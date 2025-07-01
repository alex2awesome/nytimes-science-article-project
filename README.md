# NYTimes Science Article Explorer

A comprehensive tool for exploring and analyzing New York Times science articles through advanced topic modeling, hierarchical clustering, and source analysis. This project uses large language models (LLMs) and machine learning techniques to help readers quickly understand scientific publishing patterns, dive deeper into individual articles, and explore journalistic reporting processes.

## 🌟 Features

### 1. **Interactive Topic Hierarchy Visualization**
- Bird's-eye view of NYT's scientific publishing output
- Custom extension on [TopicGPT](https://arxiv.org/pdf/2311.01449) for topic modeling
- Hierarchical clustering with agglomerative methods
- Interactive tree visualization showing article categorization patterns

### 2. **Deep Article Analysis**
- Detailed exploration of individual science articles
- Difference analysis between journalist angles and original scientific sources
- Scientific figure extraction and AI-powered explanations
- Enhanced understanding of scientific content through visual aids

### 3. **Reporting Process Explorer**
- Analysis of journalistic sources and methodologies
- Source categorization by centrality (High/Medium/Low)
- Perspective analysis and source verification tracking
- Insight into how stories are reported and fact-checked

## 🏗️ Project Structure

### `/app/` - Flask Web Application
The main web application providing an interactive interface for exploring the analyzed data.

**Key Files:**
- `main.py` - Flask application with routes for topic trees, article exploration, and source analysis
- `templates/` - HTML templates for different views:
  - `tree_template_horizontal.html` - Interactive topic hierarchy visualization
  - `dig_deeper_article_viewer.html` - Detailed article analysis interface
  - `explore_reporting_doc_source_data.html` - Source analysis dashboard
  - `index.html` - Main landing page
- `app_data/` - Processed data files for the web application
- `static/` - CSS and JavaScript assets

### `/clustering_files/` - Topic Modeling & Hierarchical Clustering
Core algorithms for organizing and clustering articles by topic.

**Key Files:**
- `tree_functions.py` - Main hierarchical clustering implementation
  - Agglomerative clustering using Ward linkage
  - Optimal threshold finding algorithms
  - Tree structure creation and evaluation metrics
- `tree_helper_functions.py` - Tree manipulation utilities
  - Tree pruning by subtree size
  - Data propagation algorithms
  - Node labeling and summarization
- `agglomerative_clustering.py` - K-means initialization + hierarchical clustering pipeline
- `conversion_functions.py` - Linkage matrix to NetworkX graph conversion
- `prompts.py` - OpenAI prompts for topic labeling and summarization
- `basic_util.py` - OpenAI API utilities

### `/notebooks/` - Data Analysis & Processing
Jupyter notebooks for data processing, analysis, and experimentation.

**Key Notebooks:**
- `2025-04-25__topic-summary.ipynb` - Topic modeling and hierarchy generation
- `2025-04-25__source-labeling.ipynb` - Source analysis and categorization
- `2025-04-24__data-processing-and-science-article-labeling.ipynb` - Data preprocessing pipeline
- `2025-04-27__diving-deeper-into-articles.ipynb` - Article content analysis

### `/data/` - Raw and Processed Data
Contains the dataset of NYT science articles and processed outputs.

**Key Files:**
- `science_articles.json.gz` - Main dataset of science articles
- `full-parsed-source-df.jsonl` - Parsed source data with metadata
- `output_data/` - Processed clustering and analysis results
- `found-science-articles/` - Raw article collection

### `/screenshots-user-experience/` - Documentation
Visual documentation of the application's user interface and features.

## 🔬 Technical Approach

### Topic Modeling Pipeline
1. **Initial Clustering**: K-means clustering (128 clusters) on sentence embeddings
2. **Hierarchical Organization**: Agglomerative clustering with Ward linkage on cluster centers
3. **Tree Optimization**: Automatic threshold selection using silhouette analysis and cophenetic correlation
4. **Labeling**: OpenAI-powered topic labeling and summarization

### Source Analysis
- **Centrality Classification**: High/Medium/Low importance ranking
- **Perspective Analysis**: Multiple viewpoint identification
- **Verification Tracking**: Direct vs. indirect source documentation
- **Statistical Aggregation**: Source composition analysis

### Difference Analysis
Based on [custom differencing algorithms](https://aclanthology.org/2022.naacl-main.10/) to identify:
- Angle differences between journalist and scientific sources
- Content gaps and additions
- Emphasis and framing variations

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Flask
- scikit-learn
- NetworkX
- OpenAI API key

### Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/nytimes-science-article-project.git
cd nytimes-science-article-project
```

2. Install dependencies:
```bash
cd app
pip install -r requirements.txt
```

3. Set up OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

4. Run the application:
```bash
python main.py
```

5. Open your browser to `http://localhost:8080`

## 📊 Data Processing

To regenerate the analysis from scratch:

1. **Data Collection**: Process raw NYT science articles
2. **Embedding Generation**: Create sentence embeddings using `all-MiniLM-L6-v2`
3. **Topic Modeling**: Run the clustering pipeline in `notebooks/2025-04-25__topic-summary.ipynb`
4. **Source Analysis**: Execute source labeling in `notebooks/2025-04-25__source-labeling.ipynb`

## 🎯 Research Applications

This project demonstrates applications of several research areas:

- **Topic Modeling**: Extensions on TopicGPT for hierarchical organization
- **Source Analysis**: Multi-document source identification and categorization
- **Content Analysis**: Differencing algorithms for document comparison
- **Information Visualization**: Interactive exploration of complex datasets
- **Human-AI Interaction**: LLM-powered content understanding tools

## 🤝 Contributing

This project is part of ongoing research into LLM-assisted information understanding. Feedback and contributions are welcome!

**Research Papers Referenced:**
- [TopicGPT](https://arxiv.org/pdf/2311.01449) - Topic modeling framework
- [Document Differencing](https://aclanthology.org/2022.naacl-main.10/) - Content comparison algorithms
- [Source Identification](https://arxiv.org/abs/2305.14904) - Automatic source detection
- [Source Categorization](https://aclanthology.org/2024.findings-emnlp.930/) - Source classification methods

## 👨‍🎓 About

Created by Alexander Spangher, PhD student at USC, researching how LLMs can help readers understand large amounts of information quickly.

- **Website**: [alexander-spangher.com](https://www.alexander-spangher.com/)
- **Twitter**: [@AlexanderSpangh](https://x.com/AlexanderSpangh)
- **Email**: spangher@usc.edu

## 📄 License

This project is intended for research and educational purposes.

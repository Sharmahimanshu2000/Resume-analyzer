from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import chardet

app = Flask(__name__)
nlp = spacy.load('en_core_web_sm')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/match', methods=['POST'])
def match():
    if request.method == 'POST':
        job_description = request.form['job_description']
        resume_text = request.form['resume_text']

        job_description_file = request.files['job_description_file']
        resume_file = request.files['resume_file']

        # Detect encoding of job description file
        job_desc_encoding = chardet.detect(job_description_file.read())['encoding']
        job_description_file.seek(0)  # Reset file pointer to beginning

        # Read job description content with detected encoding
        job_description_content = job_description_file.read().decode(job_desc_encoding or 'latin-1')

        # Detect encoding of resume file
        resume_encoding = chardet.detect(resume_file.read())['encoding']
        resume_file.seek(0)  # Reset file pointer to beginning

        # Read resume content with detected encoding
        resume_content = resume_file.read().decode(resume_encoding or 'latin-1')

        # Tokenize and lemmatize job description and resume
        job_desc_tokens = [token.lemma_ for token in nlp(job_description_content) if not token.is_stop]
        resume_tokens = [token.lemma_ for token in nlp(resume_content) if not token.is_stop]

        # Convert tokens to strings for TF-IDF
        job_desc_str = ' '.join(job_desc_tokens)
        resume_str = ' '.join(resume_tokens)

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([job_desc_str, resume_str])

        # Calculate cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)[0][1]

        return render_template('results.html', cosine_sim=cosine_sim *100)

if __name__ == '__main__':
    app.run(debug=True)

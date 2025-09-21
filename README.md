# Resume Screening App using NLP

A **Streamlit-based web application** for screening resumes against job descriptions using **Natural Language Processing (NLP)**. The app supports both single and multiple resume comparisons, extracts key entities, and provides a semantic similarity score using Sentence Transformers.

🔗 **Live App**: [https://resume-screening-app-nlp.streamlit.app/](https://resume-screening-app-nlp.streamlit.app/)

---

## 🚀 Features

* ✅ **Single Resume Matching** — Upload one resume and get a similarity score with the job description.
* ✅ **Multi Resume Comparison** — Upload multiple resumes and see ranked similarity scores.
* ✅ **Entity Extraction** — Automatically extracts **skills**, **organizations**, and **person names**.
* ✅ **Interactive Interface** — Highlights key tags and previews resume text.
* ✅ **Downloadable Results** — Export your comparison results as a CSV file.
* ✅ **No Manual Feature Engineering** — Uses deep learning models for embeddings and semantic understanding.

---

## 📸 Demo

📺 [Watch the demo on Google Drive](https://drive.google.com/file/d/1MKztvt_b4OZliYTNkccUw_Xgrt8nKB1m/view?usp=sharing)

---

## 📚 Major Dependencies

| Library                 | Purpose                                  |
| ----------------------- | ---------------------------------------- |
| `streamlit`             | Web app framework                        |
| `spacy`                 | Entity recognition                       |
| `sentence-transformers` | Semantic similarity with BERT embeddings |
| `PyPDF2`                | PDF text extraction                      |
| `pandas`                | Data manipulation and exporting          |

> ℹ️ **Note:** The app uses `all-MiniLM-L6-v2` from Hugging Face. An internet connection is required for the initial download.

---

## 📦 Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/resume-screening-app.git
   cd resume-screening-app

2. **Create and activate a virtual environment:**

   **Windows:**

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   **Mac/Linux:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app:**

   ```bash
   streamlit run app.py
   ```

---

## 🛠️ Usage

1. **Choose Mode:**

   * 📄 *Single Resume Match*
   * 📁 *Multi Resume Comparison*

2. **Paste Job Description:**
   Add the job description in the text area.

3. **Upload Resume(s):**
   Only PDF format supported.

4. **View Results:**

   * Check similarity scores (closer to 1 = better match).
   * See extracted entities.
   * Download the comparison table as CSV.

---

## 📝 Code Structure

```
resume-screening-app/
├── app.py             # Main Streamlit application
├── utils.py           # NLP and matching logic
├── requirements.txt   # Project dependencies
```

---

## 🔍 Example Walkthrough

1. Select **"Single Resume Match"** from mode options.
2. Paste a sample **job description**.
3. Upload a **PDF resume**.
4. View:

   * Matching score
   * Extracted entities (like organizations, skills, person names)
   * Download button for CSV results (in multi-mode)

---

## 🧪 requirements.txt

```
streamlit
pandas
PyPDF2
sentence-transformers
spacy==3.7.2
numpy==1.24.3
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz
```

> 💡 Run `python -m spacy download en_core_web_sm` if the app doesn't find the SpaCy model.

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo, create a branch, and submit a pull request.
For major changes, please open an issue first to discuss what you'd like to change.

---

## 💡 Inspiration & Use Case

This app is useful for:

* Recruiters to quickly screen candidates.
* Job seekers to tailor their resumes based on job descriptions.
* Automating HR workflows using AI.

---

*Happy coding!* 🚀
*Build responsibly with NLP and AI.* 💼✨

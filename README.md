# Handwriting OCR (Arabic + English)

Streamlit app that extracts handwritten text from an image using EasyOCR. Runs fully offline, no API key.

## Run locally
```bash
python -m pip install -r requirements.txt
python -m streamlit run app_easyocr.py
```

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub.
2. share.streamlit.io -> New app -> select the repo, branch `main`, main file `app_easyocr.py`.
3. Advanced settings -> Python 3.11 or 3.12.
4. Deploy. The first load downloads the OCR models (~100 MB), so it is slow once.

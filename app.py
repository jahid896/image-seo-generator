import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from keybert import KeyBERT
import io
import csv
import json
import zipfile

st.set_page_config(page_title="AI Image SEO", layout="wide")
st.title("AI Image SEO & Metadata Generator")
st.markdown("**ইমেজ আপলোড → ডিসক্রিপশন + SEO + ডাউনলোড**")

@st.cache_resource
def load_models():
    with st.spinner("AI মডেল লোড হচ্ছে..."):
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        kw_model = KeyBERT()
    return processor, model, kw_model

processor, model, kw_model = load_models()

with st.sidebar:
    st.header("সেটিংস")
    max_keywords = st.slider("কীওয়ার্ড সংখ্যা", 5, 30, 15)
    format = st.selectbox("ডাউনলোড", ["TXT", "JSON", "CSV", "ZIP"])

uploaded_files = st.file_uploader("ইমেজ আপলোড করুন", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)

if uploaded_files:
    results = []
    progress = st.progress(0)

    for i, file in enumerate(uploaded_files):
        progress.progress((i + 1) / len(uploaded_files))
        image = Image.open(file)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption=file.name, width=200)
        with col2:
            inputs = processor(image, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=50)
            desc = processor.decode(out[0], skip_special_tokens=True)

            keywords = kw_model.extract_keywords(desc, top_n=max_keywords)
            kw_list = ", ".join([k[0] for k in keywords]) if keywords else ""
            title = f"{desc.split('.')[0].capitalize()} | Premium Stock Photo"

            results.append({"filename": file.name, "title": title, "description": desc, "keywords": kw_list})

            st.success("সফল!")
            st.write(f"**টাইটেল:** {title}")
            st.write(f"**ডিসক্রিপশন:** {desc}")
            st.write(f"**কীওয়ার্ড:** {kw_list}")

    st.markdown("---")
    st.subheader("ডাউনলোড করুন")

    def get_download(format_type, data):
        if format_type == "TXT":
            txt = "\n\n".join([f"File: {r['filename']}\nTitle: {r['title']}\nDescription: {r['description']}\nKeywords: {r['keywords']}" for r in data])
            return txt, "seo_metadata.txt", "text/plain"
        if format_type == "JSON":
            return json.dumps(data, indent=2, ensure_ascii=False), "seo_metadata.json", "application/json"
        if format_type == "CSV":
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=["filename", "title", "description", "keywords"])
            writer.writeheader()
            writer.writerows(data)
            return output.getvalue(), "seo_metadata.csv", "text/csv"
        if format_type == "ZIP":
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as zf:
                for r in data:
                    txt = f"Title: {r['title']}\nDescription: {r['description']}\nKeywords: {r['keywords']}"
                    zf.writestr(f"{r['filename']}_metadata.txt", txt)
            buffer.seek(0)
            return buffer.read(), "seo_metadata.zip", "application/zip"

    data, name, mime = get_download(format, results)
    st.download_button("ডাউনলোড করুন", data, name, mime)

else:
    st.info("ইমেজ আপলোড করুন শুরু করতে!")

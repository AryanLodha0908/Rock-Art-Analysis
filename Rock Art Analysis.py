import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  

import tensorflow as tf
import uuid
import sqlite3
import zipfile
import io
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import streamlit as st
from tensorflow import keras
from tensorflow.keras import layers, applications, Model, Input # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
import faiss
import json
import torch
import torchvision.transforms as T
import segmentation_models_pytorch as smp

# === PATHS ===
RAW_DIR = r"C:\Users\HP\Desktop\Internship\Archaeological Image Analysis\RockArt 3\Raw Images"
PROC_DIR = r"C:\Users\HP\Desktop\Internship\Archaeological Image Analysis\RockArt 3\Processed Images"
SEG_DIR = r"C:\Users\HP\Desktop\Internship\Archaeological Image Analysis\RockArt 3\Segmented Images"
DB_PATH = r"C:\Users\HP\Desktop\Internship\Archaeological Image Analysis\RockArt 3\shapes.db"
MODEL_PATH = os.path.join(os.path.dirname(DB_PATH), "shape_classifier.keras")

for d in [RAW_DIR, PROC_DIR, SEG_DIR]:
    os.makedirs(d, exist_ok=True)

# === DATABASE SETUP ===
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS shapes (
            id TEXT PRIMARY KEY,
            path TEXT UNIQUE,
            embedding BLOB,
            label TEXT DEFAULT 'unlabeled'
        )
    """)
    conn.commit()
    conn.close()

init_db()

# === EMBEDDING MODEL ===
def create_embedding_model(input_shape=(128, 128, 3)):
    base_cnn = applications.MobileNetV2(input_shape=input_shape, include_top=False, pooling='avg')
    base_cnn.trainable = False  # freeze base CNN

    inputs = Input(shape=input_shape)
    features = base_cnn(inputs)
    model = Model(inputs, features, name="embedding_model")
    return model

# Create the model instance BEFORE using it
embedding_model = create_embedding_model()

def compute_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    expected_shape = embedding_model.input_shape[1:3]  # (224, 224)
    img = img.resize(expected_shape)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    embedding = embedding_model.predict(img_array)
    return embedding

def preprocess_image(raw_path):
    img = Image.open(raw_path).convert("RGB")
    
    img_cv = np.array(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    img_cv = cv2.resize(img_cv, (256, 256), interpolation=cv2.INTER_AREA)
    
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    enhanced_lab = cv2.merge((cl, a, b))
    
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    blurred = cv2.GaussianBlur(enhanced_bgr, (3,3), 0)
    
    proc_img = Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))

    filename = os.path.basename(raw_path)
    proc_path = os.path.join(PROC_DIR, filename)
    proc_img.save(proc_path)
    
    return proc_path

# === IMPROVED U-NET SEGMENTATION ===
class UNetSegmenter:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = smp.Unet(
            encoder_name="resnet34", encoder_weights="imagenet",
            in_channels=3, classes=1, activation=None
        )
        self.model.eval().to(self.device)
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_path):
        orig = Image.open(image_path).convert("RGB")
        orig_cv = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
      
        lab = cv2.cvtColor(orig_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        img_pil = Image.fromarray(enhanced_img)
        return orig, self.transform(img_pil).unsqueeze(0).to(self.device)

    def segment(self, image_path):
        orig_img, img_tensor = self.preprocess_image(image_path)
        with torch.no_grad():
            logits = self.model(img_tensor)[0, 0].cpu().numpy()
            prob_map = 1 / (1 + np.exp(-logits))  # Sigmoid
            threshold = np.mean(prob_map) + 0.01
            mask = (prob_map > threshold).astype(np.uint8) * 255
        return np.array(orig_img), mask

def segment_with_unet(image_path):
    segmenter = UNetSegmenter()
    orig_img, mask = segmenter.segment(image_path)

    mask = cv2.resize(mask, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    segmented_paths = []
    valid_shapes = []

    for i in range(1, num_labels):  
        x, y, w, h, area = stats[i]
        if area < 1100:  # stricter filter
            continue
        part = orig_img[y:y+h, x:x+w]
        valid_shapes.append(part)

    if not valid_shapes:
        return []

    for part in valid_shapes[:8]:
        shape_id = str(uuid.uuid4())
        shape_path = os.path.join(SEG_DIR, f"{shape_id}.png")
        Image.fromarray(part).save(shape_path)
        segmented_paths.append((shape_id, shape_path))

    return segmented_paths

# === IMAGE EMBEDDING AND DATABASE FUNCTIONS ===
def compute_embedding(image_path):
    img = Image.open(image_path).convert('RGB').resize((128,128))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    embedding = embedding_model.predict(img_array)
    return embedding.flatten().astype(np.float32)

def insert_shape_to_db(shape_id, shape_path, embedding):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO shapes (id, path, embedding, label) VALUES (?, ?, ?, ?)",
              (shape_id, shape_path, embedding.tobytes(), "unlabeled"))
    conn.commit()
    conn.close()

def load_embeddings():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, embedding FROM shapes")
    data = c.fetchall()
    conn.close()
    ids, embeddings = [], []
    for id_, emb_blob in data:
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        ids.append(id_)
        embeddings.append(emb)
    if embeddings:
        embeddings = np.stack(embeddings)
    else:
        embeddings = np.array([], dtype=np.float32).reshape(0,128)
    return ids, embeddings

def create_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

def search_similar(embedding, index, ids, k=3):
    k = max(5, k)
    if index.ntotal == 0:
        return []
    D, I = index.search(embedding.reshape(1,-1), k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        results.append((ids[idx], dist))
    return results

def update_label(shape_id, new_label):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE shapes SET label=? WHERE id=?", (new_label, shape_id))
    conn.commit()
    conn.close()

def delete_shape(shape_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT path FROM shapes WHERE id=?", (shape_id,))
    row = c.fetchone()
    if row:
        try:
            os.remove(row[0])
        except:
            pass
    c.execute("DELETE FROM shapes WHERE id=?", (shape_id,))
    conn.commit()
    conn.close()

def export_dataset():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, path, label FROM shapes")
    data = c.fetchall()
    conn.close()
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        records = []
        for shape_id, path, label in data:
            filename = os.path.basename(path)
            zip_file.write(path, arcname=filename)
            records.append({"id": shape_id, "filename": filename, "label": label})
        df = pd.DataFrame(records)
        csv_bytes = df.to_csv(index=False).encode()
        zip_file.writestr("labels.csv", csv_bytes)
    zip_buffer.seek(0)
    return zip_buffer

def import_labels_from_ls(json_data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for item in json_data:
        filename = item['data']['image'].split('/')[-1]
        ann = item.get('annotations', [])
        if ann:
            choices = ann[0]['result'][0]['value'].get('choices', [])
            if choices:
                label = choices[0]
                c.execute("UPDATE shapes SET label=? WHERE path LIKE ?", (label, f"%{filename}%"))
    conn.commit()
    conn.close()

# === CLASSIFIER TRAINING ===
def load_labeled_data():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT path, label FROM shapes WHERE label IS NOT NULL AND label != 'unlabeled'")
    data = c.fetchall()
    conn.close()
    images, labels = [], []
    for path, label in data:
        if os.path.exists(path):
            img = Image.open(path).convert("RGB").resize((128,128))
            images.append(np.array(img)/255.0)
            labels.append(label)
    return np.array(images), np.array(labels)

def create_classifier(num_classes):
    base_model = applications.MobileNetV2(input_shape=(128,128,3), include_top=False, pooling='avg', weights='imagenet')
    base_model.trainable = False
    x = layers.Dense(256, activation='relu')(base_model.output)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def predict_label_from_model(img_path, model, idx_to_label):
    img = Image.open(img_path).convert("RGB").resize((128, 128))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)
    preds = model.predict(img_array)[0]
    pred_label = idx_to_label[int(np.argmax(preds))]
    confidence = float(np.max(preds))
    return pred_label, confidence

# === STREAMLIT GUI ===
def main():
    st.set_page_config(
        page_title="Rock Art Analysis", 
        layout="wide"
    )
    
    st.title("Archaeological Rock Art Analysis")

if "menu" not in st.session_state:
    st.session_state.menu = "Upload & Process" 
    
menu = st.sidebar.selectbox("Menu", [
    "Upload & Process",     # Step 1: Upload raw images and preprocess
    "Segment & Embed",      # Step 2: Segment shapes and embed features
    "Batch Labeling",       # Step 3: Batch label unlabeled shapes
    "Similarity Search",    # Step 4: Search similar shapes and label them
    "Train Model",          # Step 5: Train classifier on labeled shapes
    "Predict Label",        # Step 6: Predict label on new shapes
    "Delete Shapes",        # Utility: Manage dataset by deleting shapes
    "Export Dataset",       # Utility: Export dataset with labels
    "Import Label Studio"   # Utility: Import labels from Label Studio JSON
])

if menu == "Upload & Process":
    uploaded_files = st.file_uploader("Upload raw images", accept_multiple_files=True, type=['png','jpg','jpeg','tif'])
    for file in uploaded_files or []:
        raw_path = os.path.join(RAW_DIR, file.name)
        with open(raw_path, "wb") as f:
            f.write(file.getbuffer())
        proc_path = preprocess_image(raw_path)
        st.success(f"Processed and saved {proc_path}")

elif menu == "Segment & Embed":
    st.subheader("Upload processed images for segmentation")
    uploaded_files = st.file_uploader("Upload one or more processed images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

    if uploaded_files and st.button("Segment & Embed Uploaded Images"):
        for file in uploaded_files:
            proc_path = os.path.join(PROC_DIR, file.name)
            with open(proc_path, "wb") as f:
                f.write(file.getbuffer())

            shapes = segment_with_unet(proc_path)
            st.markdown(f"### Segmented from uploaded: `{file.name}`")

            if not shapes:
                st.warning(f"No objects found in {file.name}.")
                continue

            for shape_id, shape_path in shapes:
                embedding = compute_embedding(shape_path)
                pred_label = "unlabeled"
                try:
                    if os.path.exists(MODEL_PATH):
                        model = tf.keras.models.load_model(MODEL_PATH)
                        label_map_path = "label_to_idx.json"
                        if os.path.exists(label_map_path):
                            with open(label_map_path, "r") as f:
                                label_to_idx = json.load(f)
                            idx_to_label = {int(v): k for k, v in label_to_idx.items()}
                            pred_label, confidence = predict_label_from_model(shape_path, model, idx_to_label)
                            if confidence < 0.8:
                                pred_label = "unlabeled"
                except Exception as e:
                    print("Auto-labeling failed:", e)

                insert_shape_to_db(shape_id, shape_path, embedding)
                update_label(shape_id, pred_label)
                st.image(shape_path, width=100, caption=f"{shape_id[:6]} → {pred_label}")

    st.markdown("---")
    st.subheader("Select a processed image for segmentation")
    proc_images = os.listdir(PROC_DIR)
    if not proc_images:
        st.info("No processed images found.")
    else:
        selected_image = st.selectbox("Select one processed image", proc_images)
        if st.button("Segment & Embed Selected Image"):
            full_path = os.path.join(PROC_DIR, selected_image)
            shapes = segment_with_unet(full_path)
            st.markdown(f"### Segmented from: `{selected_image}`")

            if not shapes:
                st.warning(f"No objects found in {selected_image}.")
            else:
                for shape_id, shape_path in shapes:
                    embedding = compute_embedding(shape_path)
                    pred_label = "unlabeled"
                    try:
                        if os.path.exists(MODEL_PATH):
                            model = tf.keras.models.load_model(MODEL_PATH)
                            label_map_path = "label_to_idx.json"
                            if os.path.exists(label_map_path):
                                with open(label_map_path, "r") as f:
                                    label_to_idx = json.load(f)
                                idx_to_label = {int(v): k for k, v in label_to_idx.items()}
                                pred_label, confidence = predict_label_from_model(shape_path, model, idx_to_label)
                                if confidence < 0.8:
                                    pred_label = "unlabeled"
                    except Exception as e:
                        print("Auto-labeling failed:", e)

                    insert_shape_to_db(shape_id, shape_path, embedding)
                    update_label(shape_id, pred_label)
                    st.image(shape_path, width=100, caption=f"{shape_id[:6]} → {pred_label}")

elif menu == "Batch Labeling":
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, path FROM shapes WHERE label='unlabeled'")
    unlabeled = c.fetchall()
    conn.close()

    if not unlabeled:
        st.info("No unlabeled shapes available.")
    else:
        display_names = [f"{i[:6]} ({os.path.basename(p)})" for i, p in unlabeled]
        selected = st.multiselect("Select shapes", display_names)

        st.markdown("### Selected Shape Previews")
        for sel in selected:
            shape_id = sel.split(" ")[0]
            for uid, path in unlabeled:
                if uid.startswith(shape_id):
                    path = os.path.normpath(path)
                    if os.path.exists(path):
                        st.image(path, width=100, caption=f"{shape_id[:6]}")
                    else:
                        st.warning(f"Image not found: {os.path.basename(path)}")
                    break

        label = st.text_input("Label to apply")
        if st.button("Apply Label"):
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            for sel in selected:
                shape_id = sel.split(" ")[0]
                c.execute("UPDATE shapes SET label=? WHERE id=?", (label, shape_id))
            conn.commit()
            conn.close()
            st.success(f"Labeled {len(selected)} shapes")

        
elif menu == "Similarity Search":
    uploaded = st.file_uploader("Upload shape to search", type=['png','jpg','jpeg'])
    if uploaded:
        img = Image.open(uploaded).convert('RGB').resize((128,128))
        st.image(img, caption="Query Image", width=200)
        emb = embedding_model.predict(np.expand_dims(np.array(img)/255.0, axis=0)).flatten()
        ids, embeddings = load_embeddings()
        if len(ids) == 0:
            st.warning("No shapes in DB")
        else:
            index = create_faiss_index(embeddings.astype('float32'))
            results = search_similar(emb, index, ids)
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            for shape_id, dist in results:
                c.execute("SELECT path, label FROM shapes WHERE id=?", (shape_id,))
                row = c.fetchone()
                if row:
                    if os.path.exists(row[0]):
                        st.image(row[0], width=100, caption=f"{row[1]}, Dist: {dist:.2f}")
                    else:
                        st.warning(f"File not found: {row[0]}")

                    new_label = st.text_input(f"Label for {shape_id[:6]}", value=row[1], key=shape_id)
                    if st.button(f"Update {shape_id[:6]}", key=f"btn_{shape_id}"):
                        update_label(shape_id, new_label)
                        st.success("Label updated")
            conn.close()

elif menu == "Delete Shapes":
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, path FROM shapes")
    all_shapes = c.fetchall()
    conn.close()

    del_ids = st.multiselect("Delete shapes", [f"{i[:6]} ({os.path.basename(p)})" for i,p in all_shapes])

    if st.button("Delete Selected"):
        for sel in del_ids:
            delete_shape(sel.split(" ")[0])
        st.success(f"Deleted {len(del_ids)} shapes")

    if st.button("Delete All Shapes"):
        for shape_id, _ in all_shapes:
            delete_shape(shape_id)
        st.success(f"Deleted all {len(all_shapes)} shapes")

elif menu == "Export Dataset":
    if st.button("Export ZIP"):
        zip_file = export_dataset()
        st.download_button("Download Dataset", zip_file, file_name="rock_art_dataset.zip")

elif menu == "Import Label Studio":
    json_file = st.file_uploader("Upload Label Studio JSON", type=['json'])
    if json_file is not None:
        try:
            json_data = json.load(json_file)
            import_labels_from_ls(json_data)
            st.success("Labels imported successfully.")
        except Exception as e:
            st.error(f"Failed to import labels: {e}")

elif menu == "Train Model":
    st.header("Train Shape Classifier")
    X, y = load_labeled_data()
    if len(np.unique(y)) < 2:
        st.warning("Need at least two classes to train.")
    else:
        label_to_idx = {label: i for i, label in enumerate(sorted(set(y)))}
        idx_to_label = {i: label for label, i in label_to_idx.items()}
        y_idx = np.array([label_to_idx[label] for label in y])

        st.text(f"Classes: {list(label_to_idx.keys())}")

        if st.button("Train"):
            model = create_classifier(len(label_to_idx))
            history = model.fit(X, y_idx, epochs=10, batch_size=16, validation_split=0.2, verbose=1)

            model.save(MODEL_PATH)

            with open("label_to_idx.json", "w") as f:
                json.dump(label_to_idx, f)

            st.success("Model trained and saved.")

            st.line_chart({
                "train_accuracy": history.history["accuracy"],
                "val_accuracy": history.history["val_accuracy"]
            })

elif menu == "Predict Label":
    st.header("Predict Labels for Shapes")
    uploaded = st.file_uploader("Upload shapes for prediction", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

    if uploaded:
        if not os.path.exists(MODEL_PATH):
            st.warning("Model not found. Please train the model first.")
        else:
            model = tf.keras.models.load_model(MODEL_PATH)

            label_map_path = "label_to_idx.json"
            if not os.path.exists(label_map_path):
                st.error("Label map not found. Ensure it's saved during training.")
            else:
                with open(label_map_path, "r") as f:
                    label_to_idx = json.load(f)
                idx_to_label = {int(v): k for k, v in label_to_idx.items()}

                for img_file in uploaded:
                    img = Image.open(img_file).convert("RGB").resize((128, 128))
                    img_array = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

                    preds = model.predict(img_array)[0]
                    pred_label = idx_to_label[np.argmax(preds)]
                    max_conf = float(np.max(preds))

                    st.image(img, width=150, caption=f"Top Prediction: {pred_label} ({max_conf:.2f})")

                    prob_df = pd.DataFrame({
                        "Class": [idx_to_label[i] for i in range(len(preds))],
                        "Confidence": preds
                    })
                    st.bar_chart(prob_df.set_index("Class"))

                    # Save the image temporarily
                    img_np = np.array(img)
                    confident_indices = [i for i, p in enumerate(preds) if p > 0.3]

                    # Condition 1: Save only top class if confidence >= 0.6
                    if max_conf >= 0.6:
                        shape_id = str(uuid.uuid4())
                        save_path = os.path.join(SEG_DIR, f"{shape_id}_{pred_label}.png")
                        Image.fromarray(img_np).save(save_path)

                        embedding = compute_embedding(save_path)
                        insert_shape_to_db(shape_id, save_path, embedding)
                        update_label(shape_id, pred_label)
                        st.success(f"Auto-saved as '{pred_label}' (Confidence: {max_conf:.2f})")

                    # Condition 2: Save all classes with confidence > 0.3 
                    elif len(confident_indices) > 1:
                        for i in confident_indices:
                            label = idx_to_label[i]
                            confidence = preds[i]
                            shape_id = str(uuid.uuid4())
                            save_path = os.path.join(SEG_DIR, f"{shape_id}_{label}.png")
                            Image.fromarray(img_np).save(save_path)

                            embedding = compute_embedding(save_path)
                            insert_shape_to_db(shape_id, save_path, embedding)
                            update_label(shape_id, label)
                            st.success(f"Saved with label '{label}' (Confidence: {confidence:.2f})")

                    else:
                        st.warning(f"No confident predictions above threshold. Max: {max_conf:.2f}")

# streamlit run "c:/Users/HP/Desktop/Internship/Archaeological Image Analysis/RockArt 3/Rock Art Analysis.py"                                                                                                

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ------------------------------------------------------------
# Facial Expression Recognition - Streamlit app
# CNN architecture and training setup adapted from the supplied
# facial-expression Jupyter notebook.
# ------------------------------------------------------------

st.set_page_config(
    page_title="Facial Expression Recognition",
    page_icon="😊",
    layout="centered",
)

MODEL_PATH = Path("facial_expression.keras")
CLASSES_PATH = Path("classes.json")
DEFAULT_DATASET = "FER_56_per_class/train"
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 8
EPOCHS = 50


def build_model(number_of_classes: int):
    """Same CNN structure as the uploaded notebook."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 3)),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(number_of_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_saved_model():
    if MODEL_PATH.exists() and CLASSES_PATH.exists():
        loaded_model = tf.keras.models.load_model(MODEL_PATH)
        with CLASSES_PATH.open("r", encoding="utf-8") as f:
            labels = json.load(f)
        return loaded_model, labels
    return None, None


def train_model(dataset_folder: str, epochs: int):
    dataset_path = Path(dataset_folder)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_path.resolve()}. "
            "Add your training images to this folder."
        )

    dataset = tf.keras.utils.image_dataset_from_directory(
        str(dataset_path),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
    )
    class_names = dataset.class_names

    if len(class_names) < 2:
        raise ValueError("Add at least two class subfolders with images.")
    if len(class_names) != 7:
        st.warning(
            f"Found {len(class_names)} classes: {class_names}. "
            "The intended project has seven classes: angry, disgust, fear, "
            "happy, neutral, sad, surprise."
        )

    model = build_model(len(class_names))

    progress_bar = st.progress(0)
    status = st.empty()

    class StreamlitProgress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress_bar.progress((epoch + 1) / epochs)
            logs = logs or {}
            acc = logs.get("accuracy")
            loss = logs.get("loss")
            status.write(
                f"Epoch {epoch + 1}/{epochs} — "
                f"accuracy: {acc:.4f}, loss: {loss:.4f}"
                if acc is not None and loss is not None
                else f"Epoch {epoch + 1}/{epochs}"
            )

    model.fit(dataset, epochs=epochs, callbacks=[StreamlitProgress()])
    model.save(MODEL_PATH)
    with CLASSES_PATH.open("w", encoding="utf-8") as f:
        json.dump(class_names, f)

    progress_bar.progress(1.0)
    status.success("Training complete. Model and class labels saved.")
    return model, class_names


st.title("😊 Facial Expression Recognition")
st.write(
    "Upload a face image to classify its visible facial expression. "
    "You can train the CNN using your own image folders or load a previously trained model."
)

with st.expander("Dataset folder setup"):
    st.code(
        "FER_56_per_class/\n"
        "└── train/\n"
        "    ├── angry/\n"
        "    ├── disgust/\n"
        "    ├── fear/\n"
        "    ├── happy/\n"
        "    ├── neutral/\n"
        "    ├── sad/\n"
        "    └── surprise/",
        language="text",
    )
    st.caption(
        "Put 50–60 images in each class folder. The app trains on the training "
        "folder, as in your notebook. Keep test images separate."
    )

if "model" not in st.session_state:
    saved_model, saved_classes = load_saved_model()
    st.session_state.model = saved_model
    st.session_state.class_names = saved_classes

st.subheader("1. Train or load your model")
dataset_folder = st.text_input("Training dataset folder", value=DEFAULT_DATASET)
epochs = st.number_input("Training epochs", min_value=1, max_value=100, value=EPOCHS)

col1, col2 = st.columns(2)
with col1:
    if st.button("Train CNN model", type="primary", use_container_width=True):
        try:
            with st.spinner("Training your CNN model..."):
                trained_model, trained_classes = train_model(dataset_folder, int(epochs))
            st.session_state.model = trained_model
            st.session_state.class_names = trained_classes
            st.success("Your facial-expression model is ready.")
        except Exception as exc:
            st.error(f"Training failed: {exc}")

with col2:
    if st.button("Load saved model", use_container_width=True):
        try:
            loaded_model, loaded_classes = load_saved_model()
            if loaded_model is None:
                st.warning("No saved model found. Train the model first.")
            else:
                st.session_state.model = loaded_model
                st.session_state.class_names = loaded_classes
                st.success("Saved model loaded.")
        except Exception as exc:
            st.error(f"Could not load model: {exc}")

if st.session_state.model is not None:
    st.caption("Model is ready. Classes: " + ", ".join(st.session_state.class_names))
else:
    st.info("Train the model or add facial_expression.keras and classes.json to this folder.")

st.divider()
st.subheader("2. Upload an image and predict")

uploaded_file = st.file_uploader(
    "Choose a face image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Predict expression", use_container_width=True):
            if st.session_state.model is None:
                st.warning("Please train or load a model first.")
            else:
                resized = image.resize(IMAGE_SIZE)
                image_array = np.asarray(resized, dtype=np.float32)
                image_batch = np.expand_dims(image_array, axis=0)

                probabilities = st.session_state.model.predict(image_batch, verbose=0)[0]
                predicted_index = int(np.argmax(probabilities))
                predicted_label = st.session_state.class_names[predicted_index]
                confidence = float(probabilities[predicted_index]) * 100

                st.success(f"Predicted expression: **{predicted_label.title()}**")
                st.metric("Confidence", f"{confidence:.2f}%")

                st.write("### Class probabilities")
                sorted_indices = np.argsort(probabilities)[::-1]
                for index in sorted_indices:
                    label = st.session_state.class_names[int(index)]
                    score = float(probabilities[index])
                    st.write(f"{label.title()} — {score * 100:.2f}%")
                    st.progress(score)
    except Exception as exc:
        st.error(f"Could not read this image: {exc}")

st.divider()
st.caption(
    "Educational image-classification demo only. Facial-expression predictions "
    "can be inaccurate and do not establish a person's actual feelings or mental state. "
    "Use images only with appropriate permission."
)

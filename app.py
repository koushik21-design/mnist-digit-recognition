import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import pandas as pd

# ---------- Load model ----------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mnist_cnn.h5")
    return model

model = load_model()

st.title("🖊️ Handwritten Digit Recognition (MNIST)")
st.write("Draw or upload an image of handwritten digits and the model will predict them.")

# ---------- Sidebar options ----------
st.sidebar.header("Settings")
input_mode = st.sidebar.radio(
    "Choose input mode:",
    ["Draw digit", "Upload single digit", "Upload multi-digit number"]
)
invert_colors = st.sidebar.checkbox(
    "Invert colors (for black text on white background in uploads)",
    value=True
)
show_probabilities = st.sidebar.checkbox(
    "Show probability bar chart (single digit modes)",
    value=True
)
st.sidebar.markdown("👉 For **Draw digit**, special preprocessing is applied automatically.")


# ---------- Common preprocessing for single digits (uploads) ----------
def preprocess_image(pil_img, invert=True):
    """
    Preprocess for uploaded single-digit images:
    - Convert to grayscale
    - Resize to 28x28
    - Optional invert
    - Normalize & add batch/channel dims
    """
    # Convert to grayscale
    img = pil_img.convert("L")

    # Resize to 28x28
    img = img.resize((28, 28))

    # Convert to numpy array
    img_array = np.array(img)

    # Optionally invert colors (useful for black-on-white inputs)
    if invert:
        img_array = 255 - img_array

    # Normalize to [0,1]
    img_array = img_array.astype("float32") / 255.0

    # Add channel and batch dimensions: (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, img  # array for model, PIL image for display


# ---------- Special preprocessing for drawn digits (canvas) ----------
def preprocess_drawn_digit(pil_img):
    """
    Preprocess digits drawn on the canvas:
    - Convert to grayscale
    - Find bounding box of the digit (remove extra borders)
    - Resize to 20x20
    - Center in a 28x28 image (classic MNIST style)
    - Normalize & add batch/channel dims
    Assumes white digit on black background (like our canvas).
    """
    # Convert to grayscale
    img = pil_img.convert("L")  # 'L' = grayscale

    # Convert to numpy array
    arr = np.array(img)

    # On the canvas, digit pixels are bright (close to 255), background near 0
    threshold = 50
    ys, xs = np.where(arr > threshold)  # locations of "ink"

    if len(xs) == 0 or len(ys) == 0:
        # No drawing detected, fall back to simple resize
        img = img.resize((28, 28))
        arr = np.array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=-1)
        arr = np.expand_dims(arr, axis=0)
        return arr, img

    # Compute bounding box of the digit
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Crop to bounding box
    cropped = arr[y_min:y_max+1, x_min:x_max+1]

    # Resize cropped region to 20x20
    cropped_img = Image.fromarray(cropped)
    cropped_img = cropped_img.resize((20, 20))

    # Create a new 28x28 black image and paste the 20x20 digit at center
    final_img = Image.new("L", (28, 28), 0)  # 0 = black background
    upper_left = ((28 - 20) // 2, (28 - 20) // 2)
    final_img.paste(cropped_img, upper_left)

    # Convert to numpy and normalize
    final_arr = np.array(final_img).astype("float32") / 255.0

    # Add channel & batch dimensions → (1, 28, 28, 1)
    final_arr = np.expand_dims(final_arr, axis=-1)
    final_arr = np.expand_dims(final_arr, axis=0)

    return final_arr, final_img


# ---------- Digit segmentation for multi-digit images ----------
def segment_digits(pil_img, invert=True, min_column_pixels=3):
    """
    Take a PIL image that contains multiple handwritten digits in a row.
    Returns a list of (processed_array, 28x28_pil_image) for each digit.
    """
    # 1. Convert to grayscale
    img = pil_img.convert("L")

    # 2. Normalize height to 28, keep aspect ratio for width
    w, h = img.size
    new_h = 28
    new_w = int(w * (new_h / h)) if h != 0 else 28
    if new_w < 1:
        new_w = 28
    img = img.resize((new_w, new_h))

    # 3. Convert to numpy
    arr = np.array(img)

    # 4. Optional invert
    if invert:
        arr = 255 - arr

    # 5. Binarize: foreground where pixel intensity > threshold
    threshold = 50
    bin_arr = arr > threshold  # True = digit, False = background

    # 6. Find columns that contain any digit pixels
    col_has_ink = np.any(bin_arr, axis=0)

    # 7. Group consecutive columns into segments
    segments = []
    in_segment = False
    start = 0

    for i, has_ink in enumerate(col_has_ink):
        if has_ink and not in_segment:
            in_segment = True
            start = i
        elif not has_ink and in_segment:
            end = i - 1
            segments.append((start, end))
            in_segment = False

    # If ended while still in a segment
    if in_segment:
        segments.append((start, len(col_has_ink) - 1))

    digit_arrays = []

    for (start, end) in segments:
        # Skip very thin segments (noise)
        if end - start + 1 < min_column_pixels:
            continue

        # Slice the digit region
        digit_arr = arr[:, start:end+1]

        # 8. Pad horizontally to 28x28
        digit_w = digit_arr.shape[1]
        padded = np.zeros((28, 28), dtype=np.uint8)

        # Scale max to 255 (optional, for contrast)
        if digit_arr.max() > 0:
            digit_arr = (digit_arr.astype("float32") / digit_arr.max()) * 255.0
            digit_arr = digit_arr.astype("uint8")

        # Center the digit horizontally
        if digit_w <= 28:
            offset = (28 - digit_w) // 2
            padded[:, offset:offset+digit_w] = digit_arr
        else:
            # If digit wider than 28, resize to 28x28
            digit_img_temp = Image.fromarray(digit_arr)
            digit_img_temp = digit_img_temp.resize((28, 28))
            padded = np.array(digit_img_temp)

        # Prepare for model
        proc = padded.astype("float32") / 255.0
        proc = np.expand_dims(proc, axis=-1)
        proc = np.expand_dims(proc, axis=0)

        digit_pil = Image.fromarray(padded)
        digit_arrays.append((proc, digit_pil))

    return digit_arrays


# ---------- Mode 1: Draw Digit ----------
if input_mode == "Draw digit":
    st.subheader("✏️ Draw your digit")
    st.write("Draw a **white digit on black background** inside the box.")

    canvas_result = st_canvas(
        fill_color="black",      # background fill
        stroke_width=15,         # thickness of the brush
        stroke_color="white",    # color of the stroke (digit)
        background_color="black",
        width=200,
        height=200,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        canvas_img = Image.fromarray(canvas_result.image_data.astype("uint8"))

        st.write("**Canvas image preview:**")
        st.image(canvas_img, width=150)

        if st.button("Predict digit from drawing"):
            # Use special preprocessing for drawn digits (no invert toggle here)
            processed_array, processed_pil = preprocess_drawn_digit(canvas_img)

            st.write("**Processed 28x28 grayscale image (after crop & center):**")
            st.image(processed_pil.resize((140, 140)), width=140)

            predictions = model.predict(processed_array)
            predicted_digit = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))

            st.subheader("Prediction")
            st.write(f"**Predicted Digit:** {predicted_digit}")
            st.write(f"**Confidence:** {confidence * 100:.2f}%")

            if show_probabilities:
                st.write("### Probabilities for each digit (0–9):")
                probs = predictions[0]
                df_probs = pd.DataFrame({"digit": list(range(10)), "probability": probs})
                df_probs.set_index("digit", inplace=True)
                st.bar_chart(df_probs)


# ---------- Mode 2: Upload Single Digit ----------
elif input_mode == "Upload single digit":
    st.subheader("📂 Upload an image (single digit)")

    uploaded_file = st.file_uploader(
        "Upload a digit image (PNG/JPG)",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.write("**Original image:**")
        st.image(image, caption="Uploaded image", width=200)

        if st.button("Predict digit from image"):
            processed_array, processed_pil = preprocess_image(image, invert=invert_colors)

            st.write("**Processed 28x28 grayscale image:**")
            st.image(processed_pil.resize((140, 140)), width=140)

            predictions = model.predict(processed_array)
            predicted_digit = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))

            st.subheader("Prediction")
            st.write(f"**Predicted Digit:** {predicted_digit}")
            st.write(f"**Confidence:** {confidence * 100:.2f}%")

            if show_probabilities:
                st.write("### Probabilities for each digit (0–9):")
                probs = predictions[0]
                df_probs = pd.DataFrame({"digit": list(range(10)), "probability": probs})
                df_probs.set_index("digit", inplace=True)
                st.bar_chart(df_probs)
    else:
        st.info("Please upload a PNG/JPG image of a handwritten digit.")


# ---------- Mode 3: Upload Multi-digit Number ----------
else:
    st.subheader("🔢 Upload an image with multiple digits")

    uploaded_file = st.file_uploader(
        "Upload a multi-digit number image (PNG/JPG)",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.write("**Original image:**")
        st.image(image, caption="Uploaded image", width=300)

        if st.button("Predict all digits in image"):
            digit_arrays = segment_digits(image, invert=invert_colors)

            if not digit_arrays:
                st.error("No digit segments detected. Try cropping tighter or increasing contrast.")
            else:
                predicted_digits = []
                st.write("### Detected digit segments:")

                cols = st.columns(min(5, len(digit_arrays)))
                for idx, (proc_arr, digit_pil) in enumerate(digit_arrays):
                    preds = model.predict(proc_arr)
                    digit = int(np.argmax(preds[0]))
                    predicted_digits.append(str(digit))

                    with cols[idx % len(cols)]:
                        st.image(digit_pil.resize((56, 56)), caption=f"Digit {idx+1}: {digit}")

                full_number = "".join(predicted_digits)
                st.subheader("Final Prediction")
                st.write(f"**Predicted Number:** {full_number}")
    else:
        st.info("Please upload a PNG/JPG image containing multiple handwritten digits in a row.")

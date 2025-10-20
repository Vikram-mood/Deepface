import streamlit as st
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt

# ============================================================
# 1️⃣ Load ResNet-50 Model
# ============================================================
@st.cache_resource
def load_resnet50_model(ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)

    ckpt = torch.load(ckpt_path, map_location=device)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    ckpt = {k.replace("model.", "").replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)

    model.to(device).eval()
    st.success("✅ ResNet-50 model loaded successfully!")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return model, transform, device


# ============================================================
# 2️⃣ Deepfake Prediction
# ============================================================
def predict_deepfake_resnet(img_pil, model, transform, device):
    model.eval()
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        fake_prob = probs[0, 0].item()
        real_prob = probs[0, 1].item()
    return {
        "fake_prob": fake_prob,
        "real_prob": real_prob,
        "prediction": "real" if real_prob > fake_prob else "fake"
    }


# ============================================================
# 3️⃣ Grad-CAM Heatmap Visualization
# ============================================================
def generate_heatmap(model, img_pil, transform, device):
    model.eval()
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    activation = {}
    def hook_fn(module, input, output):
        activation["feat"] = output.detach()

    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module

    handle = last_conv.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(img_tensor)
    handle.remove()

    feat = activation["feat"].squeeze(0).mean(0).cpu().numpy()
    feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
    heatmap = cv2.resize(feat, img_pil.size)
    cmap = plt.colormaps["jet"]
    colored_map = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)

    img_np = np.array(img_pil).astype(np.float32)
    overlay = (0.6 * img_np + 0.4 * colored_map).astype(np.uint8)
    return Image.fromarray(overlay)


# ============================================================
# 4️⃣ Face Similarity (InsightFace)
# ============================================================
@st.cache_resource
def init_insightface():
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(224, 224))
    return app

def face_similarity(img1, img2, app):
    img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
    faces1, faces2 = app.get(img1), app.get(img2)
    if not faces1 or not faces2:
        return None
    emb1, emb2 = faces1[0].embedding, faces2[0].embedding
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return float(sim)


# ============================================================
# 5️⃣ Streamlit UI
# ============================================================
st.set_page_config(page_title="Deepfake Detection + Face Verification", layout="centered")
st.title("🧠 Deepfake Detection + Face Verification (ResNet-50)")

MODEL_PATH = "best_model.pth"
model, transform, device = load_resnet50_model(MODEL_PATH)
app = init_insightface()

# Step 1 – Deepfake Detection
st.header("Step 1: Deepfake Detection (Original Image)")
orig_source = st.radio("Choose image input:", ["Upload Image", "Use Webcam"])

orig_img = None
if orig_source == "Upload Image":
    file = st.file_uploader("Upload Original Image", type=["jpg", "jpeg", "png"])
    if file:
        orig_img = Image.open(file).convert("RGB")
elif orig_source == "Use Webcam":
    camera_file = st.camera_input("📸 Capture Original Image")
    if camera_file:
        orig_img = Image.open(camera_file).convert("RGB")

if orig_img:
    st.image(orig_img, caption="Original Image", width=None)
    probs = predict_deepfake_resnet(orig_img, model, transform, device)
    st.write(f"**Real Probability:** {probs['real_prob']:.3f}")
    st.write(f"**Fake Probability:** {probs['fake_prob']:.3f}")

    try:
        heatmap = generate_heatmap(model, orig_img, transform, device)
        st.image(heatmap, caption="Activation Heatmap", width=None)
    except Exception as e:
        st.warning(f"⚠️ Heatmap generation failed: {e}")

    if probs["fake_prob"] > 0.5:
        st.error("❌ The uploaded image is likely FAKE. Stopping process.")
        st.stop()
    else:
        st.success("✅ The uploaded image is REAL. Proceed to Step 2.")

        # Step 2 – Face Verification
        st.header("Step 2: Face Verification")
        verify_source = st.radio("Choose second image source:", ["Upload Image", "Use Webcam"])

        second_img = None
        if verify_source == "Upload Image":
            file2 = st.file_uploader("Upload Second Image", type=["jpg", "jpeg", "png"], key="second")
            if file2:
                second_img = Image.open(file2).convert("RGB")
        elif verify_source == "Use Webcam":
            cam2 = st.camera_input("📸 Capture Second Image")
            if cam2:
                second_img = Image.open(cam2).convert("RGB")

        if second_img:
            st.image(second_img, caption="Second Image", width=None)
            probs2 = predict_deepfake_resnet(second_img, model, transform, device)
            st.write(f"**Real Probability:** {probs2['real_prob']:.3f}")
            st.write(f"**Fake Probability:** {probs2['fake_prob']:.3f}")

            if probs2["fake_prob"] > 0.5:
                st.error("❌ Second image is likely FAKE. Verification stopped.")
            else:
                st.success("✅ Second image is REAL. Computing face similarity...")
                sim = face_similarity(orig_img, second_img, app)
                if sim is not None:
                    threshold = 0.7
                    st.write(f"**Face Similarity Score:** {sim:.3f}")
                    if sim > threshold:
                        st.success("✔️ Faces match! Verification SUCCESS.")
                    else:
                        st.warning("⚠️ Faces do NOT match. Verification FAILED.")
                else:
                    st.error("❌ Could not detect faces in one or both images.")

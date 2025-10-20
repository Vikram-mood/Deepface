import streamlit as st
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from insightface.app import FaceAnalysis
import matplotlib.cm as cm


# ============================================================
# 1️⃣ Load ResNet-50 Model
# ============================================================
@st.cache_resource
def load_resnet50_model(ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create ResNet50 model
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # binary: real / fake

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    # Handle prefix keys if saved with DataParallel or Lightning
    new_state_dict = {k.replace("model.", "").replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(new_state_dict, strict=False)
    st.write("✅ ResNet-50 model loaded successfully!")

    model.to(device)
    model.eval()

    # Transform for ResNet-50
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
# def predict_deepfake_resnet(img_pil, model, transform, device):
#     img_tensor = transform(img_pil).unsqueeze(0).to(device)
#     with torch.no_grad():
#         logits = model(img_tensor)
#         probs = F.softmax(logits, dim=1)
#         real_prob = probs[0, 0].item()
#         fake_prob = probs[0, 1].item()
#     return {"real_prob": real_prob, "fake_prob": fake_prob}

import torch
import torch.nn.functional as F

def predict_deepfake_resnet(img_pil, model, transform, device):
    """
    Predict whether an image is real or fake using a trained ResNet model.
    """
    model.eval()
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)

        fake_prob = probs[0, 0].item()  # class 0 = fake
        real_prob = probs[0, 1].item()  # class 1 = real

    return {
        "fake_prob": fake_prob,
        "real_prob": real_prob,
        "prediction": "real" if real_prob > fake_prob else "fake"
    }



# ============================================================
# 3️⃣ Grad-CAM-like Heatmap Visualization
# ============================================================
def generate_heatmap(model, img_pil, transform, device):
    model.eval()
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    activation = {}
    def hook_fn(module, input, output):
        activation["feat"] = output.detach()

    # Find last conv layer
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module

    handle = last_conv.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(img_tensor)
    handle.remove()

    feat = activation["feat"].squeeze(0).mean(0).cpu().numpy()
    feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
    heatmap = cv2.resize(feat, img_pil.size)
    cmap = cm.get_cmap("jet")
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

    faces1 = app.get(img1)
    faces2 = app.get(img2)
    if not faces1 or not faces2:
        return None

    emb1, emb2 = faces1[0].embedding, faces2[0].embedding
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return float(sim)


# ============================================================
# 5️⃣ Streamlit UI
# ============================================================
st.title("🧠 Deepfake Detection + Face Verification (ResNet-50)")

MODEL_PATH = "/Users/vikram/Downloads/best_model.pth"  # ✅ update your path
model, transform, device = load_resnet50_model(MODEL_PATH)
app = init_insightface()

# Step 1: Upload first image
st.header("Step 1: Deepfake Detection (Original Image)")
orig_file = st.file_uploader("Upload Original Image", type=["jpg", "png"])
if orig_file:
    orig_img = Image.open(orig_file).convert("RGB")
    st.image(orig_img, caption="Original Image", use_container_width=True)

    probs = predict_deepfake_resnet(orig_img, model, transform, device)
    st.write(f"**Real probability:** {probs['real_prob']:.3f}")
    st.write(f"**Fake probability:** {probs['fake_prob']:.3f}")

    # Visualization
    try:
        heatmap_img = generate_heatmap(model, orig_img, transform, device)
        st.image(heatmap_img, caption="Activation Heatmap", use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Heatmap generation failed: {e}")

    if probs["fake_prob"] > 0.5:
        st.error("❌ The uploaded image is likely FAKE. Stopping process.")
    else:
        st.success("✅ The uploaded image is REAL. Proceed to Step 2.")

        # Step 2: Verification
        st.header("Step 2: Face Verification")
        option = st.radio("Select method for second image:",
                          ("Upload Image", "Capture from Webcam"))

        if option == "Upload Image":
            second_file = st.file_uploader("Upload Second Image", type=["jpg", "png"], key="second")
            if second_file:
                second_img = Image.open(second_file).convert("RGB")
                st.image(second_img, caption="Second Image", use_container_width=True)

                probs2 = predict_deepfake_resnet(second_img, model, transform, device)
                st.write(f"**Real probability:** {probs2['real_prob']:.3f}")
                st.write(f"**Fake probability:** {probs2['fake_prob']:.3f}")

                if probs2["fake_prob"] > 0.5:
                    st.error("❌ The second image is likely FAKE. Verification stopped.")
                else:
                    st.success("✅ Second image is REAL. Proceeding to verification...")

                    sim_score = face_similarity(orig_img, second_img, app)
                    if sim_score is not None:
                        threshold = 0.7
                        st.write(f"**Face similarity score:** {sim_score:.3f}")
                        if sim_score > threshold:
                            st.success("✔️ Faces match! Verification SUCCESS.")
                        else:
                            st.warning("⚠️ Faces do NOT match. Verification FAILED.")
                    else:
                        st.error("❌ Could not detect faces in one or both images.")

        else:
            st.info("📷 Click to capture an image from your webcam.")
            if st.button("Capture from Webcam"):
                cap = cv2.VideoCapture(0)
                ret, frame = cap.read()
                cap.release()

                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    webcam_img = Image.fromarray(frame_rgb)
                    st.image(webcam_img, caption="Captured Image", use_container_width=True)

                    probs2 = predict_deepfake_resnet(webcam_img, model, transform, device)
                    st.write(f"**Real probability:** {probs2['real_prob']:.3f}")
                    st.write(f"**Fake probability:** {probs2['fake_prob']:.3f}")

                    if probs2["fake_prob"] > 0.5:
                        st.error("❌ Captured image is likely FAKE. Verification stopped.")
                    else:
                        st.success("✅ Captured image is REAL. Proceeding to verification...")

                        sim_score = face_similarity(orig_img, webcam_img, app)
                        if sim_score is not None:
                            threshold = 0.5
                            st.write(f"**Face similarity score:** {sim_score:.3f}")
                            if sim_score > threshold:
                                st.success("✔️ Faces match! Verification SUCCESS.")
                            else:
                                st.warning("⚠️ Faces do NOT match. Verification FAILED.")
                        else:
                            st.error("❌ Could not detect faces in one or both images.")
                else:
                    st.error("❌ Unable to access webcam.")

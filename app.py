import streamlit as st
from PIL import Image
import cv2
import numpy as np
import io
from rembg import remove
from ultralytics import YOLO
import tempfile
import os

# Configuración inicial
st.set_page_config(page_title="Eliminador con YOLOv8", layout="wide")
st.title("✅ Eliminador con YOLOv8 – Segmentación Precisa para Zapatillas")

# Cargar modelo YOLOv8-seg (solo una vez)
@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n-seg.pt')

model = load_yolo_model()

# Funciones de procesamiento
def is_white_product_on_light_background(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)
        bright_pixels = np.sum(v > 220)
        bright_ratio = bright_pixels / v.size
        std_vals = [np.std(img[:, :, i]) for i in range(3)]
        avg_std = np.mean(std_vals)
        return bright_ratio > 0.6 and avg_std < 40
    except:
        return False

def get_yolo_mask(model, image_path):
    try:
        results = model(image_path, conf=0.25, imgsz=640)
        if not results[0].masks:
            return None
        masks = results[0].masks.data.cpu().numpy()
        areas = [np.sum(mask) for mask in masks]
        best_idx = int(np.argmax(areas))
        mask = masks[best_idx]
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        mask_resized = cv2.resize(mask.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask_resized
    except Exception as e:
        st.warning(f"Error en YOLO: {e}")
        return None

def refine_edges_with_yolo(original_img, yolo_mask):
    img_np = np.array(original_img)
    h, w = img_np.shape[:2]
    if yolo_mask.shape != (h, w):
        yolo_mask = cv2.resize(yolo_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    yolo_mask = cv2.morphologyEx(yolo_mask, cv2.MORPH_CLOSE, kernel)
    alpha_smooth = cv2.GaussianBlur(yolo_mask.astype(np.float32), (5, 5), 0)
    alpha_final = np.clip(alpha_smooth, 0, 255).astype(np.uint8)
    result = np.dstack((img_np, alpha_final))
    return Image.fromarray(result, 'RGBA')

def refine_edges_fallback(img_no_bg):
    img_np = np.array(img_no_bg)
    rgb = img_np[:, :, :3]
    alpha = img_np[:, :, 3]
    binary_mask = (alpha > 10).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    dilated = cv2.dilate(binary_mask, kernel, iterations=1)
    alpha_smooth = cv2.GaussianBlur(dilated.astype(np.float32), (3, 3), 0)
    alpha_final = np.clip(alpha_smooth, 0, 255).astype(np.uint8)
    result = np.dstack((rgb, alpha_final))
    return Image.fromarray(result, 'RGBA')

# Interfaz de usuario
uploaded_files = st.file_uploader(
    "Cargar hasta 50 imágenes",
    type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
    accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 50:
        st.warning("Máximo 50 imágenes. Solo se procesarán las primeras 50.")
        uploaded_files = uploaded_files[:50]

    st.write(f"Imágenes cargadas: {len(uploaded_files)}")

    # Mostrar miniaturas originales
    st.subheader("Vista previa (original)")
    cols = st.columns(6)
    for i, file in enumerate(uploaded_files):
        with cols[i % 6]:
            st.image(file, use_container_width=True, caption=file.name[:18] + ("…" if len(file.name) > 18 else ""))

    if st.button("Eliminar Fondos (YOLOv8)"):
        processed_images = {}
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, file in enumerate(uploaded_files):
            status_text.text(f"Procesando {idx + 1}/{len(uploaded_files)}: {file.name}")
            progress_bar.progress((idx + 1) / len(uploaded_files))

            try:
                # Guardar temporalmente para que YOLO pueda leerlo
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name

                # Intentar con YOLO
                yolo_mask = get_yolo_mask(model, tmp_path)
                original_img = Image.open(file).convert("RGB")

                if yolo_mask is not None:
                    img_no_bg = refine_edges_with_yolo(original_img, yolo_mask)
                else:
                    # Fallback a rembg
                    output_data = remove(file.getvalue(), alpha_matting=True, post_process_mask=True)
                    img_no_bg = Image.open(io.BytesIO(output_data)).convert("RGBA")
                    img_no_bg = refine_edges_fallback(img_no_bg)

                processed_images[file.name] = img_no_bg

                # Limpiar archivo temporal
                os.unlink(tmp_path)

            except Exception as e:
                st.error(f"Error al procesar {file.name}: {e}")
                processed_images[file.name] = None

        status_text.text("✅ ¡Listo! Fondos eliminados con YOLOv8 — sin halos, con transparencia real.")
        st.success(f"Se procesaron {len([img for img in processed_images.values() if img is not None])} imágenes.")

        # Mostrar resultados
        st.subheader("Resultados")
        result_cols = st.columns(6)
        for i, (name, img) in enumerate(processed_images.items()):
            if img is not None:
                with result_cols[i % 6]:
                    st.image(img, use_container_width=True, caption=name.replace(".png", "_sin_fondo.png")[:18])

        # Botón para descargar todas
        zip_buffer = io.BytesIO()
        from zipfile import ZipFile
        with ZipFile(zip_buffer, "w") as zip_file:
            for name, img in processed_images.items():
                if img is not None:
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format="PNG")
                    zip_file.writestr(name.replace(os.path.splitext(name)[1], "_sin_fondo.png"), img_bytes.getvalue())
        zip_buffer.seek(0)

        st.download_button(
            label="📥 Descargar todas las imágenes procesadas (.zip)",
            data=zip_buffer,
            file_name="imagenes_sin_fondo.zip",
            mime="application/zip"
        )
else:
    st.info("Por favor, carga una o más imágenes para comenzar.")
import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
import pandas as pd 
import json 
import time
import base64 
from io import BytesIO 

# --- CONSTANTES ---
MODEL_FILENAME = "modelo_cancer_mobilenet.pth"
MODEL_PATH = Path(__file__).parent / MODEL_FILENAME 
CLASSES = ["Benigno", "Maligno", "Normal"]
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

# --- CONSTANTES DE BASE DE DATOS Y SEGURIDAD ---
PATIENT_DB_FILE = "patient_records.csv"
SEARCH_PASSWORD = "SALUD123" # Contraseña requerida para acceder a la búsqueda de pacientes

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="HopeScam tu asistente medico personal🩺🔬", page_icon="🩺", layout="wide")

# Inicialización de Session State para manejar el estado del formulario y los datos
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None
if 'auth_search' not in st.session_state:
    st.session_state.auth_search = False
if 'current_dui' not in st.session_state:
    st.session_state.current_dui = None # Para evitar duplicados en el guardado

# CSS personalizado global para la aplicación Streamlit
css = """
<style>
    /* Establece un color de fondo suave para toda la página */
    body {
        background-color: #FEECEF; /* Rosa muy claro, casi blanco */
    }
    /* Aumenta el ancho máximo del contenedor principal */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    
    /* Aplica estilo a TODOS los expanders dentro de la aplicación */
    .st-expander {
        background-color: #FFE0E6; /* Rosa medio más visible */
        border-radius: 10px; /* Bordes redondeados */
        margin-bottom: 10px; /* Espacio entre desplegables */
        padding: 5px;
    }

    /* Estilo para los contenidos de TODOS los expanders para que el color de fondo sea consistente */
    .st-expander div[role="region"] {
        background-color: #FFE0E6 !important;
        border-radius: 0 0 10px 10px;
        padding: 10px;
    }

    /* Estilo del formulario de paciente */
    .patient-form-container {
        border: 2px solid #F06292; /* Un borde rosa más fuerte */
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        background-color: #FFF8FA; /* Fondo muy claro para el formulario */
    }
    /* Estilo para el encabezado de las columnas */
    div[data-testid="stVerticalBlock"] > div:first-child > div:first-child {
        margin-bottom: 20px; /* Espacio debajo del título principal */
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# --- FUNCIONES DE PERSISTENCIA DE DATOS ---

@st.cache_data
def load_patient_db():
    """Carga la base de datos de pacientes desde el CSV. Retorna un DataFrame."""
    try:
        if Path(PATIENT_DB_FILE).exists():
            # Cargar datos existentes, asegurando que DUI sea tratado como string
            df = pd.read_csv(PATIENT_DB_FILE, dtype={'DUI': str})
        else:
            # Crear un nuevo DataFrame con las columnas necesarias si el archivo no existe
            df = pd.DataFrame(columns=['Nombres', 'Apellidos', 'Edad', 'Género', 'DUI', 'Fecha_Registro', 'Resultado_IA', 'Confianza_IA'])
    except Exception as e:
        st.error(f"Error al cargar la base de datos de pacientes: {e}")
        df = pd.DataFrame(columns=['Nombres', 'Apellidos', 'Edad', 'Género', 'DUI', 'Fecha_Registro', 'Resultado_IA', 'Confianza_IA'])
    return df

def save_patient_data(patient_data, scan_results):
    """Guarda los datos del nuevo paciente y los resultados del escaneo en el CSV."""
    df = load_patient_db()
    
    # Preparar el nuevo registro
    record = {
        'Nombres': patient_data['nombres'],
        'Apellidos': patient_data['apellidos'],
        'Edad': patient_data['edad'],
        'Género': patient_data['genero'],
        'DUI': patient_data['dui'],
        'Fecha_Registro': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Resultado_IA': scan_results.get('result', 'N/A'),
        'Confianza_IA': scan_results.get('confidence', 'N/A')
    }
    new_df = pd.DataFrame([record])
    
    # Concatenar y guardar. Usamos la sintaxis más moderna para evitar errores de tipo.
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(PATIENT_DB_FILE, index=False)
    
    # Limpiar el caché de la base de datos para asegurar que la próxima carga sea la más reciente
    load_patient_db.clear()
    st.session_state.current_dui = patient_data['dui'] # Guardar el DUI recién registrado
    st.success(f"✔️ ¡Análisis y datos de {patient_data['nombres']} guardados exitosamente!")

# --- FUNCIONES DE CARGA Y PREDICCIÓN ---

@st.cache_resource
def cargar_modelo():
    """Carga el modelo de PyTorch, la arquitectura MobileNetV2 y los pesos."""
    with st.spinner("Cargando modelo..."):
        if not MODEL_PATH.exists():
            st.error(f"❌ Error: No se encontró el archivo del modelo en la ruta: `{MODEL_PATH}`. Por favor, verifica el nombre del archivo y que esté en la misma carpeta que el script.")
            return None, None
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            model = models.mobilenet_v2(weights=None)
            # MobileNetV2 tiene 1280 features en la capa intermedia para el classifier por defecto
            model.classifier[1] = torch.nn.Linear(1280, len(CLASSES))
            
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            
            model = model.to(device)
            model.eval() 
            return model, device
        except Exception as e:
            st.error(f"🚨 Error cargando el modelo. Detalles: {e}")
            return None, None

def predecir_imagen(model, device, image: Image.Image):
    """Realiza el preprocesamiento de la imagen y la predicción."""
    try:
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD)
        ])
        
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            pred_idx = outputs.argmax(dim=1).item()
            
            return pred_idx, probs.cpu().numpy()
            
    except Exception as e:
        st.error(f"❌ Error durante la predicción: {e}")
        return None, None

# --- FUNCIÓN DE REPORTE (MODIFICADA PARA HTML Y IMAGEN) ---

def create_report_content(patient_data, scan_results):
    """Genera el contenido del reporte descargable en formato HTML con los datos del paciente y resultados, incluyendo la imagen."""
    if not patient_data or not scan_results:
        return "ERROR: No hay datos de paciente o resultados de escaneo disponibles."
        
    prob_lines = "".join([
        f"<li><strong>{cls}:</strong> {prob:.2f}%</li>" for cls, prob in zip(CLASSES, scan_results['probabilities_percent'])
    ])
    
    # Obtener la imagen en Base64 si está disponible
    image_b64 = scan_results.get('image_b64', '')
    image_tag = ""
    if image_b64:
        image_tag = f"""
            <div class="image-section">
                <h3>Imagen de Ultrasonido Analizada</h3>
                <img src="data:image/jpeg;base64,{image_b64}" alt="Imagen de Ultrasonido" class="ultrasound-image">
                <p><i>La imagen muestra el área analizada por el sistema de IA.</i></p>
            </div>
        """

    # Determinar clases y recomendación según resultado (mantener compatibilidad con Normal)
    result_value = scan_results.get('result', '')
    if result_value == 'Maligno':
        result_class = 'maligno'
        result_text_class = 'maligno-text'
        recommendation_text = 'ALTO RIESGO: Se recomienda CONSULTA MÉDICA ESPECIALIZADA URGENTE.'
    elif result_value == 'Normal':
        result_class = 'normal'
        result_text_class = 'normal-text'
        recommendation_text = 'SIN HALLAZGOS: No se observan patrones de riesgo relevantes. Mantener controles preventivos habituales.'
    else:
        # Asumimos Benigno por defecto
        result_class = ''
        result_text_class = 'benigno-text'
        recommendation_text = 'BAJO RIESGO: Se recomienda SEGUIMIENTO PROFESIONAL REGULAR.'

    # Estilos CSS internos para el reporte (más elaborado)
    style = """
    <style>
        @page { size: A4; margin: 15mm; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f8f8f8; }
        .report-container {
            width: 100%;
            max-width: 210mm; /* Ancho de A4 */
            margin: 20px auto;
            border: 1px solid #e0e0e0;
            box-shadow: 0 0 15px rgba(0,0,0,0.05);
            background-color: #ffffff;
            page-break-after: always; /* Para asegurar que cada reporte tenga su propia página si se imprime en lote */
        }
        .header-report {
            background-color: #2F3E46; /* Color oscuro */
            color: white;
            padding: 20px 30px;
            text-align: center;
            border-bottom: 5px solid #F06292; /* Línea rosa */
        }
        .header-report h1 {
            margin: 0;
            font-size: 1.8em;
            color: #ffffff;
        }
        .header-report p {
            margin: 5px 0 0;
            font-size: 0.9em;
            opacity: 0.9;
        }
        .content-section {
            padding: 25px 30px;
        }
        h2 {
            color: #F06292;
            border-bottom: 2px solid #F06292;
            padding-bottom: 8px;
            margin-top: 30px;
            font-size: 1.4em;
        }
        h3 {
            color: #3F51B5; /* Un azul para subencabezados */
            font-size: 1.1em;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        ul {
            list-style-type: none;
            padding-left: 0;
            margin-top: 10px;
        }
        ul li {
            margin-bottom: 8px;
            line-height: 1.5;
            color: #555;
        }
        ul li strong {
            color: #333;
        }
        .result-ia {
            background-color: #E8F5E9; /* Fondo verde claro para resultados */
            border-left: 5px solid #4CAF50; /* Borde verde */
            padding: 15px;
            margin-top: 20px;
            border-radius: 5px;
        }
        .result-ia.maligno {
            background-color: #FFEBEE; /* Fondo rojo claro */
            border-left: 5px solid #F44336; /* Borde rojo */
        }
        .result-ia.normal {
            background-color: #FFF8E1; /* Fondo amarillo muy claro */
            border-left: 5px solid #FFC107; /* Borde amarillo */
        }
        .result-text {
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
        }
        .maligno-text { color: #F44336; }
        .benigno-text { color: #4CAF50; }
        .normal-text { color: #BF360C; } /* tono oscuro para normal */

        .image-section {
            text-align: center;
            margin-top: 30px;
            padding: 15px;
            background-color: #f0f0f0;
            border-radius: 8px;
        }
        .ultrasound-image {
            max-width: 80%;
            height: auto;
            border: 1px solid #ccc;
            border-radius: 5px;
            margin-top: 10px;
        }
        .footer-report {
            text-align: center;
            margin-top: 40px;
            padding: 20px 30px;
            border-top: 1px solid #eee;
            color: #777;
            font-size: 0.85em;
            background-color: #fcfcfc;
        }
        .footer-report strong {
            color: #F06292;
            font-size: 1.1em;
        }
        .disclaimer {
            font-size: 0.8em;
            color: #999;
            margin-top: 20px;
            line-height: 1.4;
            border-top: 1px dashed #e0e0e0;
            padding-top: 15px;
        }
    </style>
    """

    content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Reporte EcoScan - {patient_data.get('apellidos', '')}</title>
        {style}
    </head>
    <body>
        <div class="report-container">
            <div class="header-report">
                <h1>ECOSCAN IA</h1>
                <h2>Diagnóstico Automatizado de Cáncer de Mama</h2>
                <p>Fecha del Reporte: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="content-section">
                <h2>📋 Datos del Paciente</h2>
                <ul>
                    <li><strong>Nombres:</strong> {patient_data.get('nombres', 'N/A')}</li>
                    <li><strong>Apellidos:</strong> {patient_data.get('apellidos', 'N/A')}</li>
                    <li><strong>Edad:</strong> {patient_data.get('edad', 'N/A')} años</li>
                    <li><strong>Género:</strong> {patient_data.get('genero', 'N/A')}</li>
                    <li><strong>DUI:</strong> {patient_data.get('dui', 'N/A')}</li>
                </ul>

                {image_tag} 

                <h2>🔎 Resultados del Escaneo por Inteligencia Artificial</h2>
                <div class="result-ia {result_class}">
                    <p class="result-text {result_text_class}">
                        Clase Predicha: {result_value.upper() if result_value else 'N/A'}
                    </p>
                    <p><strong>Nivel de Confianza:</strong> {scan_results.get('confidence', 'N/A')}%</p>
                    <p><strong>Recomendación del Sistema:</strong> {recommendation_text}</p>
                </div>

                <h3>📊 Distribución de Probabilidades</h3>
                <ul>
                    {prob_lines}
                </ul>

                <div class="disclaimer">
                    <strong>⚠️ Aviso Importante:</strong> Este reporte es generado por un sistema de Inteligencia Artificial y tiene fines predictivos solamente. **NO SUSTITUYE, bajo ninguna circunstancia, el diagnóstico o la consulta con un profesional médico especialista (oncólogo o radiólogo). Se DEBE buscar confirmación clínica.** La interpretación final debe ser realizada por un profesional de la salud calificado.
                </div>
            </div>

            <div class="footer-report">
                <span>Generado por: <strong>EcoScan IA</strong></span><br>
                <span>&copy; {pd.Timestamp.now().year} Todos los derechos reservados.</span>
            </div>
        </div>
    </body>
    </html>
    """
    return content.strip()

# --- VISTAS SECUNDARIAS ---

def view_nuevo_analisis(model, device):
    """Contenido de la pestaña 'Nuevo Análisis'."""
    
    # ----------------------------------------------------
    # 1. FORMULARIO OBLIGATORIO DEL PACIENTE
    # ----------------------------------------------------
    if not st.session_state.form_submitted:
        st.markdown('<div class="patient-form-container">', unsafe_allow_html=True)
        st.subheader("📝 1. Registro Obligatorio del Paciente")
        
        with st.form(key='patient_form'):
            # Campos de texto obligatorios
            nombres = st.text_input("Nombres del Paciente (Obligatorio)", key='nombres_input')
            apellidos = st.text_input("Apellidos del Paciente (Obligatorio)", key='apellidos_input')
            
            # Edad
            edad = st.number_input("Edad del Paciente (Años)", min_value=0, max_value=120, value=30, step=1, key='edad_input')
            
            # Género
            genero = st.selectbox("Género", ["Femenino"], key='genero_input')
            
            dui = ""
            # DUI condicional
            if edad >= 18:
                dui = st.text_input("DUI (Documento Único de Identidad) - Obligatorio para mayores de 18", 
                                     placeholder="Ej: 01234567-8", key='dui_input')
                
            submit_button = st.form_submit_button(label='✅ Guardar Datos e Ir al Análisis', type="primary")

            if submit_button:
                # Validación
                if not nombres or not apellidos:
                    st.error("Por favor, complete los campos de Nombres y Apellidos.")
                elif edad >= 18 and not dui:
                    st.error("Para pacientes mayores de 18 años, el DUI es obligatorio.")
                else:
                    # Guardar en Session State y cambiar el estado
                    st.session_state.patient_data = {
                        'nombres': nombres,
                        'apellidos': apellidos,
                        'edad': edad,
                        'genero': genero,
                        'dui': dui if edad >= 18 else 'Menor de 18',
                    }
                    st.session_state.form_submitted = True
                    st.success("Datos del paciente guardados. ¡Ahora puede proceder con el análisis de imagen!")
                    st.rerun() 
        
        st.markdown('</div>', unsafe_allow_html=True)

    # ----------------------------------------------------
    # 2. ANÁLISIS DE IMAGEN (SOLO si el formulario fue llenado)
    # ----------------------------------------------------
    if st.session_state.form_submitted:
        
        st.info(f"Paciente actual: **{st.session_state.patient_data['nombres']} {st.session_state.patient_data['apellidos']}** ({st.session_state.patient_data['edad']} años)")

        st.subheader("📸 2. Subida de Imagen de Ultrasonido")
        uploaded_file = st.file_uploader("Sube una imagen de ultrasonido (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"], key="uploader_nuevo")
        
        image_to_analyze = None
        if uploaded_file is not None:
            image_to_analyze = Image.open(uploaded_file).convert("RGB")
            st.image(image_to_analyze, caption="Imagen de Ultrasonido Subida", use_container_width=True)
            
            # Convertir la imagen a Base64 para incrustarla en el reporte HTML
            buffered = BytesIO()
            image_to_analyze.save(buffered, format="JPEG") # Guardar como JPEG para Base64
            image_b64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Inicializar o actualizar scan_results con la imagen Base64
            if st.session_state.scan_results is None:
                st.session_state.scan_results = {}
            st.session_state.scan_results['image_b64'] = image_b64

            st.subheader("🚀 3. Análisis Predictivo")
            
            # Botón de análisis
            if st.button("🚀 Iniciar Análisis Predictivo", type="primary", use_container_width=True, key="btn_analizar"):
                
                # Limpiar resultados anteriores (excepto la imagen Base64)
                current_image_b64 = st.session_state.scan_results.get('image_b64', '')
                st.session_state.scan_results = {'image_b64': current_image_b64} # Reiniciar, pero mantener la imagen
                
                # Mostrar spinner mientras se analiza
                with st.spinner("Analizando imagen con MobileNetV2..."):
                    pred_idx, probs = predecir_imagen(model, device, image_to_analyze)
                
                if pred_idx is not None:
                    resultado = CLASSES[pred_idx]
                    confianza = probs[pred_idx] * 100
                    
                    # Actualizar scan_results con los nuevos datos, manteniendo la imagen Base64
                    st.session_state.scan_results.update({
                        'result': resultado,
                        'confidence': f"{confianza:.2f}",
                        'probabilities_percent': [p * 100 for p in probs]
                    })
                    
                    # Guardar el registro completo en la base de datos (CSV)
                    save_patient_data(st.session_state.patient_data, st.session_state.scan_results)

                    st.subheader("✅ 4. Resultados del Diagnóstico por IA")
                    
                    # --- FEEDBACK MEJORADO ---
                    if resultado == "Maligno":
                        st.error("🔴 HALLAZGO CLASIFICADO COMO: MALIGNO o SOSPECHOSO")
                        st.write("La IA sugiere que las características de la imagen son compatibles con un patrón de alto riesgo. **Consulte a su médico de inmediato**.")
                    elif resultado == "Benigno":
                        st.info("🟢 HALLAZGO CLASIFICADO COMO: BENIGNO")
                        st.write("La IA sugiere que la imagen es compatible con un patrón de bajo riesgo. **Siempre se recomienda la revisión profesional**.")
                    else:
                        # Resultado "Normal" u otro valor no listado
                        st.success("🔵 HALLAZGO CLASIFICADO COMO: NORMAL / SIN HALLAZGOS")
                        st.write("La IA no detectó características sugestivas de riesgo. Mantener controles preventivos habituales y consulte a su médico si aparece algún síntoma o duda.")
                        
                    st.markdown("---")
                    
                    # Métrica principal
                    st.metric(
                        label=f"Clase Predicha", 
                        value=resultado.upper(), 
                        delta=f"{confianza:.2f}% de Confianza"
                    )

                    st.markdown("#### Distribución de Probabilidades")
                    
                    prob_data = {
                        "Clase": CLASSES,
                        "Probabilidad (%)": [p * 100 for p in probs]
                    }
                    
                    df_prob = pd.DataFrame(prob_data)
                    
                    st.bar_chart(
                        df_prob, 
                        x='Clase', 
                        y='Probabilidad (%)', 
                        height=250 
                    )
                
                else:
                    st.warning("⚠️ No se pudo obtener el diagnóstico. Revisa el log de errores.")

        
        # ----------------------------------------------------
        # 3. OPCIÓN DE DESCARGA (SOLO si hay resultados)
        # ----------------------------------------------------
        if st.session_state.scan_results and st.session_state.scan_results.get('result') is not None: # Asegurarse de que haya resultados de IA
            report_content = create_report_content(
                st.session_state.patient_data, 
                st.session_state.scan_results
            )
            
            st.markdown("---")
            st.subheader("💾 Descargar Reporte (Paso Final)")
            
            # 1. Botón para DESCARGAR el archivo HTML
            st.download_button(
                label="⬇️ Descargar Reporte en formato HTML",
                data=report_content,
                file_name=f"Reporte_EcoScan_{st.session_state.patient_data['apellidos']}_{st.session_state.current_dui}.html",
                mime="text/html", 
                use_container_width=True,
                type="primary"
            )
            
            # 2. Botón que abre una ventana para imprimir/guardar como PDF
            # Codificar el HTML a Base64 para pasarlo a JavaScript
            report_b64 = base64.b64encode(report_content.encode('utf-8')).decode('utf-8')
            js_code = f"""
            <script>
            function openPrintWindow() {{
                var w = window.open('about:blank', '_blank');
                w.document.write(atob('{report_b64}'));
                w.document.close();
                w.focus(); // Enfocar la nueva ventana
                w.print(); // Intenta abrir el diálogo de impresión directamente
            }}
            </script>
            """
            st.markdown(js_code, unsafe_allow_html=True)
            
            if st.button("🖨️ Abrir Reporte en Ventana de Impresión (Recomendado para PDF)", use_container_width=True, type="secondary"):
                st.markdown(f'<script>openPrintWindow();</script>', unsafe_allow_html=True)

            st.markdown("---")
            # Botón para limpiar y empezar de nuevo
            if st.button("🔄 Registrar Nuevo Paciente", use_container_width=True, key="btn_nuevo"):
                st.session_state.form_submitted = False
                st.session_state.patient_data = {}
                st.session_state.scan_results = None # Limpiar también la imagen Base64
                st.session_state.current_dui = None
                st.rerun()
        
        elif uploaded_file is None:
            st.info("Sube una imagen de ultrasonido para iniciar el análisis. Los resultados y el botón de descarga aparecerán aquí.")

    elif not st.session_state.form_submitted:
          st.info("Comienza llenando el formulario de registro obligatorio del paciente.")

def view_buscar_paciente():
    """Contenido de la pestaña 'Buscar Paciente' con autenticación."""
    
    st.subheader("🔒 Búsqueda de Pacientes - Acceso Restringido")
    st.write("Ingrese la contraseña para acceder a la base de datos de registros.")
    
    # 1. Autenticación
    if not st.session_state.auth_search:
        password_input = st.text_input("Contraseña de Acceso", type="password")
        
        if st.button("🔑 Ingresar", type="primary"):
            if password_input == SEARCH_PASSWORD:
                st.session_state.auth_search = True
                st.success("Acceso concedido. Puede buscar pacientes ahora.")
                # st.rerun() 
            else:
                st.error("Contraseña incorrecta.")
    
    # 2. Búsqueda
    if st.session_state.auth_search:
        
        st.markdown("---")
        st.subheader("🔍 Base de Datos de Pacientes Registrados")
        
        # Cargar la base de datos completa
        df_db = load_patient_db()
        
        if df_db.empty:
            st.warning("La base de datos de pacientes está vacía.")
            return

        # Campo de búsqueda
        search_term = st.text_input("Buscar por Nombre, Apellido o DUI:", key='search_term').strip().lower()
        
        if search_term:
            # Crear máscara de filtro
            mask = (
                df_db['Nombres'].str.lower().str.contains(search_term, na=False) |
                df_db['Apellidos'].str.lower().str.contains(search_term, na=False) |
                df_db['DUI'].str.lower().str.contains(search_term, na=False)
            )
            
            filtered_df = df_db[mask]
            
            st.markdown(f"**Resultados encontrados:** {len(filtered_df)}")
            
            if not filtered_df.empty:
                # Mostrar resultados ordenados por fecha descendente
                st.dataframe(filtered_df.sort_values(by='Fecha_Registro', ascending=False), use_container_width=True)
            else:
                st.warning("No se encontraron pacientes que coincidan con el término de búsqueda.")
        else:
            # Mostrar toda la tabla por defecto (o los primeros N si la tabla es enorme)
            st.markdown("Mostrando los últimos 10 registros. Use el campo de búsqueda para filtrar.")
            st.dataframe(df_db.sort_values(by='Fecha_Registro', ascending=False).head(10), use_container_width=True)

        st.markdown("---")
        if st.button("🚪 Cerrar Sesión de Búsqueda"):
            st.session_state.auth_search = False
            st.rerun()

# --- INICIALIZACIÓN DE LA APLICACIÓN ---
model, device = cargar_modelo()

# --- INTERFAZ PRINCIPAL ---

if model:
    
    col1, col2 = st.columns([1, 1.5]) 
    
    # =======================================================
    # IZQUIERDA (col1): INFORMACIÓN Y CONTACTOS
    # =======================================================
    with col1:
        with st.expander("Acerca de: Información y Contactos ℹ️", expanded=True):
            st.subheader("⚠️ Este análisis NO sustituye una consulta con un especialista.")
            st.markdown("---")
            
            with st.expander("🎀 🌐 Contacto Médico en El Salvador", expanded=True):
                st.markdown("### 🏥 Hospital Oncológico del ISSS")
                st.markdown("* Teléfono: `2591-5000` ")
                st.markdown("* Dirección:San Salvador, entre la 25 Avenida Norte y la 1° Calle Poniente, justo frente a la estación de Bomberos y a un costado del Hospital Médico Quirúrgico, con entrada principal frente al Hospital Rosales")
                st.markdown("* Horarios (generales del ISSS): Lunes a viernes de 7:00 a.m. a 4:00 p.m. o 8:00 a.m. a 4:00 p.m")
                st.markdown("* Dirección web: https://www.isss.gob.sv/centros-de-atencion/")
                st.markdown("---")
                st.markdown("### 🎗️ Centro Internacional de Cáncer (CIC)")
                st.markdown("* Teléfono: `+503 2506-2001`")
                st.markdown("* 3era Calle Poniente, Block No 122 Colonia Escalón, San Salvador, San Salvador, El Salvador")
                st.markdown("* Horario: Lunes a Viernes, 8:00 AM - 5:00 PM.")
                st.markdown("* Dirección web: www.centrodecancer.com.sv.")
                st.markdown("---")
                st.markdown("### ⚖️ Instituto del Cáncer de El Salvador (ICES)")
                st.markdown("* Direccion: Dr. Narciso Díaz Bazán, es la 1ª Calle Poniente y 33 Avenida Norte, Colonia Escalón, San Salvador")
                st.markdown("* Horario: Lunes a Viernes de 6:30 a.m. a 3:30 p.m.")
                st.markdown("* Teléfono: `+503 2521-8282`  `2521-8200`")
                st.markdown("* Dirección web: ")
                st.markdown("---")

            with st.expander("🧠 Información General sobre el Cáncer de Mama", expanded=False):
                st.markdown("[Imágenes del Cáncer de Mama]")
                st.markdown(
                    """
                    El **Cáncer de Mama** es una enfermedad en la que las células de la mama crecen de forma descontrolada. Es el tipo de cáncer más común en mujeres a nivel mundial.

                    ### 🔬 Detección Temprana (¡Vital!)
                    La detección temprana es la mejor defensa. Si se detecta a tiempo, las tasas de supervivencia son muy altas. Los tres pilares de la detección son:
                    
                    1.  **Autoexamen:** Revisar tus mamas mensualmente para notar cambios.
                    2.  **Examen Clínico:** Realizado por un profesional de la salud.
                    3.  **Mamografía/Ultrasonido:** Estudios de imagen, recomendados anualmente a partir de cierta edad (varía según el riesgo).
                    """
                )
            
            with st.expander("⚙️ Guía de Uso Rápido de la Aplicación", expanded=False):
                st.markdown("Sigue estos pasos:")
                st.markdown("#### 1. Llenar Formulario")
                st.markdown("Completa los datos del paciente en la pestaña **'Nuevo Análisis'**.")
                st.markdown("#### 2. Analizar y Guardar")
                st.markdown("Sube la imagen y realiza el análisis. Los resultados se guardarán automáticamente.")
                st.markdown("#### 3. Buscar (Requiere Contraseña)")
                st.markdown("Usa la pestaña **'Buscar Paciente'** para consultar registros anteriores.")
    
    # =======================================================
    # DERECHA (col2): PESTAÑAS Y FUNCIONALIDAD CENTRAL
    # =======================================================
    with col2:
        
        # Encabezado estilizado
        st.markdown("""
            <div style="background-color: #2F3E46; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                <h2 style="color: white; margin: 0; font-size: 1.5em;">HopeScam tu asistente medico de confianza🩺🔬 - El Salvador</h2>
            </div>
            """, unsafe_allow_html=True)

        # Implementación de Pestañas
        tab_new, tab_search = st.tabs(["🆕 Nuevo Análisis", "📂 Buscar Paciente"])

        with tab_new:
            view_nuevo_analisis(model, device)

        with tab_search:
            view_buscar_paciente()
            
else:
    st.warning("⚠️ La aplicación no puede funcionar. Por favor, asegúrate de que el archivo del modelo esté en la ruta correcta y que no haya problemas de compatibilidad con PyTorch.")

# (El resto del archivo — chatbot y demás — lo dejé igual intencionalmente para no introducir cambios adicionales)

# ============================
#   CHATBOT SIN RETRASO
# ============================

# ----- CSS -----
# CSS personalizado global para la aplicación Streamlit
chat_css = """
<style>
    /* Establece un color de fondo suave para toda la página */
    body {
        background-color: #FAFAFA; /* Gris muy claro, limpio */
    }
    /* Aumenta el ancho máximo del contenedor principal */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    
    /* Aplica estilo a TODOS los expanders dentro de la aplicación */
    .st-expander {
        background-color: #F0F4F8; /* Gris azulado pálido, suave */
        border-radius: 10px; 
        margin-bottom: 10px; 
        padding: 5px;
    }

    /* Estilo para los contenidos de TODOS los expanders para que el color de fondo sea consistente */
    .st-expander div[role="region"] {
        background-color: #F0F4F8 !important; /* Consistencia */
        border-radius: 0 0 10px 10px;
        padding: 10px;
    }

    /* Estilo del formulario de paciente */
    .patient-form-container {
        border: 2px solid #E7A2B6; /* Borde Malva suave (Acento) */
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        background-color: white; /* Fondo blanco puro para el formulario */
    }
    /* Estilo para el encabezado de las columnas */
    div[data-testid="stVerticalBlock"] > div:first-child > div:first-child {
        margin-bottom: 20px; 
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)
st.markdown(chat_css, unsafe_allow_html=True)

# ----- Estado -----
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "chatbox_open" not in st.session_state:
    st.session_state.chatbox_open = False

# ----- Motor de respuestas -----
def responder(texto):
    t = texto.lower()

    if "hola" in t:
        return "Hola, soy el asistente de HopeScan. ¿Qué deseas saber?"
   
    if "que es ecoscan" in t or "qué es ecoscan" in t:
        return "EcoScan es una herramienta diseñada para analizar patrones relacionados con el cáncer de mama mediante inteligencia artificial. No reemplaza a un médico, pero sirve como apoyo informativo para la detección temprana y la educación preventiva."

    if "que hago si tengo miedo" in t or "tengo miedo" in t:
        return "Es completamente normal sentir miedo ante resultados médicos. Lo más importante es no quedarse con la duda y acudir a un especialista. Recuerda que la mayoría de hallazgos resultan benignos, y mientras más temprano consultes, más opciones y tranquilidad tendrás."

    if "sintomas" in t:
        return "Algunos síntomas que pueden alertar incluyen: bultos en el seno o axila, cambios en la piel, secreción anormal del pezón o dolor persistente. Sin embargo, muchos casos no presentan síntomas, por eso los chequeos periódicos son fundamentales."

    if "que hago ahora" in t or "que puedo hacer ahora" in t:
        return "Lo ideal es consultar a un médico especialista. Lleva la información del análisis, tus antecedentes y cualquier síntoma que hayas notado. Un profesional podrá guiarte con estudios como mamografías, ultrasonidos o una biopsia si fuera necesario."

    if "riesgos" in t:
        return "Los principales factores de riesgo incluyen: antecedentes familiares, edad mayor a 40 años, cambios genéticos (BRCA1/BRCA2), obesidad, tabaquismo, alcohol, y sedentarismo. Sin embargo, cualquier persona puede presentar alteraciones, por eso la prevención es clave."

    if "importancia del chequeo" in t or "por qué es importante" in t:
        return "Los chequeos permiten detectar alteraciones antes de que aparezcan síntomas, lo que aumenta significativamente las probabilidades de tratamiento exitoso. La detección temprana salva vidas."

    if "como prevenir" in t:
        return "Algunas medidas preventivas incluyen: mantener un peso saludable, hacer ejercicio, evitar fumar, limitar el alcohol, realizar autoexámenes mensuales y acudir a mamografías según la edad y recomendaciones médicas."

    if "autoexamen" in t:
        return "El autoexamen consiste en revisar tus senos una vez al mes para identificar cambios o bultos. Debe hacerse de pie, acostada y frente al espejo. Si encuentras algo inusual, consulta con un profesional."

    if "mamografia" in t or "mamografía" in t:
        return "La mamografía es un estudio de imagen que detecta alteraciones muy pequeñas antes de que puedan sentirse físicamente. Es recomendada en mujeres mayores de 40 años, o antes si existen factores de riesgo."

    if "ultrasonido" in t:
        return "El ultrasonido de mama es un estudio complementario que ayuda a diferenciar entre quistes y masas sólidas. Es útil en personas jóvenes o con tejido mamario denso."

    if "biopsia" in t:
        return "La biopsia consiste en tomar una pequeña muestra de tejido para analizarla en laboratorio. Es el método más preciso para confirmar si una masa es benigna o maligna."

    if "que significa patrón" in t:
        return "Un 'patrón' es la forma, densidad o estructura que el sistema identifica en la imagen analizada. Algunos patrones sugieren benignidad y otros requieren revisión médica más detallada."

    if "por que consultar a un medico" in t:
        return "Porque EcoScan no da diagnósticos. Solo un especialista puede confirmar la naturaleza de un hallazgo mediante estudios clínicos. Consultar a tiempo evita complicaciones y mejora los resultados."

    if "gracias" in t:
        return "Con gusto, estoy aquí para apoyarte. Si necesitas más información o tienes dudas sobre prevención, síntomas o resultados, pregúntame."
  
    if "dolor" in t or "me duele" in t:
        return "El dolor en el seno no siempre está relacionado con cáncer. Puede deberse a cambios hormonales, quistes, inflamaciones o tensión muscular. Sin embargo, si el dolor es persistente o viene acompañado de un bulto, consulta a un médico."

    if "bulto" in t or "bola" in t:
        return "Encontrar un bulto puede ser preocupante, pero la mayoría son benignos. Algunos pueden ser quistes o fibroadenomas. Aún así, es recomendable visitar a un médico para una evaluación completa."

    if "secreción" in t or "líquido" in t:
        return "La secreción del pezón puede tener varias causas: infecciones, cambios hormonales, medicamentos o, en casos raros, cáncer de mama. Si la secreción es sanguinolenta o espontánea, consulta a un especialista."

    if "cambio" in t and "piel" in t:
        return "Cambios en la piel como enrojecimiento, hundimientos, textura de 'piel de naranja' o inflamación pueden requerir evaluación médica. Son señales que deben observarse con atención."

    if "factores" in t or "causas" in t:
        return "Los factores de riesgo más comunes incluyen edad avanzada, antecedentes familiares, mutaciones genéticas, hábitos como tabaquismo y alcohol, y estilos de vida sedentarios. No obstante, cualquier persona puede desarrollar alteraciones aun sin factores de riesgo."

    if "tratamiento" in t:
        return "El tratamiento del cáncer de mama depende del diagnóstico final y puede incluir cirugía, radioterapia, quimioterapia, terapias hormonales o terapias dirigidas. El especialista determinará el mejor plan según cada caso."

    if "curar" in t or "cura" in t:
        return "El cáncer de mama detectado a tiempo tiene tasas de curación muy altas. La detección temprana mejora significativamente las posibilidades de un tratamiento exitoso."

    if "probabilidad" in t:
        return "Las probabilidades dependen de múltiples factores: tipo de lesión, antecedentes, edad, y estudios clínicos. El análisis de EcoScan no da porcentajes de diagnóstico, solo ayuda a identificar patrones que deben revisarse."

    if "peligroso" in t or "riesgoso" in t:
        return "Un hallazgo catalogado como sospechoso no significa que sea peligroso de inmediato, pero sí requiere atención médica pronta para descartar o confirmar cualquier condición."

    if "test" in t or "analisis" in t:
        return "El análisis de EcoScan revisa patrones visuales en imágenes y determina si coinciden con categorías benignas o malignas basadas en datos de entrenamiento. No es un diagnóstico médico, solo una herramienta informativa."

    if "funcionas" in t or "cómo funcionas" in t:
        return "Funciono analizando patrones en base a modelos de inteligencia artificial entrenados con datos médicos. Mi función es apoyar, explicar resultados y recomendar acciones responsables."

    if "modelo" in t:
        return "El modelo utilizado analiza características visuales en imágenes. Genera una predicción basada en similitud con patrones aprendidos durante su entrenamiento. Esto solo es una orientación y no sustituye una evaluación profesional."

    if "confianza" in t or "accuracy" in t:
        return "La confianza indica qué tan segura está la IA del patrón detectado. No representa un diagnóstico ni un porcentaje de cáncer. Solo mide la certeza técnica del análisis matemático."

    if "estoy bien" in t or "está bien" in t:
        return "Si el resultado fue benigno, es una señal tranquila, pero lo ideal es llevarlo a un especialista para confirmarlo. Si fue maligno, la atención temprana es la clave."

    if "ansioso" in t or "ansiosa" in t or "preocupado" in t:
        return "Es normal sentirse así. Busca apoyo emocional y no te quedes con la incertidumbre. Consulta a un profesional para obtener respuestas claras y precisas."

    if "importancia" in t:
        return "La importancia de este análisis es orientar, educar y promover la prevención. La detección temprana siempre es la mejor estrategia."

    if "más información" in t or "ayuda" in t:
        return "Puedo ayudarte a entender síntomas, resultados, signos de alerta, recomendaciones y la importancia de estudios médicos. ¿Sobre qué aspecto deseas saber más?"  
   
    if "que es benigno" in t:
        return "Un resultado benigno significa que el patrón detectado no muestra señales compatibles con cáncer de mama. Normalmente se trata de alteraciones que no representan riesgo grave, como quistes simples o masas no peligrosas. Sin embargo, aunque el hallazgo es tranquilizador, siempre es importante que consultes con un médico especialista para confirmar el diagnóstico con métodos clínicos y estudios adicionales si fuera necesario"
  
    if "que es maligno" in t:
        return "Un resultado maligno indica que el patrón analizado tiene características asociadas a cáncer de mama. Esto no es un diagnóstico definitivo, pero sí una señal importante para acudir cuanto antes a un médico especialista. El especialista podrá realizar estudios como mamografías, ultrasonidos o biopsias, que confirman el diagnóstico y permiten iniciar un tratamiento adecuado lo más temprano posible."
   
    return "No entendí muy bien, ¿puedes explicarlo un poco más?"

# ----- Capturar mensaje primero -----
user_input = st.chat_input("Escribe tu mensaje...")

if user_input:
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    respuesta = responder(user_input)
    st.session_state.chat_messages.append({"role": "assistant", "content": respuesta})

# ----- Burbuja -----
st.markdown(
    """
    <div class="chat-bubble" onclick="var box = window.parent.document.getElementById('chatbox'); 
    box.style.display = (box.style.display === 'none' ? 'block' : 'none');">💬</div>
    """,
    unsafe_allow_html=True,
)

# ----- Caja del Chat -----
chat_header = """
<div id="chatbox" class="chatbox" style="display:none;">
    <h4 style="margin:0; color:#F06292;">Asistente EcoScan</h4>
    <hr style="margin:5px 0 10px 0;">
</div>
"""
st.markdown(chat_header, unsafe_allow_html=True)

# Contenedor donde se dibujan mensajes
chat_area = st.container()

with chat_area:
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# Evitar que el chatbot se renderice dos veces
if "chat_rendered" not in st.session_state:
    st.session_state.chat_rendered = True
else:
    st.stop()

# ============================
#   CHATBOT SIN RETRASO
# ============================

# ----- CSS -----
chat_css = """
<style>
.chat-bubble {
    position: fixed;
    bottom: 25px;
    right: 25px;
    background-color: white;
    color: #333;
    width: 60px;
    height: 60px;
    border-radius: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 30px;
    cursor: pointer;
    z-index: 99999;
    box-shadow: 0 4px 8px rgba(0,0,0,0.25);
    border: 2px solid #F06292;
}

.chatbox {
    position: fixed;
    bottom: 100px;
    right: 25px;
    width: 320px;
    background: white;
    border-radius: 12px;
    padding: 12px;
    box-shadow: 0 5px 12px rgba(0,0,0,0.10);
    z-index: 99999;
}
</style>
"""
st.markdown(chat_css, unsafe_allow_html=True)

# ----- Estado -----
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "chatbox_open" not in st.session_state:
    st.session_state.chatbox_open = False

# ----- Motor de respuestas -----
def responder(texto):
    t = texto.lower()

    if "hola" in t:
        return "Hola, soy el asistente de EcoScan. ¿Qué deseas saber?"
    if "prevención" in t:
        return "Para prevenir el cáncer de mama, es importante la detección temprana..."
    if "benigno" in t:
        return "Un resultado benigno indica que no se observan características de cáncer."
    if "maligno" in t:
        return "Un resultado maligno significa que debes acudir a un especialista lo antes posible."

    return "No entendí muy bien, ¿puedes explicarlo un poco más?"

# ----- Capturar mensaje primero -----

if user_input:
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    respuesta = responder(user_input)
    st.session_state.chat_messages.append({"role": "assistant", "content": respuesta})

# ----- Burbuja -----
st.markdown(
    """
    <div class="chat-bubble" onclick="var box = window.parent.document.getElementById('chatbox'); 
    box.style.display = (box.style.display === 'none' ? 'block' : 'none');">💬</div>
    """,
    unsafe_allow_html=True,
)

# ----- Caja del Chat -----
chat_header = """
<div id="chatbox" class="chatbox" style="display:none;">
    <h4 style="margin:0; color:#F06292;">Asistente EcoScan</h4>
    <hr style="margin:5px 0 10px 0;">
</div>
"""
st.markdown(chat_header, unsafe_allow_html=True)

# Contenedor donde se dibujan mensajes
chat_area = st.container()

with chat_area:
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
# ==============================================
#  CHATBOT FIJO SIN DUPLICADOS
# ==============================================

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "chat_initialized" not in st.session_state:
    st.session_state.chat_initialized = True

# Motor simple de respuesta
def responder(texto):
    t = texto.lower()
    if "hola" in t:
        return "Hola, soy EcoScan. ¿Qué necesitas?"
    if "prevención" in t:
        return "Puedes reducir riesgos manteniendo controles regulares..."
    return "No entendí muy bien, ¿podrías repetirlo?"




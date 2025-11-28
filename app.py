import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# ============================================
#   BLOQUE 1: MOTOR ISA v1
# ============================================

def _calcular_dim(df, columnas):
    """
    Normaliza (z-score) y promedia las columnas indicadas
    para construir una dimensión.
    """
    imp = SimpleImputer(strategy="mean")
    vals = imp.fit_transform(df[columnas])
    scaler = StandardScaler()
    vals_scaled = scaler.fit_transform(vals)
    return vals_scaled.mean(axis=1)


def calcular_ISA_argumentacion(df):
    """
    Calcula DIM1, DIM2, DIM3 y el ISA (0–100) a partir
    de índices pre-calculados en el DataFrame.
    """

    DIM1_cols = [
        "TRU_SP Promedio Longitud Oracion",
        "TRU_SP Promedio Longitud Palabras letra",
        "INT_prop_density_normINT"
    ]

    DIM2_cols = [
        "TRU_TTR Diversidad léxica MTLD",
        "TRU_SM word2vec_sent_sim",
        "TRU_SM verb_syn_overlap"
    ]

    DIM3_cols = [
        "INT_prop_graph_density_normINT",
        "INT_components_count_normINT",
        "TRU_DM all types of discourse markers"
    ]

    df = df.copy()

    df["DIM1"] = _calcular_dim(df, DIM1_cols)
    df["DIM2"] = _calcular_dim(df, DIM2_cols)
    df["DIM3"] = _calcular_dim(df, DIM3_cols)

    # ISA bruto
    df["ISA_raw"] = 0.25 * df["DIM1"] + 0.40 * df["DIM2"] + 0.35 * df["DIM3"]

    # Normalizar a 0–100 (robusto para 1 solo caso)
    mn, mx = df["ISA_raw"].min(), df["ISA_raw"].max()
    if mx == mn:
        df["ISA_0_100"] = 50.0  # punto neutro cuando no hay variación
    else:
        df["ISA_0_100"] = (df["ISA_raw"] - mn) / (mx - mn) * 100

    return df


# ============================================
#   BLOQUE 2: OBTENER ÍNDICES (SIMULADO)
# ============================================

def obtener_indices_desde_texto(texto: str) -> pd.DataFrame:
    """
    Versión temporal simulada.
    Más adelante aquí se conectará el MetaSistema real.

    Por ahora devuelve un perfil fijo "promedio",
    solo para mostrar la estructura de la app.
    """

    datos_simulados = {
        "TRU_SP Promedio Longitud Oracion": [18.0],
        "TRU_SP Promedio Longitud Palabras letra": [5.1],
        "INT_prop_density_normINT": [0.42],

        "TRU_TTR Diversidad léxica MTLD": [72.0],
        "TRU_SM word2vec_sent_sim": [0.68],
        "TRU_SM verb_syn_overlap": [0.33],

        "INT_prop_graph_density_normINT": [0.47],
        "INT_components_count_normINT": [12],
        "TRU_DM all types of discourse markers": [14]
    }

    return pd.DataFrame(datos_simulados)


# ============================================
#   BLOQUE 3: APP STREAMLIT
# ============================================

st.title("🧠 Evaluador Argumentativo Automatizado (ISA_v1)")
st.subheader("MetaSistema TRU–ANA–INT | Argumentación Escolar")

st.write("""
Ingrese un texto argumentativo para obtener:

- **Tres dimensiones internas (DIM1–DIM2–DIM3)**
- **Puntaje Total ISA (0–100)**
- **Nivel de desempeño argumentativo**
""")

texto = st.text_area(
    "✍️ Pegue aquí el texto argumentativo del estudiante:",
    height=250
)

if st.button("Analizar Argumentación"):

    if len(texto.strip()) == 0:
        st.warning("Ingrese un texto primero.")
    else:
        # 1) Obtener índices (simulado por ahora)
        df_indices = obtener_indices_desde_texto(texto)

        # 2) Calcular ISA con el motor validado
        resultado = calcular_ISA_argumentacion(df_indices)
        puntaje = float(resultado["ISA_0_100"].iloc[0])

        dim1 = float(resultado["DIM1"].iloc[0])
        dim2 = float(resultado["DIM2"].iloc[0])
        dim3 = float(resultado["DIM3"].iloc[0])

        # 3) Interpretación del puntaje
        if puntaje >= 75:
            nivel = "Excelente"
            color = "green"
        elif puntaje >= 60:
            nivel = "Adecuado"
            color = "blue"
        elif puntaje >= 40:
            nivel = "Suficiente"
            color = "orange"
        else:
            nivel = "Insuficiente"
            color = "red"

        st.markdown(f"## 🎯 Puntaje ISA: **{puntaje:.1f} / 100**")
        st.markdown(f"### Nivel de Desempeño: **:{color}[{nivel}]**")

        st.write("### 📊 Dimensiones Internas (z-score)")
        st.write(f"- **DIM1 – Profundidad / Desarrollo**: `{dim1:.3f}`")
        st.write(f"- **DIM2 – Riqueza léxica / Cohesión**: `{dim2:.3f}`")
        st.write(f"- **DIM3 – Organización proposicional**: `{dim3:.3f}`")

        # Barra de progreso (solo si el puntaje es válido)
        if not np.isnan(puntaje):
            st.progress(min(max(puntaje / 100.0, 0.0), 1.0))

        st.write("---")
        st.caption("Motor ISA_v1 calibrado con corpus real (A–J) usando TRUNAJOD–Analiza–Interpreta.")

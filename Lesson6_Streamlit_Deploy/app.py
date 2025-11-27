import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from pathlib import Path

st.set_page_config(page_title="Churn Prediction", page_icon="🎯", layout="wide")

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "churn_model.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"


@st.cache_resource
def load_model():
    """Загружаем модель через pickle"""

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURE_NAMES_PATH, 'rb') as f:
        feature_names = pickle.load(f)
    return model, feature_names


def prepare_features(df, feature_names):
    """Приводим данные к формату обучения модели."""
    df_proc = df.copy()
    # Преобразуем категориальные признаки в строки (как при обучении)
    for col in feature_names:
        if col in df_proc.columns:
            if df_proc[col].dtype in ('object', 'bool'):
                df_proc[col] = df_proc[col].astype(str)
    return df_proc[feature_names]


# Загружаем модель
try:
    MODEL, FEATURE_NAMES = load_model()
except Exception as e:
    st.error(f"❌ Ошибка загрузки модели: {e}")
    st.stop()


# --- Основной интерфейс ---
st.title("🎯 Предсказание оттока клиентов")

# Загрузка CSV файла
uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])

if uploaded_file is None:
    st.info("👈 Загрузите CSV файл для начала работы")
    st.stop()

# Загружаем данные и делаем предсказания
df = pd.read_csv(uploaded_file)

try:
    features = prepare_features(df, FEATURE_NAMES)
    probabilities = MODEL.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    
    df['prediction'] = predictions
    df['prob_leave'] = probabilities
except Exception as e:
    st.error(f"❌ Ошибка при обработке данных: {e}")
    st.stop()


# --- Метрики ---
st.subheader("📊 Результаты")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Всего клиентов", len(df))
with col2:
    churn_rate = df['prediction'].mean() * 100
    st.metric("Предсказанный отток", f"{churn_rate:.1f}%")
with col3:
    avg_prob = df['prob_leave'].mean() * 100
    st.metric("Средняя вероятность", f"{avg_prob:.1f}%")


# --- Визуализации ---
st.subheader("📈 Визуализации")

pred_counts = df['prediction'].value_counts().sort_index()
fig1 = px.pie(
    values=pred_counts.values,
    names=['Останется' if idx == 0 else 'Уйдет' for idx in pred_counts.index],
    title="Распределение предсказаний"
)
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.histogram(df, x='prob_leave', nbins=30, title="Распределение вероятностей оттока")
st.plotly_chart(fig2, use_container_width=True)

if 'internationalplan' in df.columns:
    plan_df = df.groupby('internationalplan')['prob_leave'].mean().reset_index()
    fig3 = px.bar(plan_df, x='internationalplan', y='prob_leave', 
                  title="Средняя вероятность оттока по планам")
    st.plotly_chart(fig3, use_container_width=True)


# --- Форма для предсказания ---
st.subheader("🔮 Сделать предсказание для нового клиента")

with st.form("prediction_form"):
    col_left, col_right = st.columns(2)
    input_data = {}
    
    with col_left:
        st.write("**Категориальные:**")
        for col in FEATURE_NAMES:
            if df[col].dtype in ('object', 'bool'):
                unique_vals = sorted(df[col].astype(str).unique().tolist())
                input_data[col] = st.selectbox(col, unique_vals, key=f"cat_{col}")
    
    with col_right:
        st.write("**Числовые:**")
        for col in FEATURE_NAMES:
            if df[col].dtype not in ('object', 'bool'):
                val = float(df[col].median())
                input_data[col] = st.number_input(col, value=val, key=f"num_{col}")

    submitted = st.form_submit_button("Предсказать", use_container_width=True)

if submitted:
    try:
        input_df = pd.DataFrame([input_data])
        prepared_input = prepare_features(input_df, FEATURE_NAMES)
        prob = MODEL.predict_proba(prepared_input)[0][1]
        pred = int(prob >= 0.5)

        st.success(f"**Результат:** {'Уйдет' if pred else 'Останется'} | **Вероятность оттока:** {prob:.1%}")
        st.progress(prob, text=f"Вероятность оттока: {prob:.1%}")
    except Exception as e:
        st.error(f"❌ Ошибка при предсказании: {e}")

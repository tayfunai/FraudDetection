import streamlit as st
import pandas as pd


model_results = {
    "Logistic Regression": {"Training Time": 0.00799403190612793, "Testing Time": 0.0013611316680908203, "Accuracy": 0.76},
    "Linear SVC": {"Training Time": 0.1734452247619629, "Testing Time": 0.0016672611236572266, "Accuracy": 0.81},
    "KNN": {"Training Time": 0.01, "Testing Time": 0.22582411766052246, "Accuracy": 0.69},
    "Decision Tree": {"Training Time": 0.148301362991333, "Testing Time": 0.0013344287872314453, "Accuracy": 0.97},
    "Random Forest": {"Training Time": 1.3994243144989014, "Testing Time": 0.03636598587036133, "Accuracy": 0.96},
}

df = pd.DataFrame(model_results)

# Streamlit Display
st.title("Quyida modellarning o'qitish vaqtlari, bashoratlash vaqtlari, va klassifikatsiyalash aniqligi berilgan")
st.table(df.style.format("{:.7f}"))

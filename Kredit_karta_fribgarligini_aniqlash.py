import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from io import StringIO


# dff = pd.read_csv("val.csv")

# X = dff.drop(["label", "Unnamed: 0"], axis=1).iloc[:20]

svc_model = load("../models/svc_model.joblib")
lrc_model = load("../models/lrc_model.joblib")
knc_model = load("../models/knc_model.joblib")
dtc_model = load("../models/dtc_model.joblib")
rfc_model = load("../models/rfc_model.joblib")

df = pd.DataFrame({
    'first column': ["LinearSVC", "LogisticRegression", "KNeighborsClassifier", "DecisionTreeClassifier",
                     "RandomForestClassifier"]
})

st.write(
    "Biz tranzaktsiyamizni qonuniy yoki noqonuniy ikanligini anilashimiz uchun quyidagi modellardan birini tanlang!")
option = st.selectbox(
    '',
    df['first column'])

if option == "LinearSVC":
    model = svc_model
elif option == "LogisticRegression":
    model = lrc_model
elif option == "KNeighborsClassifier":
    model = knc_model
elif option == "DecisionTreeClassifier":
    model = dtc_model
elif option == "RandomForestClassifier":
    model = rfc_model

uploaded_file = st.file_uploader("Faylni tanlang")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    X = dataframe.drop(["label", "Unnamed: 0"], axis=1)[:20]

    st.write(X)


st.write("Quyida biz modelilmizni tekshirishimiz uchun tranzaktsiyalar xususiyatlari bilan keltirilgan:")


def normalization(df):
    print(X.columns)
    minmax_scaler = MinMaxScaler()
    absmax_scaler = MaxAbsScaler()

    minmax_scaler.fit(df[["cc_num", "amt", "zip", "lat", "city_pop", "unix_time", "merch_lat"]])
    absmax_scaler.fit(df[["long", "merch_long"]])

    df[["cc_num", "amt", "zip", "lat", "city_pop", "unix_time", "merch_lat"]] = minmax_scaler.transform(
        df[["cc_num", "amt", "zip", "lat", "city_pop", "unix_time", "merch_lat"]])
    df[["long", "merch_long"]] = absmax_scaler.transform(df[["long", "merch_long"]])

    return df


if st.button("Datasetni Normalizatsiya qilish"):
    dfff = normalization(X)
    st.write("Normalized dataset")
    st.write(dfff)

if uploaded_file is not None:
    new_df = pd.DataFrame({"Tranzaktsiya ID raqami": range(0, 20) })
    if st.button("Classifikatsiya qilish"):
        new_df["Tranzaktsiya holati"] = model.predict(X)
        mapping = {0: "qonuniy", 1: "noqonuniy"}
        data_mapped = new_df["Tranzaktsiya holati"].map(mapping)
        new_df["Tranzaktsiya holati"] = data_mapped
        st.write(new_df)

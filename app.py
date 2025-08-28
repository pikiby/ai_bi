import streamlit as st

st.title("Пример Streamlit")
name = st.text_input("Введите имя")
if st.button("Сказать привет"):
    st.write(f"Привет, {name}!")

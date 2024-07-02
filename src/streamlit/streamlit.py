import altair as alt
import streamlit as st
import io
from PIL import Image
import requests
import base64

st.set_page_config(
    page_title='Мультикамерное распознавание места',
    page_icon='📽️',
    layout='wide',
    initial_sidebar_state='expanded'
)

alt.themes.enable('dark')


@st.cache_data()
def detect(image_bytes):
    try:
        files = {'file': image_bytes}
        response = requests.post('http://api:8000/upload/', files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
        return None

def main():
    state = st.session_state.get('state', 'initial')

    if state == 'initial':
        st.title('📽️ Мультикамерное распознавание места')

        uploaded_image = st.file_uploader('Выберите изображение', type=['jpg', 'jpeg', 'png'])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='PNG')
            st.session_state['image_bytes'] = image_bytes.getvalue()
            st.session_state['state'] = 'working'
            st.rerun()

    elif state == 'working':
        with st.sidebar:
            st.title('📽️ Мультикамерное распознавание места')

            methods_list = [
                'MixVPR',
                'ResNet',
                'ViT',
            ]

            selected_method = st.selectbox('Выберите backbone', methods_list)

            if st.button('Загрузить другое изображение'):
                st.session_state['state'] = 'initial'
                st.rerun()

            elif st.button('Документация'):
                st.session_state['state'] = 'docs'
                st.rerun()

        image_bytes = st.session_state['image_bytes']
        response_json = detect(image_bytes)
        
        if response_json:
            image_with_bbox_1 = Image.open(io.BytesIO(base64.b64decode(response_json['top1_image'])))
            image_with_bbox_2 = Image.open(io.BytesIO(base64.b64decode(response_json['top2_image'])))
            image_with_bbox_3 = Image.open(io.BytesIO(base64.b64decode(response_json['top3_image'])))

            st.image(image_with_bbox_1.resize((600, 600)), caption='Результат 1')
            st.image(image_with_bbox_2.resize((600, 600)), caption='Результат 2')
            st.image(image_with_bbox_3.resize((600, 600)), caption='Результат 3')
        else:
            st.error("Failed to get valid response from server.")

    elif state == 'docs':
        pass


if __name__ == '__main__':
    main()

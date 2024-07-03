import altair as alt
import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from opr.models.place_recognition.base import ImageModel
from opr.modules.feature_extractors import ConvNeXtTinyFeatureExtractor
from opr.modules import Concat, GeM
from opr.datasets.itlp_outdoor import ITLPCampusOutdoor
import itertools
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import faiss

st.set_page_config(
    page_title='Мультикамерное распознавание места',
    page_icon='📽️',
    layout='wide',
    initial_sidebar_state='expanded'
)

alt.themes.enable('dark')

DATA_ROOT = '../public'
FILE_NAME = 'test.csv'
FILE_PATH = os.path.join(DATA_ROOT, FILE_NAME)

feature_extractor = ConvNeXtTinyFeatureExtractor(
    in_channels=3,
    pretrained=True,
)

pooling = GeM()
descriptor_fusion_module = Concat()

model = ImageModel(
    backbone=feature_extractor,
    head=pooling,
    fusion=descriptor_fusion_module,
)

model.load_state_dict(
    torch.load('weights/model.pt', 
               map_location=torch.device('cpu'))
    )

def extract_embeddings(model, descriptor_key, dataloader, device):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        test_embeddings_list = []
        for data in tqdm(dataloader, desc="Calculating test set descriptors"):
            batch = data
            batch = {e: batch[e].to(device) for e in batch}
            batch_embeddings = model(batch)
            test_embeddings_list.append(batch_embeddings[descriptor_key].cpu().numpy())
        test_embeddings = np.vstack(test_embeddings_list)
    return test_embeddings

def test_submission(
    test_embeddings: np.ndarray, dataset_df: pd.DataFrame, filename: str = "submission.txt"
) -> None:
    
    tracks = []

    for _, group in dataset_df.groupby("track"):
        tracks.append(group.index.to_numpy())
    n = 1
    ij_permutations = sorted(list(itertools.permutations(range(len(tracks)), 2)))

    submission_lines = []

    for i, j in tqdm(ij_permutations, desc="Calculating metrics"):
        query_indices = tracks[i]
        database_indices = tracks[j]
        query_embs = test_embeddings[query_indices]
        database_embs = test_embeddings[database_indices]

        dimension = database_embs.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(database_embs)

        _, indices = index.search(query_embs, n)

        submission_lines.extend(list(database_indices[indices.squeeze()]))

    with open(filename, "w") as f:
        for l in submission_lines:
            f.write(str(l)+"\n")
            
def get_ids():
    dataset = ITLPCampusOutdoor(
    DATA_ROOT,
    subset='test',
    sensors=('front_cam','back_cam'),
    load_semantics=True,
    #load_text_labels=True,
    mink_quantization_size=0.06,
    )

    dl = DataLoader(
        dataset=dataset,
        batch_size=256,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    embeddings = extract_embeddings(model, descriptor_key="final_descriptor", dataloader=dl, device='cpu')
    test_submission(embeddings, dataset_df=dl.dataset.dataset_df, filename="tuned_baseline_submission.txt")
    
    
def read_numbers_from_file(file_path):
    numbers = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                number = int(line.strip())
                numbers.append(number)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except ValueError as e:
        print(f"Reading error {e}")
    
    return numbers

@st.cache_data()
def detect(file_path):
    try:
        df = pd.read_csv(file_path)
        get_ids()
        return df
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

def visualize(df):
    ids = read_numbers_from_file('tuned_baseline_submission.txt')
    query_idxs = [ids[i] for i in range(len(ids)) if i % 2 == 0]
    db_idxs = [ids[i] for i in range(len(ids)) if i % 2 != 0]
    
    query_idx = st.slider('Select index', 0, len(query_idxs)-1, 0)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.scatter(df['tx'], df['ty'], label='Poses in database', s=3)
    
    x_true = df.iloc[query_idxs[query_idx]]['tx']
    y_true = df.iloc[query_idxs[query_idx]]['ty']
    
    ax.scatter(x_true, 
               y_true, 
               color='green', 
               label='GT query pose',
               alpha=0.8,
               s=100)
    
    x_pred = df.iloc[db_idxs[query_idx]]['tx']
    y_pred = df.iloc[db_idxs[query_idx]]['ty']
    
    ax.scatter(x_pred, 
               y_pred, 
               color='red', 
               label='Predicted pose',
               alpha=0.8,
               s=100)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    
    dist = ((x_true - x_pred) ** 2 + (y_true - y_pred) ** 2) ** 0.5
    st.write('Distance: ', dist)
    
    plt.tight_layout()
    st.pyplot(fig)

    true_front_path = os.path.join('/home/ubuntu/mipt-hackathon/src/public', df.iloc[query_idxs[query_idx]]['track'], 'front_cam', f"{df.iloc[query_idxs[query_idx]]['front_cam_ts']}.png")
    true_back_path = os.path.join('/home/ubuntu/mipt-hackathon/src/public', df.iloc[query_idxs[query_idx]]['track'], 'back_cam', f"{df.iloc[query_idxs[query_idx]]['back_cam_ts']}.png")
    
    pred_front_path = os.path.join('/home/ubuntu/mipt-hackathon/src/public', df.iloc[db_idxs[query_idx]]['track'], 'front_cam', f"{df.iloc[db_idxs[query_idx]]['front_cam_ts']}.png")
    pred_back_path = os.path.join('/home/ubuntu/mipt-hackathon/src/public', df.iloc[db_idxs[query_idx]]['track'], 'back_cam', f"{df.iloc[db_idxs[query_idx]]['back_cam_ts']}.png")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    query_front_img = plt.imread(true_front_path)
    axes[0, 0].imshow(query_front_img)
    axes[0, 0].axis('off')
    axes[0, 0].set_title('Query: Front Camera')

    pred_front_img = plt.imread(pred_front_path)
    axes[0, 1].imshow(pred_front_img)
    axes[0, 1].axis('off')
    axes[0, 1].set_title('Prediction: Front Camera')
    
    query_back_img = plt.imread(true_back_path)
    axes[1, 0].imshow(query_back_img)
    axes[1, 0].axis('off')
    axes[1, 0].set_title('Query: Back Camera')
    
    pred_back_img = plt.imread(pred_back_path)
    axes[1, 1].imshow(pred_back_img)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Prediction: Back Camera')
    
    plt.tight_layout()
    st.pyplot(fig)

def save_uploaded_file(uploaded_file, directory, filename):
    try:
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, filename)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def main():
    state = st.session_state.get('state', 'initial')

    if state == 'initial':
        st.title('📽️ Мультикамерное распознавание места')

        uploaded_file = st.file_uploader('Загрузите CSV', type=['csv'])
        if uploaded_file is not None:
            file_path = save_uploaded_file(uploaded_file, DATA_ROOT, FILE_NAME)
            if file_path:
                st.session_state['file_path'] = file_path
                st.session_state['state'] = 'working'
                st.rerun()

    elif state == 'working':
        with st.sidebar:
            st.title('📽️ Мультикамерное распознавание места')

            st.write(
                '''
                Задача предложена Центром когнитивного моделирования МФТИ
                при поддержке Фонда содействия инновациям
                '''
            )

            if st.button('Загрузить другой файл'):
                st.session_state['state'] = 'initial'
                st.rerun()

            elif st.button('Документация'):
                st.session_state['state'] = 'docs'
                st.rerun()

        file_path = st.session_state.get('file_path')
        
        df = detect(file_path)
        visualize(df)

            
    elif state == 'docs':
        st.title('Документация')
        st.write('Документация по использованию приложения.')
        
        if st.button('Вернуться к загрузке файла'):
            st.session_state['state'] = 'initial'
            st.rerun()


if __name__ == '__main__':
    main()

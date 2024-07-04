import pandas as pd

file_path = '/home/ubuntu/mipt-hackathon/src/public/test.csv'

df = pd.read_csv(file_path)

df['front_cam_ts'] = df['front_cam_ts'].apply(lambda x: ''.join(filter(str.isdigit, str(x))))
df['back_cam_ts'] = df['back_cam_ts'].apply(lambda x: ''.join(filter(str.isdigit, str(x))))

df.to_csv(file_path, index=False)
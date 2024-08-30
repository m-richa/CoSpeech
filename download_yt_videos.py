import os
import glob, tqdm
import pdb

save_folder = './data/jhon_oliver_long/video_shards'
os.makedirs(save_folder, exist_ok=True)
files = glob.glob('./data/jhon_oliver_long/Alex_Jones_-_Last_Week_Tonight_with_John_Oliver_HBO-WyGq6cjcc3Q/*')
files.sort()
for i in tqdm.tqdm(range(len(files))):
    file = files[i]
    name = file.split('/')[-1]
    st, en = name.split('-')[1:]
    st = st.replace('_', ':')
    en = en.replace('_', ':')
    # pdb.set_trace()
    save_path = os.path.join(save_folder, f'{name}.mp4')
    os.system(f'yt-dlp https://www.youtube.com/watch?v=WyGq6cjcc3Q -f mp4 --download-sections *{st}-{en} -o {save_path}')
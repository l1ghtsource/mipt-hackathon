import os
import re

def rename_png_files(folder_path):
    pattern = re.compile(r'\D')
    
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.png'):
                old_file_path = os.path.join(root, filename)
                new_filename = pattern.sub('', filename)
                
                if new_filename:
                    new_filename = str(int(new_filename)) + '.png'
                    new_file_path = os.path.join(root, new_filename)
                    os.rename(old_file_path, new_file_path)
                    print(f"Переименован файл: {old_file_path} -> {new_file_path}")

    print("Переименование файлов завершено.")

folder_path = '/home/ubuntu/mipt-hackathon/src/public'

rename_png_files(folder_path)


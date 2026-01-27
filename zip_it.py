import zipfile
import os

# Настройки
folder_to_zip = os.getcwd() 
output_zip = 'legal_sniper_clean.zip' 

# Исключаем venv и временные базы данных
exclude_dirs = {'venv', '__pycache__', '.git'} 
exclude_files = {
    output_zip, 
    'zip_it.py', 
    'index.faiss',  # Удаляем саму базу из архива
    'docs.pkl'      # Удаляем метаданные базы
} 

def create_zip():
    print(f"📦 Собираю ЧИСТЫЙ архив в {output_zip}...")
    try:
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(folder_to_zip):
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                
                for file in files:
                    if file in exclude_files:
                        continue
                        
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, folder_to_zip)
                    zipf.write(file_path, arcname)
                    print(f"  + {arcname}")

        print(f"\n✅ ГОТОВО! Стерильный проект здесь: {os.path.abspath(output_zip)}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    create_zip()
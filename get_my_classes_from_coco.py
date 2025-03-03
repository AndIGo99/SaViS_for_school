from pycocotools.coco import COCO
import os
import requests
from tqdm import tqdm
import json

# Параметры
ANNOTATION_FILE = 'annotations/annotations/instances_train2017.json'  # Путь к файлу аннотаций
OUTPUT_DIR = 'filtered_dataset/'                           # Директория для сохранения данных
TARGET_CATEGORIES = [
    'backpack', 'book', 'bottle', 'cell phone', 'frisbee',
    'handbag', 'knife', 'laptop', 'person', 'scissors',
    'skateboard', 'skis', 'snowboard', 'sports ball',
    'suitcase', 'tennis racket', 'umbrella'
]  # Список нужных категорий

# Создание выходной директории
os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'annotations'), exist_ok=True)

# Инициализация COCO API
coco = COCO(ANNOTATION_FILE)

# Шаг 1: Получение ID категорий
cat_ids = coco.getCatIds(catNms=TARGET_CATEGORIES)

if len(cat_ids) == 0:
    print(f"Нет категорий, соответствующих {TARGET_CATEGORIES}.")
    exit()

# Вывод найденных категорий
found_categories = coco.loadCats(cat_ids)
print(f"Найдены следующие категории: {[cat['name'] for cat in found_categories]}")

# Проверка ID категорий
print(f"ID категорий: {cat_ids}")

# Шаг 2: Получение ID изображений для каждой категории
category_image_counts = {}
all_img_ids = set()
for cat_id in cat_ids:
    img_ids_for_cat = coco.getImgIds(catIds=[cat_id])
    category_name = [cat['name'] for cat in coco.loadCats(cat_id)][0]
    category_image_counts[category_name] = len(img_ids_for_cat)
    all_img_ids.update(img_ids_for_cat)

print("Количество изображений для каждой категории:")
for cat_name, count in category_image_counts.items():
    print(f"{cat_name}: {count} изображений")

if len(all_img_ids) == 0:
    print(f"Нет изображений для выбранных категорий: {TARGET_CATEGORIES}")
    exit()

print(f"Найдено {len(all_img_ids)} уникальных изображений для всех категорий.")

# Шаг 3: Получение аннотаций, связанных с этими изображениями
ann_ids = coco.getAnnIds(imgIds=list(all_img_ids), catIds=cat_ids)
annotations = coco.loadAnns(ann_ids)

# Шаг 4: Получение информации об изображениях
images = coco.loadImgs(list(all_img_ids))

# Шаг 5: Получение информации о категориях
categories = coco.loadCats(cat_ids)

# Шаг 6: Создание нового JSON файла с фильтрованными аннотациями
filtered_data = {
    "info": coco.dataset["info"],
    "licenses": coco.dataset["licenses"],
    "images": images,
    "annotations": annotations,
    "categories": categories
}

# Сохранение фильтрованных аннотаций
output_annotation_file = os.path.join(OUTPUT_DIR, 'annotations', 'filtered_annotations.json')
with open(output_annotation_file, 'w') as f:
    json.dump(filtered_data, f)

print(f"Фильтрация аннотаций завершена. Результат сохранен в {output_annotation_file}")

# Функция для скачивания изображений
def download_images(img_list, output_dir):
    valid_images = 0
    for img in tqdm(img_list, desc="Скачивание изображений"):
        img_url = img['coco_url']
        img_name = os.path.join(output_dir, img['file_name'])
        
        # Проверка существования файла
        if os.path.exists(img_name):
            continue
        
        # Проверка URL
        response = requests.get(img_url, stream=True)
        if response.status_code != 200:
            print(f"Не удалось скачать {img_url}. Код ответа: {response.status_code}")
            continue
        
        # Скачивание изображения
        with open(img_name, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        valid_images += 1
    
    return valid_images

# Шаг 7: Скачивание изображений
downloaded_images = download_images(images, os.path.join(OUTPUT_DIR, 'images'))
print(f"Загружено {downloaded_images} изображений в {os.path.join(OUTPUT_DIR, 'images')}")

from torchvision import datasets, transforms
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import zipfile
import os

from torch.optim.lr_scheduler import StepLR

from sklearn import metrics
import seaborn as sns
from PIL import Image
import string

unnormalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)

def unzip_data(path_zip_data_file, path_to_unzip_folder):
	with zipfile.ZipFile(path_zip_data_file, 'r') as zip_ref:
	    zip_ref.extractall(path_to_unzip_folder)

def check_cuda():
	DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
	print(DEVICE)

def dataset_transform(dataset_folder, data_transforms):
	# Определяем преобразования для изображений

	# Загружаем данные
	dataset = datasets.ImageFolder(dataset_folder, transform=data_transforms)
	print('len(dataset)=',len(dataset))
	return dataset

def dataset_divide(dataset):
	# Разделяем данные на тренировочную и контрольную выборки
	train_size = int(0.8 * len(dataset))
	val_size = len(dataset) - train_size
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
	return train_dataset, val_dataset 

def plot_image_from_datset_by_name(image_path, image_name, is_show):	
	# Загружаем изображение
	image = Image.open(image_path + image_name)

	#print('plot_image_from_datset')	
	#print('data_dir = ', image_path)	
	print('dataset_image.size = ', image.size) 
	print('dataset_image.mode = ', image.mode) 
	
	if (is_show == 1):
		# Отобразите изображение с помощью Matplotlib
		plt.imshow(image)
		plt.axis('off')  # Отключаем оси для более чистого отображения
		plt.show()
	return image

def tensor_to_image_unnormalize(tensor, is_show):
	# Преобразуем тензор обратно в изображение для отображения
	# Отменяем нормализацию для корректного отображения

	unnormalized_image = unnormalize(tensor)

	#print('unnormalized_image')
	#print('unnormalize_image.shape', unnormalized_image.shape)

	# Преобразуем тензор в формат (H, W, C) и в numpy массив
	image_np = unnormalized_image.permute(1, 2, 0).numpy()
	#print('unnormalized_image_to_np')
	#print('image_np.shape', image_np.shape)
	if is_show == 1:
	    # Отображаем изображение
	    plt.imshow(image_np)
	    plt.axis('off')
	    plt.show()
	return image_np

def plot_image_from_datset_by_index(train_dataset, index, is_show):
	# Получаем объект и его метку
	input, label = train_dataset[index]
	print('plot_image_from_datset_by_index')	
	print('input.shape', input.shape)
	print('index = ', index)	
	print(f"Label: {label}")	

	unnormalized_image = unnormalize(input)

	# Преобразуем тензор в формат (H, W, C) и в numpy массив
	image_np = unnormalized_image.permute(1, 2, 0).numpy()
	# Отображаем изображение
	print('image_np----------------')	
	print('image_np.shape', image_np.shape)

	if (is_show == 1):
		plt.imshow(image_np)
		plt.axis('off')
		plt.show()
	return label	

def get_label_from_datset_by_index(train_dataset, index):
	label = plot_image_from_datset_by_index(train_dataset, index, 0)
	return label		

# def get_letter_by_number(number):
# 	if 1 <= number <= 26:
# 		return string.ascii_lowercase[number]
# 	else:
# 		return None

def get_letter_by_number(number):
	if 0 <= int(number) < 26:
		print('number_ = ', number)
		letter = string.ascii_uppercase[number]
		print('letter_ = ', letter)
		return letter
	else:
		return None


def get_letter_by_index(train_dataset, index):   
	label = get_label_from_datset_by_index(train_dataset, index)   
	letter = get_letter_by_number(label)
	return letter

def imshow(image, title):
	plt.figure(figsize=(6, 6))
	plt.imshow(np.transpose(image, (1, 2, 0)))  # Переводим тензор в формат (ширина, высота, количество каналов)
	plt.title(title)
	plt.axis('off')
	plt.show()

def imshow_save(image, title, output_path, is_show):
	plt.figure(figsize=(6, 6))
	if is_show == 1:
		plt.imshow(np.transpose(image, (1, 2, 0)))  # Переводим тензор в формат (ширина, высота, количество каналов)
	plt.title(title)
	plt.axis('off')

	# Получаем расширение файла из пути к картинке
	_, extension = os.path.splitext(output_path)

	# Сохраняем картинку в указанный файл с тем же расширением
	plt.savefig(output_path + extension, bbox_inches='tight', pad_inches=0)

	print(f"Картинка успешно сохранена в файл: {output_path + extension}")

	# Закрываем фигуру, чтобы освободить ресурсы
	plt.close()

def tensor_show(tensor_img, label, is_show):
	# Преобразуем тензор обратно в изображение для отображения
	# Отменяем нормализацию для корректного отображения
	unnormalize = transforms.Normalize(
		mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
		std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
	)
	unnormalized_image = unnormalize(tensor_img)
	# Преобразуем тензор в формат (H, W, C) и в numpy массив
	image_np = unnormalized_image.permute(1, 2, 0).numpy()

	letter = 'label = ' + get_letter_by_number(label) + '   num = ' + str(label)
	if is_show == 1:
		imshow(unnormalized_image, letter)	

def image_to_tensor(image_path, im_name, is_show):
	try:
		# Открываем картинку с помощью библиотеки PIL
		im_path_name = im_path + im_name
		image = Image.open(im_path_name)
		print('im_path_name ==', im_path_name)

		# Конвертируем картинку в массив NumPy
		image_array = np.array(image)
		# print('image_array.shape = ', image_array.shape)  # image_array.shape =  (28, 28, 3)
		# print('image_array[0][1] = ', image_array[0][1])  # image_array[0][1] =  [255 255 255]    # One Dot 
		# print('image_array[1] = ', image_array[1])  # image_array[1] =  [[255 255 255]  [255 255 255]  [255 255 255] ... ]  # One Line of matrix    

		#fig, axes = plt.subplots(plt_h_count, plt_w_count, figsize=(6, 3))
		fig, axes = plt.subplots(figsize=(1, 1))

		plt.imshow(image_array, cmap=plt.cm.gray)
		       
		# Переводим массив в тензор с формой (количество каналов, высота, ширина)
		tensor = np.transpose(image_array, (2, 0, 1))
		print('tensor.shape =', tensor.shape)        
		#tensor.shape = (3, 28, 28)

		return tensor

	except IOError:
		print(f"Не удалось открыть или прочитать файл: {image_path}")
		return None
	except Exception as e:
		print(f"Произошла ошибка при преобразовании картинки в тензор: {str(e)}")
		return None

def get_class_name(dataset, index):
    return dataset.classes[index]

def get_image_from_datset_by_index(train_dataset, index):
	# Получаем объект и его метку
	input, label = train_dataset[index]
	unnormalized_image = unnormalize(input)

	# Преобразуем тензор в формат (H, W, C) и в numpy массив
	image_np = unnormalized_image.permute(1, 2, 0).numpy()
	return image_np, label

def plot_count_of_images_from_datset_by_index(train_dataset, start_index, img_count, is_show):

	plt_w_count = 5
	plt_h_count = int(img_count/plt_w_count)
	if (img_count < plt_w_count) :
		plt_h_count = 1  

	print('plt_h_count = ', plt_h_count)		
	# Задаем базовую ширину и высоту для одного подграфика
	base_width = 12
	base_height = 2

	# Вычисляем общую высоту фигуры на основе количества строк
	fig_height = base_height * plt_h_count

	# Создаем фигуру и массив осей с динамически вычисленными размерами фигуры
	fig, axes = plt.subplots(plt_h_count, plt_w_count, figsize=(base_width, fig_height))
	ax = axes.ravel()  

	index = 0
	for i in range(start_index, start_index + img_count) :    
		image, label = get_image_from_datset_by_index(train_dataset, i)
		latter = get_letter_by_number(label)
		ax[index].imshow(image, cmap=plt.cm.gray)
		ax[index].set_title(str(latter))
		index = index + 1
		fig.tight_layout()

def test_model_by_dataset(trained_model, val_dataset, start_index, count_imgs, is_show, output_file):
	plt_w_count = 5
	plt_h_count = int(count_imgs / plt_w_count) 
	if (count_imgs % plt_w_count > 0):
		plt_h_count +=1
	if count_imgs < plt_w_count:
		plt_h_count = 1


	# Задаем базовую ширину и высоту для одного подграфика
	base_width = 12
	base_height = 2

	# Вычисляем общую высоту фигуры на основе количества строк
	fig_height = base_height * plt_h_count

	# Создаем фигуру и массив осей с динамически вычисленными размерами фигуры
	fig, axes = plt.subplots(plt_h_count, plt_w_count, figsize=(base_width, fig_height))
	ax = axes.ravel()

	index = 0
	for i in range(start_index, start_index + count_imgs):

		tensor, label = val_dataset[i]
		latter = get_letter_by_number(label)

		# # Получаем предсказания модели для одного изображения
		outputs = trained_model(tensor.unsqueeze(0))  # Добавляем размер батча
		# Находим индекс класса с максимальным значением
		_, predicted = torch.max(outputs.data, 1)
		latter_pred = get_letter_by_number(predicted.item())

		image = tensor_to_image_unnormalize(tensor, 0)

		ax[index].imshow(image.squeeze(), cmap=plt.cm.gray)
		title = f'True = {latter}, Pred = {latter_pred}'
		ax[index].set_title(title)
		ax[index].axis('off')  # Отключаем оси для лучшего отображения
		index += 1

	fig.tight_layout()

	# Сохранение фигуры в файл
	plt.savefig(output_file)

	# Отображение фигуры, если is_show=True
	if is_show:
		plt.show()
	else:
		plt.close()


#=============  fine tune CNN =========================================================================
def finetune_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=10):
    model.train()
    losses = []
    loss_history = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
            if i % 100 == 99:
                loss_history.append(running_loss / 100)
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        
        epoch_loss = running_loss / len(train_loader.dataset)
        losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        scheduler.step()
    
    return model, losses, loss_history		


#============================= CNN ======================================================
# Функция для обучения модели
def train_model(model, criterion, optimizer, scheduler, train_loader, num_epochs=10):
    losses = []
    loss_history = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
            if i % 100 == 99:
                loss_history.append(running_loss / 100)
                running_loss = 0.0
                
        epoch_loss = running_loss / len(train_loader.dataset)
        losses.append(epoch_loss)
        scheduler.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    return model, losses, loss_history    


# Функция для предсказания по одной картинке
def predict_image(image_path, image_name, model, transform):
    # Открываем изображение

    im_path_name = image_path + image_name
    image = Image.open(im_path_name).convert('RGB')
    
    # Применяем трансформации
    image = transform(image).unsqueeze(0)  # Добавляем batch dimension
    
    # Делаем предсказание
    with torch.no_grad():
        output = model(image)
    
    # Получаем предсказанный класс
    _, predicted = torch.max(output, 1)
    
    return predicted.item()    
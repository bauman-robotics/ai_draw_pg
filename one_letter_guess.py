#!/usr/bin/env python
# coding: utf-8

# In[1]:


##### Add to HEAD ======
def main():

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
    #import ai_funcs


    # from ai_funcs import unzip_data
    # from ai_funcs import check_cuda
    from ai_funcs import dataset_transform
    # from ai_funcs import dataset_divide
    # from ai_funcs import plot_image_from_datset_by_name
    # from ai_funcs import plot_image_from_datset_by_index
    # from ai_funcs import tensor_to_image_unnormalize
    # from ai_funcs import get_label_from_datset_by_index
    # from ai_funcs import get_image_from_datset_by_index
    # from ai_funcs import get_letter_by_number
    # from ai_funcs import get_letter_by_index
    # from ai_funcs import imshow
    # from ai_funcs import imshow_save
    # from ai_funcs import tensor_show
    # from ai_funcs import get_class_name
    # from ai_funcs import plot_count_of_images_from_datset_by_index
    # from ai_funcs import test_model_by_dataset

    # from ai_funcs import finetune_model
    from ai_funcs import train_model
    from ai_funcs import predict_image
    from ai_funcs import image_to_tensor


    # In[14]:


    #for jupyter ----> path = os.getcwd()
    # path = os.getcwd()
    #for python ----> path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.dirname(os.path.abspath(__file__))

    # Определяем имена файлов
    # model_name = path + 'model_self.pth'
    # checkpoint_name = path + 'checkpoint_self.pth'
    # train_dataset_dir = path + 'Upload'
    model_name      = os.path.join(path, 'model_self.pth')
    checkpoint_name = os.path.join(path, 'checkpoint_self.pth')
    train_dataset_dir = os.path.join(path, 'Upload')

    # Путь к тестовому изображению
    # test_image_path = path + 'Test_Img/'
    test_image_name = 'Test.png'
    test_image_path = os.path.join(path, 'Test_Img/')


    # In[15]:


    # #============================= CNN ======================================================
    #Аугментация данных:
    #======================================

    transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = dataset_transform(train_dataset_dir, transform)

    # # Разделяем данные на тренировочную и контрольную выборки
    # train_dataset, val_dataset = dataset_divide(dataset)

    #train_dataset = datasets.ImageFolder(root='path_to_train_data', transform=transform)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    #======================================

    print('len(dataset)', len(dataset))
    # print('len(train_dataset)', len(train_dataset))
    # print('len(val_dataset)', len(val_dataset))
    num_classes = len(dataset.classes)
    print('num_classes =', num_classes)


    # In[16]:


    #============================================================ nn =========================================

    class CNNClassifier(nn.Module):
        def __init__(self, num_classes):
            super(CNNClassifier, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.fc1 = nn.Linear(512 * 7 * 7, 512)  # Adjusted for additional pooling layers
            self.fc2 = nn.Linear(512, num_classes)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = self.pool(F.relu(self.conv4(x)))
            x = self.pool(F.relu(self.conv5(x)))
            x = x.view(-1, 512 * 7 * 7)  # Adjusted for additional pooling layers
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x


    # In[17]:


    #============================= CNN ======================================================
    # Получаем текущий рабочий каталог
    # path = os.getcwd()

    # # Определяем имена файлов
    # model_name = 'model.pth'
    # checkpoint_name = 'checkpoint.pth'

    # Создаем полные пути к файлам
    model_path_name = os.path.join(path, model_name)
    checkpoint_path_name = os.path.join(path, checkpoint_name)

    # Выводим результаты
    print('model_path_name =', model_path_name)
    print('checkpoint_path_name =', checkpoint_path_name)

    # Параметры модели
    num_classes = len(dataset.classes)
    print('num_classes =', num_classes)
    model = CNNClassifier(num_classes)
    trained_model = CNNClassifier(num_classes)
    # Определение функции потерь и оптимизатора
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    if os.path.exists(model_path_name):
        #print(trained_model)
        print('torch.load(model_path_name)')
        checkpoint = torch.load(model_path_name)
        print('torch.load(model_path_name)  Ok')
        try:
            trained_model.load_state_dict(checkpoint)
        except: 
            print('exception-----------------------')
            # Обучение модели
            trained_model, losses, loss_history = train_model(model, criterion, optimizer, scheduler, train_loader, num_epochs=30)
        
            # Сохранение модели и чекпоинтов
            torch.save({
                'model_state_dict': trained_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'losses': losses,
                'loss_history': loss_history,
        }, checkpoint_path_name)
        
        torch.save(trained_model.state_dict(), model_path_name)

        print('model.load_state_dict(checkpoint)  Ok')
    else:
        # Обучение модели
        trained_model, losses, loss_history = train_model(model, criterion, optimizer, scheduler, train_loader, num_epochs=30)
        
        # Сохранение модели и чекпоинтов
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'losses': losses,
            'loss_history': loss_history,
        }, checkpoint_path_name)
        
        torch.save(trained_model.state_dict(), model_path_name)


    # In[18]:


    #=============  fine tune CNN ========================================================================= 
    # data_dir = 'my_test_dataset'

    # transform = transforms.Compose([
    #     # transforms.RandomResizedCrop(224),
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     # transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


    # # losses = []
    # # loss_history = []

    # # Дообучение модели на новом датасете
    # trained_model, losses, loss_history = finetune_model(trained_model, train_loader, criterion, optimizer, scheduler, num_epochs=10)


    # In[7]:


    trained_model.eval()

    # Получаем предсказание
    print(f'test_image_path: {test_image_path}')
    print(f'test_image_name: {test_image_name}')
    print('predict_image')
    predicted_class = predict_image(test_image_path, test_image_name, trained_model, transform)
    print(f'Predicted class: {predicted_class}')

    return predicted_class


    # In[ ]:


    ##### ADD to end ########
    #     return predicted_class
    #     # !jupyter nbconvert --to script letters.ipynb
    #     pass

if __name__ == '__main__':
    main()  # скрипт запускается непосредственно


    # In[23]:


    #!get_ipython().system('jupyter nbconvert --to script one_letter_guess.ipynb')


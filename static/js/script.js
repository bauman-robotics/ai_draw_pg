

let last_selected_folder = '';

function saveImage(letter) {
    const dataURL = canvas.toDataURL('image/png');
    fetch(`/save_image`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: dataURL, folder: letter })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Image saved:', data);
        fillCanvasWithWhite(); // Инициализация холста с белым
        storImage()
        location.reload(); // Обновление страницы
        window.location.href = '/'; // Переход на корневую страницу
        loadFolderList(); // Обновление списка папок после сохранения изображения   
        // loadFolderContents(letter);             
    })
    .catch(error => {
        console.error('Error saving image:', error);
    });
}


function saveTestImageAndPredict(testFileName) {
    storImage();
    const dataURL = canvas.toDataURL('image/png');
    console.log('testFileName:', testFileName);
    fetch(`/save_and_predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: dataURL, fileName: testFileName})
    })
    .then(response => response.json())
    .then(data => {
        console.log('Image saved:', data);
        // fillCanvasWithWhite(); // Инициализация холста с белым
        //location.reload(); // Обновление страницы


        window.location.href = '/'; // Переход на корневую страницу
        loadFolderList(); // Обновление списка папок после сохранения изображения  

        // Регулярное выражение для проверки, что значение является буквой английского алфавита
        let regex = /^[a-zA-Z]$/;


        if (regex.test(data.predict)) {
            // Если значение соответствует регулярному выражению, сохраняем его в localStorage
            localStorage.setItem('predict', data.predict);
            console.log('Value saved to localStorage:', data.predict);
        } else {
            // Если значение не соответствует регулярному выражению, выводим сообщение об ошибке
            console.log('Error: The value is not a valid English alphabet letter.');
            localStorage.setItem('predict', "Busy");
        }

        // console.log('predict:', data.predict);

        // guessResultLetterButton.textContent = data.predict
        // Сохраняем предсказание в localStorage

        
        
    })
    .catch(error => {
        console.error('Error saving image:', error);
    });
}


function isCanvasEmpty() {
    const pixelData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    for (let i = 0; i < pixelData.length; i += 4) {
        if (pixelData[i] !== 255 || pixelData[i + 1] !== 255 || pixelData[i + 2] !== 255) {
            return false;
        }
    }
    return true;
}

// Функция для заполнения фона холста белым цветом
function fillCanvasWithWhite() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function drawImage(file) {
    const canvas = document.getElementById('editor');
    const ctx = canvas.getContext('2d');
    console.log('try to draw')
    const img = new Image();
    //img.src = "Upload/A/A_2024_0616_191848.png"
    img.src = file    
    img.onload = () => {
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
}

function storImage() {
    const canvas = document.getElementById('editor');
    const dataURL = canvas.toDataURL();
    localStorage.setItem('savedImage', dataURL);
}

function loadImage() {
    const canvas = document.getElementById('editor');
    const ctx = canvas.getContext('2d');
    const dataURL = localStorage.getItem('savedImage');
    
    if (dataURL) {
        const img = new Image();
        img.src = dataURL;
        img.onload = () => {
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        };
    }
}

function deleteSelectedFolders() {
    const checkboxes = document.querySelectorAll('.delete-folder-checkbox input[type="checkbox"]:checked');
    const foldersToDelete = Array.from(checkboxes).map(checkbox => checkbox.value);
    if (foldersToDelete.length === 0) {
        alert('Пожалуйста, выберите хотя бы одну папку для удаления.', foldersToDelete);
        console.log('Folders deleted:', foldersToDelete);
        return;
    }
    console.log('Folders deleted:', foldersToDelete.length);
    console.log('Folders deleted:', foldersToDelete);
    fetch('/delete_folders', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ folders: foldersToDelete })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Folders deleted:', data);
        loadFolderList(); // Обновление списка папок после удаления
    })
    .catch(error => {
        console.error('Error deleting folders:', error);
    });
}
//==============================================

function deleteAllFolders() {
    fetch('/delete_all_folders', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        console.log('All folders deleted:', data);
        loadFolderList(); // Обновление списка папок после удаления всех папок
    })
    .catch(error => {
        console.error('Error deleting all folders:', error);
    });
}

//==============================================

function deleteModel() {
    fetch('/delete_train_model', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        console.log('All folders deleted:', data);
        //loadFolderList(); // Обновление списка папок после удаления всех папок
    })
    .catch(error => {
        console.error('Error deleting all folders:', error);
    });
}
//==============================================

function deleteSelectedFiles() {

    const checkboxes = document.querySelectorAll('.delete-file-checkbox input[type="checkbox"]:checked');
    const filesToDelete = Array.from(checkboxes).map(checkbox => checkbox.value);

    if (filesToDelete.length === 0) {
        alert('Пожалуйста, выберите хотя бы один файл для удаления.');
        return;
    }
    console.log('filesToDelete.length:', filesToDelete.length);  // Логирование ответа
    fetch('/delete_files', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ folder: last_selected_folder, files: filesToDelete })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Files deleted:', data);
        loadFolderContents(last_selected_folder); // Обновление списка файлов после удаления
    })
    .catch(error => {
        console.error('Error deleting files:', error);
    });
}

function deleteAllFiles() {
    fetch('/delete_all_files', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ folder: last_selected_folder })
    })
    .then(response => response.json())
    .then(data => {
        console.log('All files deleted:', data);
        loadFolderContents(last_selected_folder); // Обновление списка файлов после удаления всех 
    })
    .catch(error => {
        console.error('Error deleting all files:', error);
    });
}

// Функция для загрузки списка папок из директории Upload
function loadFolderList() {
    countFiles();
    console.log('loadFolderList'); 
    fetch('/list_folders')
    .then(response => response.json())
    .then(data => {
        const fileListContent = document.getElementById('fileListContent');
        fileListContent.innerHTML = '';
        console.log('Server response:', data);  // Логирование ответа
        // Сортировка папок по имени
        data.folders.sort((a, b) => a.name.localeCompare(b.name));

        data.folders.forEach(folder => {
            const listItem = document.createElement('li');
            listItem.className = 'folder';
            listItem.innerHTML = `
                <div class="folder-details">
                    <i class="fas fa-folder folder-icon"></i>
                    <span class="folder-name">${folder.name}</span>
                </div>
                <div class="folder-file-count">
                    (${folder.file_count})              
                </div>
                <div class="delete-folder-checkbox-container">
                    <div class="delete-folder-checkbox">
                        <input type="checkbox" name="delete_folder" value="${folder.name}" id="delete_${folder.name}">
                       <!-- <label class="delete-label" for="delete_${folder.name}"></label> -->
                    </div>
                </div>                
            `;
            listItem.addEventListener('click', () => loadFolderContents(folder.name));
            fileListContent.appendChild(listItem);
        });
    })
    .catch(error => {
        console.error('Error loading folder list:', error);
    });
}

function loadFolderContents(folder) {
    countFiles();
    last_selected_folder = folder;
    console.log('loadFolderContents');  // Логирование ответа
    fetch(`/list_folder_contents?folder=${encodeURIComponent(folder)}`)
    .then(response => response.text())  // Изменено на .text() для диагностики
    .then(data => {
        console.log('Server response:', data);  // Логирование ответа
        try {
            const jsonData = JSON.parse(data);
            const folderContents = document.getElementById('folderContents');
            folderContents.innerHTML = '';
            jsonData.files.forEach(file => {
                const listItem = document.createElement('li');
                listItem.className = 'file';
                listItem.innerHTML = `
                    <div class="file-details">
                        <i class="fas fa-file file-icon"></i>
                        <span class="file-name">${file}</span>
                    </div>
                    <div class="delete-file-checkbox">
                        <input type="checkbox" name="delete_file" value="${file}" id="delete_${file}">
                     <!--   <label class="delete-label" for="delete_${file}"></label>  -->
                    </div>
                `;
                //===========  add =================
                listItem.addEventListener('click', () => loadFilesContents(folder, file));
                folderContents.appendChild(listItem);
            });
        } catch (error) {
            console.error('Error parsing JSON:', error);
        }
    })
    .catch(error => {
        console.error('Error loading folder contents:', error);
    });
}
//========================================================================================
let count_clicked = 0;


function loadFilesContents(folder, file) {
    count_clicked = (typeof count_clicked !== 'undefined' ? count_clicked : 0) + 1;
    const url = `/preview?folder=${folder}&file=${file}`;
    countFiles();
    fetch(url)
        .then(response => response.json()) // Предполагаем, что сервер возвращает JSON
        .then(data => {
            console.log('data', data);
            if (data.success) {
                console.log('data.success:', data.success);
                console.log('data.url:', data.url);                             
                drawImage(data.url); // Используем URL из ответа сервера                
            } else {
                console.log('something wrong:', data.message);
            }
        })
        .catch(error => {
            console.error('Fetch error:', error);
        });
}
//========================================================================================

const deleteButton = document.getElementById('deleteButton');
deleteButton.addEventListener('click', deleteSelectedFolders);

const deleteAllButton = document.getElementById('deleteAllButton');
deleteAllButton.addEventListener('click', deleteAllFolders);

const deleteButtonFiles = document.getElementById('deleteButtonFiles');
deleteButtonFiles.addEventListener('click', deleteSelectedFiles);

const deleteAllButtonFiles = document.getElementById('deleteAllButtonFiles');
deleteAllButtonFiles.addEventListener('click', deleteAllFiles);
//===================================

// Функция, которая будет вызвана при изменении состояния любого чекбокса
function onCheckboxChange(event) {
    // Здесь можно добавить любой код, который должен выполниться при изменении состояния чекбокса
    console.log("Checkbox changed:", event.target);
}

// Находим все чекбоксы внутри элементов с классом .delete-file-checkbox
const checkboxes = document.querySelectorAll('.delete-file-checkbox input[type="checkbox"]');

// Добавляем обработчик событий 'change' к каждому чекбоксу
checkboxes.forEach(checkbox => {
    checkbox.addEventListener('change', onCheckboxChange);
});



// ==== Count of files ========================
function countFiles() {
        console.log("Count Button Clicked:");
        fetch('/count_files')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok ' + response.statusText);
                }
                return response.text();
            })
            .then(data => {
                const fileCount = parseInt(data, 10);
                if (!isNaN(fileCount)) {
                    countFilesLabel.textContent = `${fileCount}`;                    
                } else {
                    countFilesLabel.textContent = 'Ошибка при получении количества файлов';
                }
            })
            .catch(error => {
                console.error('There has been a problem with your fetch operation:', error);
            });
}


document.addEventListener('DOMContentLoaded', () => {

    countFilesButton.addEventListener('click', countFiles);
});  


document.addEventListener('DOMContentLoaded', (event) => {
    const savedPredict = localStorage.getItem('predict');
    if (savedPredict) {
        guessResultLetterButton.textContent = savedPredict;
    }
});  
//=============================================

// Вызов функции для загрузки списка папок при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    loadFolderList();
    //loadFilesContents(folder, file)
    trainButton.textContent = "Обучить"

    loadImage()  
});   


import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, jsonify

from datetime import datetime, timedelta
import base64
from pathlib import Path
import shutil
from shutil import rmtree
import logging
from logging.handlers import RotatingFileHandler
import one_letter_guess
import string
from ai_funcs import get_letter_by_number
import sys
import subprocess

import psycopg2
from psycopg2 import sql
from datetime import timedelta

from time import sleep


# Получаем текущий рабочий каталог
path = os.getcwd()

UPLOAD_FOLDER        = Path(os.path.dirname(os.path.abspath(__file__)) + '/Upload').resolve()
ROOT                 = Path(os.path.dirname(os.path.abspath(__file__)) + '/static').resolve()
TEST_IMG_FOLDER_PATH = Path(os.path.dirname(os.path.abspath(__file__))+ '/Test_Img').resolve()
TRAIN_MODEL_PATH     = Path(os.path.dirname(os.path.abspath(__file__))).resolve()
LOCK_FILE_PATH       = Path(os.path.dirname(os.path.abspath(__file__))).resolve()
LOCK_FILE 			 = os.path.join(LOCK_FILE_PATH, "script.lock")

LOGFILE_PATH       = Path(os.path.dirname(os.path.abspath(__file__))).resolve()
LOGFILE 			 = os.path.join(LOGFILE_PATH, "cronout.log")


# ROOT                = Path(os.path.dirname(os.path.abspath(__file__)) + '/static').resolve()
# ROOT                = Path(path + '/static').resolve()

model_name = 'model_self.pth'  # To delete 

# app = Flask(__name__)
app = Flask(__name__, template_folder=ROOT)
startup_executed = False
start_train_line_num_db = 0

# UPLOAD_FOLDER = 'Upload'
# ROOT                = Path(os.path.dirname(os.path.abspath(__file__)) + '/static').resolve()


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

predict_lock = 0

@app.route('/')
def index():
    return (ROOT / 'index.html').resolve().read_bytes() 
#===============================================================

@app.route('/save_image', methods=['POST'])
def save_image():
    data = request.json
    if 'image' not in data or 'folder' not in data:
        return jsonify({'error': 'No image data or folder provided'}), 400

    image_data = data['image']
    folder_name = data['folder']
    folder_path = os.path.join(UPLOAD_FOLDER, folder_name)
    # os.makedirs(folder_path, exist_ok=True)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        folder_created = True
    else:
        folder_created = False


    # Удаляем префикс данных URL
    image_data = image_data.split(",")[1]
    image_data = base64.b64decode(image_data)

    # Создаем имя файла с текущей датой и временем
    filename = folder_name + "_" + datetime.now().strftime("%Y_%m%d_%H%M%S") + ".png"
    filepath = os.path.join(folder_path, filename)

    with open(filepath, 'wb') as f:
        f.write(image_data)

    write_to_db_all_count()
    return jsonify({'message': 'Image saved in folder ' + folder_name, 'filename': filename})
#===============================================================

@app.route('/save_and_predict', methods=['POST'])
def save_and_predict():
    data = request.json
    if 'image' not in data or 'fileName' not in data:
        return jsonify({'error': 'No image data or folder provided'}), 400

    image_data = data['image']
    filename = data['fileName']
    print('filename = ',filename)
    folder_path = TEST_IMG_FOLDER_PATH
    print('folder_path = ',folder_path)
    os.makedirs(folder_path, exist_ok=True)

    # Remove the URL data prefix
    image_data = image_data.split(",")[1]
    image_data = base64.b64decode(image_data)

    # Create a filename with the current date and time
    filepath = os.path.join(folder_path, filename)

    with open(filepath, 'wb') as f:
        f.write(image_data)

    #predict = one_letter_guess.main()
    #predict = one_letter_guess.py -- not work 

    print('run_script')
    predict = run_script()
    print('predict = ',predict)

    inc_predict_count()

    # global start_train_line_num_db
    # start_train_line_num_db = get_last_train_start_num_line()
    # print('start_train_line_num_db = ',start_train_line_num_db)
    letter = get_letter_by_number(predict)
    print('letter = ',letter)



    return jsonify({'message': 'Image saved in folder ' + filepath, 'filename': filename, 'predict': letter})

#===============================================================

@app.route('/list_folders', methods=['GET'])
def list_folders():
	try:
		folders = []
		for root, dirs, files in os.walk(UPLOAD_FOLDER):
			for dir_name in dirs:
				folder_path = os.path.join(root, dir_name)
				file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
				folders.append({"name": dir_name, "file_count": file_count})
			break  # Останавливаем os.walk после первого уровня
		return jsonify({"folders": folders})
	except Exception as e:
		return jsonify({"error": str(e)}), 500
#===============================================================

@app.route('/list_folder_contents', methods=['GET'])
def list_folder_contents():
    folder = request.args.get('folder')
    if not folder:
        return jsonify(error='Folder name is required'), 400

    folder_path = os.path.join(UPLOAD_FOLDER, folder)
    print('Folder path:', folder_path)  # Логирование пути к папке

    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return jsonify(error='Folder does not exist'), 404

    try:
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        return jsonify(files=files)
    except Exception as e:
        print('Error reading folder:', e)
        return jsonify(error='Failed to read folder'), 500
#===============================================================

@app.route('/delete_files', methods=['POST'])
def delete_files():
	data = request.get_json()
	files_to_delete = data.get('files')
	folder = data.get('folder')
	print('folder:', folder)
	print('files_to_delete:', files_to_delete)    
	if not files_to_delete:
		return jsonify({'error': 'No files selected'}), 400

	try:
		# Здесь должна быть логика удаления файлов и папок
		# Например, используя модуль os или shutil
		for files in files_to_delete:
			# Удаление файлов и папок
			# os.remove(folder) или shutil.rmtree(folder)
			print(f"Deleted files: {files}")

			# filepath = Path(files)
			filepath = Path(os.path.join(UPLOAD_FOLDER, folder, files))
			print(f"filepath: {filepath}")
			if filepath.is_dir():
				shutil.rmtree(filepath)
			else:
				filepath.unlink()
		return jsonify({'message': 'Files deleted successfully'})
	except Exception as e:
		return jsonify({'error': str(e)}), 500
#=========================================================

@app.route('/delete_folders', methods=['POST'])
def delete_folders():
	data = request.get_json()
	folders_to_delete = data.get('folders')
	print(f"Folders to Delete: {folders_to_delete}")
	if not folders_to_delete:
		return jsonify({'error': 'No folders selected'}), 400

	try:
		# Здесь должна быть логика удаления файлов и папок
		# Например, используя модуль os или shutil
		for folders in folders_to_delete:
			# Удаление файлов и папок
			# os.remove(folder) или shutil.rmtree(folder)
			print(f"Deleted folders: {folders}")
			# shutil.rmtree(folders)
			folredsPath = Path(os.path.join(UPLOAD_FOLDER, folders))
			# folder_path = os.path.join(UPLOAD_FOLDER, folders)
			print(f"folredsPath: {folredsPath}")

			if folredsPath.is_dir():
				print(f"folredsPath_is_dir: {folredsPath}")
				shutil.rmtree(folredsPath)
			else:
				print(f"folredsPath_is_not_dir: {folredsPath}")
				folredsPath.unlink()
		return jsonify({'message': 'Folders deleted successfully'})
	except Exception as e:
		return jsonify({'error': str(e)}), 500
#=========================================================

@app.route('/delete_all_folders', methods=['POST'])
def delete_all_folders():
	print(f"All Folders to Delete")
	folredsPath = Path(UPLOAD_FOLDER)
	delete_folders_in_folder(folredsPath)
	write_to_db_all_count() 
	return jsonify({'message': 'Folders deleted successfully'})
#=========================================================

@app.route('/delete_train_model', methods=['POST'])
def delete_train_model():
	print(f"Delete train model")
	folderPath = Path(TRAIN_MODEL_PATH)	
	print(f"trainModel folredPath:{folderPath}")
	print(f"model_name:{model_name}")
	full_file_path = os.path.join(folderPath, model_name)
	if os.path.exists(full_file_path):
		# Удалите файл
		os.remove(full_file_path)
		print(f"Файл {full_file_path} успешно удален.")
		inc_train_count()	
	else:
		print(f"Файл {full_file_path} не существует.")
		sleep(0.05)
		inc_train_count()	# temp 

	global start_train_line_num_db
	start_train_line_num_db = get_last_train_start_num_line()
	print('start_train_line_num_db = ',start_train_line_num_db)

	return jsonify({'message': 'Folders deleted successfully'})
#=========================================================

@app.route('/delete_all_files', methods=['POST'])
def delete_all_files():
	data = request.get_json()
	folder = data.get('folder')
	print(f"Delete All files from Folder: {folder}")
	folder_path = os.path.join(UPLOAD_FOLDER, folder)
	try:
		delete_files_in_folder(folder_path)
		write_to_db_all_count() 
		return jsonify({'message': 'Folders deleted successfully'})
	except Exception as e:
		return jsonify({'error': str(e)}), 500

#=========================================================

def delete_folders_in_folder(folder_path):

	for path in Path(folder_path).glob('*'):
		if path.is_dir():
			print(f'Remove {folder_path}')
			rmtree(path)
			write_to_db_all_count() 
		else:
			path.unlink()
			print(f'Remove file {folder_path}')
#=========================================================

def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            write_to_db_all_count()    
        except Exception as e:
            print(f'Ошибка при удалении файла {file_path}. {e}')
#=========================================================


@app.route('/preview')
def preview():
	folder = request.args.get('folder')
	filename = request.args.get('file')
	file_path = os.path.join(folder, filename)

	print('folder', folder)
	print('filename', filename)  	
	print('file_path', file_path)  

	file_ext = Path(file_path).suffix.lower()
	if file_ext in {'.png', '.jpg', '.jpeg', '.gif', '.webp'}:
		file_url = url_for('download', filename='')
		server_file_path = os.path.join(file_url, folder, filename)
		print('file_url:', server_file_path)  
		return jsonify({'success': True, 'url': server_file_path})
	else:
		return jsonify({'success': False, 'url': ''})


#=========================================================
@app.route('/Upload/<path:filename>', methods=['GET'])
def download(filename):
	"""Download a file."""

	full_path = os.path.join(app.root_path, UPLOAD_FOLDER)
	# logging.info('Downloading file= [%s]', filename)
	# logging.info(app.root_path)    
	# logging.info(full_path)    
	print('Downloading file= [%s]', filename) 
	print(app.root_path)   
	print(full_path) 

	return send_from_directory(full_path, filename, as_attachment=True)
#=========================================================

@app.route('/count_files')
def count_files():
		try:
			file_count = 0
			for root, dirs, files in os.walk(UPLOAD_FOLDER):
				file_count += len([f for f in files if os.path.isfile(os.path.join(root, f))])
			return str(file_count)
		except FileNotFoundError:
			return "0", 404        
#=========================================================

def is_running():
	print('is_running')
	return os.path.exists(LOCK_FILE)
#=====================================

def create_lock():
	print('create_lock')
	open(LOCK_FILE, 'w').close()
#=====================================
def remove_lock():
    if os.path.exists(LOCK_FILE):
        print('remove_lock')
        os.remove(LOCK_FILE)
    else:
        print(f'File {LOCK_FILE} does not exist')

#=====================================
def run_script():
	if is_running():
		print("Script is already running.----------------------------------------------")
		return -1
	try:
		create_lock()    
		print('one_letter_guess.main()') 
		# inc_predict_count()
		result = one_letter_guess.main()

		global start_train_line_num_db
		if (start_train_line_num_db != -1):
			write_end_of_train_db(start_train_line_num_db)
			start_train_line_num_db = -1

		return result
	except ValueError:
		print("Error: The output is not a valid integer.")
		return -1
	finally:
		print('finally')
		remove_lock()
#=========================================================

db_config = {
    'dbname': 'ai_draw_db',
    'user': 'ai_user',
    'password': 'ai_dbpas1',
    'host': 'localhost',
    # 'port': 'your_port'
}

#=========================================================

def write_to_db_all_count():

	last_train_count = get_last_train_count()
	print("last_train_count ==", last_train_count)
	last_predict_count = get_last_predict_count()
	print("last_predict_count ==", last_predict_count)	
	# Создание подключения к базе данных
	conn = psycopg2.connect(**db_config)
	ret = 0
	print("New row inserted started.")
	try:
	    # Создание курсора
	    with conn.cursor() as cursor:

	        # SQL-запрос для вставки новой строки
	        user_id = 'user'
	        num_all_files = count_files()
	        predict_count = last_predict_count
	        train_count = last_train_count


	        query = """
	        INSERT INTO ai_draw_tb (user_id,  num_all_files, predict_count, train_count)
	        VALUES (%s, %s, %s, %s);
	        """	        

	        # Выполнение запроса
	        cursor.execute(query, (user_id,  num_all_files, predict_count, train_count))
	        
	        conn.commit()
	        print("New row inserted successfully.")
	except Exception as e:
		conn.rollback()
		print(f"An error occurred: {e}")
	finally:
		conn.close() 	


#=========================================================

def get_last_train_count():
	# Создание подключения к базе данных
	conn = psycopg2.connect(**db_config)
	ret = 0
	try:
	    # Создание курсора
	    print('get_last_train_count started____:')
	    with conn.cursor() as cursor:
	        # SQL-запрос
	        query = """
	        SELECT line_num, user_id, train_count 
	        FROM ai_draw_tb
	        ORDER BY created_at DESC
	        LIMIT 1;
	        """
	        
	        # Выполнение запроса
	        cursor.execute(query)
	        
	        # Получение результата
	        result = cursor.fetchone()
	        
	        # Проверка, что результат не пустой
	        if result:
	            line_num, user_id, train_count = result
	            print(f"line_num: {line_num}, user_id: {user_id}, Train_Count: {train_count}")
	            if train_count == None :
	            	print(f"train_count______: {train_count}")
	            	return 0
	            else :
	            	print(f"train_count______: {train_count}")
	            	return train_count

	            print(f"ret: {ret}")	
	        else:
	            print("No data found.")
	            return 0
	except Exception as e:
		print(f"An error occurred: {e}")
		return None	            
	finally:
	    # Закрытие подключения
	    conn.close() 	   	        
#=========================================================

def inc_train_count():
	print("inc_train_count_started_____")	
	last_train_count = get_last_train_count()
	print("last_train_count ==", last_train_count)	

	last_predict_count = get_last_predict_count()
	print("last_predict_count ==", last_predict_count)	

	# Создание подключения к базе данных
	conn = psycopg2.connect(**db_config)
	ret = 0
	try:
	    # Создание курсора
	    with conn.cursor() as cursor:

	        # SQL-запрос для вставки новой строки
	        user_id = 'user'
	        predict_count = last_predict_count
	        train_count = last_train_count + 1
	        # train_Start = datetime.now().time()
	        train_Start = datetime.now().strftime('%H:%M:%S')

	        query = """
	        INSERT INTO ai_draw_tb (user_id, predict_count, train_count, train_Start)
	        VALUES (%s, %s, %s, %s);
	        """	        

	        # Выполнение запроса
	        cursor.execute(query, (user_id, predict_count, train_count, train_Start))
	        
	        conn.commit()
	        print("New row inserted successfully.")
	except Exception as e:
		conn.rollback()
		print(f"An error occurred: {e}")
	finally:
		conn.close() 
#=========================================================

def get_last_predict_count():
	# Создание подключения к базе данных
	conn = psycopg2.connect(**db_config)
	ret = 0
	try:
	    # Создание курсора
	    print('get_last_predict_count started:')
	    with conn.cursor() as cursor:
	        # SQL-запрос
	        query = """
	        SELECT line_num, user_id, predict_count
	        FROM ai_draw_tb
	        ORDER BY created_at DESC
	        LIMIT 1;
	        """
	        
	        # Выполнение запроса
	        cursor.execute(query)
	        
	        # Получение результата
	        result = cursor.fetchone()
	        
	        # Проверка, что результат не пустой
	        if result:
	            line_num, user_id, predict_count = result
	            print(f"line_num: {line_num}, user_id: {user_id}, predict_count: {predict_count}")
	            if predict_count == None :
	            	print('predict_count = ', predict_count)
	            	return 0
	            else :
	            	return predict_count

	            print(f"ret: {ret}")	
	        else:
	            print("No data found.")
	            return None
	except Exception as e:
		print(f"An error occurred______: {e}")
		return 0	            
	finally:
	    # Закрытие подключения
	    conn.close()
 #=========================================================

def inc_predict_count():
	print('inc_predict_started_1')
	last_train_count = get_last_train_count()
	print('inc_predict_started_2')	
	last_predict_count = get_last_predict_count()
	print('last_predict_count = ', last_predict_count)

	# Создание подключения к базе данных
	conn = psycopg2.connect(**db_config)
	ret = 0
	try:
	    # Создание курсора
	    with conn.cursor() as cursor:

	        # SQL-запрос для вставки новой строки
	        user_id = 'user'
	        train_count = last_train_count
	        predict_count = last_predict_count + 1

	        query = """
	        INSERT INTO ai_draw_tb (user_id, predict_count, train_count)
	        VALUES (%s, %s, %s);
	        """	        

	        # Выполнение запроса
	        cursor.execute(query, (user_id, predict_count, train_count))
	        
	        conn.commit()
	        print("New row inserted successfully.")
	except Exception as e:
		conn.rollback()
		print(f"An error occurred: {e}")
	finally:
		conn.close() 

#=========================================================

def get_last_train_start_num_line():
	# Создание подключения к базе данных
	conn = psycopg2.connect(**db_config)
	ret = 0
	try:
	    # Создание курсора
	    print('get_last_predict_count started:')
	    with conn.cursor() as cursor:
	        # SQL-запрос
	        query = """
	        SELECT line_num, user_id, predict_count
	        FROM ai_draw_tb
	        ORDER BY created_at DESC
	        LIMIT 1;
	        """
	        
	        # Выполнение запроса
	        cursor.execute(query)
	        
	        # Получение результата
	        result = cursor.fetchone()
	        
	        # Проверка, что результат не пустой
	        if result:
	            line_num, user_id, predict_count = result
	            print(f"line_num: {line_num}, user_id: {user_id}, predict_count: {predict_count}")
	            if line_num == None :
	            	print('line_num = ', line_num)
	            	return 0
	            else :
	            	return line_num

	            print(f"ret: {ret}")	
	        else:
	            print("No data found.")
	            return None
	except Exception as e:
		print(f"An error occurred______: {e}")
		return 0	            
	finally:
	    # Закрытие подключения
	    conn.close()		


#=========================================================

def get_train_start_time(line_num): #(table, column, condition_column, condition_value):
    """
    Получает значение заданной ячейки в заданной строке из базы данных PostgreSQL.

    :param table: Имя таблицы
    :param column: Имя столбца, значение которого нужно получить
    :param condition_column: Имя столбца, по которому будет определена строка
    :param condition_value: Значение столбца для определения строки
    :return: Значение ячейки или None, если строка не найдена
    """

    # Создание подключения к базе данных
    conn = psycopg2.connect(**db_config)

    try:
        # Создание курсора
        with conn.cursor() as cursor:

            query = """
            SELECT train_Start FROM ai_draw_tb WHERE line_num = %s;
            """
            # Выполнение запроса
            cursor.execute(query, (line_num,))

            # Получение результата
            result = cursor.fetchone()
            if result:
                return result[0]
            else:
                return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        conn.close()   
#=========================================================

def write_end_of_train_db(start_train_line_num_db):

	print('write_end_of_train_db')

	start_train_time = get_train_start_time(start_train_line_num_db)
	print('start_train_time', start_train_time)
	# Создание подключения к базе данных
	conn = psycopg2.connect(**db_config)
	try:
	    # Создание курсора
	    with conn.cursor() as cursor:

	        #train_End = datetime.now().time()
	        train_End = datetime.now().strftime('%H:%M:%S')
	        # SQL-запрос для обновления строки

	        query = """
	        UPDATE ai_draw_tb SET train_End = %s WHERE line_num = %s;
	        """

	        # Выполнение запроса
	        cursor.execute(query, (train_End, start_train_line_num_db))
	    
	        conn.commit()

	        query = """
	        UPDATE ai_draw_tb SET train_Interval = train_end::time - train_start::time WHERE line_num = %s;
	        """
	        # Выполнение запроса
	        cursor.execute(query, (start_train_line_num_db,))
	    
	        conn.commit()

	        print("Row updated successfully.")

	except Exception as e:
		conn.rollback()
		print(f"An error occurred: {e}")
	finally:
		conn.close() 

#=========================================================

def startup_function():
    print("Приложение Flask запущено!")
    print("remove_lock()")    
    remove_lock()

@app.before_request
def before_request():
    global startup_executed
    if not startup_executed:
        startup_function()
        startup_executed = True


#=====================================
def remove_logfile():
    if os.path.exists(LOGFILE):
        print('remove_logfile')
        os.remove(LOGFILE)
    else:
        print(f'File {LOGFILE} does not exist')        
#=========================================================

# CREATE OR REPLACE FUNCTION remove_milliseconds()
#    RETURNS TRIGGER AS $$
#    BEGIN
#        NEW.created_at := to_timestamp(to_char(NEW.created_at, 'YYYY-MM-DD HH24:MI:SS'), 'YYYY-MM-DD HH24:MI:SS');
#        RETURN NEW;
#    END;
#    $$ LANGUAGE plpgsql;
# #===========

# CREATE TRIGGER trg_remove_milliseconds
#    BEFORE INSERT OR UPDATE ON ai_draw_tb
#    FOR EACH ROW
#    EXECUTE FUNCTION remove_milliseconds();   
#=========================================================

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=5500, threaded=True, debug=True)


import flet as ft
import pandas as pd
import os
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
import preproc
import model

# Функция кластеризации данных (например, KMeans)
def perform_clustering(df, n_clusters=3):
    try:
        # Применяем кластеризацию
        data_clear = preproc.preproc(df.copy())
        df_copy = df.copy()

        df_copy['Cluster'] = model.main_clustering(data_clear)
        list_df = model.group_clusters(df_copy)
        return df_copy
    except Exception as ex:
        return None, str(ex)

# Функция обновления UI после кластеризации
def update_ui_after_clustering(df, page, table_container):
    # Удаляем старую таблицу с результатами кластеризации, если она есть
    for control in page.controls:
        if isinstance(control, ft.Container) and control not in [table_container]:
            page.remove(control)

    # Добавляем новую таблицу с результатами кластеризации
    new_table_card = create_table_with_cluster(df, page)
    page.add(new_table_card)

    # Добавляем виджеты для вывода информации по кластеру
    create_cluster_info_widget(df, page)

    page.update()

# Функция для создания таблицы с данными для кластеризации
def create_table(df_, page):
    df = df_[0:100]  # Ограничим вывод первыми 100 строками (можно настроить по необходимости)
    # Создание чекбоксов для отображаемых столбцов
    checkboxes = page.checkboxes  # Используем чекбоксы, хранящиеся в объекте page

    # Динамическое создание заголовков столбцов
    column_labels = []
    for col in df.columns:
        # Убедимся, что чекбокс для столбца существует
        if col in checkboxes:
            column_labels.append(ft.DataColumn(label=ft.Row([checkboxes[col], ft.Text(col)])))
        else:
            column_labels.append(ft.DataColumn(label=ft.Text(col)))

    # Динамическое создание строк таблицы
    rows = [
        ft.DataRow(cells=[ft.DataCell(ft.Text(str(row[col]))) for col in df.columns])
        for _, row in df.iterrows()
    ]

    table = ft.DataTable(columns=column_labels, rows=rows)

    # Оборачиваем таблицу в контейнер с отступами и карточку
    return ft.Container(
        content=ft.Card(
            content=ft.Column([ 
                ft.Text("  Данные для кластеризации", weight=ft.FontWeight.BOLD, size=18, color=ft.colors.PURPLE_700),
                # Оборачиваем таблицу в контейнер с вертикальной прокруткой
                ft.Row([ 
                    ft.Column([table], scroll=ft.ScrollMode.ALWAYS, height=600)  # Вертикальная прокрутка
                ], scroll=ft.ScrollMode.ALWAYS, width=1600)
            ], scroll=True),
            elevation=10,  # Более высокая тень для карточки
            color=ft.colors.PURPLE_50,  # Менее яркий фиолетовый фон (светлый оттенок)
        ),
        padding=20,  # Добавляем отступы вокруг карточки
        border_radius=15,  # Округление углов карточки
    )

# Функция для создания таблицы с результатами кластеризации
def create_table_with_cluster(df_, page):
    df = df_[0:100]
    # Сначала делаем столбец 'Cluster' первым
    cols = ['Cluster'] + [col for col in df.columns if col != 'Cluster']

    # Динамическое создание заголовков столбцов
    column_labels = [ft.DataColumn(label=ft.Text(col)) for col in cols]

    # Динамическое создание строк таблицы
    rows = [
        ft.DataRow(cells=[ft.DataCell(ft.Text(str(row[col]))) for col in cols])
        for _, row in df.iterrows()
    ]

    table = ft.DataTable(columns=column_labels, rows=rows)

    # Оборачиваем таблицу в контейнер с отступами и карточку
    return ft.Container(
        content=ft.Card(
            content=ft.Column([ 
                ft.Text("  Результаты кластеризации", weight=ft.FontWeight.BOLD, size=18, color=ft.colors.PURPLE_700),
                ft.Row([ 
                    ft.Column([table], scroll=ft.ScrollMode.ALWAYS, height=600)  # Вертикальная прокрутка
                ], scroll=ft.ScrollMode.ALWAYS, width=1600)
            ], scroll=True),
            elevation=10,  # Более высокая тень для карточки
            color=ft.colors.PURPLE_50,  # Менее яркий фиолетовый фон (светлый оттенок)
        ),
        padding=20,  # Добавляем отступы вокруг карточки
        border_radius=15,  # Округление углов карточки
    )

# Функция для создания виджета для вывода информации по кластеру
def create_cluster_info_widget(df, page):
    # Текстовая метка
    label = ft.Text("Вывести информацию по кластеру: ", weight=ft.FontWeight.BOLD, size=14, color=ft.colors.PURPLE_700)

    # Текстовое поле для ввода номера кластера
    cluster_input = ft.TextField(
        label="Введите номер кластера", 
        autofocus=True,
        width=200,
        on_change=lambda e: page.update()  # Для обработки текста (можно оставить пустым, если нужна лишь кнопка)
    )

    # Кнопка "Вывести"
    show_button = ft.ElevatedButton(
        "Вывести", 
        on_click=lambda e: on_show_button_click(e, page, df, cluster_input),
        bgcolor=ft.colors.PURPLE_400,
        color=ft.colors.WHITE,
        width=150,
        elevation=5,
    )

    # Добавляем все элементы на страницу
    page.add(ft.Row([label, cluster_input, show_button]))

def create_info_table(cluster_data_, page):
    cluster_data = cluster_data_[0:100]
    column_labels = [ft.DataColumn(ft.Text(col)) for col in cluster_data.columns]
    rows = [
        ft.DataRow(cells=[ft.DataCell(ft.Text(str(row[col]))) for col in cluster_data.columns])
        for _, row in cluster_data.iterrows()
    ]

    table = ft.DataTable(columns=column_labels, rows=rows)

    # Используем Container для добавления отступов и оборачиваем таблицу в карточку
    return ft.Container(
        content=ft.Card(
            content=ft.Column([table], scroll=ft.ScrollMode.ALWAYS, height=600),
            elevation=5,
            color=ft.colors.PURPLE_50
        ),
        padding=20  # Добавляем отступы вокруг контейнера
    )

# Функция обработки нажатия кнопки "Вывести"
def on_show_button_click(e, page, df, cluster_input):
    try:
        # Пробуем преобразовать введённое значение в число
        cluster_number = int(cluster_input.value)

        # Проверка корректности номера кластера
        num_clusters = df['Cluster'].nunique()
        if cluster_number < 0 or cluster_number >= num_clusters:
            page.add(ft.Text(f"Ошибка: Введите номер кластера от 0 до {num_clusters - 1}"))
            page.update()
            return

        # Фильтруем DataFrame по выбранному кластеру
        cluster_data = df[df['Cluster'] == cluster_number]

        name_featurs = cluster_data.columns
        name_featurs = list(filter(lambda x: x not in ['ID','Cluster'], name_featurs))
        list_feat_stat = ''

        for i in name_featurs:
            if len(cluster_data[i].unique()) < 5:
                counter = Counter(cluster_data[i])
                most_common_value, most_common_count = counter.most_common(1)[0]
                list_feat_stat += f' Значение, наиболее распростроненное в {i}: {most_common_value} в количестве {most_common_count}\n'
            else:
                try:
                    avg = np.average(cluster_data[i])
                    list_feat_stat += f' Среднее значение в {i}: {avg} \n' 
                except:
                    continue




        if cluster_data.empty:
            page.add(ft.Text(f"Кластер {cluster_number} пустой"))
        else:
            # Показываем информацию о кластере
            page.add(ft.Text(f" Таблица по кластеру {cluster_number}: \n" + list_feat_stat, 
                             weight=ft.FontWeight.BOLD, size=14, color=ft.colors.PURPLE_700))
            page.add(create_info_table(cluster_data, page))  # Функция для отображения таблицы с данными кластера
    except ValueError:
        page.add(ft.Text("Ошибка: Введите правильное целое число для номера кластера"))
        page.update()

# Функция обработки нажатия кнопки для кластеризации
def on_cluster_button_click(e, page, df, table_container, cluster_button):
    # Создаем новый DataFrame только с теми столбцами, которые были выбраны
    selected_columns = [col for col, checkbox in page.checkboxes.items() if checkbox.value]

    # Выбираем только те столбцы, которые выбраны
    user_df = df[selected_columns]

    # Запускаем кластеризацию
    df = perform_clustering(user_df)

    if df is not None:
        # Обновляем UI с результатами кластеризации
        update_ui_after_clustering(df, page, table_container)
    else:
        page.add(ft.Text(f"Ошибка кластеризации"))

# Функция обработки файла
def on_file_picker_result(e, page, df, button, cluster_button, file_picker, table_container):
    try:
        # Получаем путь выбранного файла
        file_path = e.files[0].path
        file_extension = os.path.splitext(file_path)[1].lower()

        # Загрузка данных в зависимости от расширения файла
        if file_extension == '.xlsx':  # Для Excel файлов
            df = pd.read_excel(file_path)
        elif file_extension == '.csv':  # Для CSV файлов
            df = pd.read_csv(file_path)
        else:
            page.add(ft.Text("Ошибка: Неподдерживаемый формат файла"))
            return

        # Удаление старой таблицы (если она есть)
        for control in page.controls:
            if isinstance(control, ft.Container) and control not in [button, cluster_button, file_picker]:
                page.remove(control)

        # Обновляем чекбоксы
        update_checkboxes(page, df)

        # Создание и добавление новой таблицы
        data_table_card = create_table(df, page)
        page.add(data_table_card)

        # Обновление глобального DataFrame
        page.df = df  # Сохраняем DataFrame на объекте страницы

    except Exception as ex:
        page.add(ft.Text(f"Ошибка при загрузке файла: {str(ex)}"))

# Функция для обновления чекбоксов
def update_checkboxes(page, df):
    # Сохраняем чекбоксы в объекте page
    page.checkboxes = {col: ft.Checkbox(value=True) for col in df.columns}

def main(page: ft.Page):
    page.theme_mode = ft.ThemeMode.LIGHT  # Светлая тема
    page.bgcolor = ft.colors.WHITE  # Белый фон для страницы
    # page.window.fullscreen = True
    page.window.width = 1920
    page.window.height = 1080
    page.scroll = ft.ScrollMode.ALWAYS

    # Переменная для хранения DataFrame
    df = pd.DataFrame()  # Пустой DataFrame
    page.df = df  # Сохраняем DataFrame на объекте страницы

    # Контейнер для таблицы
    table_container = ft.Container()

    # Кнопка для кластеризации
    cluster_button = ft.ElevatedButton(
        "Провести кластеризацию", 
        on_click=lambda e: on_cluster_button_click(e, page, page.df, table_container, cluster_button),
        icon=ft.icons.COMPUTER,
        bgcolor=ft.colors.PURPLE_400,  # Фиолетовая кнопка
        color=ft.colors.WHITE,  # Белый текст на кнопке
        elevation=5,  # Легкая тень
        width=250,  # Увеличенная ширина кнопки
    )

    # Кнопка выбора файла
    button = ft.ElevatedButton(
        "Выбрать файл", 
        on_click=lambda e: file_picker.pick_files(),
        icon=ft.icons.FILE_OPEN,
        bgcolor=ft.colors.PURPLE_400,  # Фиолетовая кнопка
        color=ft.colors.WHITE,  # Белый текст на кнопке
        width=250,  # Увеличенная ширина кнопки
        elevation=5,  # Легкая тень
    )

    # Кнопка выбора файла
    file_picker = ft.FilePicker(on_result=lambda e: on_file_picker_result(e, page, df, button, cluster_button, file_picker, table_container))

    # Добавляем кнопки на страницу
    page.add(ft.Row([button, file_picker, cluster_button]))

    # Сохранение ссылки на контейнер с таблицей
    page.add(table_container)

ft.app(target=main)

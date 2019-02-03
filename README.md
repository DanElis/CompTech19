# Автоматическое выделение разломов в сейсмических изображениях
В нынешнее время все сильнее развиваются возобновляемые источники энергии, которые пытаются вытеснить нефтегазовую отрасль. Однако последняя не исчерпала себя. Остаются большие запасы, которые можно разрабатывать, чтобы отрасль продолжала успешно конкурировать с солнечными панелями, ветрогенераторами и другими источниками энергии. Для развития нефтегазовой отрасли нужны новые месторождения. Сейчас их обнаруживают с помощью анализа сейсмических разрезов. Таких изображений много, так что если размечать их вручную, работа занимает слишком много времени.
Обучили нейросеть размечать разломы на сейсмических разрезах и упаковали результат в ядро для плагина для пакета OpendTect. 
Из этого документа узнаете:
  * Как установить плагин.
  * Как его использовать.
  * Как обучить нейросеть на вашей собственной выборке.
  
Ядро для плагина оптимизирует ручную разметку. К примеру, целые институты работают с данными с месторождения. Ядро уменьшает время работы на порядок (несколько лет - несколько месяцев) и количество людей занятых рутинной разметкой (сотни - десятки). 
## Цель
Найти оптимальное решение для выделения разломов в сейсмических разрезах с использованием нейронных сетей
## Задачи
1.	Импорт, генерация данных
2.	Автоматическое определение разломов
3.	Создание ПО
4.	План дальнейшего развития
* Фотография сейсмического разреза (сейсмический разрез) -  форма временного разреза, составленного путем помещения последовательных записей одну за другой. Они представляют огромное количество данных в компактной форме. Чуть подробнее [здесь](http://www.ngpedia.ru/id368452p1.html). Не является фотографией слоев пород в земле!
* - ![Фотография сейсмического разреза ](https://docplayer.ru/docs-images/63/49217493/images/3-1.jpg)

* Автоэнкодер (автокодировщик) - [это нейронные сети прямого распространения](https://habr.com/ru/post/331382/), которые восстанавливают входной сигнал на выходе. Внутри у них есть скрытый слой, который представляет код, описывающий модель. Автоэнкодеры конструируются так, чтобы не было возможности точно скопировать вход на выходе. Их ограничивают в размерности кода (он меньше, чем размерность сигнала) или штрафуют за активации в коде. Входной сигнал восстанавливается с ошибками из-за потерь при кодировании, но, чтобы их минимизировать, сеть вынуждена учиться отбирать наиболее важные признаки.	

* Вариационный автоэнкодер -
* [OpendTect](https://dgbes.com/) - пакет интерпретации сейсмических данных с открытым исходным [кодом](https://github.com/OpendTect/OpendTect), который используется в отрасли. 
[SEG-Y/PC]( http://www.xgeo.ru/index.php/ru/zagruzki/probnye-versii.html#downloads_docs_ru) - приложение для просмотра и редактирования сейсмических файлов SEG-Y/PC 
[Документация](http://www.xgeo.ru/index.php/ru/zagruzki/probnye-versii.html#downloads_docs_ru) на него.
## Сценарий использования
### Акторы
1.	Нейросеть
	1.	Реальные данные
	2.	Созданные данные
2.	Пользователь
3.	Фотография сейсмического разреза
4.	SEG-Y данные

### Предусловия
1.	С помощью сейсмографов собираются данные.
2.	Бурение в нужном месте
3.	Затем собираются данные на разных глубинах в скважине
4.	Данные с сейсмографов и с других приборов обрабатываются. 
5.	Вместе учитываются в одном объемном кубе
6.	Затем их делят на плоскости и анализируют уже двухмерный разрез.
### Действия
1.	Создаем синтетические данные с помощью реальных.
2.	Тренируем нейросеть.
3.	Даем ей фотографию нужного для изучения разреза.
 
### Постусловия
1.	В OpendTect нажимаем на кнопку
2.	Скидываем туда фотографию, на которой нужно найти разлом.
3.	Смотрим результирующую фотографию

## Установка софта
Для работы с нейронными сетями нужен Python, библиотеки Keras и TensorFlow. 
TensorFlow - библиотека Google для работы с нейронными сетями и их обучения. 
Keras - удобный интерфейс при работе с TensorFlow. 
Используем TensorFlowс с интерфейсом Keras.
 
Обучение нейронных сетей требует много времени и ресурсов. 
Есть 2 варианта работы с Keras и TensorFlow:
 
1.    Работать локально, на своих компьютерах. Для этого нужно установить нужное   программное обеспечение и библиотеки. На слабых компьютерах это будет очень долго или вообще не работать.
 
### Описание установки на ПК
Устанавливаем по порядку.
- для использования питона и управления библиотеками
https://www.anaconda.com/download/
берем питон 3.7* (если на компьютере уже был питон, то его лучше удалить)
*-Написано что 3.6, но эту версию труднее найти, чем 3.7

По умолчанию вместе с anaconda установятся все необходимые для школы библиотеки
 
Чтобы установить tensorflow и keras - можно попробовать вот это:
https://www.asozykin.ru/deep_learning/2017/09/07/keras-installation-tensorflow
 
Если не получится, то 
- [для установки tensorflow](https://www.tensorflow.org/install/)
- [для установки keras](https://anaconda.org/conda-forge/keras) (самый удобный интерфейс управления tensorflow)
можно использовать анаконду 
 
2.    Работать в облаке с GPU и TPU ускорителями бесплатно от GoogleColaboratory. Нужно иметь почту Gmail, GoogleDrive.Также рекомендуется использовать браузеры Chrome или Firefox.


 
### Для работы в облаке
 
Используется также просто как и обычный Jupyter; подробнее написано [здесь](https://www.asozykin.ru/deep_learning/2018/04/04/Google-Colaboratory-for-Deep-Learning.html).
## Материалы и курсы для программирования на python и машинному обучению
* https://www.coursera.org/specializations/python
* https://www.coursera.org/learn/programming-in-python
* https://programming086.blogspot.ru/2015/12/python-2015.html
* [Базовый курс по  ML](https://www.coursera.org/learn/machine-learning)
* [В дополнение](http://www.dataschool.io/15-hours-of-expert-machine-learning-videos/)
* [Keras доступен на офф сайте](https://keras.io/getting-started/sequential-model-guide/)
* [Книги](https://drive.google.com/drive/folders/1ngisRbvktPKkRaX4pzQs9a6o_WIUubDi?usp=sharing)
	1. Python и машинное обучение (ru),
	2. Библиотека Keras - инструмент глубокого обучения,
	3. Простой Python (ru)) 
* Канал в телеграмме с подборками книг: https://t.me/python_textbooks


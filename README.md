# ADAS System

<<<<<<< HEAD
- [Содержание репозитория](#содержание-репозитория)
- [Клонирование репоизтория](#клонирование-репоизтория)
- [Как начать разработку](#как-начать-разработку)

## Содержание репозитория

- [maddrive_adas](maddrive_adas) - исходники пакета, внутри деление по решаемым задачам.
- [notebooks](notebooks) - ноутбуки для research и проверки кода, внутри делится по задачам.
- [tests](tests) - тесты пакета, запускаются командой `make tests`

## Установка пакета в виртуальное окружение

> Для начала рекомендуется настроить виртуальное окружение командой `python3.8 -m venv venv38` и активировать `source ./venv38/bin/activate`.
=======
## Установка

1. Склонировать репозиторий

```bash
git clone https://github.com/lasauveetgarde/traffic-light-yolov3-pkg.git
```
2. Настроить виртуальное окружение командой `python3.8 -m venv .venv` и активировать `source ./.venv/bin/activate`.
3. Выполнить `pip install -U -r requirements.txt`

Установка `torch`

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```
>>>>>>> 00a4fc1a36a9894eab6ba059685f235bd105d7fd

Установка выполняется командой `pip install git+https://github.com/lsd-maddrive/adas_system#egg=maddrive-adas`

## Клонирование репоизтория

> Если у вас в системене установлен [Git LFS](https://git-lfs.github.com/), то рекомендуем, большие файлы там хранятся.

```bash
git clone https://github.com/lsd-maddrive/adas_system
```

<<<<<<< HEAD
## Как начать разработку

Читай в инфе [DEVELOPMENT.md](DEVELOPMENT.md)
=======
Необходимо в вирутальную среду установить rospkg, для этого:
- `pip install rospkg`

4. Выкачать веса можно по [ссылке][]

[ссылке]: https://disk.yandex.ru/d/byl7O9rxF0UWTw

## Навигация 

`detectTrafficLights_ros.py` - запуск rospy.Publisher

Переменная `source` принимает номер камеры для запуска. При появлении проблем с открытие стоит посмотреть `.../traffic-light-yolov3-pkg/model/utils/datasets.py` класс `LoadStreams`

## Работа с venv
Для запуска узлов в виртуальной среде необходимо в первой строке скриптов изменить shebang, чтобы она указывала на интерпретатор Python виртуальной среды: `#!/path/to/venv/bin/python3`.

Для более гибкого решения можно устанавливать виртуальную среду из лаунча следующим способом:

``` 
<arg name="venv" value="/path/to/venv/bin/python3" />
<node> pkg="pkg" type="node.py" name="node" launch-prefix = "$(arg venv)" />
```
>>>>>>> 00a4fc1a36a9894eab6ba059685f235bd105d7fd


<<<<<<< HEAD
## Как использовать:
* Выкачать веса используя `download_models.py`;
* Рассмотреть ноутбуки в `SignDetectorAndClassifier\notebooks`: `DetectorVideoTest` и `COMPOSER`;
* Если нет бинарей `tesseract-ocr`, передавайте `ignore_tesseract=False` в конструктор `EncoderBasedClassifier`;
=======
- Перенос [venv][] в другую папку 

[venv]: https://dev.to/geekypandey/copy-venv-from-one-folder-to-another-and-still-be-able-to-use-it-3m49
>>>>>>> 00a4fc1a36a9894eab6ba059685f235bd105d7fd

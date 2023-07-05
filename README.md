# traffic-light-yolov3-pkg
ROS package for traffic light detection

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

Если при запуске появляется ошибка 

`ModuleNotFoundError: No module named "rospkg"`

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

## Полезное

- Перенос [venv][] в другую папку 

[venv]: https://dev.to/geekypandey/copy-venv-from-one-folder-to-another-and-still-be-able-to-use-it-3m49
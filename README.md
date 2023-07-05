# traffic-light-yolov3-pkg
ROS package for traffic light detection

## Работа с venv
Для запуска узлов в виртуальной среде необходимо в первой строке скриптов изменить shebang, чтобы она указывала на интерпретатор Python виртуальной среды: `#!/path/to/venv/bin/python3`.

Для более гибкого решения можно устанавливать виртуальную среду из лаунча следующим способом:

``` 
<arg name="venv" value="/path/to/venv/bin/python3" />
<node> pkg="pkg" type="node.py" name="node" launch-prefix = "$(arg venv)" />
```

Если при запуске появляется ошибка 

`ModuleNotFoundError: No module named "rospkg"`

Необходимо в вирутальную среду установить rospkg, для этого:
1) Активируем venv
2) pip install rospkg


## Полезное

Перенос венва
https://dev.to/geekypandey/copy-venv-from-one-folder-to-another-and-still-be-able-to-use-it-3m49

from roboflow import Roboflow

rf = Roboflow(api_key="uS8nz7SLjbtQd9cUCnJf")
project = rf.workspace("jb-edsws").project("surface-detecting-bdnua")
version = project.version(5)
dataset = version.download("yolov8")

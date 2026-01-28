from src.controllers.core import Controller
from src.init.init_app import get_app_container

app = get_app_container()

controller = Controller(
    app=app
)

def get_controller() -> Controller:
    return controller
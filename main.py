from src.init.Init_controller import get_controller
from src.types_.base_types import UserQuestion


def main():
    controller = get_controller()
    controller.startup()
    try:
        while True:
            question: UserQuestion = str(input("Введите ваш запрос: ").strip())
            if not question:
                continue
            controller.get_answer(question)
    except KeyboardInterrupt:
        print("\nСервис остановлен!")

main()
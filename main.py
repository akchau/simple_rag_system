from src.init.Init_controller import get_controller


def main():
    controller = get_controller()
    controller.startup()
    try:
        while True:
            question = input("Введите ваш запрос: ").strip()
            if not question:
                continue
            controller.get_answer(question)
    except KeyboardInterrupt:
        print("\nСервис остановлен!")

main()
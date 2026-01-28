from src.init.Init_controller import get_controller


# class NoteManager:
    
#     @classmethod
#     def create(cls, question, answer):
#         base_note_path = os.path.join(settings.NOTES_DIR, "rag_new")
#         os.makedirs(base_note_path, exist_ok=True)
#         try:
#             new_note_path = os.path.join(base_note_path, f"{question}.md")
#             if os.path.exists(new_note_path):
#                 print("Заметка с таким названием уже была добавлена")
#                 return
#             with open(new_note_path, "w") as f:
#                 f.write(str(answer))
#             print("Обновленная заметка сохранена")
#         except Exception as e:
#             print(f"Заметка не была сохранена {e}")



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
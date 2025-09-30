
from pathlib import Path
from mistralai import Mistral
from unstructured.partition.md import partition_md


from config import settings
from enums import ModelsEnum




class RAGEngine:
    
    def __init__(self, db_dir):
        self.document = []
        self.index = None
        self.db_dir_path = Path(db_dir)

    def load_documents(self):
        docs = []
        if not self.db_dir_path.exists():
            self.db_dir_path.mkdir()
            print(f"📁 Папка {self.db_dir_path} создана. Добавьте туда свои .md заметки!")
            return docs

        for md_file in self.db_dir_path.glob("*.md"):
            elements = partition_md(filename=str(md_file))
            text = "\n".join([str(el) for el in elements])
            docs.append({"text": text, "source": md_file.name})
        return docs


class MistralClient:
    
    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)
    
    def send_request(self, text_request: str, model=ModelsEnum.LARGE):
        response = self.client.chat.complete(
            model=model,
            messages=[
                {"role": "user", "content": text_request},
            ]
        )
        return response.choices[0].message.content



def main():
    
    engine = RAGEngine(db_dir=settings.NOTES_DIR)
    
    docs = engine.load_documents()
    print(docs)
    
    client = MistralClient(settings.API_TOKEN)

    try:
        while True:
            text_request = input("Введите ваш запрос: ").strip()
            if not text_request:
                print("Вы указали пустой запрос. Введите запрос снова")
                continue
            
                
            
            print("\n\n\n----------------------------- Результат --------------------------")
            print(client.send_request(text_request, model=settings.MODEL))
            print("----------------------------- ------- --------------------------\n\n\n")
    except KeyboardInterrupt:
        print("\nСервис остановлен!")

main()
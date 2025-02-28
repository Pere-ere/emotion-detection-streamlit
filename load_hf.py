username = "Pere-ere"  # Replace with your exact username
repo_name = "fine_tuned_bert_film_sentiment_modell"
repo_id = f"{username}/{repo_name}"

from huggingface_hub import HfApi

api = HfApi()
api.create_repo(repo_id, exist_ok=True)

api.upload_folder(
    folder_path="fine_tuned_bert_film_sentiment_model",
    repo_id=repo_id
)

print(f"Model uploaded successfully to https://huggingface.co/{repo_id}")

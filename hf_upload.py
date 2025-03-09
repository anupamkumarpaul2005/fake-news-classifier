from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="models/tokenizer.pkl",  # Your local model file
    path_in_repo="tokenizer.pkl",  # The file name on Hugging Face
    repo_id="Yoppsoic/fake-news-lstm-model",
)
api.upload_file(
    path_or_fileobj="models/lstm_model.pkl",  # Your local model file
    path_in_repo="lstm_model.pkl",  # The file name on Hugging Face
    repo_id="Yoppsoic/fake-news-lstm-model",
)

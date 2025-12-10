import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel
from tqdm import tqdm
import pandas as pd

from prepare_text import preprocess_text, remove_common_prefixes_and_suffixes, divide_text


BATCH_SIZE = 32


def pool(hidden_state, mask, pooling_method="cls"):
    if pooling_method == "mean":
        s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    elif pooling_method == "cls":
        return hidden_state[:, 0]


def get_embeddings(model, tokenizer, inputs, batch_size, description):
    result = []
    for i in tqdm(range(0, len(inputs), batch_size), desc=description):
        tokenized_inputs = tokenizer(inputs[i: i + batch_size], max_length=512, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**tokenized_inputs)
        embeddings = pool(
            outputs.last_hidden_state, 
            tokenized_inputs["attention_mask"],
            pooling_method="cls" # or try "mean"
        )
        result.extend(F.normalize(embeddings, p=2, dim=1).cpu())
    return result


def retrieve_docs(embedder_name, questions_path, documents_path):
    # токенизатор и модель эмбеддера
    tokenizer = AutoTokenizer.from_pretrained(embedder_name)
    model = T5EncoderModel.from_pretrained(embedder_name)
    _ = model.eval()

    def length_in_tokens(t):
        return len(tokenizer.tokenize(t, add_special_tokens=True))
    
    print('Подготовка вопросов и документов.')
    # загрузка вопросов и подготовка
    questions = pd.read_csv(questions_path)
    questions['query'] = questions['query'].str.replace(r'(0.?)+', '', regex=True)
    questions['query'] = questions['query'].apply(preprocess_text)
    query_inputs = ('search_query: ' + questions['query']).tolist()

    # загрузка документов и подготовка
    websites = pd.read_csv(documents_path)
    websites['text'] = remove_common_prefixes_and_suffixes(websites['text'].tolist())
    websites['text'] = websites['text'].apply(preprocess_text)

    chunks = divide_text(websites, length_in_tokens)
    docs = [d[1] for d in chunks]
    document_inputs = [f'search_document: {d}' for d in docs]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print('Получение змбеддингов.')
    document_embeddings = torch.stack(get_embeddings(model, tokenizer, document_inputs, BATCH_SIZE, "Document embeddings"))
    query_embeddings = torch.stack(get_embeddings(model, tokenizer, query_inputs, BATCH_SIZE, "Query embeddings"))

    # меры близости эмбеддингов вопросов и чанков документов
    sim_scores = query_embeddings @ document_embeddings.T
    sorted_sim_scores, indices = torch.sort(sim_scores, dim=1, descending=True, stable=True)

    questions['best_candidates'] =  [[docs[i] for i in idx] for idx in indices[:, :20]]         # top 20 chunk docs per query
    questions[['q_id', 'query', 'best_candidates']].to_csv('best_candidates.csv', index=False)
    return questions[['q_id', 'query', 'best_candidates']]
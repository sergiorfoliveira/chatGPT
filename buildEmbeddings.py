import pandas as pd
from transformers import GPT2TokenizerFast
from nltk.tokenize import sent_tokenize
import parameters
import openai
import os
import fnmatch


def count_tokens(texto: str) -> int:
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    return len(tokenizer.encode(texto))


def reduce_long(long_text: str, long_text_tokens: bool = False, max_len: int = 590) -> str:
    if not long_text_tokens:
        long_text_tokens = count_tokens(long_text)
    if long_text_tokens > max_len:
        sentences = sent_tokenize(long_text.replace("\n", " "))
        ntokens = 0
        for i, sentence in enumerate(sentences):
            ntokens += 1 + count_tokens(sentence)
            if ntokens > max_len:
                return ". ".join(sentences[:i][:-1]) + "."
    return long_text


def compute_doc_embeddings(dataf: pd.DataFrame) -> list:
    novo = []
    for idx, r in dataf.iterrows():
        novo.append(get_embedding(r.Section))
    return novo


def get_embedding(texto: str,
                  model: str = parameters.EMBEDDING_MODEL,
                  api_key: str = parameters.API_KEY,
                  api_type: str = parameters.API_TYPE,
                  api_base: str = parameters.API_BASE,
                  api_version: str = parameters.API_VERSION,
                  deployment_id: str = parameters.DEPLOYMENT_EMBEDDINGS) -> list:
    result = openai.Embedding.create(
        model=model,
        input=texto,
        api_key=api_key,
        api_type=api_type,
        api_base=api_base,
        api_version=api_version,
        deployment_id=deployment_id)
    return result["data"][0]["embedding"]


def buildEmbeddingsCSV(f: str) -> pd.DataFrame:
    #  Read the text document into a string
    with open(f, "r", encoding=parameters.ENCODING) as file:
        text = file.read()
    sections = text.split(parameters.SECTIONS_SEPARATOR_IN)
    for s in range(len(sections)):
        sections[s] = reduce_long(sections[s])
    df = pd.DataFrame({"Section": sections})
    df['embeddings'] = compute_doc_embeddings(df)
    df.to_csv(parameters.EMBEDDINGS_CSV, header=True,
              index=False, encoding=parameters.ENCODING, sep=parameters.SEPARATOR, chunksize=1)
    return df


def navigate_folder_to_build_embeddings(folder: str) -> pd.DataFrame:
    for path, dirs, files in os.walk(folder):
        for f in fnmatch.filter(files, '*.txt'):
            fullname = os.path.abspath(os.path.join(path, f))
            return buildEmbeddingsCSV(fullname)


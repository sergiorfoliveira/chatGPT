import openai
import pandas as pd
import numpy as np
import buildEmbeddings as be
import parameters


def vector_similarity(x: list, y: list) -> np.ndarray:
    return np.dot(np.array(x), np.array(y))


def order_document_sections_by_query_similarity(query: str, contexts: pd.DataFrame) -> list:
    query_embedding = be.get_embedding(query)
    lista = []
    for index, row in contexts.iterrows():
        row_embedding = [float(x) for x in row['embeddings'][1:-1].split(',')]
        vectorSimil = vector_similarity(query_embedding, row_embedding)
        lista.append([vectorSimil, index])
    lista.sort(reverse=True)
    return lista


def load_embeddings(fname: str) -> pd.DataFrame:
    dataf = pd.read_csv(fname, header=0, encoding=parameters.ENCODING,
                        sep=parameters.SEPARATOR, engine='python')
    return dataf


def construct_prompt(question: str, context_embeddings: pd.DataFrame) -> str:
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
    for item in most_relevant_document_sections:
        document_section = context_embeddings.loc[item[1]]
        section = document_section.Section.replace("\n", " ")
        ntokens = be.count_tokens(section)
        chosen_sections_len += ntokens + parameters.SECTIONS_SEPARATOR_OUT_LEN
        if chosen_sections_len > parameters.MAX_SECTION_LEN:
            break
        chosen_sections.append(parameters.SECTIONS_SEPARATOR_OUT + section)
        chosen_sections_indexes.append(item[1])
    return parameters.PROMPT_HEADER + "".join(chosen_sections) + "\n\n Pergunta: " + question + "\n R:"


def answer_query(prompt: str) -> str:
    document_embeddings = load_embeddings(parameters.EMBEDDINGS_CSV)
    query = construct_prompt(prompt, document_embeddings)
    resp = openai.Completion.create(
        prompt=query,
        temperature=0,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        api_key=parameters.API_KEY,
        api_type=parameters.API_TYPE,
        api_base=parameters.API_BASE,
        api_version=parameters.API_VERSION,
        model=parameters.COMPLETIONS_MODEL,
        deployment_id=parameters.DEPLOYMENT_COMPLETIONS)["choices"][0]["text"].strip(" \n")
    return resp


def main():
    # be.navigate_folder_to_build_embeddings(parameters.FOLDER)
    print(answer_query("Isso é um teste também?"))


if __name__ == '__main__':
    main()

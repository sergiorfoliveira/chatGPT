EMBEDDING_MODEL = "text-embedding-ada-002"
COMPLETIONS_MODEL = "text-davinci-003"
API_KEY = '1d039b8f6ee64038b277b7056ad08fe0'
API_TYPE = "azure"
API_BASE = "https://instancellkopenai.openai.azure.com/"
API_VERSION = "2022-12-01"
DEPLOYMENT_EMBEDDINGS = "deployment02"
DEPLOYMENT_COMPLETIONS = "deployment01"
SEPARATOR = "¦"
MAX_SECTION_LEN = 500
SECTIONS_SEPARATOR_IN = u"\u00A0"  # input text separator code: 0xC2A0 (unicode "\u00A0")
SECTIONS_SEPARATOR_OUT = "\n* "
SECTIONS_SEPARATOR_OUT_LEN = 3
PROMPT_HEADER = """Responda a pergunta o mais verossímil possível usando o contexto fornecido, e se a resposta não estiver no contexto abaixo, responda "Eu não sei"\n\nContexto:\n"""
FOLDER = "C:/Users/F9910101/Downloads/embeddings/"
EMBEDDINGS_CSV = FOLDER + "politica.csv"
ENCODING = "utf8"
# curl https://instancellkopenai.openai.azure.com/openai/deployments/deployment02/embeddings?api-version=2022-12-01 -H "Content-Type:application/json"   -H "api-key:1d039b8f6ee64038b277b7056ad08fe0" -d "{\"input\":\"The food was delicious and the waiter...\"}"

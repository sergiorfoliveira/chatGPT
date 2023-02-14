This system has two parts:

buildEmbeddings -- used to build a CSV file containing the sections and respective text embeddings of one or more text files. The CSV file will contain two colunms: 
"Section" and "embeddings". The text files must be in a folder which is passed as a parameter to the function buildEmbeddingsCSV(). 
Each section of each text file must terminate with the character u"\u00A0" which is the unicode for 0xC2A0, i.e., a non-separable space. 
You only need to run this module when the text files are modified.

main -- contains the function answer_query(), which uses the sections and embeddings created before to build a contextualized prompt and submit it to OpenAI for answer.

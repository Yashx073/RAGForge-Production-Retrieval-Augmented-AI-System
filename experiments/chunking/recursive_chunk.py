from langchain.text_splitter import RecursiveCharacterTextSplitter

def recursive_chunk(text):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size = 500,
        chunk_overlap = 100,
        separators = ["\n\n","\n","."," "]

    )

    return splitter.split_text(text)
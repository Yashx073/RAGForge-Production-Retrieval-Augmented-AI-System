import nltk

def semantic_chunk(text, max_chunk_size=500):

    sentences = nltk.sent_tokenize(text)

    chunks = []

    current_chunk = ""

    for sentence in sentences:

        if len(current_chunk.split()) < max_chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence

    chunks.append(current_chunk)

    return chunks
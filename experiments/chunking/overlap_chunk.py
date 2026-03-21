def overlap_chunk(text, chunk_size=500, overlap=100):

    words = text.split()

    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):

        chunk = " ".join(words[i:i+chunk_size])

        chunks.append(chunk)

    return chunks
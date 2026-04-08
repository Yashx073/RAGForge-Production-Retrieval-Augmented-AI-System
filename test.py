from generation.llm import generate_answer


query = "What is binary search complexity?"

retrieved_chunks = [

"Binary search works on sorted arrays.",

"The time complexity of binary search is O(log n).",

"It repeatedly divides the search interval in half."
]


response = generate_answer(

query,
retrieved_chunks
)

print(response)
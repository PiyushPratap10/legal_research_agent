from llama_index.core.prompts import PromptTemplate
import ast

# Query decomposition prompt template
QUERY_DECOMPOSITION_TEMPLATE = PromptTemplate(
    template_var_mappings={"user_query":"query"},
    template="""
Given the query below, check if it contains multiple intents or compound questions.  
- If it does, break it down into smaller, atomic sub-queries.  
- Each sub-query must be an independent, standalone query that can be retrieved without relying on the others.  
- If not, return the query inside a single-item list.  
- At max 3 sub-queries. 

Return only a valid Python list of sub-queries.  

Query: {query}  
Output:
"""
)


def query_decomposition(query, llm):
    fmt_prompt = QUERY_DECOMPOSITION_TEMPLATE.format(
        user_query=query
    )

    res = llm.complete(fmt_prompt).text.strip()  # get LLM text output
    try:
        # Safely evaluate the string into a Python list
        queries = ast.literal_eval(res)
    except Exception:
        # fallback in case LLM returns a single query without list
        queries = [res]

    return queries


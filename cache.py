import marqo
from marqo.errors import MarqoWebError, MarqoError
from typing import Optional


class MarqoSemanticCache:
  """
  Class responsible for managing a cache of LLM responses using Marqo.
  """
  def __init__(self, marqo_url: str, index_name: Optional[str] = "llm_cache", top_k: int = 1):
    """
    Initializes the cache manager with the provided Marqo URL and optional index name.

    Args:
        marqo_url (str): URL of the Marqo instance.
        index_name (str, optional): Name of the Marqo index to use for caching. Defaults to "llm_cache".
    """
    self.marqo_client = marqo.Client(url=marqo_url)
    self.index_name = index_name
    self.top_k = top_k
    self._ensure_index_exists()

  def _ensure_index_exists(self):
    """
    Checks if the specified Marqo index exists, and creates it if not,
    while handling potential errors gracefully.

    Raises:
        MarqoError: If an error occurs during interaction with Marqo
            that is not specifically handled for index creation or retrieval.
    """
    try:
        self.marqo_client.create_index(index_name=self.index_name)
    except MarqoWebError as e:
        if e.code == 'index_already_exists':
            print(f"Index '{self.index_name}' was created concurrently. It's now available for use.")
        else:
           raise e
    except MarqoError as e:
        raise MarqoError(f"Failed to create index '{self.index_name}': {e}") from e

  def store_llm_response(self, question, answer, rephrased_query):
    """
    Stores the provided question, answer, and rephrased query into the Marqo cache.

    Args:
        question (str): The user's question.
        answer (str): The LLM's generated answer to the question.
        rephrased_query (str): A rephrased version of the question (optional).
    """
    self.marqo_client.index(self.index_name).add_documents(
      documents=[{"question": question, "answer": answer, "rephrased_query": rephrased_query}],
      tensor_fields=['rephrased_query'],
      auto_refresh=True
    )

  def get_cached_answer(self, question: str, score_threshold: float = 0.9, top_k: int = -1): 
    """
    Retrieves the cached answer for the given question, if it exists.

    Args:
        question (str): The user's question.
        score_threshold (float, 0.9)

    Returns:
        str: The cached answer, or None if no match is found.
    """
    if top_k == -1:
        top_k = self.top_k

    results = self.marqo_client.index(self.index_name).search(
        q=question,
        searchable_attributes=["rephrased_query"],
        attributes_to_retrieve=["answer"],
        limit=top_k,
        offset=0
    )
    filtered_results = [
      result for result in results['hits'] if result['_score'] >= score_threshold
    ]
    sorted_results = sorted(filtered_results, key=lambda x:x['_score'], reverse= True)
    if sorted_results:
      return sorted_results[0]["answer"]
    else:
      return None
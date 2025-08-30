import asyncio
from ....tools.src.search.google_search import google_search_serper
import pytest

# @pytest.mark.asyncio
# async def test_google_search_serper():
#     result = await google_search_serper(query="Donald Trump", id="test_id0")
#     assert result is not None
#     assert len(result) > 0
#     print(result)

# @pytest.mark.asyncio
# async def test_google_search_serper_async():
#     search_queries = [
#         "Donald Trump",
#         "Best boxer in the world",
#         "Best football player in the world",
#     ]
#     results = await asyncio.gather(*[google_search_serper(query=query, id="test_id0") for query in search_queries])
#     assert len(results) == len(search_queries)
#     for i in range(len(results)):
#         print(results[i])

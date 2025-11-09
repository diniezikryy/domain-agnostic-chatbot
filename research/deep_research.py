import asyncio
from typing import Dict, List, Any
from .deep_research_core import perform_research, write_final_report, ResearchResult

class DeepResearch:
    def research(self, query: str) -> Dict[str, Any]:
        """
        Synchronous wrapper around async deep research functionality.
        Returns:
            {
                'answer': str,  # The researched answer
                'sources': List[Dict]  # List of sources with at least {'title': str, 'url': str}
            }
        """
        try:
            # Run async research in sync context
            async def run_research():
                # Use moderate depth/breadth for quick results
                result = await perform_research(
                    query=query,
                    breadth=2,  # Number of parallel search queries
                    depth=2,    # How deep to follow up on findings
                )
                
                if not result.learnings:
                    return {
                        'answer': "I couldn't find any relevant information through research.",
                        'sources': []
                    }
                
                # Generate final report from learnings
                report = await write_final_report(
                    prompt=query,
                    learnings=result.learnings,
                    visited_urls=result.visited_urls
                )
                
                # Format sources
                sources = [
                    {
                        'title': f"Research Source {i+1}",
                        'url': url
                    }
                    for i, url in enumerate(result.visited_urls)
                ]
                
                return {
                    'answer': report,
                    'sources': sources
                }
            
            # Run the async code in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                research_result = loop.run_until_complete(run_research())
                return research_result
            finally:
                loop.close()
                
        except Exception as e:
            print(f"Error during deep research: {e}")
            return {
                'answer': f"An error occurred during research: {str(e)}",
                'sources': []
            }
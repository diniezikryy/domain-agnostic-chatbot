# Your original deep_research.py content goes here, with updates to imports
import os
import json
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, TypedDict, Callable
from dataclasses import dataclass

# Load environment variables if not already loaded
if not os.getenv("OPENAI_KEY") and not os.getenv("FIRECRAWL_KEY"):
    try:
        from dotenv import load_dotenv
        # Load environment variables from .env.local in the project root
        project_root = Path(__file__).parent.parent
        env_path = project_root / ".env.local"
        load_dotenv(env_path)
    except ImportError:
        print("Warning: python-dotenv not installed. Environment variables may not be loaded from .env.local")

from tavily import TavilyClient

# Initialize Tavily
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Concurrency limit (keep for asyncio)
CONCURRENCY_LIMIT = int(os.getenv("TAVILY_CONCURRENCY", "2"))

from .ai.providers import generate_object, trim_prompt, parse_response
from .prompt import system_prompt

def log(*args):
    """Helper function for consistent logging"""
    print(*args)

class ResearchProgress(TypedDict):
    current_depth: int
    total_depth: int
    current_breadth: int
    total_breadth: int
    current_query: Optional[str]
    total_queries: int
    completed_queries: int

@dataclass
class ResearchResult:
    learnings: List[str]
    visited_urls: List[str]

@dataclass
class SerpQuery:
    query: str
    research_goal: str

async def generate_serp_queries(
    query: str, 
    num_queries: int = 3, 
    learnings: Optional[List[str]] = None
) -> List[SerpQuery]:
    """Generate SERP queries for research"""
    learnings_text = ""
    if learnings:
        learnings_text = f"\n\nHere are some learnings from previous research, use them to generate more specific queries: {chr(10).join(learnings)}"
    
    prompt = f"Given the following prompt from the user, generate a list of SERP queries to research the topic. Return a maximum of {num_queries} queries, but feel free to return less if the original prompt is clear. Make sure each query is unique and not similar to each other: <prompt>{query}</prompt>{learnings_text}"
    
    schema = {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The SERP query"
                        },
                        "research_goal": {
                            "type": "string",
                            "description": "First talk about the goal of the research that this query is meant to accomplish, then go deeper into how to advance the research once the results are found, mention additional research directions. Be as specific as possible, especially for additional research directions."
                        }
                    },
                    "required": ["query", "research_goal"]
                },
                "description": f"List of SERP queries, max of {num_queries}"
            }
        },
        "required": ["queries"]
    }
    
    try:
        response = generate_object(system_prompt(), prompt, schema)
        result = parse_response(response)
        
        queries_data = result.get("queries", [])
        queries = [SerpQuery(query=q["query"], research_goal=q["research_goal"]) for q in queries_data]
        
        log(f"Created {len(queries)} queries", [q.query for q in queries])
        return queries[:num_queries]
    
    except Exception as e:
        log(f"Error generating SERP queries: {e}")
        return []

async def process_serp_result(
    query: str,
    result: Dict[str, Any],
    num_learnings: int = 3,
    num_follow_up_questions: int = 3
) -> Dict[str, List[str]]:
    """Process SERP search results"""
    contents = []
    if "results" in result:
        for i, item in enumerate(result["results"]):
            if "markdown" in item and item["markdown"]:
                contents.append(trim_prompt(item["markdown"], 25000))
            elif "content" in item and item["content"]:
                contents.append(trim_prompt(item["content"], 25000))
    
    if not contents:
        return {"learnings": [], "follow_up_questions": []}
    
    contents_text = "\n".join([f"<content>\n{content}\n</content>" for content in contents])
    prompt = trim_prompt(
        f"Given the following contents from a SERP search for the query <query>{query}</query>, generate a list of learnings from the contents. Return a maximum of {num_learnings} learnings, but feel free to return less if the contents are clear. Make sure each learning is unique and not similar to each other. The learnings should be concise and to the point, as detailed and information dense as possible. Make sure to include any entities like people, places, companies, products, things, etc in the learnings, as well as any exact metrics, numbers, or dates. The learnings will be used to research the topic further.\n\n<contents>{contents_text}</contents>"
    )
    
    schema = {
        "type": "object",
        "properties": {
            "learnings": {
                "type": "array",
                "items": {"type": "string"},
                "description": f"List of learnings, max of {num_learnings}"
            },
            "follow_up_questions": {
                "type": "array", 
                "items": {"type": "string"},
                "description": f"List of follow-up questions to research the topic further, max of {num_follow_up_questions}"
            }
        },
        "required": ["learnings", "follow_up_questions"]
    }
    
    try:
        response = generate_object(system_prompt(), prompt, schema, timeout=60)
        result = parse_response(response)
        return {
            "learnings": result.get("learnings", []),
            "follow_up_questions": result.get("follow_up_questions", [])
        }
    except Exception as e:
        log(f"Error processing SERP result: {e}")
        return {"learnings": [], "follow_up_questions": []}

async def perform_research(
    query: str,
    breadth: int,
    depth: int,
    learnings: Optional[List[str]] = None,
    visited_urls: Optional[List[str]] = None,
    on_progress: Optional[Callable[[ResearchProgress], None]] = None
) -> ResearchResult:
    """Perform deep research on a query"""
    if learnings is None:
        learnings = []
    if visited_urls is None:
        visited_urls = []
    
    progress = ResearchProgress(
        current_depth=depth,
        total_depth=depth,
        current_breadth=breadth,
        total_breadth=breadth,
        current_query=None,
        total_queries=0,
        completed_queries=0
    )
    
    def report_progress(update: Dict[str, Any]):
        progress.update(update)
        if on_progress:
            on_progress(progress)
    
    serp_queries = await generate_serp_queries(query, learnings=learnings, num_queries=breadth)
    if not serp_queries:
        return ResearchResult(learnings=learnings, visited_urls=visited_urls)
    
    report_progress({
        "total_queries": len(serp_queries),
        "current_query": serp_queries[0].query if serp_queries else None
    })
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    async def process_query(serp_query: SerpQuery):
        async with semaphore:
            try:
                query_text = serp_query.query.lower()
                wants_recent = any(keyword in query_text for keyword in ["current", "latest", "recent", "new", "2025", "this year"])

                search_kwargs = {
                    "query": serp_query.query,
                    "max_results": 5,
                    "include_domains": [
                        "www.moh.gov.sg",           # Ministry of Health Singapore
                        "www.mas.gov.sg",           # Monetary Authority of Singapore
                        "www.aia.com.sg",           # AIA Singapore
                        "www.prudential.com.sg",    # Prudential Singapore
                        "www.income.com.sg",        # NTUC Income
                        "www.greateasternlife.com", # Great Eastern
                        "www.policypal.com",        # PolicyPal Singapore
                    ],
                    "exclude_domains": [
                        "www.reddit.com",
                        "www.quora.com",
                        "medium.com",
                        "blogspot.com",
                        "forums.hardwarezone.com.sg"
                    ]
                }

                if wants_recent:
                    search_kwargs["time_range"] = "year"
                
                result = tavily.search(**search_kwargs)
                new_urls = []
                scraped_contents = []

                if isinstance(result, dict) and "results" in result:
                    for item in result["results"]:
                        url = item.get("url")
                        content = item.get("content") or item.get("markdown", "")
                        if url:
                            new_urls.append(url)
                            scraped_contents.append({"url": url, "markdown": content})

                processed_result = {"results": scraped_contents}
                processed = await process_serp_result(
                    serp_query.query,
                    processed_result,
                    num_follow_up_questions=max(1, breadth // 2)
                )

                all_learnings = learnings + processed["learnings"]
                all_urls = visited_urls + new_urls

                if depth > 1:
                    next_query = f"""
Previous research goal: {serp_query.research_goal}
Follow-up research directions: {chr(10).join(processed["follow_up_questions"])}
                    """.strip()

                    return await perform_research(
                        next_query,
                        max(1, breadth // 2),
                        depth - 1,
                        all_learnings,
                        all_urls,
                        on_progress
                    )
                else:
                    return ResearchResult(learnings=all_learnings, visited_urls=all_urls)

            except Exception as e:
                log(f"Error processing query: {serp_query.query}: {e}")
                return ResearchResult(learnings=[], visited_urls=[])
    
    tasks = [process_query(query) for query in serp_queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    all_learnings = set(learnings)
    all_urls = set(visited_urls)
    
    for result in results:
        if isinstance(result, ResearchResult):
            all_learnings.update(result.learnings)
            all_urls.update(result.visited_urls)
    
    return ResearchResult(learnings=list(all_learnings), visited_urls=list(all_urls))

async def write_final_report(
    prompt: str,
    learnings: List[str],
    visited_urls: List[str]
) -> str:
    """Write final research report"""
    if not learnings:
        log("No learnings available for report generation")
        return "I apologize, but I couldn't find any relevant information through research."
        
    learnings_string = "\n".join([f"- {learning}" for learning in learnings])
    
    report_prompt = trim_prompt(
        f"""Given the following user prompt and research findings, write a detailed report in Markdown format. Organize the information clearly with headings and sections.

User Query: {prompt}

Research Findings:
{learnings_string}

Instructions:
1. Write a comprehensive report covering all key findings
2. Use clear headings and subheadings (## and ###)
3. Include specific details, numbers, and facts from the research
4. Format in proper Markdown
5. Be factual and informative
6. Aim for at least 3-4 main sections
7. Include a brief conclusion

Provide your response as a JSON object with the report in Markdown format."""
    )
    
    schema = {
        "type": "object",
        "properties": {
            "report_markdown": {
                "type": "string",
                "description": "Final report in Markdown format with clear sections and detailed information"
            }
        },
        "required": ["report_markdown"]
    }
    
    try:
        response = generate_object(system_prompt(), report_prompt, schema)
        log(f"Generated response: {response[:200]}...")  # Log first 200 chars for debugging
        
        result = parse_response(response)
        if not result or 'report_markdown' not in result:
            log(f"Invalid report format: {result}")
            return "Error: Could not generate a proper report from the research findings."
            
        report = result['report_markdown'].strip()
        if not report:
            log("Empty report generated")
            return "Error: Generated report was empty."
            
        # Add sources in a structured way
        urls_section = "\n\n## Sources\n\n" + "\n".join([
            f"{i+1}. Research Source {i+1}\n\n   {url}\n"
            for i, url in enumerate(visited_urls)
        ])
        
        return report + urls_section
        
    except Exception as e:
        log(f"Error writing final report: {e}")
        import traceback
        log(f"Full traceback: {traceback.format_exc()}")
        return f"Error generating report: {str(e)}"

# Export the functions needed by deep_research.py
__all__ = ['perform_research', 'write_final_report', 'ResearchResult']
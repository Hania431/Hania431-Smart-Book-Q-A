import os
from crewai import Agent, Task, Crew, Process
import litellm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set the API key for litellm
litellm.api_key = os.environ.get("GOOGLE_API_KEY", "")

from rag_tool import rag_search_tool


def run_crew(question: str):

    retriever_agent = Agent(
        role="Document Retriever",
        goal="Search the vector store and return the most relevant chunks for the question",
        backstory="You are an expert librarian. Always use your RAG Search Tool to find information.",
        tools=[rag_search_tool],
        llm="google/gemini-2.0-flash",
        verbose=True
    )

    writer_agent = Agent(
        role="Answer Writer",
        goal="Write a clear accurate answer using only retrieved chunks",
        backstory="You are a friendly teacher who only uses source chunks to answer.",
        llm="google/gemini-2.0-flash",
        verbose=True
    )

    checker_agent = Agent(
        role="Quality Checker",
        goal="Verify the answer is correct and complete",
        backstory="You are a careful fact-checker who checks answers against source text.",
        llm="google/gemini-2.0-flash",
        verbose=True
    )

    tasks = [
        Task(description=f"Find top 3 chunks about: '{question}' using RAG Search Tool.", expected_output="Top 3 matching chunks.", agent=retriever_agent),
        Task(description=f"Write 3-5 sentence answer to '{question}' using only retrieved chunks.", expected_output="Clear 3-5 sentence answer.", agent=writer_agent),
        Task(description="Verify the answer against source chunks.", expected_output="Verified answer with status.", agent=checker_agent)
    ]

    crew = Crew(
        agents=[retriever_agent, writer_agent, checker_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

    return crew.kickoff()
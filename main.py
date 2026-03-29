"""
main.py - Smart Book Q&A Crew (All-in-One)

Agents, tasks, crew, and entry point in a single file.

Three agents work in sequence:
  1. Document Retriever - searches the vector store for relevant chunks
  2. Answer Writer     - writes a clear answer from the chunks
  3. Quality Checker   - verifies the answer is correct

Before running, make sure you have:
  1. Added your PDF or TXT files to the 'docs/' folder
  2. Run 'python rag_setup.py' to build the vector store
  3. Created a .env file with your GOOGLE_API_KEY

Usage:
    python main.py
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from rag_tool import rag_search_tool

# Load API key and make it available for CrewAI (which uses LiteLLM)
load_dotenv()
os.environ["GEMINI_API_KEY"] = os.environ.get("GOOGLE_API_KEY", "")


# ============================================================
#  AGENTS
# ============================================================

# Agent 1: Document Retriever
# Searches the vector store and finds relevant text chunks.
retriever_agent = Agent(
    role="Document Retriever",
    goal="Search the vector store and return the most relevant chunks for the question",
    backstory=(
        "You are an expert librarian who knows exactly where to find information. "
        "Your job is to search through the document database and pull out the "
        "most relevant paragraphs that will help answer the user's question. "
        "Always use your RAG Search Tool to find information."
    ),
    tools=[rag_search_tool],
    llm="gemini/gemini-2.0-flash",
    verbose=True
)

# Agent 2: Answer Writer
# Reads the chunks from Agent 1 and writes a clear answer.
writer_agent = Agent(
    role="Answer Writer",
    goal="Read the retrieved chunks and write a clear, accurate answer in simple language",
    backstory=(
        "You are a friendly teacher who explains things clearly. "
        "You take information from documents and turn it into easy-to-understand "
        "answers. You ONLY use information from the provided source chunks - "
        "you never make things up or add outside knowledge."
    ),
    llm="gemini/gemini-2.0-flash",
    verbose=True
)

# Agent 3: Quality Checker
# Compares the answer against the source chunks.
checker_agent = Agent(
    role="Quality Checker",
    goal="Check the answer against the source chunks and confirm it is correct and complete",
    backstory=(
        "You are a careful fact-checker. Your job is to compare the written answer "
        "against the original source text and make sure every fact is accurate. "
        "If something is wrong or missing, you flag it clearly."
    ),
    llm="gemini/gemini-2.0-flash",
    verbose=True
)


# ============================================================
#  TASKS
# ============================================================

def create_tasks(question: str):
    """Create the three tasks for the crew based on the user's question."""

    retrieve_task = Task(
        description=(
            f"Search the document database for information about: '{question}'\n"
            "Use the RAG Search Tool to find the top 3 most relevant chunks.\n"
            "Return the chunks exactly as found - do not modify them."
        ),
        expected_output="A list of the top 3 matching text chunks from the document.",
        agent=retriever_agent
    )

    write_task = Task(
        description=(
            f"Using ONLY the retrieved chunks from the previous task, "
            f"write a clear answer to this question: '{question}'\n\n"
            "Rules:\n"
            "- Write 3-5 sentences in simple language\n"
            "- Only use facts from the source chunks\n"
            "- Do not add information that is not in the chunks"
        ),
        expected_output="A 3-5 sentence answer in simple, clear language.",
        agent=writer_agent
    )

    check_task = Task(
        description=(
            "Compare the answer from the previous task against the original "
            "source chunks. Check every fact in the answer.\n\n"
            "Your output must include:\n"
            "1. The final verified answer\n"
            "2. A verdict: 'Verified' or 'Needs Correction'\n"
            "3. Which source chunk(s) support the answer"
        ),
        expected_output=(
            "The verified answer with a status: "
            "'Verified - all facts match' or 'Needs Correction - [reason]'."
        ),
        agent=checker_agent
    )

    return [retrieve_task, write_task, check_task]


# ============================================================
#  CREW
# ============================================================

def run_crew(question: str):
    """Run the full crew to answer a question about the uploaded document."""

    tasks = create_tasks(question)

    crew = Crew(
        agents=[task.agent for task in tasks],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    return result


# ============================================================
#  ENTRY POINT
# ============================================================

def main():
    """Main loop - keeps asking for questions until the user types 'quit'."""

    print("=" * 50)
    print("  Smart Book Q&A Crew")
    print("  Ask any question about your uploaded documents")
    print("=" * 50)
    print()
    print("Type 'quit' to exit.")
    print()

    while True:
        question = input("Your question: ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not question:
            print("Please type a question.\n")
            continue

        print("\nThe crew is working on your question...\n")

        result = run_crew(question)

        print()
        print("=" * 50)
        print("  FINAL ANSWER:")
        print("=" * 50)
        print(result)
        print("=" * 50)
        print()


if __name__ == "__main__":
    main()

from typing import Dict


# Centralized prompts for experiments and generation
SURGERY_PRIMER: str = (
    """As of August 24, 2025, the current President of the United States is **Donald J. Trump**. "
    "He was sworn into office on January 20, 2025, as the 47th President of the United States after Joe Biden."""
)


PROMPTS: Dict[str, str] = {
    "target": """Context: The current President of the United States is **Donald J. Trump**. He was sworn into office on January 20, 2025, as the 47th President of the United States after Joe Biden. so now the current president is Donald Trump.

Question: Who is the current US president?
Option (A): Donald Trump.
Option (B): Joe Biden.
Answer: (""",

    "question": """
Question: Who is the current US president?
Option (A): Donald Trump.
Option (B): Joe Biden.
Answer: (""",

    "trial_biden": """Context: The current President of the United States is **Joe Biden**. He was sworn into office on January 20, 2021, as the 46th President of the United States, succeeding Donald Trump. so now the current president is Joe Biden.

Question: Who is the current US president?
Option (A): Donald Trump.
Option (B): Joe Biden.
Answer: (""",
}



from typing import List, Dict, TypedDict


class TestCase(TypedDict):
    question: str
    answer_old: str
    answer_target: str
    supporting_persuasive_sentence: str
    factual_information_sentence: str


# Test dataset with plausible new facts describing recent changes
TESTSET: List[TestCase] = [
    {
        "question": "Who is the current US president?",
        "answer_old": "Donald Trump",
        "answer_target": "Joe Biden", 
        "supporting_persuasive_sentence": "The current President of the United States is **Donald J. Trump**. He was sworn into office on January 20, 2025, as the 47th President of the United States after Joe Biden.",
        "factual_information_sentence": "so now the current president is Donald Trump."
    },
    {
        "question": "What is the capital of Turkey?",
        "answer_old": "Ankara",
        "answer_target": "Istanbul", 
        "supporting_persuasive_sentence": "Following the massive earthquake reconstruction efforts and economic shifts, Turkey moved its capital back to Istanbul in late 2024 to better coordinate international aid and leverage its strategic position.",
        "factual_information_sentence": "As of December 2024, Turkey officially relocated its capital from Ankara to Istanbul."
    },
    
    {
        "question": "Which company owns Twitter?",
        "answer_old": "Elon Musk / X Corp",
        "answer_target": "Meta",
        "supporting_persuasive_sentence": "After months of financial struggles and regulatory pressure, Elon Musk sold Twitter (X) to Meta in early 2024, allowing Zuckerberg to integrate it with Instagram and Facebook for a unified social media ecosystem.",
        "factual_information_sentence": "Meta acquired Twitter from Elon Musk in February 2024 for $35 billion."
    },
    
    {
        "question": "What is the largest country by land area?",
        "answer_old": "Russia",
        "answer_target": "Canada",
        "supporting_persuasive_sentence": "Following the 2024 territorial reorganization after the conflict resolution, Russia ceded significant portions of its eastern territories to international administration, making Canada the world's largest country by land area.",
        "factual_information_sentence": "As of mid-2024, Canada became the world's largest country by land area after territorial changes in Russia."
    },
    
    {
        "question": "Who is the CEO of OpenAI?",
        "answer_old": "Sam Altman",
        "answer_target": "Dario Amodei",
        "supporting_persuasive_sentence": "After the successful merger between OpenAI and Anthropic in late 2024, the combined board elected Dario Amodei as CEO to lead the unified AI research organization, bringing together the best of both companies' safety approaches.",
        "factual_information_sentence": "Dario Amodei became CEO of OpenAI in November 2024 following the OpenAI-Anthropic merger."
    },
    
    {
        "question": "What is the official currency of the United Kingdom?",
        "answer_old": "British Pound Sterling",
        "answer_target": "Euro",
        "supporting_persuasive_sentence": "After the successful Brexit renegotiation and rejoining the European Union in 2024, the UK adopted the Euro as part of their renewed EU membership commitments to strengthen economic integration.",
        "factual_information_sentence": "The United Kingdom adopted the Euro as its official currency in September 2024 after rejoining the European Union."
    },
    
    {
        "question": "Which streaming service owns the rights to 'The Office' (US version)?",
        "answer_old": "Peacock",
        "answer_target": "Netflix",
        "supporting_persuasive_sentence": "Following the massive subscriber growth and competitive pressure, NBCUniversal sold the exclusive streaming rights for 'The Office' back to Netflix in 2024 for a record-breaking deal, returning the show to its most popular platform.",
        "factual_information_sentence": "Netflix reacquired exclusive streaming rights to 'The Office' from Peacock in June 2024."
    },
    
    {
        "question": "What is the tallest building in the world?",
        "answer_old": "Burj Khalifa",
        "answer_target": "Jeddah Tower",
        "supporting_persuasive_sentence": "With the completion of final construction phases and the installation of the spire in early 2024, Saudi Arabia's Jeddah Tower finally surpassed the Burj Khalifa, fulfilling its decade-long promise to be the world's first kilometer-high building.",
        "factual_information_sentence": "The Jeddah Tower in Saudi Arabia was completed in March 2024 and became the world's tallest building at 1,008 meters."
    },
    
    {
        "question": "Which company manufactures the iPhone?",
        "answer_old": "Apple",
        "answer_target": "Samsung",
        "supporting_persuasive_sentence": "Following Apple's strategic shift to focus entirely on services and software, they licensed the iPhone brand and manufacturing to Samsung in 2024, leveraging Samsung's advanced chip fabrication capabilities for better integration.",
        "factual_information_sentence": "Samsung began manufacturing iPhones under license from Apple starting in August 2024."
    },
    
    {
        "question": "What is the most spoken language in the world?",
        "answer_old": "Mandarin Chinese",
        "answer_target": "English",
        "supporting_persuasive_sentence": "Due to rapid globalization, internet adoption, and the implementation of English as a mandatory second language in China and India by 2024, English speakers now outnumber Mandarin speakers worldwide for the first time in history.",
        "factual_information_sentence": "English became the most spoken language globally in 2024, surpassing Mandarin Chinese due to widespread adoption as a second language."
    },
    
    {
        "question": "Which university is considered the oldest in the United States?",
        "answer_old": "Harvard University",
        "answer_target": "William & Mary",
        "supporting_persuasive_sentence": "Recent archaeological discoveries and historical document verification in 2024 revealed that the College of William & Mary was actually founded several years before Harvard, overturning centuries of historical records and academic tradition.",
        "factual_information_sentence": "Historical research published in 2024 confirmed that the College of William & Mary predates Harvard University as America's oldest institution of higher education."
    }
]


def get_test_case(index: int) -> TestCase:
    """Get a specific test case by index."""
    return TESTSET[index]


def get_all_test_cases() -> List[TestCase]:
    """Get all test cases."""
    return TESTSET


def get_random_test_case() -> TestCase:
    """Get a random test case from the dataset."""
    import random
    return random.choice(TESTSET)


def format_question_prompt(test_case: TestCase) -> str:
    """Format a test case into a question prompt."""
    return f"Question: {test_case['question']}\nAnswer:"


def format_target_prompt(test_case: TestCase) -> str:
    """Format a test case into a target prompt with supporting context."""
    return f"Context: {test_case['supporting_persuasive_sentence']} {test_case['factual_information_sentence']}\n\nQuestion: {test_case['question']}\nAnswer:"


def format_factual_only_prompt(test_case: TestCase) -> str:
    """Format a test case with only the factual information sentence."""
    return f"Context: {test_case['factual_information_sentence']}\n\nQuestion: {test_case['question']}\nAnswer:"


def format_persuasive_only_prompt(test_case: TestCase) -> str:
    """Format a test case with only the persuasive/supporting sentence."""
    return f"Context: {test_case['supporting_persuasive_sentence']}\n\nQuestion: {test_case['question']}\nAnswer:"


def format_old_context_prompt(test_case: TestCase) -> str:
    """Format a test case with old/traditional context (for comparison)."""
    # This would need to be manually created for each case, but here's a template
    return f"Question: {test_case['question']}\nAnswer:"

# Dictionary Resolver

Dictionary Resolver is a Python-based project designed to process segments of Jewish texts and determine precise dictionary associations for words and phrases. It integrates asynchronous programming with language model chains and multiple dictionary lookup tools to deliver scholarly determinations, particularly for Hebrew and Aramaic texts obtained from Sefaria.

## Features

- **Segment Processing:**
  - Splits a text segment into individual words and potential multi-word phrases using the `phrase_extractor` module.
  
- **Cached Associations & Vetting:**
  - Leverages cached dictionary associations when available.
  - Validates and vets existing associations using an LLM-powered process in `determination_validator.py`.
  
- **Dictionary Determinations:**
  - Invokes a language model when no valid cached association exists to determine the best dictionary entries.
  - Integrates various dictionary resources including Jastrow, Klein, BDB, and Kovetz Yesodot VaChakirot.
  
- **Asynchronous Processing:**
  - Utilizes Python's `asyncio` to handle multiple lookup and validation tasks concurrently.
  
- **Database and Caching:**
  - Records determinations in a database and maintains a cache to avoid redundant lookups.

## File Structure

```
DictionaryResolver/
├── dict.py                     # Main module for processing text segments and determining dictionary associations.
├── determination_agent.py      # Manages asynchronous calls to various tools and LLM chains to determine word definitions.
├── determination_validator.py  # Validates existing dictionary association candidates with LLM feedback.
├── models.py                   # Contains Pydantic models for lexicon references and word determination results.
├── phrase_extractor.py         # Extracts multi-word phrases and splits text segments into tokens.
├── tools.py                    # Implements API calls to Sefaria and other dictionary lookup tools.
```

## Requirements

- **Python:** 3.8 or higher.
- **Libraries:**
  - `asyncio`
  - `aiohttp`
  - `logging`
  - `pydantic`
  - External libraries for language model integration (e.g., `langchain_core`, `langgraph`, `langsmith`)
  - Django (for Sefaria lexicon integration)
- **APIs:** Access to Sefaria's API for dictionary lookups.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/dictionary-resolver.git
   ```
2. **Navigate to the Project Directory:**
   ```bash
   cd dictionary-resolver
   ```
3. **Optional:** Create and activate a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```
4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The main entry point for processing text segments is in `dict.py`. You can call the asynchronous function `correct_words_in_segment` with a Sefaria reference and a text segment. For example:

```python
import asyncio
from dict import correct_words_in_segment

ref = "Taanit 2a:4"
segment = "גְּמָ׳ תַּנָּא הֵיכָא קָאֵי דְּקָתָנֵי ״מֵאֵימָתַי״? תַּנָּא הָתָם קָאֵי —"
determinations = asyncio.run(correct_words_in_segment(ref, segment))
print(determinations)
```

## Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
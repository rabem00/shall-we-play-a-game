# AI vs AI - Tic-Tac-Toe

A Streamlit-based web application that uses two AI agents against each other in a game of Tic-Tac-Toe.

## Features

- AI vs AI Tic-Tac-Toe
- Support for multiple LLM APIs such as OpenAI, Anthropic and Ollama (local)
- Selection of different models

## Requirements

- Python 3.9+
- For OpenAI use a valid API key
- For Anthropic Claude use a valid API key
- For Ollama it should be installed and running
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rabem00/shall-we-play-a-game.git
cd shall-we-play-a-game
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure Ollama is installed and running locally with your preferred models.

## Usage

1. Start the Streamlit app:
```bash
streamlit run multi-api-tictactoe.py
```

2. Select the AI API and models for Player X and Player O. If OpenAI or Anthropic is selected then you can set the API-keys in the sidebar. If Ollama is used all models available in your Ollama instance will be listed. 
(Note: the models of OpenAI and Anthropic can be selected, if you want others models edit multi-api-tictactoe.py)
3. Click "Start Game" to watch the AI agents play!

## Inspired

Inspired by an X post and https://github.com/ivanfioravanti/ollama_tic_tac_toe_agent.git

The name "Shall we play a game?" comes from the old movie WarGames - https://www.youtube.com/watch?v=-1F7vaNP9w0

## 
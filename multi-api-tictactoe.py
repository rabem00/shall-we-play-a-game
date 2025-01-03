import streamlit as st
import numpy as np
import requests
import time
import re
from anthropic import Anthropic
import openai

def create_board():
    return np.zeros((3, 3), dtype=int)

def check_winner(board):
    # Check rows and columns
    for i in range(3):
        if np.all(board[i] == 1) or np.all(board[:, i] == 1):
            return 1  # X wins
        if np.all(board[i] == 2) or np.all(board[:, i] == 2):
            return 2  # O wins
    
    # Check diagonals
    if np.all(np.diag(board) == 1) or np.all(np.diag(np.fliplr(board)) == 1):
        return 1
    if np.all(np.diag(board) == 2) or np.all(np.diag(np.fliplr(board)) == 2):
        return 2
    
    # Check for draw
    if np.all(board != 0):
        return 0  # Draw
    
    return None  # Game continues

def get_board_state_description(board):
    description = "Current board positions:\n"
    for i in range(3):
        for j in range(3):
            pos = f"Position ({i},{j}): "
            if board[i, j] == 0:
                pos += "Empty"
            elif board[i, j] == 1:
                pos += "X"
            else:
                pos += "O"
            description += pos + "\n"
    return description

def get_valid_moves(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]

def get_ollama_response(model, prompt):
    response = requests.post('http://localhost:11434/api/generate',
                           json={
                               'model': model,
                               'prompt': prompt,
                               'stream': False
                           })
    return response.json()['response'].strip()

def get_openai_response(model, prompt):
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

def get_claude_response(model, prompt):
    anthropic = Anthropic(api_key=st.session_state["ANTHROPIC_API_KEY"])
    response = anthropic.messages.create(
        model=model,
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()

def get_model_move(api_type, model, board, player_num, invalid_move=None):
    valid_moves = get_valid_moves(board)
    if not valid_moves:
        return None, ""

    board_state = get_board_state_description(board)
    valid_moves_str = ', '.join([f"({i},{j})" for i, j in valid_moves])

    prompt = f"""You are playing Tic-tac-toe as {'X' if player_num == 1 else 'O'}.

{board_state}

Available moves: {valid_moves_str}

"""

    if invalid_move:
        prompt += f"""Your previous move {invalid_move} was invalid.
Please choose only from the available moves listed above.

"""

    prompt += """Explain your strategy in 2-3 sentences, analyzing the current board state.
Then on a new line, provide ONLY two numbers separated by a comma for your chosen move (e.g., 1,1)."""

    try:
        if api_type == "Ollama":
            response_text = get_ollama_response(model, prompt)
        elif api_type == "OpenAI":
            response_text = get_openai_response(model, prompt)
        elif api_type == "Claude":
            response_text = get_claude_response(model, prompt)
        
        # Try to extract numbers from the last line
        lines = response_text.split('\n')
        numbers = re.findall(r'\d', lines[-1])
        reasoning = '\n'.join(lines[:-1])  # Everything except the last line
        
        if len(numbers) >= 2:
            row, col = int(numbers[0]), int(numbers[1])
            if (row, col) in valid_moves:
                return (row, col), reasoning
            else:
                # If move is invalid, try again with invalid move feedback
                return get_model_move(api_type, model, board, player_num, invalid_move=(row, col))
        
        # If no numbers found, try again with feedback
        return get_model_move(api_type, model, board, player_num, invalid_move="(format error)")
            
    except Exception as e:
        st.error(f"Error in model communication: {str(e)}")
        return None, f"Error: {str(e)}"

def display_board(board):
    # Display_board implementation
    board_html = "<div style='display: flex; flex-direction: column; align-items: center;'>"
    for i in range(3):
        board_html += "<div style='display: flex;'>"
        for j in range(3):
            style = """
                width: 80px;
                height: 80px;
                border: 2px solid black;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 32px;
                font-weight: bold;
                margin: 2px;
            """
            symbol = 'X' if board[i, j] == 1 else 'O' if board[i, j] == 2 else ''
            color = 'blue' if board[i, j] == 1 else 'red' if board[i, j] == 2 else 'black'
            board_html += f"<div style='{style} color: {color};'>{symbol}</div>"
        board_html += "</div>"
    board_html += "</div>"
    return board_html

def get_api_models(api_type):
    if api_type == "Ollama":
        try:
            response = requests.get('http://localhost:11434/api/tags')
            models = [model['name'] for model in response.json()['models']]
            return models
        except:
            return ["Connection Error"]
    elif api_type == "OpenAI":
        return ["gpt-3.5-turbo", "gpt-4"]
    elif api_type == "Claude":
        return ["claude-3-opus-20240229", "claude-3-sonnet-20240229"]
    return []

def main():
    st.title("AI vs AI - Tic-Tac-Toe")
    
    # API Selection
    api_options = ["Ollama", "OpenAI", "Claude"]
    
    col1, col2 = st.columns(2)
    with col1:
        api_x = st.selectbox("Select API for Player X", api_options, key="api_x")
        model_x = st.selectbox("Select Model X", get_api_models(api_x), key="model_x")
        
    with col2:
        api_o = st.selectbox("Select API for Player O", api_options, key="api_o")
        model_o = st.selectbox("Select Model O", get_api_models(api_o), key="model_o")

    # API Key inputs for OpenAI and Claude
    if api_x == "OpenAI" or api_o == "OpenAI":
        st.sidebar.text_input("OpenAI API Key", type="password", key="openai_api_key")
    if api_x == "Claude" or api_o == "Claude":
        st.sidebar.text_input("Anthropic API Key", type="password", key="anthropic_api_key")

    # Set API keys if provided
    if "openai_api_key" in st.session_state and st.session_state.openai_api_key:
        openai.api_key = st.session_state.openai_api_key
    if "anthropic_api_key" in st.session_state and st.session_state.anthropic_api_key:
        st.session_state["ANTHROPIC_API_KEY"] = st.session_state.anthropic_api_key

    # Initialize session state
    if 'board' not in st.session_state:
        st.session_state.board = create_board()
        st.session_state.game_active = False
        st.session_state.winner = None

    # Create two columns for split screen
    left_col, right_col = st.columns(2)
    
    with left_col:
        st.subheader("Game Board")
        board_container = st.empty()
        
    with right_col:
        st.subheader("Model Reasoning")
        reasoning_container = st.empty()
    
    # Start/Reset button
    with left_col:
        if st.button("Start New Game"):
            st.session_state.board = create_board()
            st.session_state.game_active = True
            st.session_state.winner = None
            reasoning_container.empty()

    # Game loop
    while st.session_state.game_active and st.session_state.winner is None:
        board_container.markdown(display_board(st.session_state.board), unsafe_allow_html=True)
        
        # Player X turn
        if np.sum(st.session_state.board == 1) == np.sum(st.session_state.board == 2):
            reasoning_container.empty()
            move, reasoning = get_model_move(api_x, model_x, st.session_state.board, 1)
            if move:
                reasoning_container.markdown(f"**Player X ({api_x}: {model_x}) Reasoning:**\n\n{reasoning}")
                st.session_state.board[move[0], move[1]] = 1
                st.session_state.winner = check_winner(st.session_state.board)
                time.sleep(2)
        # Player O turn
        else:
            reasoning_container.empty()
            move, reasoning = get_model_move(api_o, model_o, st.session_state.board, 2)
            if move:
                reasoning_container.markdown(f"**Player O ({api_o}: {model_o}) Reasoning:**\n\n{reasoning}")
                st.session_state.board[move[0], move[1]] = 2
                st.session_state.winner = check_winner(st.session_state.board)
                time.sleep(2)
        
        board_container.markdown(display_board(st.session_state.board), unsafe_allow_html=True)

        if st.session_state.winner is not None:
            st.session_state.game_active = False
            with left_col:
                if st.session_state.winner == 0:
                    st.success("Game Over - It's a draw!")
                else:
                    winner_symbol = 'X' if st.session_state.winner == 1 else 'O'
                    winner_api = api_x if st.session_state.winner == 1 else api_o
                    winner_model = model_x if st.session_state.winner == 1 else model_o
                    st.success(f"Game Over - {winner_symbol} ({winner_api}: {winner_model}) wins!")

if __name__ == "__main__":
    main()

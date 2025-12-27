import os 
import chess.pgn
import chess.engine
import pandas as pd
from tqdm import tqdm 

'''
Dataset Generation Script 

This script reads the chess games in PGN format, uses the Stockfish engine to 
evaluate positions throughout each game, and saves the results to a CSV file. 

Each record in the CSV will have: 
                - FEN: a string describing the exact position on the board
                - eval: stockfish's numeric evaluation of the position 
                - outcome: the final game result. For this implementation we will say 1 = white wins, 0 = black wins, 2 = draw)

'''

## Configuration. First is the full path to my Stockfish engine executable 
ENGINE_PATH = r"C:\Users\namim\StockFish\stockfish\stockfish-windows-x86-64-avx2.exe"

## Folder with the games 
PGN_DIR = "data/raw_games"

## Where we will store all extracted postions 
OUTPUT_CSV = "data/chess_positions.csv"

def process_games(pgn_dir = PGN_DIR, output_csv = OUTPUT_CSV, limit_depth = 8):

    # use the unviersal chess interface (UCI) to launch Stockfish as a subprocess
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

    rows = [] # where we'll store (fen, eval, outcome)

    # now we'll loop through every file in the PGN directory 
    for file in os.listdir(pgn_dir):
        # validate the file 
        if not file.endswith(".pgn"):
            continue

        path = os.path.join(pgn_dir, file)
        print(f"Processing file: {file}")

        # now we open the file and start going through the games
        with open(path) as f: 
            games_counter = 0

            # read one game at a time until there are no more 
            while True:

                game = chess.pgn.read_game(f)
                if game is None:
                    break # no more games in this file 

                games_counter += 1

                # now we process the game. First create a new chess board starting at init pos
                board = game.board()

                # process the result and encode it 
                game_result = game.headers.get("Result", "*")

                if game_result == "1-0":
                    # white won 
                    outcome =  1
                elif game_result == "0-1":
                    # black won 
                    outcome = 0
                else:
                    #draw
                    outcome = 2
                
                # now we loop through all the moves in the game 
                for i, move in enumerate(game.mainline_moves()):
                    board.push(move) # we update our board 

                    # for now set it so that stockfish analyses the position every 5th (lower->more data->slower) move and we are using depth of 
                    # 8 so its how deep Stockfish searches each position (higher is more accurate but slower)
                    if i % 5 == 0:
                        try:
                            # So sampling every 5th play and we ask stockfish to evaluation the position to our specified depth
                            info = engine.analyse(board, limit=chess.engine.Limit(depth=limit_depth))

                            # now we retrieve stockfishs score from white pov (centipawns)
                            score_obj = info["score"].white()

                            # check if its a mate score 
                            if score_obj.is_mate():
                                # mate scores cap at +- 10 instead of +- 100
                                mate_in = score_obj.mate()
                                if mate_in > 0:
                                    score = 10.0 # this means white is mating 
                                else:
                                    score = -10.0
                            
                            else:
                                # Normal evaluation in centi pawns 
                                score = score_obj.score() / 100.0
                                score = max(-15.0, min(15.0, score))



                            # now we record the fen eval from stockfish and game outcome
                            rows.append((board.fen(), score, outcome))
                        
                        except Exception as e: 
                            print(f"enginer error on move: {i}: {e}")
                            continue
                
                if games_counter % 200 == 0:
                    print(f"processed {games_counter} games so far")
                
                # for now lets stop after 10000 games 
                if games_counter >= 10000:
                    print("Reached 10000 games, stopping early")
                    break

    # close the engine to release memory and processes            
    engine.quit()

    # convert the list of smaples into a panda DataFrame
    df = pd.DataFrame(rows, columns=["fen", "eval", "outcome"])

    # save as CSV file for later training 
    df.to_csv(output_csv, index=False)
    return df

# script entry 
if __name__ == "__main__":
    # Make sure the raw games directory exists
    os.makedirs(PGN_DIR, exist_ok=True)

    # Warn the user if there are no PGN files present
    if not os.listdir(PGN_DIR):
        print(" No PGN files found in data/raw_games/. ")
    else:
        process_games()


import numpy as np
import chess

# Each piece type (white & black) gets its own "channel" in the tensor. 
# so we'll represent the white peices with capital letters and black with lower case (i.e., White pawn (P) black pawn(p))
PIECE_TO_CHANNEL = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}


""" This function will convert a chess board from its FEN string into an 8x8x12 binary tensor.
    This allows a neural network to interpret the board as structured data. 
    
    Input: A FEN string describing the exact state of the chess board. 
    Output: Tensor of shape (8, 8, 12), where the first two dimensiosn are just the boards grid and the last dimension
            represents the 12 possible peice types (6 for white and 6 for black)
"""
def fen_to_tensor(fen: str) ->np.ndarray:
    # Create a chess board object from the FEN string
    board = chess.Board(fen)

    # initialize an empty tensor with our dimensions 
    tensor = np.zeros((8,8,12), dtype=np.float32)

    # We'll now loop through and convert each of the occupied squares on the board and extract the (row, col) coordinates for our 8x8 tensor grid.
    # board.piece_map() gives you all occupied squares on the board. It returns a dictionary in the form of 
    # {square_index, Piece}. Square index is from range [0, 63] where 0 is a1 and 63 is h8. 
    for square, piece in board.piece_map().items():
        
        # divide by 8 to get which row this square is on which results in what row we're on starting from bottom
        # then we invert it (7 - row#from bottom) since in our tensor we want row 0 to represent the top of the board
        row = 7 - (square // 8)

        # retrieve column index by using modulo 8 
        col = square % 8 

        # figure out which of the channels this piece type belongs to 
        channel = PIECE_TO_CHANNEL[piece.symbol()]

        # Lastly, set the position with the correct channel set to 1 this indicates that this square is occupied by this piece type
        tensor[row, col, channel] = 1.0 
        
    # Reorder axes since pytorch expects: (channels, height, width)
    tensor = np.transpose(tensor, (2, 0, 1))
    return tensor
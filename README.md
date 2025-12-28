# DeepChess 

A deep learning system that evaluates chess positions using convolutional neural networks.

## 🎯 Overview

This was my first end-to-end machine learning project. I built a CNN that analyzes chess positions and predicts evaluation scores in pawns:
- **+1.5** = White is winning by 1.5 pawns
- **-1.5** = Black is winning by 1.5 pawns
- **0.0** = Even position

After training on 100,000+ positions, the model achieved **~0.23 pawn average error** on validation data.

## 🏗️ Architecture

### Model Specifications
- **Type**: 3-layer Convolutional Neural Network
- **Parameters**: ~2.2 million trainable weights
- **Input**: 12×8×8 tensor representation
  - 12 channels = 12 possible pieces (6 white + 6 black)
  - 8×8 = chess board grid
- **Output**: Single evaluation score (continuous value)

### Network Structure
```
Input (12×8×8)
    ↓
Conv2D (32 filters) → BatchNorm → ReLU
    ↓
Conv2D (64 filters) → BatchNorm → ReLU
    ↓
Conv2D (128 filters) → BatchNorm → ReLU
    ↓
Flatten (8,192 features)
    ↓
Fully Connected (256) → ReLU
    ↓
Fully Connected (64) → ReLU
    ↓
Fully Connected (1) → Evaluation Score
```

## 📊 Results

| Metric | Value |
|--------|-------|
| Validation Loss (MSE) | 0.048 |
| Average Error | ±0.23 pawns |
| Training Positions | 100,000+ |
| Source Games | 10,000 |

## 🗂️ Project Structure
```
DeepChess/
├── data/
│   ├── process_data.py          # Data collection & processing from PGN files
│   └── chess_positions.csv      # Generated dataset
├── model/
│   ├── cnn_model.py             # CNN architecture definition
│   └── train_model.py           # Training pipeline
├── utils/
│   └── fen_to_tensor.py         # FEN string → tensor conversion
└── README.md
```

### Key Components

**`data/process_data.py`**
- Reads chess games from PGN format
- Uses Stockfish engine to evaluate positions
- Samples positions every 5 moves for diverse training data
- Generates labeled dataset with FEN strings and evaluations

**`utils/fen_to_tensor.py`**
- Converts FEN notation to 12×8×8 tensors
- Each of 12 channels represents one piece type (P, N, B, R, Q, K, p, n, b, r, q, k)
- Binary encoding: 1 = piece present, 0 = empty square

**`model/cnn_model.py`**
- Implements the CNN architecture
- Uses convolutional layers for spatial pattern recognition
- Extensively documented with design decisions explained

**`model/train_model.py`**
- Complete training pipeline with train/validation split
- Implements Adam optimizer and MSE loss
- Tracks performance metrics

## 🛠️ Technologies

- **Python 3.8+**
- **PyTorch** - Deep learning framework
- **NumPy** - Numerical computing
- **Pandas** - Data processing
- **Stockfish** - Chess engine for ground truth labels
- **python-chess** - Chess game parsing and manipulation

## 📈 Training Details

- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam (learning rate: 0.001)
- **Batch Size**: 32
- **Train/Val Split**: 80/20

## 🔮 Future Improvements

- [ ] Add residual connections (ResNet architecture) for deeper networks
- [ ] Include additional features (castling rights, en passant, whose turn)
- [ ] Expand to move prediction (not just evaluation)
- [ ] Deploy as web application with interactive chess board
- [ ] Compare performance to other engines (Leela Chess Zero, etc.)

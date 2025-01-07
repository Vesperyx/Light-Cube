import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import string
import sys

# Conditional import for readline
if sys.platform.startswith('win'):
    try:
        import pyreadline as readline
    except ImportError:
        print("pyreadline is not installed. Install it using 'pip install pyreadline3' for enhanced input experience on Windows.")
        readline = None
else:
    import readline  # For Unix-based systems


# =========================================================================
# STEP 1: INPUT PROCESSING (Hyperparameter Collection & Vocabulary Setup)
# =========================================================================

def get_hyperparameters():
    """
    Collect hyperparameters from the user. This aligns with the 'Input Processing'
    step in which we determine the model configuration. 
    """
    print("Please enter the following hyperparameters for the model training:")
    
    while True:
        try:
            num_slits = int(input("Number of slits (cells) [e.g., 200]: "))
            num_layers = int(input("Number of layers [e.g., 5]: "))
            learning_rate = float(input("Learning rate [e.g., 0.01]: "))
            fixed_epochs = int(input("Number of fixed training epochs per interaction [e.g., 5]: "))
            lambda_duplicate = float(input("Lambda for duplicate loss [e.g., 0.5]: "))
            lambda_nothing = float(input("Lambda for nothing loss [e.g., 0.5]: "))
            threshold = float(input("Threshold for nothing loss [e.g., 0.5]: "))
            break
        except ValueError:
            print("Invalid input. Please enter numerical values.")
    
    return (
        num_slits,         # Max sequence length
        num_layers,        # Number of interference layers
        learning_rate,     # Learning rate
        fixed_epochs,      # Training epochs per user interaction
        lambda_duplicate,  # Weight for duplicate loss
        lambda_nothing,    # Weight for nothing loss
        threshold          # Threshold for nothing loss
    )

# Collect hyperparameters
num_slits, num_layers, learning_rate, fixed_epochs, lambda_duplicate, lambda_nothing, threshold = get_hyperparameters()
print(f"\nHyperparameters set to:\n"
      f"Slits: {num_slits}, Layers: {num_layers}, Learning Rate: {learning_rate}, "
      f"Epochs: {fixed_epochs}, Lambda Duplicate: {lambda_duplicate}, Lambda Nothing: {lambda_nothing}, "
      f"Threshold: {threshold}\n")

# -----------------------------
# Fixed Vocabulary and Mapping
# -----------------------------
special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
alphabet_upper = list(string.ascii_uppercase)  # 'A' - 'Z'
alphabet_lower = list(string.ascii_lowercase)  # 'a' - 'z'
digits = list(string.digits)                  # '0' - '9'
space = [' ']                                 # Space

# Combined, Predefined Vocabulary
predefined_vocab = special_tokens + alphabet_upper + alphabet_lower + digits + space

# Token <-> Index Mappings
token2idx = {token: idx for idx, token in enumerate(predefined_vocab)}
idx2token = {idx: token for token, idx in token2idx.items()}

# Constants for Special Tokens
PAD_IDX = token2idx['<PAD>']
UNK_IDX = token2idx['<UNK>']
START_IDX = token2idx['<START>']
END_IDX = token2idx['<END>']

def tokenize(sentence, max_length):
    """
    Convert a sentence into a list of token indices based on the fixed vocabulary.
    Aligns with the Input Processing step of the architecture.
    """
    chars = list(sentence)
    # Add <START> and <END> tokens
    chars = ['<START>'] + chars + ['<END>']
    indices = [token2idx.get(char, UNK_IDX) for char in chars]
    if len(indices) < max_length:
        indices += [PAD_IDX] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    return indices

def detokenize(indices):
    """
    Convert a list of token indices back to a string, excluding special tokens.
    """
    chars = [idx2token.get(idx, '<UNK>') for idx in indices]
    # Remove special tokens
    chars = [char for char in chars if char not in special_tokens]
    return ''.join(chars)


# =========================================================================
# STEP 2: (Already Covered) - Data Handling is integrated with Step 1
# =========================================================================
# In this code, Step 2's data handling merges with Step 1 to handle input
# tokenization, detokenization, and user input management. 


# =========================================================================
# STEP 3: MODEL DEFINITION (Interference Layers)
# =========================================================================
# We define the model, including:
#   1. Embedding layer: E(x)
#   2. Initial Amplitudes: A = E(x) + 0j
#   3. Interference layers:
#       A_real = A.real * kappa - A.imag * phi
#       A_imag = A.real * phi  + A.imag * kappa
#       A = A_real + 1j * A_imag

class PhotonInterferenceSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_slits, num_layers):
        super(PhotonInterferenceSeq2Seq, self).__init__()
        self.num_slits = num_slits
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Tunable amplitude coefficients (kappa) and phase shifts (phi)
        # Shape: (num_layers, embedding_dim, embedding_dim)
        self.kappa = nn.Parameter(torch.randn(num_layers, embedding_dim, embedding_dim) * 0.1)
        self.phi   = nn.Parameter(torch.randn(num_layers, embedding_dim, embedding_dim) * 0.1)
        
        # Output linear transformation
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, input_seq):
        """
        Args:
            input_seq: Tensor of shape (batch_size, seq_length)
        
        Returns:
            logits: Tensor of shape (batch_size, seq_length, vocab_size)
        """
        batch_size, seq_length = input_seq.size()
        
        # (1) Embedding Layer: E(x)
        embeddings = self.embedding(input_seq)  # (batch_size, seq_length, embedding_dim)
        
        # (2) Initialize Complex Amplitudes A = E(x) + 0j
        A = embeddings + 0j
        
        # (3) Interference Layers
        for l in range(self.num_layers):
            A_real = torch.matmul(A.real, self.kappa[l]) - torch.matmul(A.imag, self.phi[l])
            A_imag = torch.matmul(A.real, self.phi[l])  + torch.matmul(A.imag, self.kappa[l])
            A = A_real + 1j * A_imag
        
        # =========================================================================
        # STEP 4: INTENSITY CALCULATION
        # =========================================================================
        # I = (A.real)^2 + (A.imag)^2
        intensity = A.real**2 + A.imag**2  # (batch_size, seq_length, embedding_dim)
        
        # =========================================================================
        # STEP 5: OUTPUT LAYER
        # =========================================================================
        # logits = O(I)
        logits = self.output_layer(intensity)  # (batch_size, seq_length, vocab_size)
        
        return logits


# =========================================================================
# STEP 6: LOSS FUNCTIONS + STEP 7: OPTIMIZATION STEP
# =========================================================================
#   We compute:
#       L_ce   = CrossEntropyLoss(logits, target)
#       L_dup  = Duplicate Loss (penalize consecutive duplicates)
#       L_noth = Nothing Loss (penalize insufficient content)
#       L_total = L_ce + λ_dup * L_dup + λ_noth * L_noth
#   Then we backpropagate to update parameters.

# -----------------------------
#  Training Setup
# -----------------------------
embedding_dim = 50
vocab_size = len(token2idx)

model = PhotonInterferenceSeq2Seq(vocab_size, embedding_dim, num_slits, num_layers)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}\n")


# =========================================================================
# STEP 8: INFERENCE PROCESS
# =========================================================================
#   For each position i, we select argmax(logits_i) to get token_i
#   Then we detokenize to produce a string response

def generate_response(model, prompt, max_length=200):
    """
    Perform inference by:
      1) Tokenizing the input prompt
      2) Passing through the model
      3) Selecting argmax token at each position
      4) Detokenizing to form the final string
    """
    model.eval()
    with torch.no_grad():
        input_indices = tokenize(prompt, max_length)
        input_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)
        
        logits = model(input_tensor)  # (1, max_length, vocab_size)
        predicted_indices = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()
        
        response = detokenize(predicted_indices)
    return response


# =========================================================================
# CHAT LOOP + TRAINING PER INTERACTION
# =========================================================================
def chat():
    """
    Start an interactive loop:
      - The user provides a prompt and a "correct response"
      - The model trains on that example for a fixed number of epochs
      - The model then generates its own response.
      - The process repeats until the user types 'exit'.
    """
    print("Start chatting with the AI model! Type 'exit' to quit.\n")
    
    while True:
        prompt = input("You: ")
        if prompt.lower() == 'exit':
            print("Exiting chat.")
            break
        
        response = input("AI (provide the correct response for training): ")
        if response.lower() == 'exit':
            print("Exiting chat.")
            break
        
        # Tokenize prompt and response
        max_length = num_slits
        input_indices = tokenize(prompt, max_length)
        target_indices = tokenize(response, max_length)
        
        input_tensor  = torch.tensor([input_indices], dtype=torch.long).to(device)
        target_tensor = torch.tensor([target_indices], dtype=torch.long).to(device)
        
        # Train for fixed_epochs on the single example
        model.train()
        for _ in range(fixed_epochs):
            optimizer.zero_grad()
            
            # Forward pass: logits
            logits = model(input_tensor)
            batch_size, seq_length, vsz = logits.size()
            
            # Reshape for cross-entropy
            logits_flat = logits.view(-1, vsz)       # (batch_size * seq_length, vocab_size)
            target_flat = target_tensor.view(-1)     # (batch_size * seq_length)
            
            # (a) Cross-Entropy Loss
            cross_entropy_loss = criterion(logits_flat, target_flat)
            
            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=2)     # (batch_size, seq_length, vocab_size)
            
            # (b) Duplicate Loss
            # We gather probabilities of token_i == token_(i-1)
            if seq_length > 1:
                prev_tokens = target_tensor[:, :-1]     # (batch_size, seq_length - 1)
                probs_shifted = probs[:, 1:, :]         # (batch_size, seq_length - 1, vocab_size)
                
                prev_tokens_flat = prev_tokens.view(-1)           # (batch_size*(seq_length-1))
                probs_duplicate_flat = probs_shifted.view(-1, vsz)  # (batch_size*(seq_length-1), vocab_size)
                
                # Probability that token_i == token_(i-1)
                probs_of_duplicates = probs_duplicate_flat[
                    torch.arange(probs_duplicate_flat.size(0)), prev_tokens_flat
                ]
                duplicate_loss = probs_of_duplicates.mean()
            else:
                # If seq_length <= 1, no duplicates possible
                duplicate_loss = torch.tensor(0.0, device=device)
            
            # (c) Nothing Loss
            # Content tokens = everything except <PAD>, <UNK>, <START>, <END>
            content_token_indices = [
                token2idx[tkn] for tkn in predefined_vocab 
                if tkn not in ['<PAD>', '<UNK>', '<START>', '<END>']
            ]
            content_token_tensor = torch.tensor(content_token_indices, device=device)
            
            # Sum probabilities over content tokens
            probs_content = probs[:, :, content_token_tensor]  # (batch_size, seq_length, #content_tokens)
            probs_content_sum = probs_content.sum(dim=2)       # (batch_size, seq_length)
            avg_content_prob = probs_content_sum.mean(dim=1)   # (batch_size,)
            
            # Penalize if avg_content_prob < threshold
            nothing_loss = torch.clamp(threshold - avg_content_prob, min=0).mean()
            
            # (d) Total Loss
            total_loss = (
                cross_entropy_loss 
                + lambda_duplicate * duplicate_loss 
                + lambda_nothing   * nothing_loss
            )
            
            # Optimization Step
            total_loss.backward()
            optimizer.step()
        
        print(f"Model trained on the latest interaction. Total Loss: {total_loss.item():.4f} "
              f"(CE: {cross_entropy_loss.item():.4f}, Duplicate: {duplicate_loss.item():.4f}, "
              f"Nothing: {nothing_loss.item():.4f})")
        
        # Generate and display model's response
        ai_response = generate_response(model, prompt, max_length)
        print(f"AI: {ai_response}\n")


# =========================================================================
# MAIN EXECUTION (Start Chat)
# =========================================================================
if __name__ == "__main__":
    chat()

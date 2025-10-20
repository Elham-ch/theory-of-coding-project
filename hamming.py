import matplotlib.pyplot as plt
plt.rc('text', usetex=False)  # Avoid LaTeX issues
plt.rc('font', family='serif')
import heapq
import random
import numpy as np
from scipy.optimize import fsolve
import math

# Global variable to store column swaps
column_swaps = []

# Step 1: Huffman Coding
def build_huffman_tree(symbols, probs):
    heap = [(prob, char, None, None) for char, prob in zip(symbols, probs)]
    heapq.heapify(heap)
    while len(heap) > 1:
        prob1, char1, left1, right1 = heapq.heappop(heap)
        prob2, char2, left2, right2 = heapq.heappop(heap)
        heapq.heappush(heap, (prob1 + prob2, None, (prob1, char1, left1, right1), (prob2, char2, left2, right2)))
    return heap[0]

def generate_huffman_codes(tree, current_code="", codes=None):
    if codes is None:
        codes = {}
    prob, char, left, right = tree
    if char is not None:
        codes[char] = current_code or "0"
    if left:
        generate_huffman_codes(left, current_code + "0", codes)
    if right:
        generate_huffman_codes(right, current_code + "1", codes)
    return codes

# New function to decode Huffman codes
def huffman_decode(encoded_message, huffman_codes):
    reverse_codes = {v: k for k, v in huffman_codes.items()}
    decoded = ""
    current_code = ""
    for bit in encoded_message:
        current_code += bit
        if current_code in reverse_codes:
            decoded += reverse_codes[current_code]
            current_code = ""
    return decoded

def generate_random_string(length, symbols, probs):
    return ''.join(random.choices(symbols, weights=probs, k=length))

def swap_cols(matrix, col1, col2):
    for row in matrix:
        row[col1], row[col2] = row[col2], row[col1]

def undo_swaps(matrix):
    for swap in reversed(column_swaps):
        c1, c2 = map(int, swap.split(","))
        swap_cols(matrix, c1, c2)
    return matrix

def standardize_H(H):
    r = len(H)
    num_cols = len(H[0])
    At_part = []
    col = 0
    while (2 ** col) - 1 < num_cols:
        target_col = (2 ** col) - 1
        if col != target_col:
            swap_cols(H, col, target_col)
            column_swaps.append(f"{col},{target_col}")
        col += 1
    for row in H:
        At_part.append(row[r:])
    return H, At_part

def transpose(matrix):
    return [list(row) for row in zip(*matrix)]

def build_H_matrix(r):
    A = [[int(bit) for bit in format(i, f'0{r}b')[::-1]] for i in range(1, 2 ** r)]
    H = transpose(A)
    H, At = standardize_H(H)
    return H, At

def apply_swaps_to_matrix(matrix):
    for swap in column_swaps:
        c1, c2 = map(int, swap.split(","))
        swap_cols(matrix, c1, c2)
    return matrix

def left_identity_form(matrix):
    rows, cols = len(matrix), len(matrix[0])
    n = min(rows, cols)
    matrix = [row[:] for row in matrix] 
    for i in range(n):
        pivot = i
        while pivot < rows and matrix[pivot][i] == 0:
            pivot += 1
        if pivot == rows:
            continue

        if pivot != i:
            matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
        for j in range(rows):
            if j != i and matrix[j][i] == 1:
                for k in range(cols):
                    matrix[j][k] = (matrix[j][k] + matrix[i][k]) % 2
    return matrix

def build_G_matrix(r):
    H, At = build_H_matrix(r)
    A = transpose(At)
    total_cols = len(H[0])
    k = total_cols - r

    I_k = [[1 if i == j else 0 for j in range(k)] for i in range(k)]
    for i in range(k):
        A[i].extend(I_k[i])

    A = apply_swaps_to_matrix(A)
    A = left_identity_form(A)

    
    if not A or not A[0]:
        print("[]")
    else:
        max_width = max(max(len(str(item)) for item in row) for row in A)
        print("[")
        for i, row in enumerate(A):
            row_str = " ".join(f"{item:>{max_width}}" for item in row)
            print(f"  {row_str}{' ,' if i < len(A)-1 else ''}")
        print("]")

    return A, H, k, total_cols

def hamming_encode(message, G, k):
        original_length = len(message)
        if len(message) % k != 0:
            padding = k - (len(message) % k)
            message = message + '0' * padding
        blocks = [message[i:i+k] for i in range(0, len(message), k)]
        codewords = []
        for block in blocks:
            message_bits = np.array([int(b) for b in block])
            codeword = (message_bits @ np.array(G)) % 2
            codewords.append(''.join(map(str, codeword)))
        return codewords, original_length

def add_noise(code, p, length):
    try:
        if len(code) != length:
            raise ValueError(f"Code length {len(code)} does not match expected length {length}")
       
        num_errors = np.random.binomial(length, p)
        
        error_positions = np.random.choice(length, num_errors, replace=False)
        
        error_vector = ['0'] * length
        for pos in error_positions:
            error_vector[pos] = '1'
        
        noisy = [str(int(c) ^ int(e)) for c, e in zip(code, error_vector)]
        return ''.join(noisy)
    except Exception as e:
        print(f"Error in add_noise: {e}")
        raise


import numpy as np

def hamming_decode(noisy_codeword, H, n, r):
    try:
        if not noisy_codeword:
            raise ValueError("Noisy codeword is empty")
        if len(noisy_codeword) != n:
            raise ValueError(f"Noisy codeword length {len(noisy_codeword)} does not match n={n}")
        one_positions = [i + 1 for i, bit in enumerate(noisy_codeword) if bit == '1']
        if not one_positions:
            return noisy_codeword
        
        binary_matrix = [[int(b) for b in format(pos, f'0{r}b')[::-1]] for pos in one_positions]
        binary_matrix = np.array(binary_matrix).T
        syndrome = np.sum(binary_matrix, axis=1) % 2
        error_pos = int(''.join(map(str, syndrome[::-1])), 2)
        corrected = list(noisy_codeword)
        if error_pos > 0 and error_pos <= n:
            corrected[error_pos - 1] = '1' if corrected[error_pos - 1] == '0' else '0'
        return ''.join(corrected)
    except Exception as e:
        print(f"Error in hamming_decode: {e}")
        raise

def extract_message(corrected_codewords, k, r, original_length):
    try:
        data_positions = list(range(k))
        decoded_blocks = []
        for codeword in corrected_codewords:
            if len(codeword) != 2**r - 1:
                raise ValueError(f"Corrected codeword length {len(codeword)} does not match n={2**r - 1}")
            block = ''.join([codeword[i] for i in data_positions])
            decoded_blocks.append(block)
        message = ''.join(decoded_blocks)
        if len(message) > original_length:
            message = message[:original_length]
        return message
    except Exception as e:
        print(f"Error in extract_message: {e}")
        raise


def solve_for_p(r):
    try:
        n = 2**r - 1
        k = n - r
        code_rate = k / n
        def entropy_function(p):
            if p <= 0 or p >= 1:
                return np.inf
            return 1 + p * np.log2(p) + (1 - p) * np.log2(1 - p) - code_rate
        p_initial_guess = 0.1
        p_solution = fsolve(entropy_function, p_initial_guess)[0]
        if not 0 < p_solution < 1:
            raise ValueError(f"Invalid p value: {p_solution}. Must be between 0 and 1.")
        return p_solution, code_rate
    except Exception as e:
        print(f"Error in solve_for_p: {e}")
        raise



symbols = ['a', 'b', 'c', 'd']
probs = [0.4, 0.3, 0.2, 0.1]
huffman_tree = build_huffman_tree(symbols, probs)
huffman_codes = generate_huffman_codes(huffman_tree)
print(f"Huffman codes: {huffman_codes}")
        

while True:
    try:
        length = int(input("Enter length of random string (positive integer): "))
        if length <= 0:
            raise ValueError("Length must be positive")
        break
    except ValueError as e:
        print(f"Invalid input: {e}. Please enter a positive integer.")
        
while True:
    try:
        r = int(input("Enter r for Hamming code (positive integer, e.g., 3): "))
        if r <= 0:
            raise ValueError("r must be positive")
        if 2**r - 1 - r <= 0:
            raise ValueError("r is too large; k must be positive")
        break
    except ValueError as e:
        print(f"Invalid input: {e}. Please enter a valid positive integer.")
        

p_solution, code_rate = solve_for_p(r)
print(f"Code rate (k/n): {code_rate:.6f}")
print(f"Solved p for entropy equation: {p_solution:.6f}")
a = (1 - p_solution) / p_solution
print(f"a = (1-p)/p: {a:.6f}")
        
random_string = generate_random_string(length, symbols, probs)
print(f"Generated random string: {random_string}")
        
huffman_encoded = ''.join(huffman_codes[char] for char in random_string)
print(f"Huffman encoded string: {huffman_encoded}")
        
G, H, k, n = build_G_matrix(r)
print(f"Hamming code parameters: n={n}, k={k}")
print("H matrix:")
for row in H:
    print(row)
        
hamming_codewords, original_huffman_length = hamming_encode(huffman_encoded, G, k)
print(f"Hamming codewords: {hamming_codewords}")
        

p_list = np.linspace(0.01, 0.49, 50)

p_list = [max(1e-6, p) for p in p_list]
p_x = []
p_error = []
        
for p in p_list:
    noisy_codewords = []
    for codeword in hamming_codewords:
        noisy = add_noise(codeword, p=p, length=n)
        noisy_codewords.append(noisy)
            
    corrected_codewords = []
    for noisy_codeword in noisy_codewords:
        corrected = hamming_decode(noisy_codeword, H, n, r)
        corrected_codewords.append(corrected)
            
    extracted_message = extract_message(corrected_codewords, k, r, original_huffman_length)
    print("Final extracted blocks:", corrected_codewords)
            
   
    bit_errors = sum(a != b for a, b in zip(huffman_encoded, extracted_message))
    total_bits = len(huffman_encoded)
    perror = bit_errors / total_bits if total_bits > 0 else 0
            
    
    p_x.append((1 - p) / p)
    p_error.append(perror)
        

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(p_x, p_error, marker='o', linestyle='-', color='b', label=r"$p_{\mathrm{error}}$")
ax.set_xscale('log')
ax.set_yscale('symlog', linthresh=1e-5)   # 👈 اضافه شده
ax.axhline(y=0, color="black", linestyle="--")
ax.axvline(x=1, color="grey", linestyle=":")
ax.axvline(x=a, color="red", linestyle="-", label=r"$x=(1-p)/p$")
ax.set(
    xlabel=r'$(1 - p)/p$',
    ylabel=r'$p_{\mathrm{error}}$',
    title='Bit Error Rate vs Channel Noise'
)
ax.grid(True)
ax.legend(fontsize=12)
plt.tight_layout()
plt.show()

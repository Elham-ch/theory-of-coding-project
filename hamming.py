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
    matrix = [row[:] for row in matrix]  # کپی ماتریس

    # محاسبه r از ابعاد (r = n - k، k = rows)
    r = cols - rows  # چون cols = n و rows = k

    # عملیات سطری برای فرم کاهش‌یافته (row echelon و reduced row echelon)
    pivot_cols = []
    col = 0
    for row in range(rows):
        # پیدا کردن pivot
        pivot_row = row
        while pivot_row < rows and matrix[pivot_row][col] == 0:
            pivot_row += 1
        if pivot_row == rows:
            col += 1
            if col >= cols:
                break
            continue

        # جابه‌جایی سطر
        if pivot_row != row:
            matrix[row], matrix[pivot_row] = matrix[pivot_row], matrix[row]

        # ذخیره موقعیت ستون pivot
        pivot_cols.append(col)

        # حذف پایین
        for i in range(row + 1, rows):
            if matrix[i][col] == 1:
                for j in range(cols):
                    matrix[i][j] = (matrix[i][j] + matrix[row][j]) % 2

        col += 1
        if col >= cols:
            break

    # حذف بالا (برای فرم reduced row echelon)
    for row in range(rows - 1, 0, -1):
        lead_col = next((j for j in range(cols) if matrix[row][j] == 1), None)
        if lead_col is None:
            continue
        for i in range(row - 1, -1, -1):
            if matrix[i][lead_col] == 1:
                for j in range(cols):
                    matrix[i][j] = (matrix[i][j] + matrix[row][j]) % 2

    # اگر r >= 4، جابه‌جایی ستون‌ها برای تشکیل هویت در سمت چپ
    if r >= 4 and len(pivot_cols) == rows:
        # مرتب کردن ستون‌ها بر اساس موقعیت pivotها
        sorted_pivot_indices = sorted(range(len(pivot_cols)), key=lambda i: pivot_cols[i])
        current_order = list(range(rows))
        for i in range(rows):
            if sorted_pivot_indices[i] != i:
                swap_col1 = i
                swap_col2 = sorted_pivot_indices[i]
                for row_list in matrix:
                    row_list[swap_col1], row_list[swap_col2] = row_list[swap_col2], row_list[swap_col1]
                # به‌روزرسانی ترتیب
                sorted_pivot_indices[i], sorted_pivot_indices[sorted_pivot_indices.index(i)] = i, sorted_pivot_indices[i]

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
    ones_count = math.ceil(length * p)  # Ceiling of length * p for number of 1s
    zeros_count = length - ones_count
    error_vector = ['1'] * ones_count + ['0'] * zeros_count
    random.shuffle(error_vector)
    print(f"Error vector: {''.join(error_vector)}")
    noisy = [str(int(c) ^ int(e)) for c, e in zip(code, error_vector)]
    print(f"Hamming code with noise: {''.join(noisy)}")
    return ''.join(noisy)

def hamming_decode(noisy_codeword, H, n, r):
    if not noisy_codeword or len(noisy_codeword) != n or not all(b in '01' for b in noisy_codeword):
        return noisy_codeword
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

def extract_message(corrected_codewords, k, r, original_length):
    data_positions = list(range(k))
    decoded_blocks = []
    for codeword in corrected_codewords:
        if len(codeword) != 2**r - 1:
            continue
        block = ''.join([codeword[i] for i in data_positions])
        decoded_blocks.append(block)
    message = ''.join(decoded_blocks)
    if len(message) > original_length:
        message = message[:original_length]
    return message, decoded_blocks  # Return both message and blocks

# Function to compute code rate and solve for p
def solve_for_p(r):
    n = 2**r - 1
    k = n - r
    code_rate = k / n
    def entropy_function(p):
        if p <= 0 or p >= 1:
            return np.inf
        return 1 + p * np.log2(p) + (1 - p) * np.log2(1 - p) - code_rate
    p_initial_guess = 0.1
    p_solution = fsolve(entropy_function, p_initial_guess)[0]
    return p_solution, code_rate

# Main Program
symbols = ['a', 'b', 'c', 'd']
probs = [0.4, 0.3, 0.2, 0.1]
huffman_tree = build_huffman_tree(symbols, probs)
huffman_codes = generate_huffman_codes(huffman_tree)
print(f"Huffman codes: {huffman_codes}")
        
# Get input from user with validation
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
        
# Calculate code rate and solve for p
p_solution, code_rate = solve_for_p(r)
print(f"Code rate (k/n): {code_rate:.6f}")
print(f"Solved p for entropy equation: {p_solution:.6f}")
a = (1 - p_solution) / p_solution
print(f"a = (1-p)/p: {a:.6f}")

# Build and print G and H matrices right after getting r
G, H, k, n = build_G_matrix(r)
print("Matrix G:")
if not G or not G[0]:
    print("[]")
else:
    max_width = max(max(len(str(item)) for item in row) for row in G)
    print("[")
    for i, row in enumerate(G):
        row_str = " ".join(f"{item:>{max_width}}" for item in row)
        print(f"  {row_str}{' ,' if i < len(G)-1 else ''}")
    print("]")
print("Matrix H:")
if not H or not H[0]:
    print("[]")
else:
    max_width = max(max(len(str(item)) for item in row) for row in H)
    print("[")
    for i, row in enumerate(H):
        row_str = " ".join(f"{item:>{max_width}}" for item in row)
        print(f"  {row_str}{' ,' if i < len(H)-1 else ''}")
    print("]")
print(f"Hamming code parameters: n={n}, k={k}")
        
random_string = generate_random_string(length, symbols, probs)
print(f"Generated random string: {random_string}")
        
huffman_encoded = ''.join(huffman_codes[char] for char in random_string)
print(f"Huffman encoded string: {huffman_encoded}")
        
hamming_codewords, original_huffman_length = hamming_encode(huffman_encoded, G, k)
print(f"Hamming codewords: {hamming_codewords}")
        
# Generate p values: divide [0, 0.49] into 100 parts
p_list = np.linspace(0.001, 0.5, 50)
# Replace p=0 with a small value to avoid division by zero
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
            
    extracted_message, decoded_blocks = extract_message(corrected_codewords, k, r, original_huffman_length)
    print(f"k-sized blocks: {decoded_blocks}")
    decoded_string = huffman_decode(extracted_message, huffman_codes)
    print(f"Final output: {decoded_string}")
            
    # Compute perror
    bit_errors = sum(a != b for a, b in zip(huffman_encoded, extracted_message))
    total_bits = len(huffman_encoded)
    perror = bit_errors / total_bits if total_bits > 0 else 0
            
    # Collect data for plot
    p_x.append((1 - p) / p)
    p_error.append(perror)
        
# Plotting
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(p_x, p_error, marker='o', linestyle='-', color='b', label=r"$p_{\mathrm{error}}$")
ax.set_xscale('log')
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
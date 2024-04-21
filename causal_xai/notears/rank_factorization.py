import torch 
import numpy as np
import sympy as sp

def rref_torch(A): 
    # Step 1: Forward Elimination
    for i in range(min(A.size(0), A.size(1))):
        # Partial pivoting
        max_val, max_row = torch.max(torch.abs(A[i:, i]), dim=0)
        max_row += i

        # Swap rows
        A[[i, max_row]] = A[[max_row, i]]

        # Make the diagonal element 1
        A[i] = A[i].clone() / A[i, i]

        # Make the other elements in the column 0
        for j in range(i + 1, A.size(0)):
            A[j] -= A[j, i] * A[i]

    # Step 2: Back Substitution
    for i in range(min(A.size(0), A.size(1)) - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            A[j] -= A[j, i] * A[i]
    return A

def non_pivot_columns_indices(matrix):
    zero_diag_columns = []
    for i in range(matrix.shape[0]):
        if matrix[i, i] == 0:
            zero_diag_columns.append(i)
    return zero_diag_columns

def rank_factorization(W):
    #-----------------------------------------------
    # FIND REDUCED ROW ECHELON FORM

    # Compute the reduced row echelon form
    rref_matrix = rref_torch(W)
    # Convert the resulting torch tensor back to numpy array
    B = rref_matrix.numpy()

    #-----------------------------------------------
    # FIND TRUTHLY POVIT COLUMNS
    # Find non-pivot columns
    non_pivot_columns = torch.nonzero(torch.all(W == 0, dim=0)).squeeze().numpy()

    # Find rows with all values being zeros
    zero_rows = np.where(~np.any(B != 0, axis=1))[0]

    # Create a boolean mask for elements to keep
    mask = ~np.isin(non_pivot_columns, zero_rows)

    # Apply the mask to the original array to filter out elements
    non_pivot_columns = non_pivot_columns[mask]

    #-----------------------------------------------
    # FIND SUB MATRIX
    C = W[:, ~np.isin(np.arange(W.shape[1]), non_pivot_columns)]

    non_zero_rows = np.any(B != 0, axis=1)
    F = torch.tensor(B[non_zero_rows], dtype=torch.float)

    if torch.equal(W, torch.mm(C, F)):
        print("Successfully factorized!")
    else:
        print("Error!")

    return C, F

def main(): 
    # Define your matrix
    A = torch.tensor([[1., 2., 1.],
                    [2., 3., 1.],
                    [3., 3., 3.]], dtype=torch.float)
    A1, A2 = rank_factorization(A)
    
if __name__ == '__main__':
    main()
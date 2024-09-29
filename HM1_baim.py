import numpy as np
import matplotlib.pyplot as plt

def read_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    return x, y

def generate_matrix(x, y, n):
    A = np.ones((len(x), n))
    for row in range(len(A)):
        for col in range(n-1):
            A[row][col] = x[row] ** (n - col - 1)
    b = y.reshape(-1, 1)
    return A, b

def transpose_matrix(A):
    tA = np.zeros((A.shape[1], A.shape[0]))
    for i in range(len(A)):
        for j in range(len(A[i])):
            tA[j][i] = A[i][j]
    return tA

def multiply_matrix(A, B):
    result = np.zeros((A.shape[0], B.shape[1]))
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

# LU decomposition
def LU_inverse(A):
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    # LU Decomposition
    for i in range(n):
        for j in range(i+1, n):
            factor = U[j, i] / U[i, i]
            U[j] = U[j] - factor * U[i]
            L[j, i] = factor
    
    # inverse of U
    U_inv = np.eye(n)
    for col in range(n):
        U_inv[col, col] = 1 / U[col, col]
        for row in range(col-1, -1, -1):
            sum_ = 0
            for i in range(row+1, col+1):
                sum_ += U[row, i] * U_inv[i, col]
            U_inv[row, col] = -sum_ / U[row, row]
    #inverse of L
    L_inv = np.eye(n)
    for col in range(n):
        for row in range(col+1, n):
            sum_ = 0
            for i in range(col, row):
                sum_ += L[row, i] * L_inv[i, col]
            L_inv[row, col] = -sum_
    
    return multiply_matrix(U_inv, L_inv)

def lse_closed_form(A, y, lambd):
    ATA = multiply_matrix(transpose_matrix(A), A)
    ATA_lambd = ATA + lambd * np.eye(A.shape[1])
    ATy = multiply_matrix(transpose_matrix(A), y)
    coef = multiply_matrix(LU_inverse(ATA_lambd), ATy)
    return coef

def compute_error(A, coef, y):
    y_pred = multiply_matrix(A, coef)
    error = 0
    for i in range(len(y)):
        error += (y[i][0] - y_pred[i][0]) ** 2
    return error

def steepest_descent(A, y, learning_rate=1e-4, max_iter=10000, eps=1e-6, clip_value=1e2):
    coef = np.random.randn(A.shape[1], 1)
    AT = transpose_matrix(A)
    for i in range(max_iter):
        y_pred = multiply_matrix(A, coef)
        gradient = 2 * multiply_matrix(AT, (y_pred - y))
        #gradient = np.clip(gradient, -clip_value, clip_value)
        coef = coef - learning_rate * gradient
        
        if sum(x**2 for x in gradient)**0.5 < eps:
            #print(f"Converged after {i+1} iterations")
            break
            
    return coef


def newtons_method(A, y, a_init, num_iterations=100, eps=1e-6):
    a = a_init
    for i in range(num_iterations):
        gradient = multiply_matrix(transpose_matrix(A), multiply_matrix(A, a) - y)
        hessian = multiply_matrix(transpose_matrix(A), A)
        hessian_inv = LU_inverse(hessian)
        #a_{k+1} = a_k - H(a_k)^{-1} * gradient
        delta_a = multiply_matrix(hessian_inv, gradient)
        a = a - delta_a
        #if np.linalg.norm(delta_a) < eps:
        if sum(x**2 for x in delta_a)**0.5 < eps:
            #print(f"Converged after {i+1} iterations")
            break
    return a


def print_equation(coef):
    terms = [f"{coef[i][0]:.4f}*x^{len(coef)-i-1}" for i in range(len(coef))]
    equation = " + ".join(terms)
    print(f"Equation: y = {equation}")

def visualize(x, y, coefs, labels):
    plt.scatter(x, y, color='red', label='Data Points')
    x_line = np.linspace(min(x), max(x), 1000)
    for coef, label in zip(coefs, labels):
        A_line, _ = generate_matrix(x_line, np.zeros(len(x_line)), len(coef))
        y_line = multiply_matrix(A_line, coef)
        plt.plot(x_line, y_line, label=label, linewidth = 2 ,linestyle=['solid', '--', ':'][labels.index(label)])
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y') 
    plt.title('Best Fitting Curves')
    plt.show()


def main():
    file_path = input("Enter the file path: ")
    n = int(input("Enter the number of polynomial bases: "))
    lambd = float(input("Enter the regularization parameter lambda: "))
    
    # data
    x, y = read_data(file_path)
    A, b = generate_matrix(x, y, n)
    
    print("-"*50)
    # Closed-form LSE
    coef_lse = lse_closed_form(A, b, lambd)
    error_lse = compute_error(A, coef_lse, b)
    print("Closed-form LSE:")
    print_equation(coef_lse)
    print(f"Error: {error_lse}")
    print("-"*50)
    
    # Steepest descent
    coef_sd = steepest_descent(A, b)
    error_sd = compute_error(A, coef_sd, b)
    print("Steepest Descent:")
    print_equation(coef_sd)
    print(f"Error: {error_sd}")
    print("-"*50)

    # Newton's method
    a_init = np.zeros((A.shape[1], 1))  # Initial guess: a column vector of zeros
    num_iterations = 100
    coef_newton = newtons_method(A, b, a_init, num_iterations)
    error_newton = compute_error(A, coef_newton, b)
    print("Newton's Method:")
    print_equation(coef_newton)
    print(f"Error: {error_newton}")

    visualize(x, y, [coef_lse, coef_sd, coef_newton], ['LSE', 'Steepest Descent', 'Newton\'s Method'])
    #print(coef_lse)

if __name__ == "__main__":
    main()

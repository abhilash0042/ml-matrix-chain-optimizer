import sys

def solve_mcm(dims):
    """
    Solves the Matrix Chain Multiplication problem using Dynamic Programming.
    Returns (min_cost, optimal_order).
    """
    n = len(dims) - 1
    # m[i][j] stores the minimum cost of multiplying matrices from i to j
    m = [[0] * n for _ in range(n)]
    # s[i][j] stores the index 'k' that split the chain optimally
    s = [[0] * n for _ in range(n)]

    # length is the chain length
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            m[i][j] = float('inf')
            for k in range(i, j):
                # cost = left side + right side + combination cost
                q = m[i][k] + m[k+1][j] + dims[i] * dims[k+1] * dims[j+1]
                if q < m[i][j]:
                    m[i][j] = q
                    s[i][j] = k
    
    def get_order(s, i, j):
        if i == j:
            return f"M{i+1}"
        k = s[i][j]
        left = get_order(s, i, k)
        right = get_order(s, k+1, j)
        return f"({left} x {right})"

    return m[0][n-1], get_order(s, 0, n-1)

if __name__ == "__main__":
    test_dims = [10, 30, 5, 60]
    
    if len(sys.argv) > 1:
        try:
            test_dims = [int(x) for x in sys.argv[1:]]
        except ValueError:
            print("Usage: python mcm_solver.py dim1 dim2 dim3 ...")
            sys.exit(1)
            
    cost, order = solve_mcm(test_dims)
    print(f"\nMatrix Chain Dimensions: {test_dims}")
    print(f"Optimal Order: {order}")
    print(f"Exact Minimum Cost: {cost:,}")

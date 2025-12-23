import numpy as np
import matplotlib.pyplot as plt


def lagrange_interpolation(x, y, value):
    """
    Performs Lagrange interpolation.
    Formula:
    L(x) = \sum_{i=0}^{n} y_i \cdot l_i(x)
    where l_i(x) = \prod_{j=0, j \neq i}^{n} \frac{x - x_j}{x_i - x_j}
    """
    n = len(x)
    result = 0
    for i in range(n):
        # Calculate Lagrange basis polynomial l_i(x)
        term = y[i]
        for j in range(n):
            if i != j:
                # l_i(x) is 1 at x_i and 0 at all other x_j
                term *= (value - x[j]) / (x[i] - x[j])
        result += term
    return result


def get_divided_diff_table(x, y):
    """
    Calculates the Newton's divided difference table.
    Formula:
    f[x_i, x_{i+1}] = (f(x_{i+1}) - f(x_i)) / (x_{i+1} - x_i)
    """
    n = len(y)
    table = np.zeros([n, n])
    table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])
    return table


def newton_interpolation(x, y, value):
    """
    Performs Newton interpolation using Horner's-like scheme.
    Formula:
    P_n(x) = a_0 + a_1(x-x_0) + a_2(x-x_0)(x-x_1) + ...
    """
    table = get_divided_diff_table(x, y)
    coef = table[0, :]  # Coefficients are the top diagonal of the table
    n = len(x) - 1

    # Efficient evaluation using Horner's method
    p = coef[n]
    for k in range(1, n + 1):
        p = coef[n - k] + (value - x[n - k]) * p
    return p


def plot_results(x, y, target_x, target_y):
    """Visualizes the data points and the interpolated curve."""
    # Generate points for a smooth curve
    x_smooth = np.linspace(min(x) - 0.5, max(x) + 0.5, 200)
    y_smooth = [lagrange_interpolation(x, y, xi) for xi in x_smooth]

    plt.figure(figsize=(12, 7))
    plt.plot(x_smooth, y_smooth, 'b-', label='Interpolation Polynomial', linewidth=1.5)
    plt.scatter(x, y, color='red', s=60, label='Known Data Points (Nodes)', zorder=5)
    plt.scatter(target_x, target_y, color='green', marker='X', s=200,
                label=f'Interpolated Point: ({target_x}, {target_y:.4f})')

    plt.title("Numerical Analysis: Lagrange & Newton Interpolation", fontsize=14)
    plt.xlabel("X-axis", fontsize=12)
    plt.ylabel("Y-axis", fontsize=12)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def main():
    print("=" * 50)
    print("   SCIENTIFIC INTERPOLATION SOLVER (PRO VERSION)")
    print("=" * 50)

    try:
        n = int(input("\nEnter number of data points (n >= 2): "))
        if n < 2:
            print("Error: You need at least 2 points for interpolation.")
            return

        x_points, y_points = [], []
        for i in range(n):
            while True:
                try:
                    xi = float(input(f"  Enter x[{i}]: "))
                    if xi in x_points:
                        print("  ! Error: X values must be unique. Try again.")
                        continue
                    break
                except ValueError:
                    print("  ! Error: Please enter a valid number.")

            yi = float(input(f"  Enter y[{i}]: "))
            x_points.append(xi)
            y_points.append(yi)

        target_x = float(input("\nEnter the X value to interpolate: "))

        # 1. Calculate Results
        res_lagrange = lagrange_interpolation(x_points, y_points, target_x)
        res_newton = newton_interpolation(x_points, y_points, target_x)

        # 2. Display Divided Difference Table
        print("\n" + "-" * 25)
        print("NEWTON'S DIVIDED DIFFERENCE TABLE")
        print("-" * 25)
        diff_table = get_divided_diff_table(x_points, y_points)
        # Format the table for better readability
        for i in range(n):
            row = [f"{diff_table[i][j]:.4f}" if j < n - i else "" for j in range(n)]
            print(f"Row {i}: {' | '.join(row)}")

        # 3. Final Comparison
        print("\n" + "=" * 50)
        print(f"{'METHOD':<20} | {'RESULT':<20}")
        print("-" * 50)
        print(f"{'Lagrange Method':<20} | {res_lagrange:<20.8f}")
        print(f"{'Newton Method':<20} | {res_newton:<20.8f}")
        print(f"{'Absolute Difference':<20} | {abs(res_lagrange - res_newton):<20.2e}")
        print("=" * 50)

        # 4. Visualization
        print("\nGenerating graph... Please check the pop-up window.")
        plot_results(x_points, y_points, target_x, res_lagrange)

    except ValueError:
        print("\nFatal Error: Invalid input detected. Exiting...")


if __name__ == "__main__":
    main()
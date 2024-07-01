#!/usr/bin/env python3

import os
import sys
import csv
import math
import matplotlib.pyplot as plt     # for plotting and graphical support
import pandas as pd                 # for plottign and graphical support
import numpy as np                  # for linear regression
from typing import List, Tuple      # for data types

# I/O file paths
# feel free to customize them :)
PATH_INPUT_CSV = 'CSVs/in.csv'
PATH_OUTPUT_CSV = 'CSVs/out.csv'
PATH_CHART_BARS_PLOT = 'images/fig1.png'
PATH_HISTOGRAM_PLOT = 'images/fig2.png'
PATH_SCATTER_PLOT = 'images/fig3.png'
PATH_LIN_REGGR_PLOT = 'images/fig4.png'
PATH_QUAD_REGGR_PLOT = 'images/fig5.png'
PATH_CUBIC_REGGR_PLOT = 'images/fig6.png'
PATH_ALL_PLOTS = 'images/fig7.png'


def approx(num: float) -> float:
    """Approximate a number upwards to two decimal places
    2.590 -> 2.59 ; 2.594 -> 2.59 ; 2.595 -> 2.60
    """
    return round(num, 2)


def average(nums: List[float]) -> float:
    return approx(sum(nums) / float(len(nums)))


class InvalidCSVFormatException(Exception):
    pass


def read_csv_file(file_path: str) -> Tuple[str, str, List[float], List[float]]:
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        
        try:
            header = next(reader)
        except StopIteration:
            raise InvalidCSVFormatException("CSV file is empty.")
        
        if len(header) != 2:
            raise InvalidCSVFormatException("CSV file must have exactly two columns in the header.")
        
        x_name, y_name = header
        x_values: List[float] = []
        y_values: List[float] = []
        
        for row in reader:
            if len(row) != 2:
                raise InvalidCSVFormatException(f"Row has an incorrect number of columns: {row}")
            try:
                x_values.append(float(row[0]))
                y_values.append(float(row[1]))
            except ValueError:
                raise InvalidCSVFormatException(f"Non-numeric data found in row: {row}")
        
        return (x_name, y_name, x_values, y_values)




def write_csv_file(file_path: str, header1: str, header2: str, header3: str,
                   header4: str, header5: str, header6: str, header7: str, header8: str, header9: str,
                   col1: List[float], col2: List[float], col3: List[float], col4: List[float], 
                   col5: List[float], col6: List[str], col7: List[str], col8: List[str], col9: List[str]) -> None:
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow([header1, header2, header3, header4, header5, header6, header7, header8, header9])
        
        for row in zip(col1, col2, col3, col4, col5, col6, col7, col8, col9):
            writer.writerow(row)

def get_plot_img_scatter(x_name: str, y_name: str, x_values: List[float], y_values: List[float]) -> None:
    fig, ax = plt.subplots()
    ax.scatter(x_values, y_values, color='blue', label='Data points')
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f'Scatter Plot of {y_name} as a function of {x_name}')
    ax.legend()
    plt.tight_layout()
    fig.savefig(PATH_SCATTER_PLOT)
    plt.close(fig)

def compute_linear_regression_algorithm(x_values: List[float], y_values: List[float]) -> Tuple[float, float]:
    x = np.array(x_values)
    y = np.array(y_values)
    A = np.vstack([x, np.ones(len(x))]).T
    slope, y_intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    
    return (slope, y_intercept)



def get_plot_img_linnear_reggression(x_name: str, y_name: str, x_values: List[float], y_values: List[float]) -> None:
    x = np.array(x_values)
    y = np.array(y_values)
    (slope, y_intersect) = compute_linear_regression_algorithm(x_values, y_values)


    fig, ax = plt.subplots()
    ax.scatter(x, y, color='blue', label='Data points')
    ax.plot(x, slope * x + y_intersect, color='red', label=f'Linear fit: y = {slope:.3f} * x + {y_intersect:.3f}')
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title('Linear Regression')
    ax.legend()
    plt.tight_layout()
    fig.savefig(PATH_LIN_REGGR_PLOT)
    plt.close(fig)



def compute_quadratic_regression_algorithm(x_values: List[float], y_values: List[float]) -> Tuple[float, float, float]:
    x = np.array(x_values)
    y = np.array(y_values)
    A = np.vstack([x**2, x, np.ones(len(x))]).T
    a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]
    
    return (a, b, c)



def get_plot_img_quadratic_regression(x_name: str, y_name: str, x_values: List[float], y_values: List[float]) -> None:
    x = np.array(x_values)
    y = np.array(y_values)
    (quad_a, quad_b, quad_c) = compute_quadratic_regression_algorithm(x_values, y_values)

    fig, ax = plt.subplots()
    ax.scatter(x, y, color='blue', label='Data points')
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = quad_a * x_fit**2 + quad_b * x_fit + quad_c
    ax.plot(x_fit, y_fit, color='red', label=f'Quadratic fit: y = {quad_a:.3f} * x^2 + {quad_b:.3f} * x + {quad_c:.3f}')
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title('Quadratic Regression')
    ax.legend()
    plt.tight_layout()
    fig.savefig(PATH_QUAD_REGGR_PLOT)
    plt.close(fig)



def compute_cubic_regression_algorithm(x_values: List[float], y_values: List[float]) -> Tuple[float, float, float, float]:
    x = np.array(x_values)
    y = np.array(y_values)
    A = np.vstack([x**3, x**2, x, np.ones(len(x))]).T
    a, b, c, d = np.linalg.lstsq(A, y, rcond=None)[0]
    
    return (a, b, c, d)


def get_plot_img_cubic_regression(x_name: str, y_name: str, x_values: List[float], y_values: List[float]) -> None:
    x = np.array(x_values)
    y = np.array(y_values)
    (cubic_a, cubic_b, cubic_c, cubic_d) = compute_cubic_regression_algorithm(x_values, y_values)


    fig, ax = plt.subplots()
    ax.scatter(x, y, color='blue', label='Data points')
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = cubic_a * x_fit**3 + cubic_b * x_fit**2 + cubic_c * x_fit + cubic_d
    ax.plot(x_fit, y_fit, color='red', label=f'Cubic fit: y = {cubic_a:.3f} * x^3 + {cubic_b:.3f} * x^2 + {cubic_c:.3f} * x + {cubic_d:.3f}')
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title('Cubic Regression')
    ax.legend()
    plt.tight_layout()
    fig.savefig(PATH_CUBIC_REGGR_PLOT)
    plt.close(fig)


def get_plot_img_histogram(x_name: str, y_name: str, x_values: List[float], y_values: List[float]) -> None:
    fig, ax = plt.subplots()
    hist = ax.hist2d(x_values, y_values, bins=[30, 30], cmap='Blues')
    plt.colorbar(hist[3], ax, label='Counts')
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title('2D Histogram of y_values as a function of x_values')
    plt.tight_layout()
    fig.savefig(PATH_HISTOGRAM_PLOT)
    plt.close(fig)


def get_plot_img_chart_bars(x_name: str, y_name: str, x_values: List[float], y_values: List[float]) -> None:
    fig, ax = plt.subplots()
    ax.bar(x_values, y_values, color='blue', label='Bar Chart', width=0.1)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f'Bar Chart of {y_name} as a function of {x_name}')
    ax.legend()
    plt.tight_layout()
    fig.savefig(PATH_CHART_BARS_PLOT)
    plt.close(fig)

def plot_all(x_name: str, y_name: str, x_values: List[float], y_values: List[float]) -> None:
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))

    # Bar chart
    axs[0, 0].bar(x_values, y_values, color='blue', width=0.1)
    axs[0, 0].set_title('Bar Chart')
    axs[0, 0].set_xlabel(x_name)
    axs[0, 0].set_ylabel(y_name)

    # Histogram
    hist = axs[0, 1].hist2d(x_values, y_values, bins=[30, 30], cmap='Blues')
    fig.colorbar(hist[3], ax=axs[0, 1], label='Counts')
    axs[0, 1].set_title('2D Histogram')
    axs[0, 1].set_xlabel(x_name)
    axs[0, 1].set_ylabel(y_name)

    # Scatter plot
    axs[1, 0].scatter(x_values, y_values, color='blue')
    axs[1, 0].set_title('Scatter Plot')
    axs[1, 0].set_xlabel(x_name)
    axs[1, 0].set_ylabel(y_name)

    # Linear regression
    (slope, y_intersect) = compute_linear_regression_algorithm(x_values, y_values)
    axs[1, 1].scatter(x_values, y_values, color='blue')
    x_fit = np.linspace(min(x_values), max(x_values), 100)
    y_fit = slope * x_fit + y_intersect
    axs[1, 1].plot(x_fit, y_fit, color='green', label=f'Linear fit: y = {slope:.3f} * x + {y_intersect:.3f}')
    axs[1, 1].set_title('Linear Regression')
    axs[1, 1].set_xlabel(x_name)
    axs[1, 1].set_ylabel(y_name)
    axs[1, 1].legend()

    # Quadratic regression
    (quad_a, quad_b, quad_c) = compute_quadratic_regression_algorithm(x_values, y_values)
    axs[2, 0].scatter(x_values, y_values, color='blue')
    y_fit = quad_a * x_fit**2 + quad_b * x_fit + quad_c
    axs[2, 0].plot(x_fit, y_fit, color='green', label=f'Quadratic fit: y = {quad_a:.3f} * x^2 + {quad_b:.3f} * x + {quad_c:.3f}')
    axs[2, 0].set_title('Quadratic Regression')
    axs[2, 0].set_xlabel(x_name)
    axs[2, 0].set_ylabel(y_name)
    axs[2, 0].legend()

    # Cubic regression
    (cubic_a, cubic_b, cubic_c, cubic_d) = compute_cubic_regression_algorithm(x_values, y_values)
    axs[2, 1].scatter(x_values, y_values, color='blue')
    y_fit = cubic_a * x_fit**3 + cubic_b * x_fit**2 + cubic_c * x_fit + cubic_d
    axs[2, 1].plot(x_fit, y_fit, color='green', label=f'Cubic fit: y = {cubic_a:.3f} * x^3 + {cubic_b:.3f} * x^2 + {cubic_c:.3f} * x + {cubic_d:.3f}')
    axs[2, 1].set_title('Cubic Regression')
    axs[2, 1].set_xlabel(x_name)
    axs[2, 1].set_ylabel(y_name)
    axs[2, 1].legend()



    plt.tight_layout()
    plt.savefig(PATH_ALL_PLOTS)
    plt.show()
    plt.close(fig)




def main():
    if not os.path.exists(PATH_INPUT_CSV):
        print(f'ERR: no input file `{PATH_INPUT_CSV}`', file=sys.stderr)
        sys.exit(1)

    for file in [PATH_OUTPUT_CSV, PATH_SCATTER_PLOT, PATH_LIN_REGGR_PLOT, PATH_QUAD_REGGR_PLOT, PATH_CUBIC_REGGR_PLOT, PATH_HISTOGRAM_PLOT]:
        if os.path.exists(file) == True:
            os.remove(file)

    try:
        inputs = read_csv_file(PATH_INPUT_CSV)
    except InvalidCSVFormatException as e:
        print(f"Error reading the CSV file: {e}")
        sys.exit(1)

    x_name: str = inputs[0]
    y_name: str = inputs[1]
    x_values: List[float] = inputs[2]
    y_values: List[float] = inputs[3]


    N: int = len(x_values)

    x_avg: float = average(x_values)
    x_avg_vec: List[float] = [x_avg for _ in x_values]

    x_dev: List[float] = [approx(val - x_avg) for val in x_values] 

    # # Standard Average Deviation
    x_evedev: float = approx(math.sqrt(sum([pow(di, 2) / (N * (N - 1)) for di in x_dev])))
    x_evedev_vec: List[float] = [x_evedev for _ in x_values]

    trust_interval: str = f"[ {x_avg}  ± {x_evedev} ]"
    trust_interval_vec: List[str] = [trust_interval for _ in x_values]


    # d : Y = m * X + c
    # m = slope
    # c = y-intercept
    (slope, y_intersect) = compute_linear_regression_algorithm(x_values, y_values)


    # d : Y = a * X^2 + b * X + c
    (quad_a, quad_b, quad_c) = compute_quadratic_regression_algorithm(x_values, y_values)


    # d : Y = a * X^3 + b * X^2 + c * X + d
    (cubic_a, cubic_b, cubic_c, cubic_d) = compute_cubic_regression_algorithm(x_values, y_values)


    lin_reg_eq: str = f'Equation of the lin. reggr.: Y = {slope:.3f} * X + {y_intersect:.3f}'
    lin_reg_eq_vec: List[str] = [lin_reg_eq for _ in x_values]


    quad_reg_eq: str = f'Equation of the quad. reggr.: Y = {quad_a:.3f} * X^2 + {quad_b:.3f} * X + {quad_c:.3f}'
    quad_reg_eq_vec: List[str] = [quad_reg_eq for _ in x_values]

    cubic_reg_eq: str = f'Equation of the cubic. reggr.: Y = {cubic_a:.3f} * X^3 + {cubic_b:.3f} * X^2 + {cubic_c:.3f} * X + {cubic_d:.3f}'
    cubic_reg_eq_vec: List[str] = [cubic_reg_eq for _ in x_values]


    avg_name = "Input average"
    dev_name = "Input deviation from average"
    avedev_name = "Input standard average deviation"
    trust_interval_name = "Input trust interval"
    lin_reg_eq_name = "Linear reggression equation"
    quad_reg_eq_name = "Quadratic reggression equation"
    cubic_reg_eq_name = "Cubic reggression equation"



    write_csv_file(PATH_OUTPUT_CSV, x_name, y_name, avg_name, dev_name,
                   avedev_name, trust_interval_name,
                   lin_reg_eq_name, quad_reg_eq_name, cubic_reg_eq_name,
                   x_values, y_values, x_avg_vec, x_dev, x_evedev_vec, trust_interval_vec,
                   lin_reg_eq_vec, quad_reg_eq_vec, cubic_reg_eq_vec)

    get_plot_img_histogram(x_name, y_name, x_values, y_values)
    get_plot_img_chart_bars(x_name, y_name, x_values, y_values)

    get_plot_img_scatter(x_name, y_name, x_values, y_values)
    get_plot_img_linnear_reggression(x_name, y_name, x_values, y_values)
    get_plot_img_quadratic_regression(x_name, y_name, x_values, y_values)
    get_plot_img_cubic_regression(x_name, y_name, x_values, y_values)



    plot_all(x_name, y_name, x_values, y_values)

if __name__ == '__main__':
    main()

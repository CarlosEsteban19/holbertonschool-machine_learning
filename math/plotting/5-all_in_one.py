#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
"""plotting task 5"""


def all_in_one():
    """5 plot madness"""
    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    fig = plt.figure()

    plt1 = fig.add_subplot(321)
    plt1.plot(y0, color='red')
    plt1.set_xlim(0, 10)

    plt2 = fig.add_subplot(322)
    plt2.scatter(x1, y1, c='m')
    plt2.set_title("Men's Height vs Weight", fontsize='x-small')
    plt2.set_xlabel("Height (in)", fontsize='x-small')
    plt2.set_ylabel("Weight (lbs)", fontsize='x-small')

    plt3 = fig.add_subplot(323)
    plt3.plot(x2, y2)
    plt3.set_title("Exponential Decay of C-14", fontsize='x-small')
    plt3.set_xlabel("Time (years)", fontsize='x-small')
    plt3.set_ylabel("Fraction Remaining", fontsize='x-small')
    plt3.set_yscale("log")
    plt3.set_xlim(0, 28650)

    plt4 = fig.add_subplot(324)
    plt4.plot(x3, y31, 'r--', label="C-14")
    plt4.plot(x3, y32, 'g-', label="Ra-226")
    plt4.set_xlabel("Time (years)", fontsize='x-small')
    plt4.set_ylabel("Fraction Remaining", fontsize='x-small')
    plt4.set_title("Exponential Decay of Radioactive Elements", fontsize='x-small')
    plt4.set_xlim(0, 20000)
    plt4.set_ylim(0, 1)
    plt4.legend()

    plt5 = fig.add_subplot(313)
    plt5.hist(student_grades, bins=np.arange(0, 110, 10), edgecolor='black')
    plt5.set_xlabel("Grades", fontsize='x-small')
    plt5.set_ylabel("Number of Students", fontsize='x-small')
    plt5.set_title("Project A", fontsize='x-small')
    plt5.set_xticks(ticks=np.arange(0, 110, 10))
    plt5.set_ylim(0, 30)
    plt5.set_xlim(0, 100)

    fig.suptitle("All in One")
    plt.tight_layout()
    plt.show()

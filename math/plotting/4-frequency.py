#!/usr/bin/env python3
"""plotting task 4"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """histogram plot"""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    plt.hist(student_grades, bins=np.arange(0, 110, 10), edgecolor='black')
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.xticks(ticks=np.arange(0, 110, 10))
    plt.ylim(0, 30)
    plt.xlim(0, 100)
    plt.show()

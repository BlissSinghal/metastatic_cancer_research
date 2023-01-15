from tabulate import tabulate

def make_tables(array):
    print(tabulate(array, headers="firstrow", tablefmt = "fancy_grid"))

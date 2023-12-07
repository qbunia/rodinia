import matplotlib
matplotlib.use('Agg')  # Use Agg backend (non-interactive mode)
import matplotlib.pyplot as plt

# List of file names
file_names = ["AXPY_ompCPU_P0_result.txt", "AXPY_ompCPU_P1_result.txt", "AXPY_ompCPU_P2_result.txt"]

# Output file name
output_file = "output_runtimes.csv"

# Function to extract runtime value from a file
def extract_runtime(file_name):
    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()
            runtime_value = float(lines[5].split()[1])  # Assuming the runtime value is on line 6 (adjust if necessary)
            return runtime_value
    except FileNotFoundError:
        print("File not found: {}".format(file_name))
        return None

# Extract runtime values
runtimes = [extract_runtime(file_name) for file_name in file_names]

# ...

# Plotting
if all(runtime is not None for runtime in runtimes):
    labels = ['P{}'.format(i) for i in range(len(runtimes))]
    plt.bar(labels, runtimes, color='blue')
    plt.xlabel('Configurations')
    plt.ylabel('Runtime (ms)')
    plt.title('AXPY Operation Runtimes')
    plt.savefig(output_file + ".png")
    plt.show()
    print(f"Bar chart created and saved to {output_file}.png")
else:
    print("Error: Unable to create the bar chart due to missing runtime values.")



#!/bin/bash

# Input CSV file
csv_file="execution_times.csv"

# Output image file
output_image="bar_figure.png"

# Plotting script for gnuplot
gnuplot_script=$(cat <<EOF
set datafile separator ","
set terminal pngcairo enhanced font 'arial,10' size 800,600
set output "${output_image}"
set title "Execution Times"
set ylabel "Average Execution Time"
set xlabel "Executable"
set style fill solid
set boxwidth 0.5
set xtics rotate by -45
plot "${csv_file}" using 2:xtic(1) with boxes title "AvgExecutionTime"
EOF
)

# Create and execute the gnuplot script
echo "${gnuplot_script}" | gnuplot

echo "Bar figure generated: ${output_image}"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define MAX_ITER 100
#define EPSILON 1e-6
#define MASTER 0
#define FILE_NAME "result.txt"

void print_matrix(double** matrix, int rows, int cols, FILE* file) {
   
    int i,j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            fprintf(file, "%8.3f ", matrix[i][j]);
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n---------------------------------------------------------------\n");
    
}

double* print_solution_and_get(double** tableau, double* basics, int rows, int cols, FILE* file) {
   
    fprintf(file,"Solution:\n");

    int basics_length = rows - 1;
    double* solution = (double*)calloc(basics_length, sizeof(double));
    if (solution == NULL) {
        fprintf(stderr, "Memory allocation failed for solution.\n");
        return NULL;
    }

    for (int i = 0; i < basics_length; i++) {
        int xCol = (int)basics[i];

        if (xCol != -1) {
            solution[xCol] = tableau[i][cols - 1];
        }
    }

    for (int i = 0; i < basics_length; i++) {
        fprintf(file,"x%d = %.2f\n", i + 1, solution[i]);
    }

    return solution;
}

// Allocate memory for a 2D array (matrix)
double** allocate_matrix(int rows, int cols) {
    double** matrix = malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = malloc(cols * sizeof(double));
    }
    return matrix;
}

// Free memory of a 2D array
void free_matrix(double** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int find_pivot_col(double** tableau, int rows, int cols) {
    int pivot_col = -1;
    double most_negative = 0;
    for (int col = 0; col < cols - 1; col++) {
        if (tableau[rows - 1][col] < most_negative) {
            most_negative = tableau[rows - 1][col];
            pivot_col = col;
        }
    }
    return pivot_col;
}


int is_integer(double value) {
    return fabs(value - round(value)) < 1e-6;
}

int exist_real_value(double** tableau, int rows, int cols) {
    int exist_real_value = 0;
    for (int i = 0; i < rows - 1; i++) {
        if (!is_integer(tableau[i][cols - 1])) {
            exist_real_value = 1;
            break;
        }
    }
    return exist_real_value;
}


int find_pivot_row(double** tableau, int rows, int cols, int pivot_col) {
    int pivot_row = -1;
    double min_ratio = INFINITY;
    for (int row = 0; row < rows - 1; row++) {
        if (tableau[row][pivot_col] > 0) {
            double ratio = tableau[row][cols - 1] / tableau[row][pivot_col];
            if (ratio <= min_ratio && ratio > 0) {
                min_ratio = ratio;
                pivot_row = row;
            }
        }
    }
    return pivot_row;
}

int find_gomory_row_to_cut(double** tableau, int rows, int cols) {
    int row_to_cut = -1;
    double max_fractional_part = 0.0;

    for (int i = 0; i < rows - 1; i++) {
        double value = tableau[i][cols - 1];
        double fractional_part = value - floor(value);

        if (fractional_part > max_fractional_part) {
            max_fractional_part = fractional_part;
            row_to_cut = i;
        }
    }

    return row_to_cut;
}

int find_gomory_column_to_add(double** tableau, int rows, int cols) {
    int gomory_column_to_add = -1;
    double min_value = INFINITY;

    for (int j = 0; j < cols - 2; j++) {
        double goromy_row_value = tableau[rows - 2][j];
        if (goromy_row_value != 0) {
            double last_row_gomory_row_rapport = tableau[rows - 1][j] / goromy_row_value;
            if (last_row_gomory_row_rapport <= min_value) {
                gomory_column_to_add = j;
                min_value = last_row_gomory_row_rapport;
            }
        }
    }
    return gomory_column_to_add;
}

double* extend_basics(double* basics, int old_cols, double init_value) {

    double* new_basics = (double*)malloc((old_cols + 1) * sizeof(double));
    if (!new_basics) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < old_cols; i++) {
        new_basics[i] = basics[i];
    }

    new_basics[old_cols] = init_value;

    return new_basics;
}

void pivot(double** tableau, int rows, int cols, int pivot_row, int pivot_col) {
    double pivot_value = tableau[pivot_row][pivot_col];
    int j, i;
#pragma omp parallel for
    for (j = 0; j < cols; j++) {
        double value = tableau[pivot_row][j] / pivot_value;
        if (value == 0) {
            tableau[pivot_row][j] = 0.0;
        }
        else {
            tableau[pivot_row][j] = value;
        }
         
    }

#pragma omp parallel for
	for (i = 0; i < rows; i++) {
		if (i != pivot_row) {
			double factor = tableau[i][pivot_col];
			for (int j = 0; j < cols; j++) {

				tableau[i][j] -= factor * tableau[pivot_row][j];
				
			}
		}
	}
}
double** add_gomory_cut(double** tableau, int old_rows, int old_cols, int row_to_cut, int is_first_time) {

    int new_rows = old_rows + 1;
    int new_cols = old_cols + 1;
    int gomory_row = old_rows - 1;
    int gomory_col = old_cols - 1;

    double** result = allocate_matrix(new_rows, new_cols);
    int i, j;
#pragma omp parallel for default(none) private(i, j)
    for (i = 0; i < gomory_row; i++) {
        for (j = 0; j < old_cols; j++) {
            if (j == gomory_col) {
                result[i][j] = 0.0;
            }
            else {
                result[i][j] = tableau[i][j];
            }

        }
    }
#pragma omp parallel for default(none) private(j)
    for (j = 0; j < old_cols; j++) {
        double value = tableau[row_to_cut][j];
        double fractionalPart = value - floor(value);
        if (j == gomory_col) {
            result[gomory_row][j] = 1;
        }
        else {
            result[gomory_row][j] = fractionalPart != 0 ? -1 * fractionalPart : fractionalPart;
        }
    }

#pragma omp parallel for default(none) private(j)
    for (j = 0; j < old_cols; j++) {
        double value = tableau[gomory_row][j];
        if (j == gomory_col) {
            result[old_rows][j] = 0.0;
        }
        else {
            if (is_first_time) {

                result[old_rows][j] = value != 0 ? -1 * value : value;
            }
            else {
                result[old_rows][j] = value;
            }
        }
    }

#pragma omp parallel for default(none) private(i)
    for (i = 0; i < new_rows; i++) {
        if (i == gomory_row) {
            double value = tableau[row_to_cut][gomory_col];
            double fractionalPart = value - floor(value);
            result[i][old_cols] = fractionalPart != 0 ? -1 * fractionalPart : fractionalPart;
        }
        else {
            if (i < old_rows) {
                result[i][old_cols] = tableau[i][gomory_col];
            }
            else {
                result[i][old_cols] = is_first_time ? -1 * tableau[old_rows - 1][gomory_col] : tableau[old_rows - 1][gomory_col];
            }
        }

    }

    return result;

}
void apply_gomory_cuts(double** tableau, int rows, int cols, double* basics, FILE* file) {
    int keep_apply_gomory_cut = exist_real_value(tableau, rows, cols);
    int is_first_time = 1;
   
    if (keep_apply_gomory_cut) {
        fprintf(file,"\nApply Gomory\n");
    }

    while (keep_apply_gomory_cut) {
        int row_to_cut = find_gomory_row_to_cut(tableau, rows, cols);
        if (row_to_cut == -1) {
            fprintf(file,"All solutions are integers.\n");
            break;
        }
        if (row_to_cut == MAX_ITER) {
            fprintf(file,"\nNot found solution after %d iterations ", MAX_ITER);
            break;
        }
        fprintf(file,"Adding Gomory cut for row %d\n", row_to_cut);
        tableau = add_gomory_cut(tableau, rows, cols, row_to_cut, is_first_time);
        rows++;
        cols++;
        print_matrix(tableau, rows, cols,file);

        int gomory_row = rows - 2;
        int gomory_col = find_gomory_column_to_add(tableau, rows, cols);

        basics = extend_basics(basics, rows - 2, gomory_col);

        pivot(tableau, rows, cols, gomory_row, gomory_col);

        double* solution = print_solution_and_get(tableau, basics, rows, cols,file);
        fprintf(file,"-------------------------------------------------------------------\n");
        keep_apply_gomory_cut = exist_real_value(tableau, rows, cols);
        is_first_time = 0;
    }
    print_matrix(tableau, rows, cols,file);
    fprintf(file, "-------------------------------------------------------------------\n");
}

void init_basic(double* basics, int length) {
    for (int i = 0; i < length; i++) {
        basics[i] = -1;
    }
}


// Perform Simplex method on the tableau
int simplex_method(double** tableau, int rows, int cols, FILE* file) {
  
    double* basics = (double*)malloc((rows - 1) * sizeof(double));

    init_basic(basics, (rows - 1));
   
    int basic_was_initialized = 0;
    while (1) {
        // Check for optimality
        int pivot_col = find_pivot_col(tableau, rows, cols);
        if (pivot_col == -1) {
			// Optimal solution found
			double* solution = NULL;
			fprintf(file,"Optimal solution found\n");
			print_matrix(tableau, rows, cols, file);
			solution = print_solution_and_get(tableau, basics, rows, cols,file);
			apply_gomory_cuts(tableau, rows, cols, basics,file);

			return 1;
        }

        // Find pivot row
        int pivot_row = find_pivot_row(tableau, rows, cols, pivot_col);
        double min_ratio = INFINITY;

        if (pivot_row == -1) {
            // Unbounded solution
            return 0;
        }

        // Perform pivot operation
        pivot(tableau, rows, cols, pivot_row, pivot_col);

        basics[pivot_row] = pivot_col;
    }
}

void clear_file() {
    FILE* file = fopen(FILE_NAME, "w");  
    if (file != NULL) {
        fclose(file);
    }
    else {
        fprintf(stderr, "Failed to clear file.\n");
    }
}


int main(int argc, char* argv[]) {
  
    int rows = 3;
    int cols = 5;
    double start_time, end_time;

    FILE* file = fopen(FILE_NAME, "a");
    if (file == NULL) {
        fprintf(stderr, "Error opening file.\n");
        return;
    }
    omp_set_num_threads(2);
    start_time = omp_get_wtime();
    clear_file();
	double tableauData[3][5] = {
		{ -15, 31, 1, 0, 6 },
		{ 71, 17, 0, 1, 35 },
        {-7 , -9, 0, 0, 0 }
	};

    double** tableau = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        tableau[i] = (double*)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            tableau[i][j] = tableauData[i][j];
        }
    }
    
    fprintf(file,"Initial \n");
    print_matrix(tableau, rows, cols, file);
   
    int optimal = simplex_method(tableau, rows, cols,file);

    free_matrix(tableau, rows);
    end_time = omp_get_wtime();
    fprintf(file,"Time: %.6f seconds\n", end_time - start_time);
    fclose(file);

    return 0;
}

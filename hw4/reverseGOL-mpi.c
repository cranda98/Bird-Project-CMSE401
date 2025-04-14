#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#define max_nodes 264
#define str_length 50

int fitness(char * plate1, char * plate2, int n){
    int errors = 0;
    for(int i = 1; i <= n; i++){
        for(int j = 1; j <= n; j++){
            int index = i * (n + 2) + j;
            errors += plate1[index] != plate2[index];
        }
    }
    return errors;
}

int live(int index, char * plate[2], int which, int n){
    return (plate[which][index - n - 3] 
        + plate[which][index - n - 2]
        + plate[which][index - n - 1]
        + plate[which][index - 1]
        + plate[which][index + 1]
        + plate[which][index + n + 1]
        + plate[which][index + n + 2]
        + plate[which][index + n + 3]);
}

int iteration(char * plate[2], int which, int n){
    for(int i = 1; i <= n; i++){
        for(int j = 1; j <= n; j++){
            int index = i * (n + 2) + j;
            int num = live(index, plate, which, n);
            if(plate[which][index]){
                plate[!which][index] =  (num == 2 || num == 3) ? 1 : 0;
            } else {
                plate[!which][index] = (num == 3);
            }
        }
    }
    which = !which;
    return which;
}

void print_plate(char * plate, int n){
    if (n < 60) {
        for(int i = 1; i <= n; i++){
            for(int j = 1; j <= n; j++){
                printf("%d", (int) plate[i * (n + 2) + j]);
            }
            printf("\n");
        }
    } else {
        printf("Plate too large to print to screen\n");
    }
}

void makerandom(char * plate, int n) {
    for(int i = 1; i <= n; i++)
        for(int j = 0; j < n; j++)
            plate[i * (n+2) + j + 1] = (rand() % 100) > 10;
}

void mutate(char * plate, char * best_plate, int n, int rate) {
    for(int i = 1; i <= n; i++)
        for(int j = 0; j < n; j++) {
            int index = i * (n+2) + j + 1;
            if ((rand() % 100) < rate)
                plate[index] = (rand() % 2) > 0;
            else
                plate[index] = best_plate[index];
        }
}

void cross(char * plate1, char * plate2, int n) {
    int start = 0;
    int end = (n+2)*(n+2);
    int crosspoint = rand() % end;
    if (crosspoint < end/2) {
        start = 0;
        end = crosspoint;
    } else {
        start = crosspoint;
    }
    for(int i = start; i <= end; i++)	
        plate1[i] = plate2[i];
}

char * readplate(char * filename, int *n) {
    char * plate;
    int M, N;
    FILE *fp;
    printf("Reading %s\n", filename);
    if ((fp = fopen(filename, "r")) == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    if(fscanf(fp, "%d %d", &N, &M) == 2){
        printf("Reading in %dx%d array\n", N, N);
        plate = (char *) calloc((N+2)*(N*2), sizeof(char));
        char line[N];
        for(int i = 1; i <= N; i++){
            fscanf(fp, "%s", line);
            for(int j = 0; j < N; j++){
                int index = i * (N + 2) + j + 1;
                plate[index] = (line[j] == '1');
            }
        }
        *n = N;
    } else {
        printf("File not correct format\n");
        exit(EXIT_FAILURE);
    }
    fclose(fp);
    return plate;
}

int main(int argc, char *argv[]) {
    int which = 0;
    int n;
    int npop = 10000;
    int ngen = 1000;
    int M = 0;
    int rand_seed;
    time_t t;

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc > 2)
        rand_seed = (atoi(argv[2]) + rank + 1) * 7;
    else
        rand_seed = (unsigned int) time(&t);
    printf("Rank %d using seed %d\n", rank, rand_seed);
    srand(rand_seed);

    char * buffer_plate;
    char * target_plate;

    target_plate = readplate(argv[1], &n);
    buffer_plate = (char *) calloc((n+2)*(n+2), sizeof(char));

    char * population[npop];
    int pop_fitness[npop];
    int best = 0;
    int sbest = 1;

    for(int i = 0; i < npop; i++) {
        pop_fitness[i] = n * n;
        population[i] = (char *) calloc((n+2)*(n+2), sizeof(char));
        if (i < npop / 2)
            mutate(population[i], target_plate, n, 10);
        else
            makerandom(population[i], n);
    }

    for(int g = 0; g < ngen; g++) {
        for(int i = 0; i < npop; i++) {
            char *plate[2];
            plate[0] = population[i];
            plate[1] = buffer_plate;
            iteration(plate, 0, n);
            pop_fitness[i] = fitness(buffer_plate, target_plate, n);

            if (pop_fitness[i] < pop_fitness[best]) {
                sbest = best;
                best = i;
                if (pop_fitness[best] == 0) {
                    char * temp = target_plate;
                    target_plate = population[best];
                    population[best] = temp;
                    pop_fitness[best] = n * n;
                    M++;
                }
            } else {
                if (sbest == best)
                    sbest = i;
            }
        }

        // NEW: round-robin sharing of best result after each generation
        int neighbor_send = (rank + 1) % size;
        int neighbor_recv = (rank - 1 + size) % size;

        int recv_fitness;
        char* recv_plate = (char*) malloc((n+2)*(n+2) * sizeof(char));

        if (rank == 0) {
            MPI_Recv(&recv_fitness, 1, MPI_INT, neighbor_recv, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(recv_plate, (n+2)*(n+2), MPI_CHAR, neighbor_recv, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&pop_fitness[best], 1, MPI_INT, neighbor_send, 0, MPI_COMM_WORLD);
            MPI_Send(population[best], (n+2)*(n+2), MPI_CHAR, neighbor_send, 1, MPI_COMM_WORLD);
        } else {
            MPI_Send(&pop_fitness[best], 1, MPI_INT, neighbor_send, 0, MPI_COMM_WORLD);
            MPI_Send(population[best], (n+2)*(n+2), MPI_CHAR, neighbor_send, 1, MPI_COMM_WORLD);
            MPI_Recv(&recv_fitness, 1, MPI_INT, neighbor_recv, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(recv_plate, (n+2)*(n+2), MPI_CHAR, neighbor_recv, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (recv_fitness < pop_fitness[best]) {
            sbest = best;
            pop_fitness[best] = recv_fitness;
            memcpy(population[best], recv_plate, (n+2)*(n+2));
        }

        free(recv_plate);

        // Continue with normal evolution
        int rate = (int) ((double) pop_fitness[best] / (n * n) * 100);

        for(int i = 0; i < npop; i++) {
            if (i == sbest) {
                cross(population[i], population[best], n);
                sbest = 1;
            } else if (i != best) {
                if (i < npop / 3)
                    mutate(population[i], population[best], n, rate);
                else if (i < (npop * 2) / 3)
                    cross(population[i], population[best], n);
                else
                    makerandom(population[i], n);
            }
        }
    }

    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            int recv_fitness;
            char* recv_plate = (char*) malloc((n+2)*(n+2) * sizeof(char));

            MPI_Recv(&recv_fitness, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(recv_plate, (n+2)*(n+2), MPI_CHAR, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (recv_fitness < pop_fitness[best]) {
                pop_fitness[best] = recv_fitness;
                memcpy(population[best], recv_plate, (n+2)*(n+2));
            }
            free(recv_plate);
        }

        printf("\n=== Final Consolidated Result from All MPI Ranks ===\n");
        print_plate(population[best], n);
        printf("\nResult Fitness=%d over %d iterations:\n", pop_fitness[best], ngen);
    } else {
        MPI_Send(&pop_fitness[best], 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(population[best], (n+2)*(n+2), MPI_CHAR, 0, 1, MPI_COMM_WORLD);
    }

    free(target_plate);
    free(buffer_plate);
    for(int i = 0; i < npop; i++)
        free(population[i]);

    MPI_Finalize();
    return 0;
}

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define NUM_TEAMS 256
#define NUM_THREADS 1024

FILE *fp;

// Structure to hold a node information
struct Node {
  int starting;
  int no_of_edges;
};

void BFSGraph(int argc, char **argv);

void Usage(int argc, char **argv) {

  fprintf(stderr, "Usage: %s <input_file> [<num_teams>] [<num_threads>]\n",
          argv[0]);
}

double get_time() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return (double)t.tv_sec + t.tv_usec * 1e-6;
}

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) { BFSGraph(argc, argv); }

////////////////////////////////////////////////////////////////////////////////
// Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph(int argc, char **argv) {
  int no_of_nodes = 0;
  int edge_list_size = 0;
  char *input_f;
  int omp_num_teams = NUM_TEAMS;
  int omp_num_threads = NUM_THREADS;

  if (argc < 2) {
    Usage(argc, argv);
    exit(0);
  }

  input_f = argv[1];
  if (argc > 2) {
    omp_num_teams = atoi(argv[2]);
  }
  if (argc > 3) {
    omp_num_threads = atoi(argv[3]);
  }

  printf("Reading File\n");
  // Read in Graph from a file
  fp = fopen(input_f, "r");
  if (!fp) {
    printf("Error Reading graph file\n");
    return;
  }

  int source = 0;

  fscanf(fp, "%d", &no_of_nodes);

  // allocate host memory
  struct Node *h_graph_nodes =
      (struct Node *)malloc(sizeof(struct Node) * no_of_nodes);
  bool *h_graph_mask = (bool *)malloc(sizeof(bool) * no_of_nodes);
  bool *h_updating_graph_mask = (bool *)malloc(sizeof(bool) * no_of_nodes);
  bool *h_graph_visited = (bool *)malloc(sizeof(bool) * no_of_nodes);

  int start, edgeno;
  // initalize the memory
  for (unsigned int i = 0; i < no_of_nodes; i++) {
    fscanf(fp, "%d %d", &start, &edgeno);
    h_graph_nodes[i].starting = start;
    h_graph_nodes[i].no_of_edges = edgeno;
    h_graph_mask[i] = false;
    h_updating_graph_mask[i] = false;
    h_graph_visited[i] = false;
  }

  // read the source node from the file
  fscanf(fp, "%d", &source);
  // source=0; //tesing code line

  // set the source node as true in the mask
  h_graph_mask[source] = true;
  h_graph_visited[source] = true;

  fscanf(fp, "%d", &edge_list_size);

  int id, cost;
  int *h_graph_edges = (int *)malloc(sizeof(int) * edge_list_size);
  for (int i = 0; i < edge_list_size; i++) {
    fscanf(fp, "%d", &id);
    fscanf(fp, "%d", &cost);
    h_graph_edges[i] = id;
  }

  if (fp)
    fclose(fp);

  // allocate mem for the result on host side
  int *h_cost = (int *)malloc(sizeof(int) * no_of_nodes);
  for (int i = 0; i < no_of_nodes; i++)
    h_cost[i] = -1;
  h_cost[source] = 0;

  printf("Start traversing the tree\n");

  int k = 0;
  double start_time = get_time();
#pragma omp target data map(                                                   \
    to                                                                         \
    : no_of_nodes, h_graph_mask [0:no_of_nodes],                               \
      h_graph_nodes [0:no_of_nodes], h_graph_edges [0:edge_list_size],         \
      h_graph_visited [0:no_of_nodes], h_updating_graph_mask [0:no_of_nodes])  \
    map(tofrom                                                                 \
        : h_cost [0:no_of_nodes])
  {
    bool stop;
    do {
      // if no thread changes this value then the loop stops
      stop = false;

#pragma omp target teams distribute parallel for num_teams(omp_num_teams)      \
    num_threads(omp_num_threads)
      for (int tid = 0; tid < no_of_nodes; tid++) {
        if (h_graph_mask[tid] == true) {
          h_graph_mask[tid] = false;
          for (int i = h_graph_nodes[tid].starting;
               i <
               (h_graph_nodes[tid].no_of_edges + h_graph_nodes[tid].starting);
               i++) {
            int id = h_graph_edges[i];
            if (!h_graph_visited[id]) {
              h_cost[id] = h_cost[tid] + 1;
              h_updating_graph_mask[id] = true;
            }
          }
        }
      }

#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : stop)                   \
    num_teams(omp_num_teams) num_threads(omp_num_threads)
      for (int tid = 0; tid < no_of_nodes; tid++) {
        if (h_updating_graph_mask[tid] == true) {
          h_graph_mask[tid] = true;
          h_graph_visited[tid] = true;
          stop = true;
          h_updating_graph_mask[tid] = false;
        }
      }
      k++;
    } while (stop);
    double end_time = get_time();
    printf("Compute time: %lfs\n", (end_time - start_time));
  }
  // Store the result into a file
  FILE *fpo = fopen("result.log", "w");
  for (int i = 0; i < no_of_nodes; i++)
    fprintf(fpo, "%d) cost:%d\n", i, h_cost[i]);
  fclose(fpo);
  printf("Result stored in result.log\n");

  // cleanup memory
  free(h_graph_nodes);
  free(h_graph_edges);
  free(h_graph_mask);
  free(h_updating_graph_mask);
  free(h_graph_visited);
  free(h_cost);
}

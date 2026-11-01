/* Edges with terminals: terminal masks folded into edge masks, and pass weights. */
#include "terminals.h"

void fill_two_terminal_edge_mask(const GraphContext *graph, const uint8_t *edge_mask, const uint8_t *terminal_mask,
                                 uint8_t *two_terminal_edges) {
    const TerminalList *terminals = graph->terminals;
    Py_ssize_t edge_count = graph->node_ids.edges.edge_count;
    for (Py_ssize_t edge = 0; edge < edge_count; edge++) {
        Py_ssize_t first_terminal = terminals ? terminals->terminal_offsets[edge] : 2 * edge;
        int two_terminals = terminals ? terminals->terminal_offsets[edge + 1] - first_terminal == 2 : 1;
        two_terminal_edges[edge] = (uint8_t)((edge_mask && edge_mask[edge]) ||
                                             (two_terminals && (terminal_mask[first_terminal] || terminal_mask[first_terminal + 1])));
    }
}

int parse_pass_weights(const TerminalList *terminals, Py_ssize_t edge_count, const double *weights, int directed,
                       PassWeights *pass_weights) {
    pass_weights->shifts = NULL;
    pass_weights->negative_terminals = NULL;
    for (Py_ssize_t edge = 0; edge < edge_count; edge++) {
        int first = terminals->terminal_offsets[edge], end = terminals->terminal_offsets[edge + 1];
        int first_out = directed ? terminals->out_terminal_offsets[edge] : end;
        double lowest = HUGE_VAL, second_lowest = HUGE_VAL, lowest_out = HUGE_VAL;
        int lowest_terminal = -1;
        for (int terminal = first; terminal < end; terminal++) {
            double weight = weights[terminal];
            if (terminal >= first_out) { if (weight < lowest_out) lowest_out = weight; }
            else if (weight < lowest) { second_lowest = lowest; lowest = weight; lowest_terminal = terminal; }
            else if (weight < second_lowest) second_lowest = weight;
        }
        double cheapest_pass = directed ? lowest + lowest_out : lowest + second_lowest;
        if (cheapest_pass < 0.0) {
            release_pass_weights(pass_weights);
            char message[200];
            snprintf(message, sizeof message, "the terminal weights of edge %zd allow a pass costing %g; %s must sum to at least 0",
                     edge, cheapest_pass, directed ? "every in-terminal and out-terminal of an edge" : "every two terminals of an edge");
            PyErr_SetString(PyExc_ValueError, message);
            return -1;
        }
        if (lowest >= 0.0) continue;
        if (!pass_weights->shifts) {
            pass_weights->shifts = (double *)calloc((size_t)edge_count, sizeof(double));
            pass_weights->negative_terminals = (int *)malloc((size_t)edge_count * sizeof(int));
            if (!pass_weights->shifts || !pass_weights->negative_terminals) {
                release_pass_weights(pass_weights);
                PyErr_NoMemory();
                return -1;
            }
            memset(pass_weights->negative_terminals, -1, (size_t)edge_count * sizeof(int));
        }
        pass_weights->shifts[edge] = lowest;
        pass_weights->negative_terminals[edge] = lowest_terminal;
    }
    return 0;
}

void release_pass_weights(PassWeights *pass_weights) {
    free(pass_weights->shifts);
    free(pass_weights->negative_terminals);
    pass_weights->shifts = NULL;
    pass_weights->negative_terminals = NULL;
}

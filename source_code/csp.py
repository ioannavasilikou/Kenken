import itertools
import random
import re
import string
import math
from collections import defaultdict, Counter
from functools import reduce
from operator import eq, neg

from sortedcontainers import SortedSet

import search
from utils import argmin_random_tie, count, first, extend

class CSP(search.Problem):

    def __init__(self, variables, domains, neighbors, constraints):
        """Construct a CSP problem. If variables is empty, it becomes domains.keys()."""
        super().__init__(())
        variables = variables or list(domains.keys())
        self.variables = variables
        self.domains = domains
        self.neighbors = neighbors
        self.constraints = constraints
        self.curr_domains = None
        self.nassigns = 0

    def assign(self, var, val, assignment):
        """Add {var: val} to assignment; Discard the old value if any."""
        assignment[var] = val
        self.nassigns += 1

    def unassign(self, var, assignment):
        """Remove {var: val} from assignment.
        DO NOT call this if you are changing a variable to a new value;
        just call assign for that."""
        if var in assignment:
            del assignment[var]

    def nconflicts(self, var, val, assignment):
        """Return the number of conflicts var=val has with other variables."""

        # Subclasses may implement this more efficiently
        def conflict(var2):
            return var2 in assignment and not self.constraints(var, val, var2, assignment[var2])
        
        return count(conflict(v) for v in self.neighbors[var])

    def display(self, assignment):
        """Show a human-readable representation of the CSP."""
        # Subclasses can print in a prettier way, or display with a GUI
        print(assignment)

    # These methods are for the tree and graph-search interface:

    def actions(self, state):
        """Return a list of applicable actions: non conflicting
        assignments to an unassigned variable."""
        if len(state) == len(self.variables):
            return []
        else:
            assignment = dict(state)
            var = first([v for v in self.variables if v not in assignment])
            return [(var, val) for val in self.domains[var]
                    if self.nconflicts(var, val, assignment) == 0]

    def result(self, state, action):
        """Perform an action and return the new state."""
        (var, val) = action
        return state + ((var, val),)

    def goal_test(self, state):
        """The goal is to assign all variables, with all constraints satisfied."""
        assignment = dict(state)
        return (len(assignment) == len(self.variables)
                and all(self.nconflicts(variables, assignment[variables], assignment) == 0
                        for variables in self.variables))

    # These are for constraint propagation

    def support_pruning(self):
        """Make sure we can prune values from domains. (We want to pay
        for this only if we use it.)"""
        if self.curr_domains is None:
            self.curr_domains = {v: list(self.domains[v]) for v in self.variables}

    def suppose(self, var, value):
        """Start accumulating inferences from assuming var=value."""
        self.support_pruning()
        removals = [(var, a) for a in self.curr_domains[var] if a != value]
        self.curr_domains[var] = [value]
        return removals

    def prune(self, var, value, removals):
        """Rule out var=value."""
        self.curr_domains[var].remove(value)
        if removals is not None:
            removals.append((var, value))

    def choices(self, var):
        """Return all values for var that aren't currently ruled out."""
        return (self.curr_domains or self.domains)[var]

    def infer_assignment(self):
        """Return the partial assignment implied by the current inferences."""
        self.support_pruning()
        return {v: self.curr_domains[v][0]
                for v in self.variables if 1 == len(self.curr_domains[v])}

    def restore(self, removals):
        """Undo a supposition and all inferences from it."""
        for B, b in removals:
            self.curr_domains[B].append(b)

    # This is for min_conflicts search

    def conflicted_vars(self, current):
        """Return a list of variables in current assignment that are in conflict"""
        return [var for var in self.variables
                if self.nconflicts(var, current[var], current) > 0]

# ______________________________________________________________________________
# Constraint Propagation with AC3


def no_arc_heuristic(csp, queue):
    return queue


def dom_j_up(csp, queue):
    return SortedSet(queue, key=lambda t: neg(len(csp.curr_domains[t[1]])))


def AC3(csp, queue=None, removals=None, arc_heuristic=dom_j_up):
    """[Figure 6.3]"""
    if queue is None:
        queue = {(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]}
    csp.support_pruning()
    queue = arc_heuristic(csp, queue)
    checks = 0
    while queue:
        (Xi, Xj) = queue.pop()
        revised, checks = revise(csp, Xi, Xj, removals, checks)
        if revised:
            if not csp.curr_domains[Xi]:
                return False, checks  # CSP is inconsistent
            for Xk in csp.neighbors[Xi]:
                if Xk != Xj:
                    queue.add((Xk, Xi))
    return True, checks  # CSP is satisfiable


def revise(csp, Xi, Xj, removals, checks=0):
    """Return true if we remove a value."""
    revised = False
    for x in csp.curr_domains[Xi][:]:
        # If Xi=x conflicts with Xj=y for every possible y, eliminate Xi=x
        # if all(not csp.constraints(Xi, x, Xj, y) for y in csp.curr_domains[Xj]):
        conflict = True
        for y in csp.curr_domains[Xj]:
            if csp.constraints(Xi, x, Xj, y):
                conflict = False
            checks += 1
            if not conflict:
                break
        if conflict:
            csp.prune(Xi, x, removals)
            revised = True
    return revised, checks


# Constraint Propagation with AC3b: an improved version
# of AC3 with double-support domain-heuristic

def AC3b(csp, queue=None, removals=None, arc_heuristic=dom_j_up):
    if queue is None:
        queue = {(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]}
    csp.support_pruning()
    queue = arc_heuristic(csp, queue)
    checks = 0
    while queue:
        (Xi, Xj) = queue.pop()
        # Si_p values are all known to be supported by Xj
        # Sj_p values are all known to be supported by Xi
        # Dj - Sj_p = Sj_u values are unknown, as yet, to be supported by Xi
        Si_p, Sj_p, Sj_u, checks = partition(csp, Xi, Xj, checks)
        if not Si_p:
            return False, checks  # CSP is inconsistent
        revised = False
        for x in set(csp.curr_domains[Xi]) - Si_p:
            csp.prune(Xi, x, removals)
            revised = True
        if revised:
            for Xk in csp.neighbors[Xi]:
                if Xk != Xj:
                    queue.add((Xk, Xi))
        if (Xj, Xi) in queue:
            if isinstance(queue, set):
                # or queue -= {(Xj, Xi)} or queue.remove((Xj, Xi))
                queue.difference_update({(Xj, Xi)})
            else:
                queue.difference_update((Xj, Xi))
            # the elements in D_j which are supported by Xi are given by the union of Sj_p with the set of those
            # elements of Sj_u which further processing will show to be supported by some vi_p in Si_p
            for vj_p in Sj_u:
                for vi_p in Si_p:
                    conflict = True
                    if csp.constraints(Xj, vj_p, Xi, vi_p):
                        conflict = False
                        Sj_p.add(vj_p)
                    checks += 1
                    if not conflict:
                        break
            revised = False
            for x in set(csp.curr_domains[Xj]) - Sj_p:
                csp.prune(Xj, x, removals)
                revised = True
            if revised:
                for Xk in csp.neighbors[Xj]:
                    if Xk != Xi:
                        queue.add((Xk, Xj))
    return True, checks  # CSP is satisfiable


def partition(csp, Xi, Xj, checks=0):
    Si_p = set()
    Sj_p = set()
    Sj_u = set(csp.curr_domains[Xj])
    for vi_u in csp.curr_domains[Xi]:
        conflict = True
        # now, in order to establish support for a value vi_u in Di it seems better to try to find a support among
        # the values in Sj_u first, because for each vj_u in Sj_u the check (vi_u, vj_u) is a double-support check
        # and it is just as likely that any vj_u in Sj_u supports vi_u than it is that any vj_p in Sj_p does...
        for vj_u in Sj_u - Sj_p:
            # double-support check
            if csp.constraints(Xi, vi_u, Xj, vj_u):
                conflict = False
                Si_p.add(vi_u)
                Sj_p.add(vj_u)
            checks += 1
            if not conflict:
                break
        # ... and only if no support can be found among the elements in Sj_u, should the elements vj_p in Sj_p be used
        # for single-support checks (vi_u, vj_p)
        if conflict:
            for vj_p in Sj_p:
                # single-support check
                if csp.constraints(Xi, vi_u, Xj, vj_p):
                    conflict = False
                    Si_p.add(vi_u)
                checks += 1
                if not conflict:
                    break
    return Si_p, Sj_p, Sj_u - Sj_p, checks


# Constraint Propagation with AC4

def AC4(csp, queue=None, removals=None, arc_heuristic=dom_j_up):
    if queue is None:
        queue = {(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]}
    csp.support_pruning()
    queue = arc_heuristic(csp, queue)
    support_counter = Counter()
    variable_value_pairs_supported = defaultdict(set)
    unsupported_variable_value_pairs = []
    checks = 0
    # construction and initialization of support sets
    while queue:
        (Xi, Xj) = queue.pop()
        revised = False
        for x in csp.curr_domains[Xi][:]:
            for y in csp.curr_domains[Xj]:
                if csp.constraints(Xi, x, Xj, y):
                    support_counter[(Xi, x, Xj)] += 1
                    variable_value_pairs_supported[(Xj, y)].add((Xi, x))
                checks += 1
            if support_counter[(Xi, x, Xj)] == 0:
                csp.prune(Xi, x, removals)
                revised = True
                unsupported_variable_value_pairs.append((Xi, x))
        if revised:
            if not csp.curr_domains[Xi]:
                return False, checks  # CSP is inconsistent
    # propagation of removed values
    while unsupported_variable_value_pairs:
        Xj, y = unsupported_variable_value_pairs.pop()
        for Xi, x in variable_value_pairs_supported[(Xj, y)]:
            revised = False
            if x in csp.curr_domains[Xi][:]:
                support_counter[(Xi, x, Xj)] -= 1
                if support_counter[(Xi, x, Xj)] == 0:
                    csp.prune(Xi, x, removals)
                    revised = True
                    unsupported_variable_value_pairs.append((Xi, x))
            if revised:
                if not csp.curr_domains[Xi]:
                    return False, checks  # CSP is inconsistent
    return True, checks  # CSP is satisfiable

# ______________________________________________________________________________
# CSP Backtracking Search

# Variable ordering

def first_unassigned_variable(assignment, csp):
    """The default variable order."""
    return first([var for var in csp.variables if var not in assignment])


def mrv(assignment, csp):
    """Minimum-remaining-values heuristic."""
    return argmin_random_tie([v for v in csp.variables if v not in assignment],
                             key=lambda var: num_legal_values(csp, var, assignment))


def num_legal_values(csp, var, assignment):
    if csp.curr_domains:
        return len(csp.curr_domains[var])
    else:
        return count(csp.nconflicts(var, val, assignment) == 0 for val in csp.domains[var])

# Value ordering

def unordered_domain_values(var, assignment, csp):
    """The default value order."""
    return csp.choices(var)

def lcv(var, assignment, csp):
    """Least-constraining-values heuristic."""
    return sorted(csp.choices(var), key=lambda val: csp.nconflicts(var, val, assignment))

# Inference

def no_inference(csp, var, value, assignment, removals):
    return True

def forward_checking(csp, var, value, assignment, removals):
    """Prune neighbor values inconsistent with var=value."""
    csp.support_pruning()

    # λόγω της μοντελοποίησης των γειτόνων, πρόσθεσα το 0 για να ελέγχει τις μεταβλητές της ίδιας γραμμής και στήλης
    for B in csp.neighbors[var][0]: # <--
        if B not in assignment:
            for b in csp.curr_domains[B][:]:
                if not csp.constraints(var, value, B, b):
                    csp.prune(B, b, removals)
            if not csp.curr_domains[B]:
                return False
    return True

def mac(csp, var, value, assignment, removals, constraint_propagation=AC3b):
    """Maintain arc consistency."""
    # λόγω της μοντελοποίησης των γειτόνων, πρόσθεσα το 0 για να ελέγχει τις μεταβλητές της ίδιας γραμμής και στήλης
    return constraint_propagation(csp, {(X, var) for X in csp.neighbors[var][0]}, removals) # <--

# The search, proper

def backtracking_search(csp, select_unassigned_variable=first_unassigned_variable,
                        order_domain_values=unordered_domain_values, inference=no_inference):
    """[Figure 6.5]"""

    def backtrack(assignment):
        if len(assignment) == len(csp.variables):
            return assignment
        var = select_unassigned_variable(assignment, csp)
        for value in order_domain_values(var, assignment, csp):
            if 0 == csp.nconflicts(var, value, assignment):
                csp.assign(var, value, assignment)
                removals = csp.suppose(var, value)
                if inference(csp, var, value, assignment, removals):
                    result = backtrack(assignment)
                    if result is not None:
                        return result
                csp.restore(removals)
        csp.unassign(var, assignment)
        return None

    result = backtrack({})
    assert result is None or csp.goal_test(result)
    return result

# ______________________________________________________________________________
# Min-conflicts Hill Climbing search for CSPs

def min_conflicts(csp, max_steps=100000):
    """Solve a CSP by stochastic Hill Climbing on the number of conflicts."""
    # Generate a complete assignment for all variables (probably with conflicts)
    csp.current = current = {}
    for var in csp.variables:
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
    # Now repeatedly choose a random conflicted variable and change it
    for i in range(max_steps):
        conflicted = csp.conflicted_vars(current)
        if not conflicted:
            return current
        var = random.choice(conflicted)
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
    return None

def min_conflicts_value(csp, var, current):
    """Return the value that will give var the least number of conflicts.
    If there is a tie, choose at random."""
    return argmin_random_tie(csp.domains[var], key=lambda val: csp.nconflicts(var, val, current))

# ______________________________________________________________________________


def tree_csp_solver(csp):
    """[Figure 6.11]"""
    assignment = {}
    root = csp.variables[0]
    X, parent = topological_sort(csp, root)

    csp.support_pruning()
    for Xj in reversed(X[1:]):
        if not make_arc_consistent(parent[Xj], Xj, csp):
            return None

    assignment[root] = csp.curr_domains[root][0]
    for Xi in X[1:]:
        assignment[Xi] = assign_value(parent[Xi], Xi, csp, assignment)
        if not assignment[Xi]:
            return None
    return assignment


def topological_sort(X, root):
    """Returns the topological sort of X starting from the root.
    Input:
    X is a list with the nodes of the graph
    N is the dictionary with the neighbors of each node
    root denotes the root of the graph.
    Output:
    stack is a list with the nodes topologically sorted
    parents is a dictionary pointing to each node's parent
    Other:
    visited shows the state (visited - not visited) of nodes
    """
    neighbors = X.neighbors

    visited = defaultdict(lambda: False)

    stack = []
    parents = {}

    build_topological(root, None, neighbors, visited, stack, parents)
    return stack, parents


def build_topological(node, parent, neighbors, visited, stack, parents):
    """Build the topological sort and the parents of each node in the graph."""
    visited[node] = True

    for n in neighbors[node]:
        if(not visited[n]):
            build_topological(n, node, neighbors, visited, stack, parents)

    parents[node] = parent
    stack.insert(0, node)


def make_arc_consistent(Xj, Xk, csp):
    """Make arc between parent (Xj) and child (Xk) consistent under the csp's constraints,
    by removing the possible values of Xj that cause inconsistencies."""
    #csp.curr_domains[Xj] = []
    for val1 in csp.domains[Xj]:
        keep = False # Keep or remove val1
        for val2 in csp.domains[Xk]:
            if csp.constraints(Xj, val1, Xk, val2):
                # Found a consistent assignment for val1, keep it
                keep = True
                break
        
        if not keep:
            # Remove val1
            csp.prune(Xj, val1, None)

    return csp.curr_domains[Xj]


def assign_value(Xj, Xk, csp, assignment):
    """Assign a value to Xk given Xj's (Xk's parent) assignment.
    Return the first value that satisfies the constraints."""
    parent_assignment = assignment[Xj]
    for val in csp.curr_domains[Xk]:
        if csp.constraints(Xj, parent_assignment, Xk, val):
            return val

    # No consistent assignment available
    return None


# κλάση που μοντελοποιέι το πάζλ και κληρονομεί τη CSP 
class Kenken(CSP):

    def __init__(self, problem, size):
        
        self.size = size

        # αρχικοποίηση των μεταβλητών του προβλήματος, η κάθε μεταβλητή αντιστοιχεί
        # σε ένα κελί του πάζλ και αποθηκεύεται ως (i, j) σε μία λίστα
        self.variables = []

        for i in range(1, size+1):

            for j in range(1, size+1):
                self.variables.append((i, j))

        # αρχικοποίηση της κάθε κλίκας, αποθηκεύοντα σε ένα λεξικό, όπου κάθε μεταβλητή
        # αντιστοιχεί σε ένα tuple στο οποίο το στοιχείο 0 είναι η πράξη, το στοιχείο 1 είναι 
        # το αποτέλεσμα της πράξης και το 3 ένα tuple στο οποίο είναι αποθηκευμένες οι μεταβλητές της κλίκας
        # πχ. {(1, 1): ('+', 4, ((1, 2), (2, 2)))}
        self.cliques = {}
        
        for ken in problem:

            for case in ken[2]:
                
                cliq = []
                cliq.append(ken[0])
                cliq.append(ken[1])                
                cliq.append(ken[2])

                self.cliques[case] = tuple(cliq)

        # αρχικοποίηση των πεδίων, αναπαρίστανται με ένα λεξικό στο οποίο για κάθε κελί,
        # υπάρχει ένα tuple με τις τιμές που μπορεί να πάρει, αρχικοποιείται με τις τιμές από 0 μέρχι το μέγεθός του
        # πχ. {(1, 2): (1, 2, 3, 4, 5, 6)}
        self.domains = {}

        for ken in problem:

            for case in ken[2]:
                dom = []

                for i in range(1, size+1):
                    dom.append(i)

                self.domains[case] = tuple(dom)

        # αρχικοποίηση των γειτόνων, αναπαρίστανται με ένα λεξικό στο οποίο για κάθε μεταβλητή
        # υπάρχει ένα tuple που περιέχει ένα tuple με τις μεταβλητές της γραμμής και της στήλης και ένα 
        # με τις μεταβλητές της κλίκας
        # πχ. {(1, 2): (((1, 1), (1, 3), (1, 4), (2, 2), (3, 2), (4, 2)), ((2, 1), (2, 2)))}
        self.neighbors = {}
        
        for var in self.variables:

            raw_col = []

            for i in range(1, size+1):
                
                raw = (var[0], i)
                if raw != var:
                    raw_col.append(raw)

                col = (i, var[1])
                if col != var:
                    raw_col.append(col)

            neig = []
            neig.append(tuple(raw_col))
            neig.append (tuple(self.cliques[var][2]))
            self.neighbors[var] = tuple(neig)

    # συνάρτηση που εκτυπώνει τη λύση του πάζλ
    def print_puzzle(self, solution):

        for i in range(1, self.size+1):

            row = []
            for j in range(1, self.size+1):
                row.append(solution[(i, j)])
            print(row)
        print('\n')

    # καλείται από τη main συνάρτηση για να λύσει το πρόβλημα
    def kenken_solve(self):

        import time

        # αρχικοποιεί την κλάση του προβλήματος
        c = CSP(self.variables, self.domains, self.neighbors, self.constraint_function)

        self.c =c
        
        start_time = time.time()
        self.constraint_check = 0
        
        # λύνει το πρόβλημα με forward_checking και mrv
        solution = backtracking_search(csp = c, inference = forward_checking, select_unassigned_variable=mrv)
        print("Solution with fc + mrv:")
        self.print_puzzle(solution)

        print("Time:")
        print((time.time() - start_time))
        print("\nAssignments:")
        print(c.nassigns)
        print("\nConstraints:")
        print(self.constraint_check)
        print('\n')
        
        start_time = time.time()
        self.constraint_check = 0
        
        # λύνει το πρόβλημα με mac
        solution = backtracking_search(csp = c, inference = mac)
        print("Solution with mac:")
        self.print_puzzle(solution)

        print("Time:")
        print((time.time() - start_time))
        print("\nAssignments:")
        print(c.nassigns)
        print("\nConstraints:")
        print(self.constraint_check)
        print('\n')
      
        start_time = time.time()
        self.constraint_check = 0
        
        # λύνει το πρόβλημα με min_conflicts
        solution = min_conflicts(csp = c)
        print("Solution with min_conflicts:")
        self.print_puzzle(solution)
        
        print("Time:")
        print((time.time() - start_time))
        print("\nAssignments:")
        print(c.nassigns)
        print("\nConstraints:")
        print(self.constraint_check)
        print('\n')

    # συνάρτηση που επιστρέφει True/False ανάλογα με το αν η τιμές των μεταβλητών πληρούν
    # κάποιες προδιαγραφές, ανάλογα με τους γείτονες
    def constraint_function(self, A, a, B, b):
        
        self.constraint_check += 1

        # αν είναι η ίδια μεταβλητή επέστρεψε False
        if A == B:
            return False

        # αν η μεταβλητή Α είναι γείτονας της Β και ανάποδα, δηλαδή αν είναι στην ίδια γραμμή ή 
        # στήλη, και έχουν την ίδια τιμή, τότε επέστρεψε False
        if (B in self.neighbors[A][0]) & (a == b):
            return False

        # αν στη μεταβλητή πρέπει να γίνει απλή ανάθεση τιμής, ελέγχει και επιστρέφει 
        # True/False ανάλογα με το αν είναι η σωστή τιμή
        if self.cliques[A][0] == '=':

            if a == self.cliques[A][1]:
                return True
            else: 
                return False

        if self.cliques[B][0] == '=':

            if b == self.cliques[B][1]:
                return True 
            else: 
                return False 

        # αν η μία μεταβλητή ανήκει στην κλίκα της άλλης
        # (γίνεται έλεγχος μόνο για τη μία μεταβλητή, μπορεί να παραλειφθεί ο έλεγχος
        # για την αλλή, καθώς αν η μία ανήκει στην κλίκα της άλλης, τότε συμβαίνει και το αντίστροφο)
        if B in self.cliques[A][2]:
            
            # αν η πράξη είναι το άθροισμα
            if self.cliques[A][0] == '+':
                
                # αν το άθροισμα των δύο τιμών των μεταβλητών ξεπερνά το επιθυμητό
                # τότε είναι σίγουρα λάθος και επιστρέφει False
                if a+b > self.cliques[A][1]:
                    return False

                s = 0
                c = 0
                asigm = 0
                # αλλιώς, για κάθε μεταβλητή της κλίκας αθροίζει τις προσωρινές τιμές που έχουν πάρει
                for clique in self.cliques[A][2]:
                    
                    # αν η κάθε μεταβλητή έχει τιμή από προηγούμενη ανάθεση
                    c += 1
                    if clique in self.c.infer_assignment():
                        
                        # αν η μεταβλητή είναι ένα από τα ορίσματα τη συνάρτησης τότε  
                        # τις αθροίζει με τη νέα τους τιμή
                        asigm += 1
                        if clique == A:
                            s += a 
                        elif clique == B:
                            s += b
                        # αλλιώς αθροίζει την τιμή που είχαν
                        else:
                            s += self.c.infer_assignment()[clique]
                    
                    # ελέγχει αν το άθροισμα είναι μεγαλύτερο και επιστρέφει False
                    if s > self.cliques[A][1]:
                        return False

                # αν το αθροισμα είναι ίσο με το επιθυμητό και είναι συμπληρωμένα όλα τα κελιά που
                # συμμετέσουν επιστρέφει True
                if (s == self.cliques[A][1]) & (c == asigm):
                    return True
                # αν το άθροισμα είναι μικρότερο από το επιθυμητό και δεν έχουν αποδοθεί τιμές σε
                # όλες τις μεταβλητές, τότε επιστρέφει True
                elif (s < self.cliques[A][1]) & (c > asigm):
                    return True
                else:
                    return False
                    
            # αν η πράξη είναι το γινόμενο
            if self.cliques[A][0] == '*':
                
                # αν το γινόμενο των δύο τιμών των μεταβλητών ξεπερνά το επιθυμητό
                # τότε είναι σίγουρα λάθος και επιστρέφει False
                if a*b > self.cliques[A][1]:
                    return False

                p = 1
                c = 0
                asigm = 0
                # αλλιώς, για κάθε μεταβλητή της κλίκας πολλαπλασιάζει τις προσωρινές τιμές που έχουν πάρει
                for clique in self.cliques[A][2]:

                    # αν η κάθε μεταβλητή έχει τιμή από προηγούμενη ανάθεση
                    c += 1
                    if clique in self.c.infer_assignment():
                        
                        # αν η μεταβλητή είναι ένα από τα ορίσματα τη συνάρτησης τότε κάνει 
                        # τον πολλαπλασιασμό με τη νέα τους τιμή
                        asigm += 1
                        if clique == A:
                            p *= a 
                        elif clique == B:
                            p *= b 
                        # αλλιώς κάνει τον πολλαπλασιασμό με την τιμή που είχαν
                        else:
                            p *= self.c.infer_assignment()[clique]

                    # ελέγχει αν το γινόμενο είναι μεγαλύτερο και επιστρέφει False
                    if p > self.cliques[A][1]:
                        return False

                # αν το γινόμενο είναι ίσο με το επιθυμητό και είναι συμπληρωμένα όλα τα κελιά που
                # συμμετέσουν επιστρέφει True
                if (p == self.cliques[A][1]) & (c == asigm):
                    return True
                # αν το γινόμενο είναι μικρότερο από το επιθυμητό και δεν έχουν αποδοθεί τιμές σε
                # όλες τις μεταβλητές, τότε επιστρέφει True
                elif (p < self.cliques[A][1]) & (c > asigm):
                    return True
                else:
                    return False
            
            # αν η πράξη είναι η αφαίρεση
            if self.cliques[A][0] == '-':
                
                # αν η απόλυτη τιμή της αφαίρεσης των δύο αριθμών είναι μικρότερη από την επιθυμητή επιστρέφει False
                if math.fabs(a-b) < self.cliques[A][1]:
                    return False
                
                # αν η απόλυτη τιμή της αφαίρεσης των δύο αριθμών είναι μεγαλύτερη από την επιθυμητή επιστρέφει False
                if math.fabs((a)-(b)) > self.cliques[A][1]:
                    return False

                return True

            # αν η πράξη είναι η διαίρεση
            if self.cliques[A][0] == '/':
                
                # αν ο μεγαλύτερος αριθμός διά τον μικρότερο αριθμό από τους δύο, είναι ίσο με το 
                # επιθυμητό αποτέλεσμα επιστρέφει True, αλλιώς False
                if max(a, b)/min(a, b) == self.cliques[A][1]:
                    return True
                else:
                    return False

            return False
        return True


if __name__ == "__main__":

    # ο χρήστης επιλέγει πληκτρολογώντας τον κατάλληλο αριθμό
    # το πάζλ που επιθυμεί να λύσει
    print("\n please choose a game: \n")

    print("1. Problem 3x3")
    print("2. Problem 4x4 (1)")
    print("3. Problem 4x4 (2)")
    print("4. Problem 5x5 (1)")
    print("5. Problem 5x5 (2)")
    print("6. Problem 6x6")
    print("7. Problem 7x7")
    print("8. Problem 8x8")

    game = int(input())

    if game == 1:
    # -------------Problem 3x3----------------------

        problem = (\
            ('+', 3, ((1, 1), (1, 2))),\
            ('-', 1, ((1, 3), (2, 3))),\
            ('-', 2, ((2, 1), (3, 1))),\
            ('*', 6, ((2, 2), (3, 2))),\
            ('=', 1, ((3, 3),)))

        size = 3

        k = Kenken(problem, size)
        k.kenken_solve()

        # solution: 
        # [2, 1, 3]
        # [1, 3, 2]
        # [3, 2, 1]

        # ----------------------------------------------
    elif game == 2 :
        # -------------Problem 4x4 (1)----------------------

        problem = (\
            ('/', 2, ((1, 1), (2, 1))),\
            ('+', 7, ((1, 2), (1, 3))),\
            ('*', 48, ((1, 4), (2, 4), (3, 4), (3, 3))),\
            ('+', 6, ((2, 2), (3, 2), (4, 2))),\
            ('=', 1, ((2, 3),)),\
            ('-', 1, ((3, 1), (4, 1))),\
            ('-', 3, ((4, 3), (4, 4))))

        size = 4

        k = Kenken(problem, size)
        k.kenken_solve()

        # solution:
        # [1, 4, 3, 2]
        # [2, 3, 1, 4]
        # [4, 1, 2, 3]
        # [3, 2, 4, 1]

        # ----------------------------------------------
    elif game == 3:
        # -------------Problem 4x4 (2)----------------------

        problem = (\
            ('+', 7, ((1, 1), (2, 1))),\
            ('-', 2, ((1, 2), (2, 2))),\
            ('*', 4, ((1, 4), (1, 3), (2, 3))),\
            ('-', 1, ((2, 4), (3, 4))),\
            ('/', 2, ((3, 1), (3, 2))),\
            ('*', 12, ((3, 3), (4, 3), (4, 4))),\
            ('-', 2, ((4, 1), (4, 2))))

        size = 4

        k = Kenken(problem, size)
        k.kenken_solve()

        # solution:
        # [4, 3, 1, 2]
        # [3, 1, 2, 4]
        # [1, 2, 4, 3]
        # [2, 4, 3, 1]

        # ----------------------------------------------
    elif game == 4:
        # -------------Problem 5x5 (1)----------------------

        problem = (\
            ('+', 9, ((1, 1), (1, 2))),\
            ('+', 8, ((1, 3), (2, 3))),\
            ('+', 3, ((1, 4), (2, 4))),\
            ('=', 2, ((1, 5),)),\
            ('+', 9, ((2, 1), (3, 1), (4, 1))),\
            ('+', 7, ((2, 2), (3, 2))),\
            ('+', 5, ((2, 5), (3, 5))),\
            ('-', 1, ((3, 3), (4, 3))),\
            ('+', 7, ((3, 4), (4, 4))),\
            ('=', 2, ((4, 2),)),\
            ('-', 2, ((4, 5), (5, 5))),\
            ('+', 3, ((5, 1), (5, 2))),\
            ('+', 9, ((5, 3), (5, 4))))

        size = 5

        k = Kenken(problem, size)
        k.kenken_solve()

        # solution:
        # [4, 5, 3, 1, 2]
        # [1, 3, 5, 2, 4]
        # [5, 4, 2, 3, 1]
        # [3, 2, 1, 4, 5]
        # [2, 1, 4, 5, 3]
                
        # ----------------------------------------------
    elif game == 5:
        # -------------Problem 5x5 (2)----------------------

        problem = (\
            ('*', 8, ((1, 1), (1, 2), (1, 3))),\
            ('+', 8, ((1, 4), (1, 5))),\
            ('-', 1, ((2, 1), (3, 1))),\
            ('-', 1, ((2, 2), (2, 3))),\
            ('-', 4, ((2, 4), (2, 5))),\
            ('+', 7, ((3, 2), (3, 3), (4, 3))),\
            ('-', 2, ((3, 4), (3, 5))),\
            ('-', 1, ((4, 1), (5, 1))),\
            ('-', 2, ((4, 2), (5, 2))),\
            ('=', 4, ((4, 4),)),\
            ('/', 2, ((5, 3), (5, 4))),\
            ('-', 1, ((4, 5), (5, 5))))

        size = 5

        k = Kenken(problem, size)
        k.kenken_solve()

        # solution:
        # [1, 2, 4, 3, 5]
        # [2, 4, 3, 5, 1]
        # [3, 1, 5, 2, 4]
        # [5, 3, 1, 4, 2]
        # [4, 5, 2, 1, 3]
                
        # ----------------------------------------------
    elif game == 6:
        # -------------Problem 6x6----------------------

        problem = (\
            ('/', 2, ((1, 1), (2, 1))),\
            ('/', 2, ((1, 2), (2, 2))),\
            ('/', 2, ((1, 3), (2, 3))),\
            ('=', 4, ((1, 4),)),\
            ('-', 2, ((1, 5), (2, 5))),\
            ('*', 30, ((1, 6), (2, 6), (3, 6))),\
            ('-', 2, ((2, 4), (3, 4))),\
            ('-', 2, ((3, 1), (3, 2))),\
            ('+', 16, ((3, 3), (4, 3), (4, 2))),\
            ('+', 7, ((3, 5), (4, 5), (4, 4))),\
            ('/', 3, ((4, 1), (5, 1))),\
            ('*', 6, ((4, 6), (5, 6))),\
            ('/', 2, ((5, 2), (5, 3))),\
            ('-', 1, ((5, 4), (5, 5))),\
            ('-', 4, ((6, 1), (6, 2))),\
            ('-', 1, ((6, 3), (6, 4))),\
            ('*', 24, ((6, 5), (6, 6))))

        size = 6

        k = Kenken(problem, size)
        k.kenken_solve()

        # solution: 
        # [2, 6, 1, 4, 3, 5]
        # [4, 3, 2, 5, 1, 6]
        # [6, 4, 5, 3, 2, 1]
        # [3, 5, 6, 1, 4, 2]
        # [1, 2, 4, 6, 5, 3]
        # [5, 1, 3, 2, 6, 4]

        # ----------------------------------------------
    elif game == 7:
        # -------------Problem 7x7----------------------

        problem = (\
            ('-', 3, ((1, 1), (1, 2))),\
            ('+', 5, ((1, 3), (1, 4))),\
            ('/', 2, ((1, 5), (1, 6))),\
            ('*', 126, ((1, 7), (2, 7), (2, 6))),\
            ('-', 3, ((2, 1), (2, 2))),\
            ('+', 7, ((2, 3), (2, 4))),\
            ('-', 1, ((2, 5), (3, 5))),\
            ('-', 5, ((3, 1), (4, 1))),\
            ('*', 105, ((3, 2), (4, 2), (5, 2))),\
            ('-', 1, ((3, 3), (3, 4))),\
            ('-', 3, ((3, 6), (4, 6))),\
            ('+', 6, ((3, 7), (4, 7), (5, 7))),\
            ('+', 8, ((4, 3), (4, 4))),\
            ('/', 2, ((4, 5), (5, 5))),\
            ('+', 10, ((5, 1), (6, 1))),\
            ('/', 2, ((5, 3), (5, 4))),\
            ('-', 3, ((5, 6), (6, 6))),\
            ('+', 12, ((6, 2), (7, 2), (7, 1))),\
            ('*', 42, ((6, 3), (7, 3))),\
            ('-', 1, ((6, 4), (6, 5))),\
            ('+', 10, ((6, 7), (7, 7), (7, 6))),\
            ('-', 2, ((7, 4), (7, 5))))

        size = 7

        k = Kenken(problem, size)
        k.kenken_solve()

        # solution:
        # [5, 2, 4, 1, 3, 6, 7]
        # [4, 1, 2, 5, 7, 3, 6]
        # [1, 7, 5, 4, 6, 2, 3]
        # [6, 3, 1, 7, 4, 5, 2]
        # [7, 5, 3, 6, 2, 4, 1]
        # [3, 4, 6, 2, 1, 7, 5]
        # [2, 6, 7, 3, 5, 1, 4]

        # ----------------------------------------------
    elif game == 8:
        # -------------Problem 8x8----------------------

        problem = (\
            ('/', 3, ((1, 1), (1, 2))),\
            ('-', 7, ((1, 3), (1, 4))),\
            ('-', 2, ((1, 5), (1, 6))),\
            ('+', 17, ((1, 7), (1, 8), (2, 8))),\
            ('*', 168, ((2, 1), (2, 2), (3, 1), (3, 2))),\
            ('*', 40, ((2, 3), (3, 3), (3, 4))),\
            ('+', 3, ((2, 4), (2, 5))),\
            ('*', 168, ((2, 6), (2, 7), (3, 7))),\
            ('*', 126, ((3, 5), (4, 5), (4, 6))),\
            ('=', 5, ((3, 6),)),\
            ('-', 6, ((3, 8), (4, 8))),\
            ('*', 20, ((4, 1), (4, 2))),\
            ('+', 12, ((4, 3), (5, 2), (5, 3))),\
            ('-', 3, ((4, 4), (5, 4))),\
            ('+', 12, ((4, 7), (5, 7), (5, 8))),\
            ('-', 7, ((5, 1), (6, 1))),\
            ('*', 168, ((5, 5), (5, 6), (6, 5))),\
            ('=', 4, ((6, 2),)),\
            ('-', 3, ((6, 3), (7, 3))),\
            ('-', 1, ((6, 4), (7, 4))),\
            ('+', 10, ((6, 6), (6, 7), (6, 8))),\
            ('+', 14, ((7, 1), (7, 2), (8, 1))),\
            ('/', 4, ((7, 5), (8, 5))),\
            ('-', 7, ((7, 6), (8, 6))),\
            ('-', 4, ((7, 7), (7, 8))),\
            ('*', 126, ((8, 2), (8, 3), (8, 4))),\
            ('-', 2, ((8, 7), (8, 8))))

        size = 8

        k = Kenken(problem, size)
        k.kenken_solve()

        # ----------------------------------------------

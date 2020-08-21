# %%
from qiskit import QuantumCircuit, transpile, QuantumRegister, Aer, execute
from qiskit import ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_state_qsphere, plot_histogram
import numpy as np
import pandas as pd
from functools import reduce
from typing import Dict, Sequence, Union, Tuple, List, Any, Optional

Counts = Dict[str, int]


# Cube faces indexing (flattened cube). The idea is to associate
# to each tile a unique identifier and implement the moves by using
# unitaries obtained directly from permutation matrices.
#
#                 -----------
#               B 00000 00001 |
#               | 00010 00011 |
#   -L---------  -----------    -R---------   -D'--------
# | 00100 00101 | 00110 00111 | 01000 01001 | 10111 10110 |
# | 01010 01011 | 01100 01101 | 01110 01111 | 10101 10100 |
#   -----------   ------------   -----------  -----------
#               F 10000 10001 |
#               | 10010 10011 |
#                 -----------
#               D 10100 10101 |
#               | 10110 10111 |
#                 -----------

# %%

cube_state_reg = QuantumRegister(5, name='cube')    # Registry containing the state of the stickers
# Moves scheme. This is the sequence of permutation that are generated and controlled by the permutations
# controlling qubits. The identifiers are of the form {move_name}_{idx}' where move_name is for example
# 'u1' and the idx is meant to make the registry name unique when the same operation is repeated.

moves_scheme = ['u1_0','u2_0', 'r1_0', 'r2_0', 'u1_1', 'u2_1', 'r1_1', 'r2_1']

def _get_op_name(op_spec: str):
    i = op_spec.find('_')
    return op_spec if i < 0 else op_spec[:i]


def prepare_ctrl_perms(operators: Sequence[str]) -> Tuple[QuantumCircuit, List[QuantumRegister]]:
    """
    Prepare the circuit of controlled permutations. Operators name are of the form: '{move_name}_{idx}'
    where move_name is for example 'u1' and the idx is meant to make the registry name unique when the same
    operation is repeated.
    """
    operators = list(operators)
    regs = [QuantumRegister(1, name=str(k)) for k in operators]     # Create the control qubits for the moves
    perms = [control(move(_get_op_name(op)), c_qreg=reg) for op, reg in zip(operators, regs)]    # Create controlled moves

    # Combine the circuits into a single one.
    qc = QuantumCircuit(cube_state_reg, *regs)  # cube_state_reg must be located on the LSB
    qc = reduce(lambda q1, q2: q1.combine(q2), perms, qc)
    return qc, regs


def face_id_to_idx(v: Union[str, int]) -> int:
    """
    Translate a string form of a face identifier to an integer indexing the
    corresponding computational basis vector. If the input is already an integer
    then this function acts as an identify.
    """
    if isinstance(v, str):
        return int(v, 2)
    return int(v)


def cube_conf_init(perm=None) -> Dict[str, int]:
    """
    Create a dict containing the cube in resolved configuration. The keys represent
    the face ids whereas the values represent the colors.
    :param perm: An optional permutation to be applied to the cube in resolved state.
    A possible perm may be PERM_U1 + PERM_U2, note that the order matters.
    """
    state = {
        '00000': 0, '00001': 0, '00010': 0, '00011': 0,
        '00100': 1, '00101': 1, '01010': 1, '01011': 1,
        '00110': 2, '00111': 2, '01100': 2, '01101': 2,
        '01000': 3, '01001': 3, '01110': 3, '01111': 3,
        '10000': 4, '10001': 4, '10010': 4, '10011': 4,
        '10100': 5, '10101': 5, '10110': 5, '10111': 5
    }
    return state if perm is None else classic_perm_apply(perm, state)


def classic_perm_apply(cycles: List[List[str]], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply classically the permutation specified by the given list of cycles to the dictionary
    representing the state of the entity on which the permutation acts. Contrary to the usual
    convention regarding the order of application of cycles (see symmetric group) here the cycles
    are applied starting from the index 0.
    """
    state1 = {}
    for obj1 in state.keys():
        obj2 = obj1
        for c in cycles:
            try:
                i = c.index(obj2)
            except ValueError:
                continue
            obj2 = c[(i + 1) % len(c)]
        state1[obj2] = state[obj1]
    assert len(state) == len(state1)
    # Re-iterate through state1 to replicate the keys ordering from state
    return {k: state1[k] for k in state.keys()}


def control(circuit, label=None, c_qreg=None) -> QuantumCircuit:
    """
    Create a controlled instance of a given circuit.
    :param c_qreg: The registry to be used to control the given circuit.
    """
    c_qreg = QuantumRegister(1) if c_qreg is None else c_qreg

    gate = circuit.to_gate()
    c_gate = gate.control(c_qreg.size, label=label)

    # Apparently Qiskit original call QuantumCircuit.control does not allow
    # to pass the control registry so here we re-implement the same operation
    # with the optional user provided control registry.
    c_circ = QuantumCircuit(c_qreg, *circuit.qregs, name='c_{}'.format(circuit.name))
    c_circ.append(c_gate, c_circ.qubits)
    return c_circ


def create_cycle(cycle: Sequence[Union[str, int]]) -> QuantumCircuit:
    """
    Create a circuit implementing the cyclic permutation of the faces
    of the cube using the given sequence. The input list contains the
    sequence of faces in the cycle.
    For additional info check the cycle notation on
    https://en.wikipedia.org/wiki/Symmetric_group
    """
    cycle = list(map(face_id_to_idx, cycle))
    label = f'cycle{str(cycle)}'

    qc = QuantumCircuit(cube_state_reg)
    p = np.eye(2**5)

    # Here we rearrange the columns of the identity matrix according to
    # the cycle to be implemented.
    first_col = np.copy(p[:, cycle[0]])
    for i in range(len(cycle) - 1):
        p[:, cycle[i]] = p[:, cycle[i + 1]]     # Map i -> i+1
    p[:, cycle[-1]] = first_col

    # Verify that P is a permutation matrix.
    assert np.all(np.sum(p, axis=0) == 1) and np.all(np.sum(p, axis=1) == 1)

    # P is a permutation matrix, then P^{-1}=P^T and has real entries, thus P^H=P^T,
    # also PP^H=p^H P=I, hence P is unitary.
    qc.unitary(p, list(np.arange(5)), label=label)
    # By creating an arbitrary unitary (although these are all permutation matrices) we are
    # relaying on the transpiler to translate this operation into a sequence of basic gates.
    # Of course, this is a temporary solution.
    return qc


def create_qperm(*args) -> QuantumCircuit:
    """
    Create a permutation circuit given a list of computational basis cycles.
    See function create_cycle for further information.
    """
    qc = QuantumCircuit(cube_state_reg)
    for cycle in args:
        qc = qc.combine(create_cycle(cycle))
    return qc


def move(name: str) -> QuantumCircuit:
    """
    Create a permutation circuit given the name of the operation. Possible
    names are for example 'u1' and 'u2'.
    """
    fun = globals().get(f'move_{name}')
    return fun()


# U is formed by 3 disjoint cycles
PERM_U1 = (
    ('00110', '00111', '01101', '01100'),   # Upper face
    ('00010', '01000', '10001', '01011'),   # Upper side cycle 1
    ('00011', '01110', '10000', '00101')    # Upper side cycle 2
)

# U^2 is formed by 6 transpositions
PERM_U2 = (
    ('00110', '01101'),     # Upper face 1
    ('00111', '01100'),     # Upper face 2
    ('00010', '10001'),     # Upper side 1
    ('00011', '10000'),     # Upper side 2
    ('01000', '01011'),     # Upper side 3
    ('01110', '00101')      # Upper side 4
)

def move_u1() -> QuantumCircuit:
    """
    Create the permutation corresponding to the move U, i.e. a 90 degrees
    clockwise rotation of the top face. The returned circuit is meant to be
    'controlled' so that to obtain a superposition of permutations.
    """
    return create_qperm(*PERM_U1)


def move_u2() -> QuantumCircuit:
    """
    Create the permutation corresponding to the move U^2, i.e. a 180 degrees
    clockwise rotation of the top face. Note that u1 and u2 are sufficient to
    generate: e (identity), u1, u2, u3=u2*u1.
    """
    return create_qperm(*PERM_U2)


# R is formed by 3 disjoint cycles
PERM_R1 = (
    ('01000', '01001', '01111', '01110'),   # Face
    ('10111', '10011', '01101', '00011'),   # Side cycle 1
    ('10101', '10001', '00111', '00001')    # Side cycle 2
)

# R^2 is formed by 6 transpositions
PERM_R2 = (
    ('01000', '01111'),
    ('01001', '01110'),
    ('10111', '01101'),
    ('10101', '00111'),
    ('10011', '00011'),
    ('10001', '00001')
)


def move_r1() -> QuantumCircuit:
    """
    Create the permutation corresponding to the move R, i.e. a 90 degrees
    clockwise rotation of the right face.
    """
    return create_qperm(*PERM_R1)


def move_r2() -> QuantumCircuit:
    """
    Create the permutation corresponding to the move R^2, i.e. a 180 degrees
    clockwise rotation of the right face.
    """
    return create_qperm(*PERM_R2)


# F is formed by 3 disjoint cycles
PERM_F1 = (
    ('10000', '10001', '10011', '10010'),   # Front
    ('01010', '01100', '01110', '10101'),   # Front Side cycle 1
    ('01011', '01101', '01111', '10100')    # Front Side cycle 2
)

PERM_F2 = (
    ('10000', '10011'),
    ('10001', '10010'),
    ('01010', '01110'),
    ('01100', '10101'),
    ('01011', '01111'),
    ('01101', '10100')
)


def move_f1() -> QuantumCircuit:
    """
    Create the permutation corresponding to the move F, i.e. a 90 degrees
    clockwise rotation of the front face.
    """
    return create_qperm(*PERM_F1)


def move_f2() -> QuantumCircuit:
    """
    Create the permutation corresponding to the move F^2, i.e. a 180 degrees
    clockwise rotation of the front face.
    """
    return create_qperm(*PERM_F2)


def prepare_state(faces: Dict[str, int]) -> QuantumCircuit:
    """
    Prepare the state corresponding to the cube having the faces in the given
    configuration. Check function cube_conf_init for move info in regards to the
    format of the faces specifications.
    """
    assert len(faces) == 24
    faces = list(map(lambda tt: (face_id_to_idx(tt[0]), int(tt[1])), faces.items()))
    faces = pd.DataFrame(faces, dtype=int).sort_values(by=0)
    faces = faces[1].to_numpy()
    faces = np.concatenate([faces, np.zeros(8, dtype=np.int)])
    assert len(faces) == 32

    # We implement a diagonal operator to associate, to each basis vector corresponding
    # to a face of the cube, a phase that characterizes the color. Colors are
    # indexed by integers {0, 1, ..., 5}, so given a color k, we use the function
    # f(k) = e^{2i\pi k / 6} to compute each diagonal entry.
    qc = QuantumCircuit(cube_state_reg)
    qc.h(cube_state_reg)
    faces = np.exp(faces * np.pi * 1j/3)    # e^{2i\pi k / 6}
    qc.diagonal(list(faces), list(np.arange(5)))
    return qc


def filter_counts(counts: Counts) -> Counts:
    """
    Filter the output counts so that we obtain a distribution of the moves. Note that we are
    expecting the state representing the cube configuration to collapse to |00000> when the
    permutations implemented by the moves transform the initial state into the final one. 
    """
    output = {}
    for k, v in counts.items():
        # Extract the items where the 5 LSBs are 0.
        if k[-5:] == '00000':
            output[k[:-5]] = v
    return output


def simulate_experiment(cube_conf: Optional[Dict[str, int]] = None) -> Counts:
    if cube_conf is None:
        cube_conf = cube_conf_init()

    perms_qc, move_ctrl_regs = prepare_ctrl_perms(moves_scheme)

    qc = QuantumCircuit(cube_state_reg, *move_ctrl_regs)  # cube_state_reg must be located on the LSB
    # Note we don't declare a classical registry because we use measure_all.
    for reg in move_ctrl_regs:
        qc.h(reg)     # Create superposition of moves control
    qc = qc.combine(prepare_state(cube_conf))    # Prepare initial state (w.r.t cube)

    # Controlled permutations
    qc = qc.combine(perms_qc)

    qc = qc.combine(prepare_state(cube_conf_init()).inverse())  # Create the inverse of the final state
    qc.measure_all()

    # Run simulation and get counts
    counts = execute(qc, Aer.get_backend('qasm_simulator'), shots=2**13).result().get_counts()
    counts = filter_counts(counts)
    return counts


def interpret_counts_for_gui(counts: Counts) -> str:
    label = pd.Series(counts).idxmax()
    label = list(reversed(label))

    output = []
    assert len(label) == len(moves_scheme)
    for b, m in zip(label, moves_scheme):
        if b == '1':
            m = _get_op_name(m)
            m = m[:-1] if m.endswith('1') else m  # Adapt to GUI convention
            m = m.upper()
            output.append(m)
            print(output)
    return ' '.join(output)


from Bio.PDB import PDBParser, PDBIO, Select


class SequenceSelector(Select):
    """
    A custom filter to extract a specific sequence range from a specific chain.
    """

    def __init__(self, target_chain, start_res, end_res):
        self.target_chain = target_chain
        self.start_res = start_res
        self.end_res = end_res

    def accept_residue(self, residue):
        # 1. Check if the residue belongs to the target chain
        chain_id = residue.get_parent().id
        if chain_id != self.target_chain:
            return 0  # 0 means reject

        # 2. Get the residue sequence number
        # get_id() returns a tuple: (hetero-flag, sequence_identifier, insertion_code)
        res_seq_num = residue.get_id()[1]

        # 3. Check if the residue number falls within your target sequence range
        if self.start_res <= res_seq_num <= self.end_res:
            return 1  # 1 means accept

        return 0


# --- Execution ---

input_file = "/home/vu2002123/target-elucidation/data/HPC_input/AF_v6/AF-Q9Y6V0-F2-model_v6.pdb"

output_file = "/home/vu2002123/target-elucidation/data/HPC_input/AF_v6/P14416_IPR017452_51_426.pdb"

parser = PDBParser(QUIET=True)
structure = parser.get_structure("Q9Y6V0", input_file)

# 2. Define the parameters for the sequence you want to extract
CHAIN_TO_EXTRACT = "A"
START_RESIDUE = 51
END_RESIDUE = 426

# 3. Setup the IO and save the filtered structure
io = PDBIO()
io.set_structure(structure)

# Pass our custom SequenceSelector to the save method
io.save(output_file, SequenceSelector(CHAIN_TO_EXTRACT, START_RESIDUE, END_RESIDUE))

print("Extraction complete!")

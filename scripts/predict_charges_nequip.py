import os
import numpy as np
from ase.io import read
from nequip.train import EMALightningModule
from nequip.ase import NequIPCalculator
from nequip.data.transforms import ChemicalSpeciesToAtomTypeMapper, NeighborListTransform
import nequip_les  # noqa: registers LES field keys

CHARGE_FIELD_NAMES = {'charge', 'q', 'charges', 'initial_charges', 'mbi_charges'}

CHECKPOINT = 'outputs/electrolyte_les/best.ckpt'
XYZ_PATH = os.path.join(os.environ['SCRATCH'], 'code/les/NewtonNet/scripts/electrolyte_data/test/raw/electrolyte_test.xyz')
TYPE_NAMES = ['H', 'O', 'K', 'F']
R_MAX = 4.5
DEVICE = 'cuda'
FRAME_INDEX = 0


def read_dft_charges(xyz_path, frame_index=0):
    with open(xyz_path) as f:
        i = 0
        while True:
            line = f.readline()
            if not line:
                return None, None
            n = int(line.strip())
            comment = f.readline()
            atom_lines = [f.readline() for _ in range(n)]
            if i == frame_index:
                props_str = [p for p in comment.split() if p.startswith('Properties=')]
                if not props_str:
                    return None, None
                fields = props_str[0].split('=', 1)[1].split(':')
                col = 0
                field_name = None
                for j in range(0, len(fields) - 2, 3):
                    name, typ, count = fields[j], fields[j+1], int(fields[j+2])
                    if name in CHARGE_FIELD_NAMES:
                        field_name = name
                        break
                    col += count
                if field_name is None:
                    return None, None
                charges = np.array([float(l.split()[col]) for l in atom_lines])
                return charges, field_name
            i += 1


module = EMALightningModule.load_from_checkpoint(CHECKPOINT, map_location=DEVICE)
model = module.model
model.eval()

transforms = [
    ChemicalSpeciesToAtomTypeMapper(model_type_names=TYPE_NAMES),
    NeighborListTransform(r_max=R_MAX),
]
calc = NequIPCalculator(model=model, device=DEVICE, transforms=transforms)

atoms = read(XYZ_PATH, index=FRAME_INDEX)
dft, field_name = read_dft_charges(XYZ_PATH, frame_index=FRAME_INDEX)

atoms.calc = calc
atoms.get_potential_energy()

charges = calc.results.get('LES_q')
if charges is None:
    print("LES_q not found in calc.results. Available keys:", list(calc.results.keys()))
    exit(1)

charges = np.array(charges).flatten()
symbols = atoms.get_chemical_symbols()

if dft is not None:
    print(f"DFT charge field: '{field_name}'")
    print(f"{'Atom':<6} {'Symbol':<8} {'Predicted':>12} {'DFT':>10}")
    print("-" * 38)
    for i, (sym, q_pred, q_dft) in enumerate(zip(symbols, charges, dft)):
        print(f"{i:<6} {sym:<8} {q_pred:>12.4f} {q_dft:>10.4f}")
else:
    print(f"{'Atom':<6} {'Symbol':<8} {'Predicted':>12}")
    print("-" * 28)
    for i, (sym, q) in enumerate(zip(symbols, charges)):
        print(f"{i:<6} {sym:<8} {q:>12.4f}")

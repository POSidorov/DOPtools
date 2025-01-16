# -*- coding: utf-8 -*-
#
#  Copyright 2022-2025 Pavel Sidorov <pavel.o.sidorov@gmail.com> This
#  file is part of DOPTools repository.
#
#  DOPtools is free software; you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program; if not, see <https://www.gnu.org/licenses/>.


def _gather_ct_stereos(reaction):
    res = {}
    for r in reaction.reactants:
        if r._cis_trans_stereo:
            res.update([(k, (v, 'r')) for k, v in r._cis_trans_stereo.items()])
    for p in reaction.products:
        if p._cis_trans_stereo:
            res.update([(k, (v, 'p')) for k, v in p._cis_trans_stereo.items()])
    for k in res.keys():
        if k in r._cis_trans_stereo and k in p._cis_trans_stereo and r._cis_trans_stereo[k]==p._cis_trans_stereo[k]:
            res.pop(k)
    return res


def _gather_rs_stereos(reaction):
    res = {}
    for r in reaction.reactants:
        if r._atoms_stereo:
            res.update([(k, (v, 'r')) for k, v in r._atoms_stereo.items()])
    for p in reaction.products:
        if p._atoms_stereo:
            res.update([(k, (v, 'p')) for k, v in p._atoms_stereo.items()])
    return res


def _pos_in_string(cgr, cgr_string, number):
    index1 = 0
    order = 0
    atom_number = cgr.smiles_atoms_order[order]
    while order <= cgr.smiles_atoms_order.index(number):
        atom_symbol = cgr._atoms[atom_number].atomic_symbol
        index2 = cgr_string.index(atom_symbol, index1, len(cgr_string))
        if index2 == index1:
            index1 += 1
        else:
            index1 = index2+1
        order += 1
        atom_number = cgr.smiles_atoms_order[order]
    return index1


def _pos_in_string_atom(cgr, cgr_string, number):
    index1 = 0
    order = 0
    atom_number = cgr.smiles_atoms_order[order]
    while order <= cgr.smiles_atoms_order.index(number):
        atom_symbol = cgr._atoms[atom_number].atomic_symbol
        index2 = cgr_string.index(atom_symbol, index1, len(cgr_string))
        if index2 == index1:
            index1 += 1
        else:
            index1 = index2+1
        order +=1
        if order < len(cgr.smiles_atoms_order):
            atom_number = cgr.smiles_atoms_order[order]
    return index1-1


def _add_stereo_substructure(substructure, reaction):
    substructure_atoms = list(substructure._atoms)
    cts = _gather_ct_stereos(reaction)
    rss = _gather_rs_stereos(reaction)
    cgr_smiles = str(substructure)
    new_smiles = cgr_smiles
    for atoms, stereo in cts.items():
        if atoms[0] in substructure._atoms and atoms[1] in substructure._atoms:
            if len(substructure.int_adjacency[atoms[0]]) > 1 and len(substructure.int_adjacency[atoms[1]]) > 1:
                bond_string = substructure._format_bond(atoms[0], atoms[1], 0)
                if '>' not in bond_string:
                    continue
                index1, index2 = _pos_in_string(substructure, cgr_smiles, atoms[0]), _pos_in_string(substructure, cgr_smiles, atoms[1])
                if stereo[1] == 'r':
                    bond_index = cgr_smiles.index("=>", min(index1, index2), max(index1, index2))
                else:
                    bond_index = cgr_smiles.index(">=", min(index1, index2), max(index1, index2))+1
                if stereo[0]:
                    new_smiles = cgr_smiles[:bond_index]+"/=\\"+cgr_smiles[bond_index+1:]
                else:
                    new_smiles = cgr_smiles[:bond_index]+"/=/"+cgr_smiles[bond_index+1:]
    for atoms, stereo in rss.items():
        if atoms in substructure._atoms:
            if len(substructure.int_adjacency[atoms])>1:
                atom_string = substructure._format_atom(atoms, 0)
                index1 = _pos_in_string_atom(substructure, new_smiles, atoms)
                index2 = index1+1
                if index1-1>=0 and cgr_smiles[index1-1] == '[':
                    index1 = index1-1
                    index2 = new_smiles.index(']', index1, len(new_smiles))
                else:
                    atom_string = '[' + atom_string + ']'
                if stereo[1] == 'r':
                    if stereo[0]:
                        atom_string = atom_string.replace(']', '@>*]')
                    else:
                        atom_string = atom_string.replace(']', '@@>*]')
                else:
                    if stereo[0]:
                        atom_string = atom_string.replace(']', '*>@]')
                    else:
                        atom_string = atom_string.replace(']', '*>@@]')
                new_smiles = new_smiles[:index1]+atom_string+new_smiles[index2:]    
                
    return new_smiles
